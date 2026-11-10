"""Persistent GRPO receiver lifecycle and stale-prefix rejection contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from qwen3.checkpoints import atomic_json
from qwen3.vllm_decode import (VLLMDecoder, bridge_digest, decoder_session,
                               training_decoder_scope)


class FakeWorker:
    def __init__(self, bridge, directory, **options):
        self.bridge = bridge
        self.directory = Path(directory) / 'worker'
        self.directory.mkdir(parents=True)
        self.counter = 0
        self.process = SimpleNamespace(pid=123)
        self.events = []
        self.closed = False
        self.sleeping = False

    def activate(self):
        assert not self.closed
        self.identity = bridge_digest(self.bridge)
        self.sleeping = False
        self.events.append(('activate', self.identity))

    def sleep(self):
        assert not self.closed
        self.sleeping = True
        self.events.append(('sleep', self.identity))

    def close(self):
        assert not self.closed
        self.closed = True
        self.events.append(('close', None))


def test_reuses_receiver_refreshes_bridge_and_sleeps_between_updates():
    bridge = SimpleNamespace(P=torch.nn.Linear(2, 2, bias=False))
    with TemporaryDirectory() as directory, patch(
            'qwen3.vllm_decode.VLLMDecoder', side_effect=FakeWorker) as factory:
        root = Path(directory)
        identities = []
        with training_decoder_scope(bridge, root / 'engine', max_model_len=3072):
            for step in range(3):
                with torch.no_grad():
                    bridge.P.weight.add_(1.)
                with decoder_session(bridge, root / str(step), max_model_len=64) as worker:
                    assert not worker.sleeping
                    identities.append(worker.identity)
                    worker.counter += 1
                assert worker.sleeping and not worker.closed
                record = json.loads((root / str(step) / 'worker-reuse.json').read_text())
                assert record['first_request'] == record['last_request'] == step + 1
                assert record['checkpoint_sha256'] == identities[-1]
            assert factory.call_count == 1
        assert len(set(identities)) == 3
        assert worker.closed
        assert [event for event, _ in worker.events] == ['activate', 'sleep'] * 3 + ['close']


@pytest.mark.parametrize('failure', ['generate', 'activate', 'sleep'])
def test_failure_closes_worker_and_clears_scope(failure):
    bridge = SimpleNamespace(P=torch.nn.Linear(1, 1))
    with TemporaryDirectory() as directory:
        created = []
        def factory(*args, **kwargs):
            worker = FakeWorker(*args, **kwargs)
            created.append(worker)
            if failure != 'generate':
                def fail():
                    raise RuntimeError('injected failure')
                setattr(worker, failure, fail)
            return worker
        with patch('qwen3.vllm_decode.VLLMDecoder', side_effect=factory):
            with pytest.raises(RuntimeError, match='injected failure'):
                with training_decoder_scope(bridge, directory, max_model_len=100):
                    with decoder_session(bridge, Path(directory) / 'phase', max_model_len=50):
                        if failure == 'generate':
                            raise RuntimeError('injected failure')
        assert created[0].closed
        # A later fit/resume gets a fresh scope; a failed fit leaves no live pool.
        with training_decoder_scope(bridge, directory, max_model_len=100):
            pass


def test_restart_mode_retains_one_context_per_use():
    bridge = object()
    with patch('qwen3.vllm_decode.VLLMDecoder') as factory:
        with training_decoder_scope(bridge, '/unused', max_model_len=100, mode='restart'):
            for step in range(3):
                with decoder_session(bridge, str(step), max_model_len=50):
                    pass
        assert factory.call_count == 3
        assert factory.return_value.__exit__.call_count == 3


def test_connector_rejects_previous_checkpoint_after_activation():
    from qwen3.vllm_connector import PLatourPrefixConnector, memory_key
    with TemporaryDirectory() as directory:
        root = Path(directory)
        connector = PLatourPrefixConnector.__new__(PLatourPrefixConnector)
        connector.directory = root
        connector.checkpoint = 'a' * 64
        connector.checkpoint_state = str(root / 'active.json')
        tokens = [1, 2, 3]
        first = memory_key(tokens[:-1], 'a' * 64)
        second = memory_key(tokens[:-1], 'b' * 64)
        for key in (first, second):
            (root / (key + '.safetensors')).touch()
        row = SimpleNamespace(prompt_token_ids=tokens, kv_transfer_params=dict(
            platour_prefix_memory=True, prefix_tokens=2,
            checkpoint_sha256='a' * 64, memory_key=first))
        atomic_json(dict(checkpoint_sha256='a' * 64), root / 'active.json')
        assert connector.get_num_new_matched_tokens(row, 0) == (2, False)
        atomic_json(dict(checkpoint_sha256='b' * 64), root / 'active.json')
        with pytest.raises(ValueError, match='Checkpoint identity'):
            connector.get_num_new_matched_tokens(row, 0)
        row.kv_transfer_params.update(checkpoint_sha256='b' * 64, memory_key=first)
        with pytest.raises(FileNotFoundError, match='mismatched'):
            connector.get_num_new_matched_tokens(row, 0)
        row.kv_transfer_params['memory_key'] = second
        assert connector.get_num_new_matched_tokens(row, 0) == (2, False)


def test_gpu_connector_rejects_stale_prefix_file_metadata():
    from qwen3.vllm_connector import PLatourPrefixConnector, PrefixMetadata, PrefixRequest
    with TemporaryDirectory() as directory:
        root = Path(directory)
        connector = PLatourPrefixConnector.__new__(PLatourPrefixConnector)
        connector.directory = root
        connector.checkpoint_state = str(root / 'active.json')
        atomic_json(dict(checkpoint_sha256='b' * 64), root / 'active.json')
        save_file({'unused': torch.zeros(1)}, root / 'key.safetensors', metadata=dict(
            format='platour-hf-prefix-v1', checkpoint_sha256='a' * 64,
            prefix_tokens='2', memory_key='key'))
        connector._get_connector_metadata = lambda: PrefixMetadata([PrefixRequest('key', 2, ())])
        with pytest.raises(ValueError, match='provenance mismatch'):
            connector.start_load_kv(None)


def test_decoder_rejects_stale_response_identity():
    with TemporaryDirectory() as directory:
        decoder = VLLMDecoder.__new__(VLLMDecoder)
        decoder.directory = Path(directory)
        decoder.memory = Path(directory)
        decoder.prefill_batch_size = 1
        decoder.counter = 0
        decoder.persistent = True
        decoder.identity = 'b' * 64
        decoder.process = SimpleNamespace(stdin=SimpleNamespace(write=lambda _: None, flush=lambda: None))
        decoder._export_batch = lambda rows: [{}]
        decoder._wait = lambda *a, **kw: dict(rows=[dict(index=0, sample_index=0, checkpoint_sha256='a' * 64)])
        with pytest.raises(RuntimeError, match='stale bridge'):
            decoder.generate([dict(index=0, prompt_token_ids=[1, 2])], count=1, cap=2, sampling={})


def test_portable_runner_signs_worker_mode_and_rejects_resume_change():
    from qwen3.runner import train
    bridge = torch.nn.Module()
    bridge.P = torch.nn.Linear(1, 1, bias=False)
    config = dict(method='grpo', task='gsm8k', smoke=True, total_steps=1,
        steps_per_epoch=1, batch_size=1, group_size=2, learning_rate=1e-5,
        eval_steps=[1], eval_samples=1, response_cap=2, rollout_worker_mode='persistent')
    rows = [dict(index=0, prompt_token_ids=[1, 2], gold='3')]
    def update(model, *_args):
        model.P.weight.grad = torch.ones_like(model.P.weight)
        return 0., 1
    with TemporaryDirectory() as directory, patch('qwen3.runner.grpo_update', update), patch(
            'qwen3.runner.evaluate_vllm', return_value=dict(correct=1, responses=1, accuracy=1.)):
        root = Path(directory)
        train(config, bridge, None, rows, rows, root)
        protocol = json.loads((root / 'training_protocol.json').read_text())
        assert protocol['resume_identity']['recipe']['rollout_worker_mode'] == 'persistent'
        checkpoint = root / 'checkpoints/step-0001'
        with pytest.raises(RuntimeError, match='protocol changed'):
            train(dict(config, rollout_worker_mode='restart'), bridge, None, rows, rows,
                  root, resume=checkpoint)
