"""Qwen3.5 frozen receiver reuse, hybrid memory versions and failure cleanup."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from qwen35.checkpoints import atomic_json, sha256
from qwen35.vllm_decode import (VLLMDecoder, decoder_session, training_decoder_scope,
                                weight_identity)


class FakeWorker:
    def __init__(self, case, receiver, weight, output, *, actor=None, **options):
        self.case = case
        self.directory = Path(output) / ('worker-' + str(options['max_num_seqs']))
        self.directory.mkdir(parents=True)
        self.process = SimpleNamespace(pid=123)
        self.counter = 0
        self.sleeping = False
        self.closed = False
        self.events = []

    def activate(self, weight, *, weight_source):
        self.identity = weight_identity(self.case, weight, weight_source)
        self.sleeping = False
        self.events.append('activate')

    def sleep(self):
        self.sleeping = True
        self.events.append('sleep')

    def close(self):
        assert not self.closed
        self.closed = True
        self.events.append('close')


def test_reuse_tracks_new_weights_and_preserves_separate_runtime_budgets(tmp_path):
    case, actor = SimpleNamespace(method='bridge'), object()
    weight = tmp_path / 'current.safetensors'
    created = []
    def factory(*args, **kwargs):
        worker = FakeWorker(*args, **kwargs)
        created.append(worker)
        return worker
    identities = []
    with patch('qwen35.vllm_decode.VLLMDecoder', side_effect=factory) as make:
        with training_decoder_scope(tmp_path / 'workers', max_model_len=3072):
            for step in range(3):
                save_file({'P.0.weight': torch.full((2, 2), float(step))}, weight)
                with decoder_session(case, tmp_path, weight, tmp_path / str(step), actor=actor,
                        weight_source='inference', max_model_len=100 + step) as worker:
                    assert not worker.sleeping
                    identities.append(worker.identity)
                    worker.counter += 1
                assert worker.sleeping and not worker.closed
                record = json.loads((tmp_path / str(step) / 'worker-reuse.json').read_text())
                assert record['first_request'] == record['last_request'] == step + 1
                assert record['checkpoint_sha256'] == identities[-1]
                with decoder_session(case, tmp_path, weight, tmp_path / ('val' + str(step)),
                        actor=actor, weight_source='inference', max_model_len=3072,
                        max_num_seqs=8, memory_fraction=.30) as validation:
                    assert worker.sleeping
                    validation.counter += 1
                assert validation.sleeping
            assert make.call_count == 2
            assert make.call_args_list[0].kwargs['max_num_seqs'] == 16
            assert make.call_args_list[1].kwargs['max_num_seqs'] == 8
    assert len(set(identities)) == 3
    assert all(w.closed and w.events == ['activate', 'sleep'] * 3 + ['close'] for w in created)


@pytest.mark.parametrize('failure', ['generate', 'activate', 'sleep'])
def test_failure_closes_all_profiles_and_resets_scope(tmp_path, failure):
    case, actor = SimpleNamespace(method='bridge'), object()
    weight = tmp_path / 'current.safetensors'
    save_file({'P.0': torch.zeros(1)}, weight)
    workers = []
    def factory(*args, **kwargs):
        worker = FakeWorker(*args, **kwargs)
        workers.append(worker)
        if len(workers) == 2 and failure != 'generate':
            def fail(*args, **kwargs):
                raise RuntimeError('injected failure')
            setattr(worker, failure, fail)
        return worker
    with patch('qwen35.vllm_decode.VLLMDecoder', side_effect=factory):
        with pytest.raises(RuntimeError, match='injected failure'):
            with training_decoder_scope(tmp_path, max_model_len=3072):
                for seqs in (8, 16):
                    with decoder_session(case, tmp_path, weight, tmp_path / str(seqs),
                            actor=actor, weight_source='inference', max_num_seqs=seqs):
                        if seqs == 16 and failure == 'generate':
                            raise RuntimeError('injected failure')
    assert all(worker.closed for worker in workers)
    with training_decoder_scope(tmp_path, max_model_len=3072):
        pass


def test_restart_and_receiver_training_keep_per_use_lifecycle(tmp_path):
    for mode, method in (('restart', 'bridge'), ('persistent', 'full'), ('persistent', 'lora')):
        with patch('qwen35.vllm_decode.VLLMDecoder') as factory:
            with training_decoder_scope(tmp_path, max_model_len=3072, mode=mode):
                for step in range(3):
                    with decoder_session(SimpleNamespace(method=method), tmp_path, tmp_path / 'weight',
                                         tmp_path / str(step), actor=object()):
                        pass
            assert factory.call_count == factory.return_value.__exit__.call_count == 3


def test_connector_checks_active_version_in_scheduler_and_gpu_import(tmp_path):
    from qwen35.vllm_connector import (Qwen35HybridMemoryConnector, memory_key,
                                      HybridMemoryMetadata, MemoryRequest)
    connector = Qwen35HybridMemoryConnector.__new__(Qwen35HybridMemoryConnector)
    connector.directory, connector.mode = tmp_path, 'full_hidden'
    connector.checkpoint = 'a' * 64
    connector.checkpoint_state = str(tmp_path / 'active.json')
    tokens = [1, 2, 3]
    keys = [memory_key(tokens[:-1], letter * 64, connector.mode) for letter in 'ab']
    for key, letter in zip(keys, 'ab'):
        save_file({'unused': torch.zeros(1)}, tmp_path / (key + '.safetensors'), metadata=dict(
            format='qwen35-hf-bridge-hybrid-memory-v1', checkpoint_sha256=letter * 64,
            bridge_mode=connector.mode, prefix_tokens='2', memory_key=key))
    row = SimpleNamespace(prompt_token_ids=tokens, kv_transfer_params=dict(qwen35_bridge_memory=True,
        prefix_tokens=2, checkpoint_sha256='a' * 64, memory_key=keys[0]))
    atomic_json(tmp_path / 'active.json', dict(checkpoint_sha256='a' * 64))
    assert connector.get_num_new_matched_tokens(row, 0) == (2, False)
    atomic_json(tmp_path / 'active.json', dict(checkpoint_sha256='b' * 64))
    with pytest.raises(ValueError, match='checkpoint identity'):
        connector.get_num_new_matched_tokens(row, 0)
    row.kv_transfer_params['checkpoint_sha256'] = 'b' * 64
    with pytest.raises(FileNotFoundError):
        connector.get_num_new_matched_tokens(row, 0)
    row.kv_transfer_params['memory_key'] = keys[1]
    assert connector.get_num_new_matched_tokens(row, 0) == (2, False)
    connector._get_connector_metadata = lambda: HybridMemoryMetadata([MemoryRequest(keys[0], 2, ())])
    with pytest.raises(ValueError, match='provenance mismatch'):
        connector.start_load_kv(None)


def test_reused_signed_checkpoint_is_reverified(tmp_path):
    case = SimpleNamespace(method='bridge')
    root = tmp_path / 'step-0001'
    root.mkdir()
    weight = root / 'model.safetensors'
    save_file({'P.0': torch.ones(1)}, weight)
    identity = sha256(weight)
    atomic_json(root / 'complete.json', dict(complete=True, step=1, sha256={weight.name: identity}))
    assert weight_identity(case, weight, 'checkpoint') == identity
    save_file({'P.0': torch.zeros(1)}, weight)
    with pytest.raises(ValueError, match='weight SHA differs'):
        weight_identity(case, weight, 'checkpoint')


def test_decoder_rejects_stale_response(tmp_path):
    decoder = VLLMDecoder.__new__(VLLMDecoder)
    decoder.directory = decoder.memory = tmp_path
    decoder.actor = None
    decoder.case = SimpleNamespace(task='dapo')
    decoder.counter = 0
    decoder.persistent = True
    decoder.identity = 'b' * 64
    decoder.process = SimpleNamespace(stdin=SimpleNamespace(write=lambda _: None, flush=lambda: None))
    decoder._wait = lambda *_: dict(rows=[dict(index=0, sample_index=0, sampling_seed=1234,
                                             checkpoint_sha256='a' * 64)])
    with pytest.raises(RuntimeError, match='stale bridge'):
        decoder.generate([dict(index=0, prompt_token_ids=[1, 2])], count=1, cap=2)


def test_api_signs_worker_mode_and_rejects_switching_a_completed_run(tmp_path):
    from tests.test_vllm_training import VLLMTrainingTests
    fixture = VLLMTrainingTests()
    fixture.setUp()
    session = fixture.fresh()
    session.case.method = 'bridge'
    options = dict(output=tmp_path / 'fit', max_steps=1, smoke=True, batch_size=2,
                   response_cap=4, validation_samples=2)
    with patch('qwen35.cli.train_step', return_value={}):
        receipt = session.fit(fixture.train, fixture.val, **options)
        assert json.loads((tmp_path / 'fit/protocol.json').read_text())['rollout_worker_mode'] == 'persistent'
        assert session.fit(fixture.train, fixture.val, **options) == receipt
        with pytest.raises(RuntimeError, match='different model/data/budget'):
            session.fit(fixture.train, fixture.val, **options, rollout_worker_mode='restart')


def test_cli_signs_worker_mode_and_rejects_resume_switch(tmp_path):
    from tests.test_qwen35 import TinyActor, portable_args
    from qwen35.cli import run_train
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text('{}')
    case = SimpleNamespace(identifier=19, tag='tiny', task='dapo', method='bridge', sender=None,
        steps=1, checkpoint_steps=(0, 1), validation_steps=(1,), val_samples=1,
        train_questions=1, batch_size=1, response_cap=2, learning_rate=.01)
    args = portable_args(output=tmp_path / 'run', data_root=data, no_final_test=True,
                         code_cases=None, official_checker=None, docker_image=None)
    rows = {'train': [dict(index=0, prompt_token_ids=[1, 2], gold='3')],
            'validation': [dict(index=0, prompt_token_ids=[1, 3], gold='3')]}
    with patch('qwen35.cli.inputs', return_value=rows), \
         patch('qwen35.cli._model', side_effect=lambda *_a, **_k: (TinyActor(), None)), \
         patch('qwen35.cli.train_step', return_value={}), \
         patch('qwen35.vllm_decode.decoder_session'), \
         patch('qwen35.cli.evaluate_vllm', return_value=dict(complete=True, accuracy=1.,
                                                           clipping_rate=0., correct=1, responses=1)):
        run_train(case, args)
        assert json.loads((args.output / 'training_protocol.json').read_text())['rollout_worker_mode'] == 'persistent'
        args.rollout_worker_mode = 'restart'
        with pytest.raises(RuntimeError, match='protocol changed'):
            run_train(case, args)


def test_legacy_signed_api_protocol_resumes_only_with_restart(tmp_path):
    from tests.test_vllm_training import VLLMTrainingTests
    from qwen3.parallel import protocol_root
    fixture = VLLMTrainingTests()
    fixture.setUp()
    session = fixture.fresh()
    session.case.method = 'bridge'
    options = dict(output=tmp_path / 'fit', max_steps=1, smoke=True, batch_size=2,
                   response_cap=4, validation_samples=2)
    # Generate a genuinely signed checkpoint using the old protocol schema.
    def legacy_protocol(root, config):
        config.pop('rollout_worker_mode')
        protocol_root(root, config)
    with patch('qwen35.cli.train_step', return_value={}):
        with patch('platour_api.protocol_root', side_effect=legacy_protocol):
            receipt = session.fit(fixture.train, fixture.val, **options, rollout_worker_mode='restart')
        assert 'rollout_worker_mode' not in json.loads((tmp_path / 'fit/protocol.json').read_text())
        assert session.fit(fixture.train, fixture.val, **options, rollout_worker_mode='restart') == receipt
        with pytest.raises(RuntimeError, match='different model/data/budget'):
            session.fit(fixture.train, fixture.val, **options)
