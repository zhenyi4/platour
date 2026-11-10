"""Export Qwen3 HF bridge prefill to a separate native vLLM 0.17.1 decoder."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid

import torch
from safetensors.torch import save_file

from .checkpoints import atomic_json


def prepare_worker_memory(device, directory):
    """Release idle allocator blocks before a separate worker checks free VRAM."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        atomic_json(dict(device=str(device), free_bytes=free, total_bytes=total,
                         allocated_bytes=torch.cuda.memory_allocated(device),
                         reserved_bytes=torch.cuda.memory_reserved(device)),
                    Path(directory) / 'memory_before_worker.json')


def bridge_digest(bridge) -> str:
    digest = hashlib.sha256()
    for name, value in bridge.P.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prefix_tensors(cache, tokens: int, batch_index: int = 0) -> dict:
    layers = ((layer.keys, layer.values) for layer in cache.layers) if hasattr(cache, 'layers') else iter(cache)
    result = {}
    for index, (key, value) in enumerate(layers):
        if (key.ndim != 4 or key.shape[0] <= batch_index or key.shape != value.shape
                or key.shape[2] < tokens):
            raise ValueError('Expected dense receiver prefix cache and valid batch index')
        result[f'layer.{index}.kv'] = torch.stack((key[batch_index, :, :tokens].transpose(0, 1),
            value[batch_index, :, :tokens].transpose(0, 1))).contiguous().cpu()
    if not result:
        raise ValueError('Empty receiver prefix cache')
    return result


@torch.no_grad()
def check_parity(bridge, decoder, rows):
    """Compare first eight native decode logprobs with HF on a signed nonzero bridge."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not rows or not any(bool(parameter.count_nonzero()) for parameter in bridge.P.parameters()):
        raise ValueError('Parity requires signed prompts and a nonzero checkpoint')
    chosen = list(dict.fromkeys((0, min(1, len(rows) - 1),
        max(range(len(rows)), key=lambda index: len(rows[index]['prompt_token_ids'])))))
    probes = [rows[index] for index in chosen]
    answers = decoder.generate(probes, count=1, cap=16, parity=True,
        sampling=dict(temperature=0., top_p=1., top_k=-1, min_p=0.,
                      presence_penalty=0., frequency_penalty=0., repetition_penalty=1.))
    if [(answer['index'], answer['sample_index']) for answer in answers] != [
            (row['index'], 0) for row in probes]:
        raise RuntimeError('Qwen3 vLLM parity probe coverage differs')
    errors, matches = [], []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for row, answer in zip(probes, answers, strict=True):
            tokens, log_probs = answer['response_token_ids'], answer.get('log_probs')
            if not tokens or log_probs is None or len(tokens) != len(log_probs):
                raise ValueError('vLLM must return token-level parity logprobs')
            output = bridge.prefix_cache_batch([row['prompt_token_ids']], return_output=True)
            for offset, (token, log_prob) in enumerate(zip(tokens[:8], log_probs[:8], strict=True)):
                logits = bridge.receiver.lm_head(output.last_hidden_state[:, -1])[0].float()
                if offset == 0:
                    matches.append(int(logits.argmax()) == token)
                hf_log_prob = float(logits[token] - logits.logsumexp(-1))
                errors.append(abs(hf_log_prob - log_prob))
                output = bridge.receiver.model(input_ids=torch.tensor([[token]],
                    device=bridge.receiver.device), past_key_values=output.past_key_values,
                    use_cache=True)
    mean, maximum = sum(errors) / len(errors), max(errors)
    return dict(mean_abs_logprob_error=mean, max_abs_logprob_error=maximum,
                first_greedy_matches=sum(matches), probes=len(probes), compared_tokens=len(errors),
                probe_question_indices=[row['index'] for row in probes])


class VLLMDecoder:
    def __init__(self, bridge, directory: Path, *, max_model_len: int,
                 memory_fraction: float = .25, max_num_seqs: int = 16,
                 max_num_batched_tokens: int | None = None, prefill_batch_size: int = 8,
                 worker_python: str | None = None):
        if worker_python is None and importlib.util.find_spec('vllm') is None:
            raise RuntimeError('vLLM is required for evaluation; install the vllm extra or '
                               'supply --vllm-python for an isolated vLLM 0.17.1 runtime')
        if bridge.receiver.device.type != 'cuda':
            raise ValueError('Native vLLM evaluation requires a CUDA receiver')
        if getattr(bridge.receiver.config, 'use_sliding_window', False):
            raise ValueError('Only full-attention Qwen3 receiver cache is supported')
        if prefill_batch_size < 1 or max_num_seqs < 1:
            raise ValueError('Positive batch and active-sequence budgets are required')
        self.prefill_batch_size = prefill_batch_size
        self.bridge = bridge
        self.identity = bridge_digest(bridge)
        self.directory = Path(directory) / ('vllm-' + uuid.uuid4().hex[:12])
        self.memory = self.directory / 'memory'
        self.memory.mkdir(parents=True)
        self.counter = 0
        self.process = None
        self.log = None
        config = dict(model=str(bridge.receiver.config._name_or_path),
                      checkpoint_sha256=self.identity, max_model_len=max_model_len,
                      memory_fraction=memory_fraction, max_num_seqs=max_num_seqs,
                      max_num_batched_tokens=max_num_batched_tokens)
        atomic_json(config, self.directory / 'config.json')
        env = dict(os.environ)
        visible = env.get('CUDA_VISIBLE_DEVICES')
        index = bridge.receiver.device.index or 0
        env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
        for key in list(env):
            if key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                       'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
        root = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        self.log = (self.directory / 'worker.log').open('a')
        try:
            prepare_worker_memory(bridge.receiver.device, self.directory)
            self.process = subprocess.Popen([worker_python or sys.executable, '-u', '-m',
                'qwen3.vllm_worker', '--directory', str(self.directory)], env=env,
                stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            self.runtime = self._wait(self.directory / 'ready.json')
        except BaseException:
            self.close()
            raise

    def _wait(self, path: Path, timeout: float = 1800):
        deadline = time.monotonic() + timeout
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM worker exited; inspect {self.directory / "worker.log"}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'vLLM worker timed out: {path}')
            time.sleep(.1)
        value = json.loads(path.read_text())
        if not value.get('complete'):
            raise RuntimeError(f'vLLM worker failed: {value.get("error")}; '
                               f'inspect {self.directory / "worker.log"}')
        return value

    def _export_batch(self, rows: list[dict]) -> list[dict]:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                cache = self.bridge.prefix_cache_batch([r['prompt_token_ids'] for r in rows])
                transfers = []
                for index, row in enumerate(rows):
                    prompt = row['prompt_token_ids']
                    key = hashlib.sha256(json.dumps([prompt[:-1], self.identity],
                        separators=(',', ':')).encode()).hexdigest()
                    path = self.memory / (key + '.safetensors')
                    tensors = prefix_tensors(cache, len(prompt) - 1, index)
                    save_file(tensors, path, metadata=dict(format='platour-hf-prefix-v1',
                        checkpoint_sha256=self.identity, prefix_tokens=str(len(prompt) - 1),
                        memory_key=key))
                    transfers.append(dict(memory_key=key, prefix_tokens=len(prompt) - 1,
                        checkpoint_sha256=self.identity, platour_prefix_memory=True))
                return transfers
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def generate(self, rows: list[dict], *, count: int, cap: int, sampling: dict,
                 parity: bool = False, seed_override: int | None = None, seed_base: int = 1234) -> list[dict]:
        if seed_override is not None and (type(seed_override) is not int or
                                          len(rows) != 1 or count != 1):
            raise ValueError('Independent sampling seed override requires one prompt and one reply')
        transfers = []
        for start in range(0, len(rows), self.prefill_batch_size):
            transfers.extend(self._export_batch(rows[start:start + self.prefill_batch_size]))
        requests = []
        for row, transfer in zip(rows, transfers, strict=True):
            for sample in range(count):
                requests.append(dict(index=row['index'], sample_index=sample,
                    sampling_seed=(seed_override if seed_override is not None else
                                   seed_base + count * row['index'] + sample),
                    prompt_token_ids=row['prompt_token_ids'], transfer=transfer))
        self.counter += 1
        request = self.directory / f'request-{self.counter:05d}.json'
        result = self.directory / f'response-{self.counter:05d}.json'
        atomic_json(dict(rows=requests, cap=cap, sampling=sampling, parity=parity), request)
        self.process.stdin.write(json.dumps(dict(request=str(request), result=str(result))) + '\n')
        self.process.stdin.flush()
        output = self._wait(result, timeout=3600)['rows']
        expected = [(row['index'], row['sample_index']) for row in requests]
        if [(row['index'], row['sample_index']) for row in output] != expected:
            raise RuntimeError('vLLM returned incomplete or reordered responses')
        for path in self.memory.glob('*.safetensors'):
            path.unlink()
        return output

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                    process.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=10)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ReceiverOnlyDecoder(VLLMDecoder):
    """Export an independent full/LoRA receiver snapshot for identical vLLM scoring."""

    def __init__(self, actor, model_path, directory, *, max_model_len,
                 memory_fraction=.25, worker_python=None, max_num_seqs=16):
        if worker_python is None and importlib.util.find_spec('vllm') is None:
            raise RuntimeError('Native vLLM 0.17.1 is required for receiver-only scoring')
        if actor.student.device.type != 'cuda' or not 0 < memory_fraction < 1:
            raise ValueError('Receiver-only vLLM requires CUDA and a valid memory fraction')
        if actor.experiment.method not in ('full', 'lora'):
            raise ValueError('Receiver-only decoding requires full or LoRA weights')
        self.directory = Path(directory) / ('vllm-receiver-' + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True)
        self.process = self.log = None
        self.counter = 0
        model_dir = self.directory / 'model'
        model_dir.mkdir()
        source = Path(model_path).resolve()
        for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
                     'merges.txt', 'added_tokens.json'):
            file = source / name
            if file.is_file():
                shutil.copyfile(file, model_dir / name)
        if not (model_dir / 'config.json').is_file() or not (
                model_dir / 'tokenizer.json').is_file():
            raise FileNotFoundError('Receiver config and tokenizer are needed for native vLLM')
        weights = {}
        with torch.no_grad():
            for key, tensor in actor.student.state_dict().items():
                if '.lora_' in key:
                    continue
                if '.base.' in key:
                    key = key.replace('.base.', '.')
                weights[key] = tensor.detach().cpu().contiguous().clone()
            if actor.experiment.method == 'lora':
                for name, layer in actor.adapters.items():
                    delta = layer.lora_B.detach().float().cpu() @ layer.lora_A.detach().float().cpu()
                    key = name + '.weight'
                    weights[key].add_((delta * (layer.alpha / layer.rank)).to(weights[key].dtype))
        snapshot = model_dir / 'model.safetensors'
        save_file(weights, snapshot, metadata={'format': 'pt'})
        del weights
        digest = hashlib.sha256()
        with snapshot.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        self.identity = digest.hexdigest()
        atomic_json(dict(model=str(model_dir), checkpoint_sha256=self.identity,
                         receiver_only=True, max_model_len=max_model_len,
                         memory_fraction=memory_fraction, max_num_seqs=max_num_seqs,
                         max_num_batched_tokens=None), self.directory / 'config.json')
        env = dict(os.environ)
        visible = env.get('CUDA_VISIBLE_DEVICES')
        index = actor.student.device.index or 0
        env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
        for key in list(env):
            if key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                       'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        self.log = (self.directory / 'worker.log').open('a')
        try:
            prepare_worker_memory(actor.student.device, self.directory)
            self.process = subprocess.Popen([str(worker_python or sys.executable), '-u', '-m',
                'qwen3.vllm_worker', '--directory', str(self.directory)], env=env,
                stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            self.runtime = self._wait(self.directory / 'ready.json')
        except BaseException:
            self.close()
            raise

    def generate(self, rows, *, count, cap, sampling, seed_override=None, logprobs=False):
        if not rows or count < 1 or cap < 1 or (seed_override is not None and
                (type(seed_override) is not int or len(rows) != 1 or count != 1)):
            raise ValueError('Receiver-only vLLM needs valid prompts, counts and seeds')
        requests = [dict(index=row['index'], sample_index=sample,
                         sampling_seed=(seed_override if seed_override is not None else
                                        1234 + count * row['index'] + sample),
                         prompt_token_ids=row['prompt_token_ids'])
                    for row in rows for sample in range(count)]
        self.counter += 1
        request = self.directory / f'request-{self.counter:05d}.json'
        response = self.directory / f'response-{self.counter:05d}.json'
        atomic_json(dict(rows=requests, cap=cap, sampling=sampling, parity=logprobs), request)
        self.process.stdin.write(json.dumps(dict(request=str(request), result=str(response))) + '\n')
        self.process.stdin.flush()
        output = self._wait(response, timeout=3600)['rows']
        if [(item['index'], item['sample_index'], item['sampling_seed']) for item in output] != [
                (item['index'], item['sample_index'], item['sampling_seed']) for item in requests]:
            raise RuntimeError('Receiver-only vLLM returned incomplete or reordered answers')
        return output

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if exc[0] is None:
            # Successful training/validation calls retain hashes and responses, not
            # a full receiver export for every optimizer step. Preserve failed runs.
            (self.directory / 'model' / 'model.safetensors').unlink(missing_ok=True)
