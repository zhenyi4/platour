"""Qwen3.5 HF hybrid prefill / native-vLLM-0.17.1 evaluation adapter."""

from contextlib import contextmanager
from contextvars import ContextVar

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoints import atomic_json

FORMAT = 'qwen35-hf-bridge-hybrid-memory-v1'


def memory_key(ids, checkpoint, mode):
    return hashlib.sha256(json.dumps([ids, checkpoint, mode],
        separators=(',', ':')).encode()).hexdigest()


def sample_first(logits, seed: int, cap: int) -> int:
    """Original GENERAL_SAMPLING first response token (top-k=20, top-p=.95)."""
    uniforms = torch.rand(cap, device=logits.device,
        generator=torch.Generator(device=logits.device).manual_seed(seed))
    values, indices = logits.float().sort(descending=False)
    threshold = values[values.shape[-1] - min(20, values.shape[-1])]
    values = values.masked_fill(values < threshold, -torch.inf)
    probabilities = values.softmax(-1)
    excluded = probabilities.cumsum(-1) <= .05
    excluded[-1] = False
    probabilities = values.masked_fill(excluded, -torch.inf).softmax(-1)
    index = (probabilities.cumsum(-1) < uniforms[0]).sum()
    return int(indices[index.clamp_max(indices.shape[-1] - 1)])


def hybrid_tensors(cache):
    """Use the original Qwen3.5 HF/FLA -> vLLM GDN+KV layout."""
    result = {}
    for index, layer in enumerate(cache.layers):
        if hasattr(layer, 'recurrent_states'):
            result[f'layer.{index}.recurrent'] = layer.recurrent_states[0][0].transpose(-1, -2).contiguous().cpu()
            result[f'layer.{index}.conv'] = layer.conv_states[0][0, :, -3:].transpose(-1, -2).contiguous().cpu()
        else:
            result[f'layer.{index}.kv'] = torch.stack((layer.keys, layer.values), 1)[0].transpose(1, 2).contiguous().cpu()
    if len(result) < 32:
        raise ValueError('Incomplete Qwen3.5 hybrid prefix memory')
    return result


def export_lora(checkpoint: Path, destination: Path, receiver: Path):
    with safe_open(checkpoint, framework='pt', device='cpu') as handle:
        from .hf_export import custom_lora_name
        keys = list(handle.keys())
        names = [key if key.startswith('model.layers.') else custom_lora_name(key) for key in keys]
        if len(names) != 496 or len(set(names)) != 496 or not all(name.startswith('model.layers.') and name.endswith(
                ('.lora_A', '.lora_B')) for name in names):
            raise ValueError('Qwen3.5 release LoRA must have the original 496 matrices')
        tensors = {}
        targets = set()
        for name, key in zip(names, keys, strict=True):
            tensor = handle.get_tensor(key)
            stem, suffix = name.rsplit('.', 1)
            module = stem.rsplit('.', 1)[-1]
            qualified = 'base_model.model.model.language_model.' + name[len('model.'):] + '.weight'
            if module == 'in_proj_qkv':
                parts = ([tensor] * 3 if suffix == 'lora_A'
                         else list(tensor.split([2048, 2048, 4096], dim=0)))
                for key, part in zip(('q', 'k', 'v'), parts, strict=True):
                    tensors[qualified.replace('.in_proj_qkv.', '.in_proj_' + key + '.')] = part.contiguous().clone()
                targets.update(('in_proj_q', 'in_proj_k', 'in_proj_v'))
            else:
                tensors[qualified] = tensor.contiguous()
                targets.add(module)
    destination.mkdir()
    save_file(tensors, destination / 'adapter_model.safetensors')
    atomic_json(destination / 'adapter_config.json',
                {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
                 'base_model_name_or_path': str(receiver), 'r': 128,
                 'lora_alpha': 128, 'lora_dropout': 0., 'bias': 'none',
                 'inference_mode': True, 'target_modules': sorted(targets),
                 'use_rslora': False, 'use_dora': False})


def weight_identity(case, weight, weight_source):
    weight = Path(weight)
    identity = None
    if weight_source not in ('release', 'checkpoint', 'inference'):
        raise ValueError('Unknown Qwen3.5 weight source')
    if weight_source == 'checkpoint':
        from .checkpoints import read, sha256
        marker = weight.parent / 'complete.json'
        record = read(marker) if marker.is_file() else {}
        identity = sha256(weight)
        if (weight.is_symlink() or not record.get('complete') or
                weight.parent.name != f"step-{record.get('step', -1):04d}" or
                record.get('sha256', {}).get(weight.name) != identity):
            raise ValueError('Training checkpoint is incomplete or its weight SHA differs')
    elif weight_source == 'release' and weight.name in ('model.safetensors', 'adapter_model.safetensors'):
        from .model import verify_standard_weight
        verify_standard_weight(case, weight)
    from .checkpoints import sha256
    return identity or sha256(weight)


class VLLMDecoder:
    def __init__(self, case, receiver: Path, weight: Path, output: Path, *, actor=None,
                 memory_fraction=.22, max_model_len=36864, max_num_seqs=16,
                 worker_python: Path | None = None, device=None, weight_source='release',
                 persistent=False):
        if worker_python is None and importlib.util.find_spec('vllm') is None:
            raise RuntimeError('vLLM 0.17.1 is required for Qwen3.5 evaluation; '
                               'install with python scripts/install_environment.py')
        if case.method in ('bridge', 'random') and actor is None:
            raise ValueError('Qwen3.5 bridge needs the frozen HF prefill actor')
        if not 0 < memory_fraction < 1 or max_num_seqs < 1:
            raise ValueError('Invalid vLLM memory/sequence budgets')
        if persistent and case.method not in ('bridge', 'random'):
            raise ValueError('Persistent decoding requires a frozen receiver')
        self.persistent = bool(persistent)
        self.sleeping = False
        self.control_counter = 0
        self.case = case
        self.actor = actor
        self.weight = Path(weight)
        identity = weight_identity(case, self.weight, weight_source)
        self.directory = Path(output) / ('vllm-' + uuid.uuid4().hex[:12])
        self.memory = self.directory / 'memory'
        self.memory.mkdir(parents=True)
        self.counter = 0
        self.process = None
        self.log = None
        from .checkpoints import sha256
        identity = identity or sha256(self.weight)
        self.identity = identity
        config = dict(model=str(receiver), checkpoint=str(weight), checkpoint_sha256=identity,
                      method=case.method, max_model_len=max_model_len,
                      max_num_seqs=max_num_seqs, memory_fraction=memory_fraction,
                      persistent=self.persistent)
        if case.method == 'lora':
            directory = self.directory / 'adapter'
            export_lora(self.weight, directory, Path(receiver))
            config['adapter_dir'] = str(directory)
        atomic_json(self.directory / 'config.json', config)
        env = dict(os.environ)
        env['PATH'] = str(Path(worker_python or sys.executable).absolute().parent) + os.pathsep + env.get('PATH', '')
        for key in list(env):
            if key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
                       'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'} or key.startswith('TORCHELASTIC_'):
                env.pop(key, None)
        if actor is not None or device is not None:
            visible = env.get('CUDA_VISIBLE_DEVICES')
            index = (actor.student.device if actor is not None else device).index or 0
            env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
        env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        self.log = (self.directory / 'worker.log').open('a')
        try:
            from qwen3.vllm_decode import prepare_worker_memory
            prepare_worker_memory(actor.student.device if actor is not None else device, self.directory)
            self.process = subprocess.Popen([str(worker_python or sys.executable), '-u', '-m',
                'qwen35.vllm_worker', '--directory', str(self.directory)],
                env=env, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            self.runtime = self._wait(self.directory / 'ready.json')
        except BaseException:
            self.close()
            raise

    def _wait(self, path, timeout=3600):
        deadline = time.monotonic() + timeout
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'Qwen3.5 vLLM worker exited; inspect {self.directory / "worker.log"}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Qwen3.5 vLLM worker timed out; inspect {self.directory / "worker.log"}')
            time.sleep(.1)
        data = json.loads(path.read_text())
        if not data.get('complete'):
            raise RuntimeError(f'Qwen3.5 vLLM worker failed: {data.get("error")}')
        return data

    def _control(self, command, **payload):
        if not self.persistent:
            raise RuntimeError('Worker controls require a persistent frozen receiver')
        self.control_counter += 1
        destination = self.directory / f'control-{self.control_counter:05d}.json'
        self.process.stdin.write(json.dumps(dict(command=command, result=str(destination), **payload)) + '\n')
        self.process.stdin.flush()
        return self._wait(destination)

    def activate(self, weight, *, weight_source):
        identity = weight_identity(self.case, weight, weight_source)
        self._control('set_checkpoint', checkpoint_sha256=identity)
        self.identity, self.weight = identity, Path(weight)
        if self.sleeping:
            from qwen3.vllm_decode import prepare_worker_memory
            prepare_worker_memory(self.actor.student.device, self.directory)
            self._control('wake')
            self.sleeping = False

    def sleep(self):
        if not self.sleeping:
            self._control('sleep')
            self.sleeping = True

    def _export(self, row, cap, count, seed_override=None, seed_base=1234, training=False):
        prompt = row['prompt_token_ids']
        cache, logits = self.actor.prefix_cache(prompt)
        key = memory_key(prompt, self.identity, 'full_hidden')
        path = self.memory / (key + '.safetensors')
        if not path.exists():
            save_file(hybrid_tensors(cache), path, metadata={
                'format': FORMAT, 'checkpoint_sha256': self.identity,
                'bridge_mode': 'full_hidden', 'prefix_tokens': str(len(prompt)),
                'memory_key': key})
        del cache
        def first(seed):
            if not training:
                return sample_first(logits, seed, cap)
            probabilities = logits.float().softmax(-1)
            uniform = torch.rand((), device=logits.device,
                generator=torch.Generator(device=logits.device).manual_seed(seed))
            return int((probabilities.cumsum(-1) < uniform).sum().clamp_max(probabilities.numel()-1))
        result = [dict(index=row['index'], sample_index=sample,
            sampling_seed=(seed_override if seed_override is not None else
                           seed_base + count * row['index'] + sample),
            prompt_token_ids=prompt,
            first_token=first(seed_override if seed_override is not None else
                              seed_base + count * row['index'] + sample),
            transfer={'memory_key': key, 'prefix_tokens': len(prompt),
                      'checkpoint_sha256': self.identity, 'qwen35_bridge_memory': True})
            for sample in range(count)]
        if training:
            log_probs = logits.float().log_softmax(-1)
            for request in result:
                request['first_log_prob'] = float(log_probs[request['first_token']])
        return result

    def generate(self, rows, *, count, cap, batch_size=None, seed_override=None, seed_base=1234, training=False):
        if seed_override is not None and (type(seed_override) is not int or
                                          len(rows) != 1 or count != 1):
            raise ValueError('Independent seed override requires exactly one Qwen3.5 answer')
        if batch_size is None:
            batch_size = (2 if self.case.task == 'dapo' else
                          4 if self.case.method in ('bridge', 'random') else 16)
        if not rows or count < 1 or cap < 1 or batch_size < 1:
            raise ValueError('Invalid vLLM test inputs/budgets')
        result = []
        eos = self.actor.student.config.eos_token_id if self.actor is not None else None
        if isinstance(eos, int):
            eos = [eos]
        eos = set(eos or [])
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            requests = []
            for row in batch:
                if self.actor is not None:
                    requests.extend(self._export(row, cap, count, seed_override, seed_base, training))
                else:
                    requests.extend(dict(index=row['index'], sample_index=sample,
                        sampling_seed=(seed_override if seed_override is not None else
                                       seed_base + count * row['index'] + sample),
                        prompt_token_ids=row['prompt_token_ids']) for sample in range(count))
            for request in requests:
                if self.actor is not None:
                    request['handoff_is_eos'] = request['first_token'] in eos
                    request['handoff_only'] = cap == 1
            self.counter += 1
            request = self.directory / f'request-{self.counter:05d}.json'
            response = self.directory / f'response-{self.counter:05d}.json'
            atomic_json(request, {'rows': requests, 'cap': cap, 'logprobs': training,
                'sampling': ({'temperature': 1., 'top_p': 1., 'top_k': -1,
                              'presence_penalty': 0.} if training else
                             {'temperature': 1., 'top_p': .95, 'top_k': 20,
                              'presence_penalty': 1.5})})
            self.process.stdin.write(json.dumps({'request': str(request), 'result': str(response)}) + '\n')
            self.process.stdin.flush()
            received = self._wait(response)['rows']
            if self.persistent and any(row.get('checkpoint_sha256') != self.identity for row in received):
                raise RuntimeError('Qwen3.5 response uses a stale bridge checkpoint')
            if [(row['index'], row['sample_index'], row['sampling_seed']) for row in received] != [
                    (row['index'], row['sample_index'], row['sampling_seed']) for row in requests]:
                raise RuntimeError('Qwen3.5 vLLM response order/seed/coverage differs')
            result.extend(received)
            for file in self.memory.glob('*.safetensors'):
                file.unlink()
        if len(result) != len(rows) * count:
            raise RuntimeError('Qwen3.5 full test denominator incomplete')
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"command":"stop"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        try:
                            os.killpg(self.process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            self.process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.process.pid, signal.SIGKILL)
                            self.process.wait(timeout=10)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_training_decoder = ContextVar('qwen35_training_decoder', default=None)


class TrainingDecoders:
    """Reuse frozen receivers per rank/runtime budget; all are asleep between phases.

    CLI validation has different memory/sequence budgets from rollout. Keep one
    engine per profile without changing either budget. API fit uses one profile.
    """

    def __init__(self, output, max_model_len):
        self.output = Path(output)
        self.max_model_len = max_model_len
        self.workers = {}
        self.busy = False

    @contextmanager
    def borrow(self, case, receiver, weight, output, *, actor, **options):
        if self.busy:
            raise RuntimeError('Concurrent use of training decoders is unsupported')
        weight_source = options.pop('weight_source', 'release')
        normalized = dict(memory_fraction=.22, max_model_len=36864, max_num_seqs=16,
                          worker_python=None, device=None)
        normalized.update(options)
        normalized['max_model_len'] = max(normalized['max_model_len'], self.max_model_len)
        # Context is fixed across training minibatches. Other runtime limits, the
        # actor and the frozen backbone must match before an engine can be reused.
        key = (id(actor), case.method, str(Path(receiver).resolve()),
               tuple((name, str(value)) for name, value in sorted(normalized.items()) if name != 'device'))
        self.busy = True
        try:
            if key not in self.workers:
                self.workers[key] = VLLMDecoder(case, receiver, weight, self.output,
                    actor=actor, weight_source=weight_source, persistent=True, **normalized)
            decoder = self.workers[key]
            decoder.activate(weight, weight_source=weight_source)
            start = decoder.counter
            yield decoder
            decoder.sleep()
            atomic_json(Path(output) / 'worker-reuse.json', dict(complete=True,
                worker_directory=str(decoder.directory), worker_pid=decoder.process.pid,
                checkpoint_sha256=decoder.identity, first_request=start + 1,
                last_request=decoder.counter, sleeping=True))
        except BaseException:
            self.close()
            raise
        finally:
            self.busy = False

    def close(self):
        workers, self.workers = self.workers, {}
        try:
            for decoder in workers.values():
                decoder.close()
        finally:
            self.busy = False


@contextmanager
def training_decoder_scope(output, *, max_model_len, mode='persistent'):
    if mode not in ('persistent', 'restart'):
        raise ValueError('rollout_worker_mode must be persistent or restart')
    if _training_decoder.get() is not None:
        raise RuntimeError('Nested Qwen3.5 training decoder scopes are unsupported')
    pool = TrainingDecoders(output, max_model_len) if mode == 'persistent' else None
    token = _training_decoder.set(pool)
    try:
        yield
    finally:
        _training_decoder.reset(token)
        if pool is not None:
            pool.close()


def reserve_training_context(max_model_len):
    """Fix the pool's context floor after CLI preflight loads the signed splits."""
    pool = _training_decoder.get()
    if pool is not None:
        if pool.workers:
            raise RuntimeError('Cannot change context after starting training workers')
        pool.max_model_len = max_model_len


@contextmanager
def decoder_session(case, receiver, weight, output, *, actor=None, **options):
    pool = _training_decoder.get()
    if pool is not None and actor is not None and case.method in ('bridge', 'random'):
        with pool.borrow(case, receiver, weight, output, actor=actor, **options) as decoder:
            yield decoder
    else:
        with VLLMDecoder(case, receiver, weight, output, actor=actor, **options) as decoder:
            yield decoder


@contextmanager
def training_decoder(actor, case, receiver, output, *, cap, prompt_length, worker_python=None):
    """Export this update's weights and isolate worker/cache per rank and attempt."""
    directory = Path(output) / ('snapshot-' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    weight = directory / 'current.safetensors'
    with torch.no_grad():
        save_file({key: value.detach().cpu().contiguous().clone()
                   for key, value in actor.inference_state().items()}, weight)
    try:
        with decoder_session(case, receiver, weight, directory,
                actor=actor if case.method in ('bridge', 'random') else None,
                worker_python=worker_python, device=actor.student.device, weight_source='inference',
                max_model_len=max(3072, cap + prompt_length)) as decoder:
            yield decoder
    except BaseException:
        raise  # Keep failed snapshots for diagnosis.
    else:
        weight.unlink()
