"""Historical four-replica SFT with private recovery and final-only public weights."""
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import get_scheduler

from .historical_models import load_historical_bridge
from .checkpoints import atomic_json, load_step, save_step


def weight_digest(bridge):
    digest = hashlib.sha256()
    for name, value in bridge.P.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def local_chunks(indices, rows, rank, world, micro_batch):
    local = indices[rank::world]
    local = sorted(local, key=lambda i: len(rows[i]['prompt_token_ids']) + len(rows[i]['response_token_ids']), reverse=True)
    return [local[i:i + micro_batch] for i in range(0, len(local), micro_batch)]


def sync_gradients(parameters):
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    flat = torch.cat([p.grad.reshape(-1) for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if not torch.isfinite(flat).all():
        raise FloatingPointError('Nonfinite global bridge gradient')
    norm = torch.linalg.vector_norm(flat)
    flat.mul_((1. / (norm + 1e-6)).clamp(max=1.))
    offset = 0
    for parameter in parameters:
        parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()
    return norm


def reconcile_sft_history(output, checkpoint):
    """Restore the signed metric prefix, preserving any torn or future lines."""
    marker = json.loads((checkpoint / 'complete.json').read_text())
    step = marker['step']
    path = output / 'metrics.jsonl'
    original = path.read_bytes() if path.exists() else b''
    lines = original.splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(lines) - 1 or original.endswith(b'\n'):
                raise RuntimeError('Committed SFT metric history is corrupt') from None
    prefix = [row for row in records if 1 <= row.get('step', -1) <= step]
    if [row['step'] for row in prefix] != list(range(1, len(prefix) + 1)):
        raise RuntimeError('Signed SFT metric history is not contiguous')
    if step and len(prefix) == step - 1:
        recovered = marker.get('metric')
        if recovered is None or recovered.get('step') != step:
            raise RuntimeError('Signed SFT step lacks its recoverable metric')
        prefix.append(recovered)
    if len(prefix) != step:
        raise RuntimeError('SFT metric history does not match the signed optimizer step')
    contents = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in prefix).encode()
    if contents != original:
        if path.exists():
            archive = output / 'recovery'
            archive.mkdir(exist_ok=True)
            shutil.copyfile(path, archive / f'metrics-before-resume-{time.time_ns()}.jsonl')
        temporary = path.with_name(path.name + '.pending')
        temporary.write_bytes(contents)
        temporary.replace(path)
    return prefix


def final_test_protocol(config, test):
    """Bind optional TEST data and decoding settings to this training run."""
    if test is None:
        return {}
    if not test or config.get('eval_backend', 'vllm') != 'vllm':
        raise ValueError('Historical SFT final TEST requires nonempty data and vLLM')
    return {'final_test': dict(
        sha256=hashlib.sha256(json.dumps(test, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        questions=len(test), samples=config['eval_samples'], response_cap=config['response_cap'],
        backend='vllm', checkpoint_policy='final_sft_checkpoint_v1')}


def complete_aligned_run(config, bridge, tokenizer, validation, test, output, optimizer, completion):
    """Commit completion only after optional final-checkpoint TEST succeeds.

    A failed TEST leaves the final signed training checkpoint resumable. The
    shared evaluator restores terminal optimizer/RNG state even on failure,
    retries partial decoding in a fresh folder and reuses a completed summary.
    """
    from .runner import evaluate_selected_once, on_rank_zero
    step = completion['steps']
    if step != (3 if config.get('smoke') else config['total_steps']):
        raise ValueError('Historical SFT final TEST requires the terminal training step')
    if test is not None:
        final_test_protocol(config, test)
        def selection():
            marker = json.loads((output / 'checkpoints' / f'step-{step:04d}' / 'complete.json').read_text())
            nll = (marker.get('metric') or {}).get('heldout_nll')
            if (not marker.get('complete') or marker.get('step') != step or
                    not isinstance(nll, float) or not math.isfinite(nll)):
                raise ValueError('Historical SFT final checkpoint lacks signed held-out NLL')
            return dict(policy='final_sft_checkpoint_v1', selected_step=step,
                        validation_heldout_nll=nll, test_used_for_selection=False,
                        candidates=[dict(step=step, heldout_nll=nll)],
                        smoke_only=bool(config.get('smoke')))
        selected = on_rank_zero(selection)
        evaluate_selected_once(dict(config, total_steps=step), bridge, tokenizer,
                               validation, test, output, optimizer, selection=selected)
        def label_test():
            path = output / 'test' / 'summary.json'
            summary = json.loads(path.read_text())
            summary.update(smoke_only=bool(config.get('smoke')),
                           training_backend='historical-sft', training_steps=step)
            atomic_json(summary, path)
        on_rank_zero(label_test)
    on_rank_zero(lambda: atomic_json(completion, output / 'complete.json'))


def train_aligned(config, sender, receiver, rows, validation, output, *,
                  micro_batch=8, resume=None, final_test=None):
    test_protocol = final_test_protocol(config, final_test)
    rank, world, local = [int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    accumulation=config.get('gradient_accumulation_steps', 2)
    if config['method'] != 'sft' or world != 4 or world * micro_batch * accumulation != config['batch_size']:
        raise ValueError('Historical SFT requires four replicas and micro-batch × accumulation × world = 64')
    if len(rows) != config['train_examples'] or math.ceil(len(rows) / 64) != config['steps_per_epoch']:
        raise ValueError('Training input count differs from experiment')
    requested_stop = os.environ.get('PLATOUR_SFT_SMOKE_INTERRUPT_AFTER_SIGNED_STEP')
    if requested_stop is not None and (not config.get('smoke') or requested_stop not in ('1', '2') or
                                       resume is not None):
        raise ValueError('Controlled SFT interruption is only for a fresh, signed smoke step')
    primary, teacher_device = 2 * local, 2 * local + 1
    if torch.cuda.device_count() < 8:
        raise ValueError('Historical mode requires eight visible GPUs')
    torch.cuda.set_device(primary)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Other ranks wait while rank zero runs the optional, potentially long TEST.
    timeout = timedelta(hours=float(os.environ.get('PLATOUR_DIST_TIMEOUT_HOURS', '24')))
    dist.init_process_group('nccl', timeout=timeout, device_id=torch.device('cuda', primary))
    output = Path(output).absolute()
    if resume is None:
        if rank == 0:
            if output.exists():
                raise FileExistsError('Use a fresh historical SFT output directory')
            output.mkdir(parents=True)
    else:
        resume = Path(resume).absolute()
        if (not output.is_dir() or resume.parent != output / 'checkpoints' or
                resume.is_symlink() or (output / 'complete.json').exists()):
            raise ValueError('Native SFT resume requires this run’s incomplete signed step')
    dist.barrier()
    audit_name = (f'model_runtime.rank-{rank:02d}.json' if resume is None else
                  f'model_runtime.resume-{resume.name}.rank-{rank:02d}.json')
    bridge, tokenizer = load_historical_bridge(sender, receiver, config['layer_mapping'],
        config.get('initialization', 'zero'), sender_family=config['sender'],
        primary=primary, secondary=teacher_device, audit_path=output / audit_name)
    source, target = bridge.sender, bridge.receiver
    parameters = tuple(bridge.P.parameters())
    def assert_same_weights():
        digest = weight_digest(bridge)
        hashes = [None] * world
        dist.all_gather_object(hashes, digest)
        if len(set(hashes)) != 1:
            raise RuntimeError('Training replicas diverged')
        return digest
    initial = assert_same_weights()
    smoke = config.get('smoke', False)
    recovery_stride = 1 if smoke else 16
    protocol = dict(config=config, world_size=world, gpu_count=8,
        receiver_gpu_ids=[0, 2, 4, 6],
        sender_gpu_groups=([[0, 1], [2, 3], [4, 5], [6, 7]] if config['sender'].startswith('Qwen3-235B')
                           else [[1], [3], [5], [7]]), micro_batch=micro_batch,
        gradient_accumulation_steps=accumulation, global_batch_size=64,
        published_checkpoint_only_final=True, private_recovery_stride=recovery_stride,
        data_parallel_partition='global batch rank stride, then local descending length sort',
        teacher_forward='full right-padded prompt, explicit attention mask',
        injection='prompt[:-1] only; final prompt token native', lm_head_chunk_tokens=128,
        objective='global valid-response-token mean cross entropy',
        gradient_reduction='flattened SUM then global L2 norm clipping to 1',
        warmup='HF linear scheduler, first optimizer update LR=0',
        sdpa_backend='math', tf32=False, bf16_reduced_precision_reduction=False,
        wandb_enabled=False, torch=torch.__version__, initial_sha256=initial, **test_protocol)
    if resume is not None:
        if json.loads((output / 'protocol.json').read_text()) != protocol:
            raise ValueError('Native SFT resume protocol, model or initialization differs')
    elif rank == 0:
        atomic_json(protocol, output / 'protocol.json')
    dist.barrier()
    def count(data, indices):
        return sum(len(data[i]['response_token_ids']) for i in indices)
    def loss(data, indices):
        return bridge.batch_response_loss([data[i] for i in indices], tokenizer.pad_token_id)
    @torch.no_grad()
    def validate():
        stats = torch.zeros(2, device=primary, dtype=torch.float64)
        for chunk in local_chunks(list(range(len(validation))), validation, rank, world, micro_batch):
            tokens = count(validation, chunk)
            stats[0] += loss(validation, chunk).double() * tokens
            stats[1] += tokens
        dist.all_reduce(stats)
        if int(stats[1]) != count(validation, list(range(len(validation)))):
            raise RuntimeError('Validation token coverage incomplete')
        return float(stats[0] / stats[1])
    with torch.no_grad():
        zero = loss(rows, [0])
        row = rows[0]
        ids = torch.tensor([row['prompt_token_ids'] + row['response_token_ids']], device=primary)
        labels = ids.clone()
        labels[:, :len(row['prompt_token_ids'])] = -100
        native = bridge.receiver(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        torch.testing.assert_close(zero, native, atol=2e-5, rtol=2e-5)
    sanity = dict(passed=True, zero_bridge_ce=float(zero), native_ce=float(native))
    if resume is not None:
        previous = json.loads((output / 'sanity.json').read_text())
        if not all(math.isclose(previous[key], sanity[key], abs_tol=2e-5, rel_tol=2e-5)
                   for key in ('zero_bridge_ce', 'native_ce')):
            raise ValueError('Native SFT baseline answer CE differs on resume')
    elif rank == 0:
        atomic_json(sanity, output / 'sanity.json')
    del zero, native, ids, labels
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=(.9, .999), weight_decay=0.)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['total_steps'])
    initial_nll = validate()
    if resume is not None:
        before = json.loads((output / 'initial_validation.json').read_text())
        if not math.isclose(initial_nll, before['nll'], abs_tol=2e-5, rel_tol=2e-5):
            raise ValueError('Native SFT held-out baseline changed before resume')
        start_step = load_step(resume, bridge, optimizer, scheduler)
        if start_step != int(resume.name.split('-')[-1]) or start_step > (3 if smoke else config['total_steps']):
            raise ValueError('Native SFT resume step and signed checkpoint differ')
        assert_same_weights()
        if rank == 0:
            reconcile_sft_history(output, resume)
        dist.barrier()
    else:
        if rank == 0:
            atomic_json(dict(nll=initial_nll), output / 'initial_validation.json')
        save_step(output, 0, bridge, optimizer, scheduler,
                  cuda_devices=(primary, teacher_device))
        start_step = 0
    started = time.monotonic()
    step = 0
    for epoch in range(1 if smoke else 3):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        visited = []
        if smoke:
            longest = max(range(len(rows)), key=lambda i: len(rows[i]['prompt_token_ids']) + len(rows[i]['response_token_ids']))
            groups = [order[:64], order[64:128], [longest]]
        else:
            groups = [order[start:start + 64] for start in range(0, len(order), 64)]
        for group_index, group in enumerate(groups):
            visited.extend(group)
            step += 1
            if step <= start_step:
                continue
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            tokens = count(rows, group)
            total = torch.zeros((), device=primary, dtype=torch.float64)
            chunks = local_chunks(group, rows, rank, world, micro_batch)
            for chunk in chunks:
                n = count(rows, chunk)
                value = loss(rows, chunk)
                if not torch.isfinite(value):
                    raise FloatingPointError('Nonfinite SFT loss')
                total += value.detach().double() * n
                (value * (n / tokens)).backward()
            norm = sync_gradients(parameters)
            if any(p.grad is not None or p.requires_grad for m in (source, target) for p in m.parameters()):
                raise RuntimeError('Frozen backbone gradient detected')
            lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            dist.all_reduce(total)
            timing = torch.tensor(time.monotonic() - tick, device=primary)
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            metric = dict(step=step, total_steps=3 if smoke else config['total_steps'], schedule_total_steps=config['total_steps'],
                smoke=smoke, epoch=epoch + 1,
                train_nll=float(total) / tokens, learning_rate=lr, grad_norm=float(norm),
                examples=len(group), response_tokens=tokens, world_size=world, gpu_count=8,
                micro_batch=micro_batch, local_micro_batches=len(chunks),
                seconds=time.monotonic() - started, step_seconds=float(timing))
            if smoke or group_index == len(groups) - 1:
                metric['heldout_nll'] = validate()
                metric['weights_sha256'] = assert_same_weights()
            if smoke or step % recovery_stride == 0 or group_index == len(groups) - 1:
                save_step(output, step, bridge, optimizer, scheduler,
                          cuda_devices=(primary, teacher_device), record=metric)
            if rank == 0:
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metric) + '\n')
                atomic_json(metric, output / 'live.json')
                print(json.dumps(metric), flush=True)
            if requested_stop == str(step):
                dist.barrier()
                if rank == 0:
                    atomic_json(dict(step=step, checkpoint=f'checkpoints/step-{step:04d}',
                                     signed=True, smoke=True), output / 'intentional_interruption.json')
                dist.barrier()
                raise RuntimeError('Controlled native SFT smoke stopped after a signed optimizer step')
        if not smoke and sorted(visited) != list(range(len(rows))):
            raise RuntimeError('Epoch training example coverage incomplete')
    final = assert_same_weights()
    if step != (3 if smoke else config['total_steps']):
        raise RuntimeError('Training step budget differs')
    if rank == 0:
        final_file = output / 'final.safetensors'
        pending = output / 'final.safetensors.pending'
        if final_file.exists() or pending.exists():
            if resume is None:
                raise FileExistsError('Existing native final bridge without a signed resume')
            archive = output / 'recovery'
            archive.mkdir(exist_ok=True)
            for path in (final_file, pending):
                if path.exists():
                    shutil.copyfile(path, archive / f'{path.name}-before-resume-{time.time_ns()}')
        save_file({f'P.{name}': value.detach().cpu().contiguous() for name, value in bridge.P.state_dict().items()},
            pending, metadata=dict(experiment=str(config['id']),
                layer_mapping=bridge.mapping,
                layer_map_0_based=json.dumps(bridge.indices, separators=(',', ':'))))
        pending.replace(final_file)
    dist.barrier()
    try:
        complete_aligned_run(config, bridge, tokenizer, validation, final_test, output, optimizer,
            dict(complete=True, smoke=smoke, steps=step, epochs=1 if smoke else 3,
                 schedule_total_steps=config['total_steps'], singleton_tail_verified=smoke,
                 seconds=time.monotonic() - started, final_sha256=final, initial_sha256=initial,
                 checkpoint='final.safetensors', replica_exact_weights=True, backbones_frozen_verified=True))
    finally:
        dist.destroy_process_group()
