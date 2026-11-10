"""Eight-GPU native4B LoRA GRPO: matched code52/math132 budgets, independent native initialization."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist

from full4b_core import log_probs
from full4b_train import valid_backend_logprobs
from lora4b_core import LoRADecoder, adapter_digest, base_digest, checkpoint, generate, load_checkpoint, validation_update
from qwen35_bridge_grpo_core import group_advantages, policy_loss
from qwen35_bridge_opd_core import ROOT, TRAIN_SAMPLING, EVAL_SAMPLING, save
from qwen35_native_actor_grpo import load_bridge
from train_qwen35_bridge_opd_vllm import RPC as LocalRPC


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def status(run, rank, phase, step):
    save(run / f'status.rank-{rank:02d}.json', dict(phase=phase, step=step, time=time.time()))
    if rank == 0:
        save(run / 'status.json', dict(phase=phase, step=step, time=time.time()))


def verify_replicas(actor, world):
    identity = adapter_digest(actor)
    identities = [None] * world
    dist.all_gather_object(identities, identity)
    if len(set(identities)) != 1:
        raise RuntimeError('LoRA factors differ across training replicas')
    return identity


def audit(actor, decoder, row, run, rank, step):
    request = dict(row, sample_index=0, sampling_seed=42)
    response = generate(actor, decoder, [request], dict(EVAL_SAMPLING, temperature=0.), 1,
                        run / f'audit.rank-{rank:02d}/step-{step:04d}', True)[0]
    with torch.no_grad():
        hf = float(log_probs(actor.student, row['prompt_token_ids'], response['response_token_ids'], checkpointing=False)[0])
    native = response['rollout_log_probs'][0]
    passed = valid_backend_logprobs(hf, native)
    save(run / f'audit.rank-{rank:02d}/comparison-{step:04d}.json', dict(passed=passed,
        hf_logprob=hf, vllm_logprob=native, absolute_error=abs(hf - native),
        native_adapter_loaded_and_audited=True, adapter_state_sha256=adapter_digest(actor), maximum_logprob_gap=.6931471805599453))
    if not passed:
        raise RuntimeError('Native LoRA/HF probability sanity exceeds existing token IS cap2')


def validate(actor, decoder, scorer, validation, run, step, rank, world, protocol, wandb_run, smoke):
    chosen = validation[:8] if smoke else validation
    samples = 2 if smoke else protocol['eval_samples']
    requests = [dict(row, sample_index=j, sampling_seed=1234 + samples * row['index'] + j)
                for row in chosen[rank::world] for j in range(samples)]
    directory = run / 'evaluation' / f'step-{step:04d}'
    started = time.monotonic()
    status(run, rank, 'validation', step)
    generate(actor, decoder, requests, EVAL_SAMPLING, protocol['max_response_length'], directory / f'rank-{rank:02d}')
    dist.barrier()
    summary = None
    if rank == 0:
        request = dict(inputs=[str(directory / f'rank-{r:02d}/responses.json') for r in range(world)],
                       expected_responses=len(chosen) * samples, result=str(directory / 'summary.json'))
        if protocol['dataset'] == 'dapo':
            request['metric'] = 'answer'
        summary = scorer.call(request)
        if not summary['complete'] or summary['gold_missing'] or summary['responses'] != len(chosen) * samples:
            raise RuntimeError('Incomplete LoRA validation')
        summary.update(step=step, samples_per_question=samples, wall_seconds=time.monotonic() - started,
                       adapter_state_sha256=adapter_digest(actor))
        save(directory / 'summary.json', summary)
        if wandb_run:
            wandb_run.log({f'val/avg@{samples}': summary['accuracy'], f'val/pass@{samples}': summary['pass_at_k'],
                           'val/clipping_rate': summary['clipping_rate'], 'val/mean_tokens': summary['mean_tokens'],
                           'training/global_step': step}, step=step)
    holder = [summary]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    protocol = read(args.root / 'protocol.json')
    for name, digest in read(args.root.parent / 'source_manifest.json').items():
        if sha(args.root.parent / name) != digest:
            raise RuntimeError('LoRA frozen source/input changed: ' + name)
    if (protocol['learning_rate'] != 1e-5 or protocol['lora_rank'] != 128 or protocol['lora_alpha'] != 128 or
            protocol['lora_dropout'] != 0 or protocol['total_steps'] != (52 if protocol['dataset'] == 'lcb' else 264) or
            (protocol['dataset'] == 'dapo' and protocol['patience'] != 10**9)):
        raise RuntimeError('LoRA run differs from approved model-side configuration or matched budget')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if rank != int(os.environ['LOCAL_RANK']) or world != 8:
        raise RuntimeError('Native LoRA expects a single eight-GPU node')
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=timedelta(minutes=90))
    run = args.root / ('smoke' if args.smoke else 'train')
    run.mkdir(parents=True, exist_ok=True)
    decoder = scorer = wandb_run = None
    try:
        status(run, rank, 'load_native_lora', 0)
        actor, runtime = load_bridge(ROOT, protocol)
        params = actor.bridge_parameters()
        if sum(p.numel() for p in params) != 259719168 or len(actor.target_modules) != 248:
            raise RuntimeError('Actual LoRA factors/target module count differs from approved configuration')
        actor.assert_trainable_parameters()
        assert all(p.dtype == torch.float32 for p in params)
        if not all(torch.count_nonzero(p).item() == 0 for name, p in actor.bridge_state_tensors().items() if name.endswith('lora_B')):
            raise RuntimeError('Initial LoRA B factors are not zero')
        frozen = base_digest(actor)
        optimizer = actor.make_optimizer(protocol['learning_rate'])
        training = read(args.root / 'train_prompts.json')
        validation = read(args.root / 'eval_prompts.json')
        step0 = args.resume_step or 0
        rng = None
        state = None
        if args.resume_step is not None:
            assert not args.smoke
            metadata, rng = load_checkpoint(actor, optimizer, run / f'checkpoints/step-{step0:04d}', rank, world, frozen, sha)
            state = metadata['early_state']
            rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            if not rows or rows[-1]['step'] != step0:
                raise RuntimeError('Resume needs checkpoint-aligned metrics; preserve and diagnose later partial steps first')
        identity = verify_replicas(actor, world)
        save(run / f'parameters.rank-{rank:02d}.json', dict(runtime, frozen_base_sha256=frozen,
             adapter_state_sha256=identity, initialization='A Kaiming uniform/B zero; native base', teacher=False))
        decoder = LoRADecoder(run, rank)
        if rank == 0:
            scorer = LocalRPC('lcb_score_worker.py' if protocol['dataset'] == 'lcb' else 'qwen35_bridge_opd_score_worker.py',
                              run / 'scorer', boxed_only=False)
        audit(actor, decoder, training[rank], run, rank, step0)
        if args.smoke:
            status(run, rank, 'stress_lora_16k', 0)
            actor.prepare_optimizer_for_stress(optimizer)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            eos = actor.student.generation_config.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            probability = log_probs(actor.student, training[0]['prompt_token_ids'], [7] * 16383 + [eos])
            (-probability.mean()).backward()
            actor.assert_trainable_parameters()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            save(run / f'stress.rank-{rank:02d}.json', dict(passed=True, response_tokens=16384,
                seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                native_lora=True, frozen_base=True, optimizer_state_resident=True))
            optimizer.zero_grad(set_to_none=True)
            del probability
            torch.cuda.empty_cache()
            dist.barrier()
        if not args.smoke and rank == 0:
            import wandb
            wandb_run = wandb.init(project='native-lora4b-dapo', config=protocol,
                                   dir=str(run), mode='disabled')
            wandb_run.summary.update(dict(queue_status='running', training_started=True, trainable_parameters=259719168))
        if step0 == 0:
            baseline = validate(actor, decoder, scorer, validation, run, 0, rank, world, protocol, wandb_run, args.smoke)
            state, _ = validation_update(None, 0, baseline, protocol['dataset'], protocol['total_steps'])
        else:
            torch.set_rng_state(rng['cpu'])
            torch.cuda.set_rng_state(rng['cuda'])
        checkpoint(actor, optimizer, run, step0, state, rank, world, frozen, sha)
        total = 2 if args.smoke else protocol['total_steps']
        started = time.monotonic()
        reason = None
        effective_updates = max((int(state_value['step'].item()) if torch.is_tensor(state_value['step']) else int(state_value['step'])
                                 for state_value in optimizer.state.values() if 'step' in state_value), default=0)
        for step in range(step0 + 1, total + 1):
            tick = time.monotonic()
            status(run, rank, 'rollout', step)
            epoch = (step - 1) // protocol['steps_per_epoch']
            offset = (step - 1) % protocol['steps_per_epoch'] * 64
            order = torch.randperm(len(training), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            selected = order[:64] if args.smoke and step == 1 else order[:1] if args.smoke else order[offset:offset + 64]
            requests = [dict(training[index], sample_index=j, sampling_seed=1_000_000 + step * 512 + i * 8 + j)
                        for i, index in enumerate(selected) if i % world == rank for j in range(8)]
            directory = run / 'rollouts' / f'step-{step:04d}'
            local_rows = generate(actor, decoder, requests, TRAIN_SAMPLING, 16384, directory / f'rank-{rank:02d}', True)
            dist.barrier()
            summary = None
            if rank == 0:
                summary = scorer.call(dict(inputs=[str(directory / f'rank-{r:02d}/responses.json') for r in range(world)],
                    expected_responses=len(selected) * 8, format_penalty=True, result=str(directory / 'summary.json')))
            holder = [summary]
            dist.broadcast_object_list(holder, src=0)
            summary = holder[0]
            all_rows = [json.loads(line) for line in (directory / 'scored.jsonl').read_text().splitlines()]
            advantages, counts = group_advantages(all_rows, 8)
            optimizer.zero_grad(set_to_none=True)
            status(run, rank, 'lora_backward', step)
            for row in local_rows:
                advantage = advantages[row['index'], row['sample_index']]
                if advantage == 0:
                    continue
                lp = log_probs(actor.student, row['prompt_token_ids'], row['response_token_ids'])
                loss, _ = policy_loss(lp, advantage, row['rollout_log_probs'])
                (loss / (len(selected) * 8)).backward()
                del loss, lp
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
            actor.assert_trainable_parameters()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if norm > 0:
                optimizer.step()
                effective_updates += 1
            optimizer.zero_grad(set_to_none=True)
            identity = verify_replicas(actor, world)
            torch.cuda.empty_cache()
            audit(actor, decoder, training[rank], run, rank, step)
            record = dict(step=step, epoch=epoch + 1, examples=len(selected), responses=len(selected) * 8,
                          train_accuracy=summary['accuracy'], accuracy_unpenalized=summary['accuracy_unpenalized'],
                          clipping_rate=summary['clipping_rate'], mean_tokens=summary['mean_tokens'], grad_norm=float(norm),
                          effective_optimizer_updates=effective_updates, adapter_state_sha256=identity,
                          step_seconds=time.monotonic() - tick)
            if rank == 0:
                with (run / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                save(run / 'live.json', record)
                if wandb_run:
                    wandb_run.log({'training/global_step': step, 'response_length/clip_ratio': record['clipping_rate'],
                        'response_length/mean': record['mean_tokens'], 'critic/score/mean': record['train_accuracy'],
                        'actor/grad_norm': record['grad_norm'], 'actor/lr': 1e-5}, step=step)
            if args.smoke and step == 2 or not args.smoke and step in protocol['eval_steps']:
                summary = validate(actor, decoder, scorer, validation, run, step, rank, world, protocol, wandb_run, args.smoke)
                state, decided = validation_update(state, step, summary, protocol['dataset'], protocol['total_steps'])
                reason = None if args.smoke else decided
                checkpoint(actor, optimizer, run, step, state, rank, world, frozen, sha)
                if rank == 0:
                    save(run / 'early_stop.json', dict(state=state, last_step=step, reason=reason))
            if reason:
                break
        if base_digest(actor) != frozen:
            raise RuntimeError('Frozen native base changed during LoRA training')
        if effective_updates <= 0:
            raise RuntimeError('LoRA training must have at least one effective update')
        if rank == 0:
            save(run / 'complete.json', dict(complete=True, steps_run=step, best_step=state['best_step'],
                best_accuracy=state['best_accuracy'], effective_optimizer_updates=effective_updates, reason=reason,
                world_size=world, frozen_base_verified=True, native_lora=True, smoke=args.smoke,
                adapter_state_sha256=identity, elapsed_seconds=time.monotonic() - started))
        dist.barrier()
    except BaseException as error:
        save(run / f'failure.rank-{rank:02d}.json', dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        if decoder is not None:
            decoder.close()
        if scorer is not None:
            scorer.close()
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
