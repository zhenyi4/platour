"""Fresh uniform-endpoint zero hidden-bridge SFT on saved Qwen3 trajectories."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
HERE = Path(__file__).absolute().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(HERE)]
import generate_qwen35_train_suite as suite
from c2c_sft_17b_common import DATA, BASELINES, PYTHON, CPU, environment
PYTHON=Path(os.environ.get('ARC_SFT_PYTHON',str(PYTHON)))
CPU=Path(os.environ.get('ARC_SFT_SCORING_PYTHON',str(CPU)))

BASE = ROOT / 'outputs/llm_communicate/qwen3_4b_8b_to_17b_hidden_zero_uniform_gsm8k_math_sft_ep3'
SAMPLING = dict(temperature=.7, top_p=.8, top_k=20, min_p=0.,
                presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)


def env(gpus):
    value=environment(gpus, compat=False)
    value['WANDB_MODE']=os.environ.get('WANDB_MODE','disabled')
    value['WANDB_DISABLED']=os.environ.get('WANDB_DISABLED','true')
    value['PLATOUR_GENERATION_PYTHON']=os.environ.get('PLATOUR_GENERATION_PYTHON',str(PYTHON))
    value['PLATOUR_SCORING_PYTHON']=os.environ.get('PLATOUR_SCORING_PYTHON',str(CPU))
    return value


def digest(bridge):
    import torch
    h = hashlib.sha256()
    for name, value in bridge.maps.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_bridge(base, checkpoint=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from direct_residual_chat_bridge import DirectResidualChatBridge
    protocol = json.loads((base / 'protocol.json').read_text())
    options = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
                   device_map={'': torch.cuda.current_device()})
    student = AutoModelForCausalLM.from_pretrained(protocol['student_checkpoint'], **options)
    teacher = AutoModelForCausalLM.from_pretrained(protocol['teacher_checkpoint'], **options)
    if protocol.get('teacher_grouped_mm',False):
        from qwen3_frozen_grouped_moe import accelerate_frozen_teacher
        teacher.requires_grad_(False).eval()
        audit=accelerate_frozen_teacher(teacher)
        suite.save(base/'teacher_runtime.json',audit)
    bridge = DirectResidualChatBridge(student, teacher, variant='teacher_residual',
        initialization='zero', lm_chunk_tokens=128).eval()
    bridge.pad_token_id = AutoTokenizer.from_pretrained(protocol['student_checkpoint'], local_files_only=True).pad_token_id
    assert bridge.layer_map == protocol['layer_map']
    assert sum(p.numel() for p in bridge.bridge_parameters()) == protocol['trainable_parameters']
    if checkpoint:
        bridge.load_projectors(checkpoint)
    else:
        assert all(p.count_nonzero() == 0 for p in bridge.bridge_parameters())
    return bridge


def prepare(root, receiver='Qwen3-1.7B', senders=('Qwen3-4B', 'Qwen3-8B'), baselines=BASELINES):
    import numpy as np
    from transformers import AutoTokenizer
    from hidden_bridge import bridge_layer_map
    from nemotron_chat_data import INDEX_DTYPE
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models' / receiver, local_files_only=True)
    student_config = json.loads((ROOT / 'data/models' / receiver / 'config.json').read_text())
    receiver_tag = receiver.replace('Qwen3-', '').lower().replace('.', '')
    exports = json.loads((DATA / 'sft_exports.json').read_text())
    jobs = []
    for sender in senders:
        teacher_tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models' / sender, local_files_only=True)
        assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
        teacher_config = json.loads((ROOT / 'data/models' / sender / 'config.json').read_text())
        for task, count, cap, prompt_cap, micro in [('gsm8k', 7473, 2048, 1024, 8),
                                                   ('math', 7500, 4096, 2048, 4)]:
            name = sender.replace('Qwen3-', '').lower() + '_to_' + receiver_tag + '_' + task
            base = root / name
            base.mkdir(exist_ok=True)
            source = DATA / task / sender / 'scored.jsonl'
            export = exports['exports'][task][sender]
            assert suite.sha(Path(export['path'])) == export['sha256']
            rows = suite.read(source)
            assert len(rows) == count and [r['index'] for r in rows] == list(range(count))
            order = __import__('torch').randperm(count, generator=__import__('torch').Generator().manual_seed(42)).tolist()
            train_size = int(.99 * count)
            train, heldout = order[:train_size], order[train_size:]
            data = base / 'data'
            data.mkdir(exist_ok=True)
            audit, index, offset = [], [], 0
            with (data / 'tokens.bin.tmp').open('wb') as binary:
                for r in rows:
                    prefix = tokenizer.apply_chat_template([dict(role='user', content=r['input_text'])],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                    if hasattr(prefix, 'keys'): prefix = prefix['input_ids']
                    assert prefix == r['prompt_token_ids'] and 2 <= len(prefix) <= prompt_cap
                    response = r['response_token_ids']
                    assert response and len(response) == r['tokens'] and len(response) <= cap
                    assert r['finish_reason'] in ('stop', 'length')
                    assert r['finish_reason'] != 'length' or len(response) == cap
                    # Keep original generated token IDs, including natural EOS.
                    # Length-capped responses receive no synthetic EOS.
                    ids = prefix + response
                    np.asarray(ids, dtype=np.uint32).tofile(binary)
                    index.append((offset, len(prefix), len(ids), 0, int(r['index'] in heldout), r['index']))
                    offset += len(ids)
                    audit.append(dict(index=r['index'], prefix=len(prefix), response=len(response),
                                      clipped=r['finish_reason'] == 'length'))
            (data / 'tokens.bin.tmp').replace(data / 'tokens.bin')
            np.asarray(index, dtype=INDEX_DTYPE).tofile(data / 'index.bin')
            suite.immutable(data / 'split.json', dict(train=train, validation=heldout))
            suite.save(data / 'complete.json', dict(records=count, source=str(source), source_sha256=suite.sha(source),
                tokens_sha256=suite.sha(data / 'tokens.bin'), index_sha256=suite.sha(data / 'index.bin'),
                original_generated_ids=True, synthetic_eos_added=False, all_source_trajectories_retained=True))
            steps = math.ceil(train_size / 64)
            mapping = bridge_layer_map(student_config['num_hidden_layers'], teacher_config['num_hidden_layers'], 'uniform_endpoints')
            protocol = dict(method='hidden_bridge_sft', sender=sender, receiver=receiver, dataset=task,
                teacher_checkpoint=str(ROOT / 'data/models' / sender),
                student_checkpoint=str(ROOT / 'data/models' / receiver),
                formula='h_s[i] + P_i(h_t[layer_map[i]])', hidden_space='decoder block input before RMSNorm',
                layer_mapping='uniform_endpoints', layer_map=mapping, mapping_rule='round(i * 35 / 27), zero-based',
                initialization='zero', bias=False, projection_dtype='float32', frozen_backbones=True,
                trainable_parameters=len(mapping) * student_config['hidden_size'] * teacher_config['hidden_size'],
                prefix_only=True, final_prompt_token_native=True, teacher_response_forward=False,
                objective='next-token CE, global valid-response-token mean',
                epochs=3, examples=count, train_examples=train_size, heldout_examples=count - train_size,
                batch_size=64, micro_batch=micro, world_size=8, gpu_ids=list(range(8)),
                steps_per_epoch=steps, total_steps=steps * 3, learning_rate=1e-4,
                optimizer='AdamW', betas=[.9, .999], weight_decay=0., grad_clip=1.,
                scheduler='linear', warmup_ratio=.1, warmup_steps=math.ceil(.1 * steps * 3), seed=42,
                data_order='torch.randperm(train_count), seed42+epoch; every train record once per epoch',
                split='seed42 torch.randperm(source_count), first99% train; same split for both senders',
                max_response_length=cap, max_prompt_length=prompt_cap, max_total_length=cap + prompt_cap,
                source=str(source), source_sha256=suite.sha(source), data_metadata=json.loads((data / 'complete.json').read_text()),
                filter='none; incorrect and clipped answers retained', resume=False,
                eval_sampling=SAMPLING, enable_thinking=False, eval_samples=2 if task == 'gsm8k' else 4,
                eval_questions=1319 if task == 'gsm8k' else 500, eval_response_cap=cap,
                eval_epochs=[3], checkpoints_each_epoch=True,
                wandb_project='hidden-bridge-gsm8k-math-sft',
                wandb_run_name=f'qwen3_{name}_hidden_zero_uniform_endpoints_lr1e4_bs64_3epoch_warmup10_linear_8gpu')
            suite.immutable(base / 'protocol.json', protocol)
            eval_task = 'gsm8k' if task == 'gsm8k' else 'math500'
            prompts = json.loads((baselines / eval_task / receiver / 'prompts.json').read_text())
            assert not {tuple(r['prompt_token_ids']) for r in rows} & {tuple(r['prompt_token_ids']) for r in prompts}
            suite.immutable(base / 'eval_prompts.json', prompts)
            suite.save(base / 'data_audit.json', dict(passed=True, rows=count,
                receiver_prompt_exact=True, original_answer_ids=True, labels_response_only=True,
                clipped=sum(r['clipped'] for r in audit), max_prompt=max(r['prefix'] for r in audit),
                max_response=max(r['response'] for r in audit), response_tokens=sum(r['response'] for r in audit)))
            jobs.append(dict(name=name, root=str(base), sender=sender, task=task))
    suite.immutable(root / 'plan.json', dict(jobs=jobs, epochs=3, gpu_ids=list(range(8)),
        mapping='uniform_endpoints', initialization='zero', source_generation=str(DATA)))
    return jobs


def profile(base):
    import torch
    from nemotron_chat_data import ChatTokens, collate
    torch.set_num_threads(4)
    torch.manual_seed(42)
    bridge = load_bridge(base)
    protocol = json.loads((base / 'protocol.json').read_text())
    dataset = ChatTokens(base / 'data')
    params = bridge.bridge_parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
    # Allocate Adam state before testing exact configured maximum lengths.
    batch = collate(dataset, [0], bridge.pad_token_id, 'cuda')
    _, loss, _ = bridge.chat_loss(batch)
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    del batch, loss
    prompt = dataset[0][0][:dataset[0][1]].tolist()
    n, response, micro = protocol['max_prompt_length'], protocol['max_response_length'], protocol['micro_batch']
    ids = (prompt * math.ceil((n + response) / len(prompt)))[:n + response]
    class Worst:
        def __getitem__(self, index):
            return __import__('numpy').asarray(ids, dtype='int64'), n
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    batch = collate(Worst(), list(range(micro)), bridge.pad_token_id, 'cuda')
    _, loss, _ = bridge.chat_loss(batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    assert torch.isfinite(loss) and float(norm) > 0
    assert all(p.grad is None and not p.requires_grad for m in (bridge.student, bridge.teacher) for p in m.parameters())
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    report = dict(passed=True, safe=peak < .85 * torch.cuda.get_device_properties(0).total_memory / 2**30,
        micro_batch=micro, prompt_tokens=n, response_tokens=response, total_tokens=n + response,
        padding_tokens=0, adam_states_allocated=True, loss=float(loss.detach()), grad_norm=float(norm),
        peak_allocated_gib=peak, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        seconds=time.monotonic() - started, discarded_weights=True)
    suite.save(base / 'worst_case.json', report)
    assert report['safe'], report
    print(json.dumps(report), flush=True)


def train(base, smoke):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from transformers import get_scheduler
    from nemotron_chat_data import ChatTokens, collate
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == 8
    torch.cuda.set_device(local); torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda', local))
    protocol = json.loads((base / 'protocol.json').read_text())
    directory = base / ('smoke' if smoke else 'train')
    directory.mkdir(exist_ok=True)
    assert not (directory / 'metrics.jsonl').exists(), 'Fresh run refuses previous updates'
    bridge = load_bridge(base)
    params = bridge.bridge_parameters()
    initial = digest(bridge)
    hashes = [None] * world
    dist.all_gather_object(hashes, initial); assert len(set(hashes)) == 1
    dataset = ChatTokens(base / 'data')
    split = json.loads((base / 'data/split.json').read_text())
    train_ids, validation = split['train'], split['validation']
    micro = protocol['micro_batch']
    def chunks(group):
        ordered = sorted(group, key=lambda i: int(dataset.index[i]['length']), reverse=True)
        local_ids = ordered[rank::world]
        return [local_ids[i:i+micro] for i in range(0, len(local_ids), micro)]
    def count(i):
        return int(dataset.index[i]['length'] - dataset.index[i]['prefix'])
    def loss(ids):
        batch = collate(dataset, ids, bridge.pad_token_id, 'cuda')
        _, value, _ = bridge.chat_loss(batch)
        assert torch.isfinite(value)
        return value
    @torch.no_grad()
    def validate():
        total = torch.zeros(2, device='cuda', dtype=torch.float64)
        for ids in chunks(validation[:16] if smoke else validation):
            n = sum(count(i) for i in ids)
            total[0] += loss(ids).double() * n; total[1] += n
        dist.all_reduce(total)
        return float(total[0] / total[1])
    # Exact zero bridge must retain the native student response CE.
    batch = collate(dataset, [train_ids[rank]], bridge.pad_token_id, 'cuda')
    with torch.no_grad():
        _, zero, _ = bridge.chat_loss(batch)
        native = bridge.student(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            labels=batch['labels'], use_cache=False).loss
        assert torch.allclose(zero, native, atol=2e-5, rtol=2e-5), (zero, native)
    sanity = dict(native_ce=float(native), zero_bridge_ce=float(zero), zero_preserves_native_ce=True)
    del batch, zero, native
    batches = []
    for epoch in range(3):
        order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(42+epoch)).tolist()
        shuffled = [train_ids[i] for i in order]
        batches.extend(shuffled[i:i+64] for i in range(0, len(shuffled), 64))
    if smoke:
        # Real source longest prompt/reply, ordinary full batches, and singleton tail.
        longest_reply = max(train_ids, key=count)
        longest_prompt = max(train_ids, key=lambda i: int(dataset.index[i]['prefix']))
        chosen = list(dict.fromkeys([longest_reply, longest_prompt, *train_ids[:130]]))[:128]
        batches = [chosen[:64], chosen[64:], chosen[-1:]]
    optimizer = torch.optim.AdamW(params, lr=protocol['learning_rate'], betas=(.9, .999), weight_decay=0.)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=1 if smoke else protocol['warmup_steps'],
        num_training_steps=len(batches))
    run = None
    if rank == 0:
        bridge.save_projectors(directory / 'initial')
        suite.save(directory / 'initialization.json', dict(initial_sha256=initial, replica_hashes=hashes,
            all_projection_weights_zero=True, parent_checkpoint=None, fresh_optimizer=True))
        suite.save(directory / 'sanity.json', sanity)
        if not smoke and protocol.get('wandb_enabled',True) and os.environ.get('WANDB_MODE')!='disabled':
            import wandb
            run = wandb.init(project=protocol['wandb_project'], name=protocol['wandb_run_name'],
                config=protocol, dir=str(directory), mode='online')
            assert not run.offline
            suite.save(directory / 'wandb.json', dict(id=run.id, url=run.url))
    visited, epochs, epoch_nll, epoch_tokens = [], [], 0., 0
    start = time.monotonic()
    try:
        initial_validation = validate()
        if rank == 0:
            suite.save(directory / 'initial_validation.json', dict(nll=initial_validation))
        for step, group in enumerate(batches, 1):
            tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            tokens = sum(count(i) for i in group)
            total = torch.zeros((), device='cuda', dtype=torch.float64)
            for ids in chunks(group):
                n = sum(count(i) for i in ids)
                value = loss(ids)
                total += value.detach().double() * n
                (value * (n / tokens)).backward()
            for p in params:
                if p.grad is None: p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
            dist.all_reduce(total)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            assert all(not p.requires_grad and p.grad is None for m in (bridge.student, bridge.teacher) for p in m.parameters())
            used_lr = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            visited.extend(group); epoch_nll += float(total); epoch_tokens += tokens
            duration = torch.tensor(time.monotonic()-tick, device='cuda')
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            record = dict(step=step, total_steps=len(batches), train_nll=float(total / tokens),
                learning_rate=used_lr, grad_norm=float(norm), examples=len(group), response_tokens=tokens,
                records_seen=len(visited), seconds=time.monotonic()-start, step_seconds=float(duration),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            epoch_end = step % protocol['steps_per_epoch'] == 0 if not smoke else step == len(batches)
            if epoch_end:
                epoch = step // protocol['steps_per_epoch'] if not smoke else 1
                heldout = validate()
                epoch_row = dict(epoch=epoch, train_nll=epoch_nll/epoch_tokens, heldout_nll=heldout, step=step)
                epochs.append(epoch_row); epoch_nll, epoch_tokens = 0., 0
                record['heldout_nll'] = heldout
                if rank == 0:
                    bridge.save_projectors(directory / f'epoch-{epoch}')
                    torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step),
                        directory / f'epoch-{epoch}/optimizer.pt')
                    suite.save(directory / 'epochs.json', epochs)
                dist.barrier()
                torch.save(dict(cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                    directory / f'epoch-{epoch}/rng.rank-{rank:02d}.pt')
            if rank == 0:
                with (directory / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                suite.save(directory / 'live.json', record)
                if run: run.log({'train/nll': record['train_nll'], 'train/lr': used_lr,
                    'train/grad_norm': float(norm), 'timing/step_seconds': float(duration),
                    **({'validation/nll': record['heldout_nll']} if 'heldout_nll' in record else {})}, step=step)
                print(json.dumps(record), flush=True)
        if not smoke:
            n = len(train_ids)
            assert len(visited) == 3 * n
            assert all(sorted(visited[e*n:(e+1)*n]) == sorted(train_ids) for e in range(3))
        final = digest(bridge)
        dist.all_gather_object(hashes, final); assert len(set(hashes)) == 1 and final != initial
        if rank == 0:
            checkpoint = directory / f'epoch-{1 if smoke else 3}/bridge.pt'
            suite.save(directory / 'complete.json', dict(complete=True, smoke=smoke, world_size=world,
                steps=len(batches), epochs=1 if smoke else 3, records_seen=len(visited), epoch_nll=epochs,
                initial_sha256=initial, final_sha256=final, replica_hashes=hashes, frozen_backbones_verified=True,
                zero_init_preserves_ce=True, all_projection_gradients_nonzero=True,
                singleton_tail_verified=smoke, seconds=time.monotonic()-start,
                selected_checkpoint=str(checkpoint), checkpoint_sha256=suite.sha(checkpoint)))
            if run: run.finish()
        dist.barrier()
    except BaseException:
        if run: run.finish(exit_code=1)
        raise
    finally:
        dist.destroy_process_group()


def sources():
    paths = [Path(__file__).absolute(), HERE / 'eval_qwen3_17b_hidden_sft.py',
        HERE / 'qwen3_17b_hidden_sft_vllm_worker.py', HERE / 'c2c_sft_17b_common.py',
        HERE / 'generate_qwen35_train_suite.py', HERE / 'prepare_qwen35_bridge_vllm.py',
        HERE / 'qwen35_hybrid_memory_connector.py', ROOT / 'scripts/nemotron_chat_data.py',
        ROOT / 'scripts/direct_residual_chat_bridge.py', ROOT / 'scripts/shared_subspace_chat_bridge.py',
        ROOT / 'scripts/shared_subspace_bridge.py', ROOT / 'scripts/shared_subspace_residual_bridge.py',
        ROOT / 'scripts/hidden_bridge.py']
    return {str(p.relative_to(ROOT)): suite.sha(p) for p in paths}


def queue(root):
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = json.loads((root / 'plan.json').read_text())['jobs']
    pinned = json.loads((root / 'source_sha256.json').read_text())
    active, completed = None, []
    def stop(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def execute(command, log, phase, job, gpus):
        nonlocal active
        assert sources() == pinned, 'Pinned code changed'
        suite.save(root / 'status.json', dict(phase=phase, job=job['name'], completed=completed, time=time.time()))
        with log.open('a') as stream:
            active = subprocess.Popen(command, cwd=ROOT, env=env(gpus), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            suite.save(root / 'active.json', dict(pid=active.pid, command=command))
            code = active.wait(); active = None
        assert code == 0, f'{phase} failed with {code}: {log}'
    try:
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert len(memory.splitlines()) == 8 and all(int(x) < 512 for x in memory.splitlines())
        for job in jobs:
            base = Path(job['root'])
            if not (base / 'worst_case.json').exists():
                execute([str(PYTHON), '-u', str(Path(__file__).absolute()), 'profile', '--root', str(base)],
                    base / 'profile.log', 'profile', job, [0])
            assert json.loads((base / 'worst_case.json').read_text())['safe']
            for smoke in (True, False):
                phase = 'smoke' if smoke else 'train'
                command = [str(PYTHON), '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                    str(Path(__file__).absolute()), 'train', '--root', str(base)]
                if smoke: command.append('--smoke')
                execute(command, base / f'{phase}.log', phase, job, list(range(8)))
                done = json.loads((base / phase / 'complete.json').read_text())
                assert done['complete'] and done['world_size'] == 8
                output = base / ('smoke_eval' if smoke else 'eval')
                output.mkdir(exist_ok=True)
                execute([str(PYTHON), '-u', str(HERE / 'eval_qwen3_17b_hidden_sft.py'), 'queue',
                    '--root', str(base), '--output', str(output), '--checkpoint', done['selected_checkpoint'],
                    *(['--smoke'] if smoke else [])], base / f'{phase}_eval.log', phase+'_eval', job, [])
                assert json.loads((output / 'summary.json').read_text())['complete']
            completed.append(job['name'])
            suite.save(root / 'results.json', {name: json.loads((root / name / 'eval/summary.json').read_text()) for name in completed})
        suite.save(root / 'status.json', dict(phase='complete', completed=completed, time=time.time()))
    except BaseException as error:
        suite.save(root / 'status.json', dict(phase='stopped' if isinstance(error, SystemExit) else 'failed',
            error=repr(error), completed=completed, time=time.time()))
        raise
    finally:
        if active is not None and active.poll() is None: suite.stop_job(active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'seal', 'queue', 'profile', 'train'])
    parser.add_argument('--root', type=Path, default=BASE)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--receiver', choices=['Qwen3-1.7B', 'Qwen3-0.6B'], default='Qwen3-1.7B')
    parser.add_argument('--senders', nargs='+', choices=['Qwen3-4B', 'Qwen3-8B'], default=['Qwen3-4B', 'Qwen3-8B'])
    parser.add_argument('--baselines', type=Path, default=BASELINES)
    args = parser.parse_args(); root = args.root.absolute()
    if args.stage == 'prepare': print(json.dumps(prepare(root, args.receiver, args.senders, args.baselines.absolute())))
    elif args.stage == 'profile': profile(root)
    elif args.stage == 'train': train(root, args.smoke)
    elif args.stage == 'seal':
        fingerprints = sources(); suite.immutable(root / 'source_sha256.json', fingerprints)
        for name in fingerprints:
            dest = root / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    elif args.detach:
        assert not (root / 'controller.json').exists(), 'Inspect prior controller before relaunch'
        with (root / 'pipeline.log').open('a') as log:
            process = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).absolute()), 'queue', '--root', str(root)],
                cwd=ROOT, env=env([]), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        suite.save(root / 'controller.json', dict(pid=process.pid, started=time.time()))
        print(json.dumps(dict(pid=process.pid, root=str(root))), flush=True)
    else: queue(root)


if __name__ == '__main__': main()
