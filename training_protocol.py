"""Shared formal training budget and vLLM rollout contracts."""
import math


def training_learning_rate(objective, method, requested):
    """Resolve formal defaults while preserving explicit caller choices."""
    if requested is None:
        return (1e-5 if method == 'full' else 1e-4) if objective == 'sft' else (1e-6 if method == 'full' else 1e-5)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('Learning rate must be finite and positive')
    return requested


def sft_schedule(objective, total, warmup_steps=None):
    """Sign the schedule; published recipes may supply their exact warmup count."""
    if objective != 'sft':
        if warmup_steps is not None:
            raise ValueError('warmup_steps is only supported for SFT')
        return {}
    warmup = math.ceil(total * .1) if warmup_steps is None else warmup_steps
    if type(warmup) is not int or not 0 <= warmup <= total:
        raise ValueError('SFT warmup_steps must be an integer between zero and total steps')
    return dict(scheduler='linear', warmup_steps=warmup, schedule_version=1)


def sft_lr_factor(completed, total, warmup):
    if completed < warmup:
        return completed / max(1, warmup)
    return max(0., (total - completed) / max(1, total - warmup))


def qwen3_sft_warmup(task, total):
    """Match published PLatour-SFT counts when controls use the same budget."""
    import json
    from pathlib import Path
    catalog = json.loads((Path(__file__).parent / 'qwen3/experiments.json').read_text())
    counts = {entry['warmup_steps'] for entry in catalog['experiments']
              if entry['method'] == 'sft' and entry['task'] == task and entry['total_steps'] == total}
    return counts.pop() if len(counts) == 1 else math.ceil(total * .1)


def set_sft_learning_rate(optimizer, protocol, completed):
    """Reconstruct LR from the signed step, including after sparse checkpoint resume.

    Receiver checkpoints store optimizer state but no scheduler. This stateless
    schedule has no hidden counter: step n stores the LR for update n+1.
    """
    if protocol['objective'] == 'sft':
        lr = protocol['learning_rate'] * sft_lr_factor(
            completed, protocol['max_steps'], protocol['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = lr


def qwen3_batch_size(objective, task):
    """Use the published recipe's effective batch for Qwen3 controls."""
    import json
    from pathlib import Path
    catalog = json.loads((Path(__file__).parent / 'qwen3/experiments.json').read_text())
    if 'batch_size' in catalog[objective]:
        return catalog[objective]['batch_size']
    sizes = {entry['batch_size'] for entry in catalog['experiments']
             if entry['method'] == objective and entry['task'] == task}
    if len(sizes) != 1:
        raise ValueError('No unique published Qwen3 batch size; specify --batch-size for this custom task')
    return sizes.pop()


def qwen3_micro_batch(objective, task):
    """PACT SFT micro-batches: cap GSM8K at four; GRPO keeps its own layout."""
    import json
    from pathlib import Path
    catalog = json.loads((Path(__file__).parent / 'qwen3/experiments.json').read_text())
    if objective != 'sft':
        return 1
    sizes = {entry['micro_batch'] for entry in catalog['experiments']
             if entry['method'] == objective and entry['task'] == task and entry['sender'] == 'Qwen3-8B'}
    size = sizes.pop() if len(sizes) == 1 else 1
    return min(size, 4) if task == 'gsm8k' else size


def checkpoint_steps(validation, total, interval=0, *, save_initial=True):
    """Save validation/final states; omit reference-only initial measurements."""
    if type(interval) is not int or interval < 0:
        raise ValueError('Checkpoint interval must be a nonnegative integer')
    steps = {0, total, *validation} | (set(range(interval, total + 1, interval)) if interval else set())
    if not save_initial:
        steps.discard(0)
    return sorted(steps)


def restart_before_first_checkpoint(root):
    """Archive unsaved history; a fresh model must repeat its initial validation."""
    import time
    from pathlib import Path
    from qwen35.checkpoints import reconcile_history
    root = Path(root)
    reconcile_history(root, 0, None)
    for path in (root / 'evaluation').glob('step-0000*'):
        recovery = root / 'recovery'
        recovery.mkdir(exist_ok=True)
        path.rename(recovery / f'{path.name}-before-restart-{time.time_ns()}')


def rollout_log_probs(row, cap):
    import torch
    tokens, values = row.get('response_token_ids'), row.get('log_probs')
    if (not tokens or len(tokens) > cap or values is None or len(values) != len(tokens)
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
        raise RuntimeError('vLLM must return finite log-probabilities for every rollout token')
    return torch.tensor(values, dtype=torch.float32)


def budget(size, batch_size, epochs, expected_epochs, max_steps=None, smoke=False):
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or size < 1):
        raise ValueError('Positive epoch, batch and TRAIN sizes required')
    if not smoke and (epochs != expected_epochs or max_steps is not None):
        raise ValueError(f'Formal training requires {expected_epochs} epochs; max_steps requires smoke=True')
    per_epoch = math.ceil(size / batch_size)
    steps = max_steps if max_steps is not None else epochs * per_epoch
    if type(steps) is not int or steps < 1:
        raise ValueError('Positive optimizer step budget required')
    return per_epoch, steps


def validation_backend(backend, objective, task):
    """SFT always measures teacher-forced NLL; GRPO generates answers."""
    expected = 'hf' if objective == 'sft' else 'vllm'
    if backend is not None and backend != expected:
        raise ValueError('SFT validation requires HF held-out NLL; GRPO validation requires vLLM')
    return expected


def default_validation_schedule(objective, family='qwen3'):
    return 'epoch' if objective == 'sft' else 'half_epoch' if family == 'qwen3' else 'recipe'


def validation_steps(schedule, objective, backend, task, per_epoch, total, *, smoke=False,
                     family='qwen3', case=None):
    if schedule == 'auto':
        schedule = default_validation_schedule(objective, family)
    if schedule not in ('step', 'epoch', 'half_epoch', 'recipe', 'initial', 'final'):
        raise ValueError('Unknown validation schedule')
    if schedule == 'final':
        if not smoke or objective != 'grpo' or backend != 'vllm':
            raise ValueError('Final-only validation is restricted to vLLM GRPO smoke training')
        return [total]
    if schedule == 'initial':
        if not smoke or objective != 'grpo' or backend != 'vllm' or family != 'qwen3':
            raise ValueError('Initial-only validation is restricted to vLLM GRPO smoke training')
        return [0]
    if schedule == 'epoch':
        if objective != 'sft':
            raise ValueError('Epoch validation requires SFT')
        return [0, *sorted(set(range(per_epoch, total + 1, per_epoch)) | {total})]
    if schedule == 'half_epoch':
        if objective != 'grpo' or family != 'qwen3':
            raise ValueError('Half-epoch validation requires Qwen3 GRPO')
        steps = {(i * per_epoch + 1) // 2 for i in range(1, 2 * math.ceil(total / per_epoch) + 1)}
        return sorted({0, total, *(step for step in steps if step <= total)})
    if schedule == 'recipe':
        if objective != 'grpo' or family != 'qwen35' or task not in ('dapo', 'deepcoder'):
            raise ValueError('Recipe validation requires Qwen3.5 GRPO')
        # Preserve the published DeepCoder exception: only IDs 26/28 validate
        # at both 294 and 295. Independent sessions identify these by method/shape.
        from qwen35.catalog import MATH_EVAL, CODE_EVAL, CODE_EVAL_WITH_294
        if task == 'dapo' and total == 264:
            steps = MATH_EVAL
        elif task == 'deepcoder' and total == 295:
            identifier = getattr(case, 'identifier', None)
            extra = identifier in (26, 28) if identifier is not None else (
                getattr(case, 'method', None) == 'random' or
                (getattr(case, 'method', None) == 'bridge' and getattr(case, 'teacher_dim', None) == 2048))
            steps = CODE_EVAL_WITH_294 if extra else CODE_EVAL
        else:
            steps = range(33 if task == 'dapo' else 42, total + 1, 33 if task == 'dapo' else 42)
        return sorted({0, total, *steps})
    return list(range(total + 1))


def training_only_record(root, weight_name, steps, identity, initial):
    """Keep the final smoke state; an initial-only measurement cannot select a model."""
    from qwen35.checkpoints import sha256
    checkpoint = f'checkpoints/step-{steps:04d}'
    return dict(policy='final_training_state_no_selection_v1', metric=None,
        best_step=steps, best_score=None, training_steps=steps,
        checkpoint=checkpoint, checkpoint_sha256=sha256(root / checkpoint / weight_name),
        protocol_sha256=identity, test_used_for_selection=False, candidates=[],
        selection_performed=False, initial_validation=initial)
