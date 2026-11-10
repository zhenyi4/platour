"""DeepCoder complete held-out validation with accuracy-only selection."""

import math


def training_seed(epoch, step, question_index, sample_index):
    """Exact original DeepCoder eight-reply per-question sampling seed."""
    if (not all(type(value) is int and value >= 0
                for value in (epoch, step, question_index, sample_index)) or
            step < 1 or sample_index >= 8):
        raise ValueError('Training seed needs a valid update and one of eight replies')
    return 42 + 100000 * epoch + 1000 * step + 8 * question_index + sample_index


def common_policy(*_args, **_kwargs):
    raise RuntimeError('Historical private DeepCoder preparation is not bundled; '
                       'use qwen35.native_multinode.prepare with the signed public inputs')


def validation_update(state, step, summary, maximum):
    accuracy, clip = summary['accuracy'], summary['clipping_rate']
    if (not summary.get('complete') or not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
            not math.isfinite(clip) or not 0 <= clip <= 1 or not 0 <= step <= maximum):
        raise RuntimeError('Only complete finite held-out validation may select a checkpoint')
    if state is None:
        if step != 0:
            raise RuntimeError('Initial fixed validation must precede optimization')
        return dict(best_step=0, best_accuracy=accuracy, best_clipping=clip, fails=0), None
    if step <= state.get('last_eval_step', state['best_step']):
        raise RuntimeError('Duplicate or reversed validation step')
    improved = accuracy > state['best_accuracy']
    updated = dict(best_step=step if improved else state['best_step'],
                   best_accuracy=accuracy if improved else state['best_accuracy'],
                   best_clipping=clip if improved else state['best_clipping'],
                   fails=0 if improved else state['fails'] + 1,
                   last_eval_step=step, clip_guard=False)
    reason = 'max_steps' if step >= maximum else None
    return updated, reason
