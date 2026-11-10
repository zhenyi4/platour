"""Validation-best checkpoint selection for portable Qwen3.5 training.

Training always runs the full step budget; there is no early stopping. Step zero is the
baseline and every scheduled validation point is a candidate. A point becomes the best
checkpoint when its accuracy strictly exceeds the best so far (ties keep the earlier
step). Clipping rate is diagnostic only and never disqualifies a checkpoint.

TEST is never read.
"""

import math
from pathlib import Path

from .checkpoints import read

POLICY = 'validation_best_accuracy_full_budget_v2'


def update(case, state, step, summary):
    """Add one complete validation result to the selection state."""
    accuracy, clip = summary.get('accuracy'), summary.get('clipping_rate')
    if (not summary.get('complete') or not isinstance(accuracy, (int, float)) or
            not isinstance(clip, (int, float)) or not math.isfinite(accuracy) or
            not math.isfinite(clip) or not 0 <= accuracy <= 1 or not 0 <= clip <= 1):
        raise RuntimeError('Only a complete validation with accuracy and clipping rate may select')
    point = {'step': step, 'accuracy': accuracy, 'clipping_rate': clip}
    if state is None:
        if step != 0:
            raise RuntimeError('Step-zero validation must be the selection baseline')
        return dict(best_step=0, best_accuracy=accuracy, best_clipping=clip, last_eval_step=0,
                    candidates=[dict(point, clip_guard=False, best_so_far=True)])
    if step <= state['last_eval_step']:
        raise RuntimeError('Duplicate or reversed validation step')
    improved = accuracy > state['best_accuracy']
    state = dict(state, last_eval_step=step,
                 candidates=[*state['candidates'], dict(point, clip_guard=False, best_so_far=improved)])
    if improved:
        state.update(best_step=step, best_accuracy=accuracy, best_clipping=clip)
    return state


def replay(case, output, upto, *, steps=None):
    """Recompute the selection state from the run's signed validation files."""
    evaluation = Path(output) / 'evaluation'
    state = None
    for step in (0, *(steps if steps is not None else case.validation_steps)):
        if step > upto:
            break
        path = evaluation / f'step-{step:04d}.json'
        if not path.is_file():
            raise RuntimeError(f'Validation result for step {step} is missing')
        state = update(case, state, step, read(path))
    return state


def selection(case, output, state):
    if state is None:
        raise RuntimeError('No validation history to select a checkpoint from')
    return {'policy': POLICY, 'selected_step': state['best_step'],
            'validation_accuracy': state['best_accuracy'],
            'validation_clipping_rate': state['best_clipping'],
            'last_validation_step': state['last_eval_step'],
            'candidates': state['candidates'], 'test_used_for_selection': False,
            'checkpoint': f"checkpoints/step-{state['best_step']:04d}",
            'experiment': case.identifier}
