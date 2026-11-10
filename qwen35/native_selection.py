"""Signed accuracy-only selection for new native runs; frozen bundles stay intact."""
import hashlib
import math
from pathlib import Path

POLICY = 'validation_best_accuracy_full_budget_v2'


def contract():
    return dict(selection_policy=POLICY,
                selection_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def verify(protocol):
    if any(protocol.get(key) != value for key, value in contract().items()):
        raise ValueError('Native selection policy changed; prepare a fresh run')


def early_stop_update(state, step, accuracy, clipping, config):
    """Retain native history format, select by accuracy, and finish the full budget."""
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (accuracy, clipping)):
        raise RuntimeError('Invalid validation accuracy/clipping rate')
    state = dict(state or dict(best_step=None, best_accuracy=None, best_clipping=None,
                              fails=0, history=[]))
    improved = state['best_accuracy'] is None or accuracy > state['best_accuracy']
    if improved:
        state.update(best_step=step, best_accuracy=accuracy, best_clipping=clipping, fails=0)
    else:
        state['fails'] += 1
    samples = config.get('samples_per_question', 4)
    state.update(samples_per_question=samples, last_eval_step=step, clip_guard=False)
    state['history'] = [*state.get('history', []), dict(step=step,
        **{f'avg_at_{samples}': accuracy}, clipping_rate=clipping, improved=improved,
        clip_guard=False, fails=state['fails'])]
    maximum = config.get('max_steps')
    return state, 'max_steps' if maximum is not None and step >= maximum else None


def validation_update(state, step, summary, dataset, maximum):
    return early_stop_update(state, step, summary['accuracy'], summary['clipping_rate'],
                             dict(max_steps=maximum, samples_per_question=4 if dataset == 'dapo' else 16))


def install(trainer):
    """Replace the entry module's imported selectors before its main() starts."""
    if hasattr(trainer, 'early_stop_update'):
        trainer.early_stop_update = early_stop_update
    if hasattr(trainer, 'validation_update'):
        trainer.validation_update = validation_update
