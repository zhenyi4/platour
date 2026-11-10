"""Isolated 35B-to-4B formal protocols: monotone B mapping, validation and early stop."""
import hashlib
import json
import math
from pathlib import Path
import tempfile

MAPPING = 'equal_width_full_to_full_monotone_b'
RULE = ('endpoint-aligned round-half-up; snap student full to nearest teacher full, ties shallower; '
        'lower preceding linear targets at inversions to make the mapping nondecreasing')


def layer_map_b(student_types, teacher_types):
    ns, nt = len(student_types), len(teacher_types)
    if ns < 2 or nt < ns:
        raise ValueError('Need at least two student layers and a no-shallower teacher')
    mapping = [(2*i*(nt-1)+ns-1)//(2*(ns-1)) for i in range(ns)]
    full = [j for j, kind in enumerate(teacher_types) if kind == 'full_attention']
    if 'full_attention' in student_types and not full:
        raise ValueError('Teacher has no full-attention layers')
    for i, j in enumerate(mapping):
        if student_types[i] == 'full_attention' and teacher_types[j] != 'full_attention':
            mapping[i] = min(full, key=lambda k: (abs(k-j), k))
    for i in range(ns-2, -1, -1):
        if mapping[i] > mapping[i+1]:
            if student_types[i] != 'linear_attention':
                raise ValueError('Monotone repair would modify a full-attention target')
            mapping[i] = mapping[i+1]
    assert all(a <= b for a, b in zip(mapping, mapping[1:]))
    assert all(teacher_types[j] == 'full_attention' for i, j in enumerate(mapping)
               if student_types[i] == 'full_attention')
    return mapping


def validation_schedule(examples, batch_size, epochs, interval=33, max_steps=None):
    epoch_steps = math.ceil(examples/batch_size)
    steps = sorted({epoch*epoch_steps+offset for epoch in range(epochs)
                    for offset in [*range(interval, epoch_steps, interval), epoch_steps]})
    if max_steps is not None:
        if max_steps not in steps:
            raise ValueError('Hard cap must coincide with a validation/checkpoint')
        steps = [s for s in steps if s <= max_steps]
    return epoch_steps, steps


def prepare(base, dataset):
    from qwen35_bridge_opd_core import ROOT, prepare as core_prepare, early_stop_config, save
    from prepare_qwen35_lcb_grpo import build
    base = Path(base).absolute()
    if dataset not in ('dapo', 'lcb'):
        raise ValueError(dataset)
    if (base/'protocol.json').exists():
        raise ValueError('Use a fresh immutable run directory')
    with tempfile.TemporaryDirectory(dir=str(base.parent)) as scratch:
        template = Path(scratch)/'template'
        if dataset == 'dapo':
            protocol = core_prepare(template, 'grpo', 'Qwen3.5-35B-A3B', 'relative_depth_groups4',
                                    gpu_count=8, dataset='dapo', max_response_length=16384,
                                    epochs=1, evaluations_per_epoch=8,
                                    tuning=dict(learning_rate=1e-5, format_penalty=True))
        else:
            protocol = build(template, 4, 1e-5, True, '')
        for name in ('train_prompts.json', 'eval_prompts.json', 'dataset.json'):
            save(base/name, json.loads((template/name).read_text()))
    student = json.loads((ROOT/'data/models/Qwen3.5-4B/config.json').read_text())['text_config']
    teacher = json.loads((ROOT/'data/models/Qwen3.5-35B-A3B/config.json').read_text())['text_config']
    mapping = layer_map_b(student['layer_types'], teacher['layer_types'])
    pairs = [dict(student_layer=i, teacher_layer=j, student_attention=student['layer_types'][i],
                  teacher_attention=teacher['layer_types'][j]) for i, j in enumerate(mapping)]
    cap = 132 if dataset == 'dapo' else 52
    epoch_steps, eval_steps = validation_schedule(protocol['examples_per_epoch'], protocol['batch_size'],
                                                 protocol['epochs'], max_steps=cap)
    samples = 4 if dataset == 'dapo' else 16
    early = early_stop_config(min(33, epoch_steps), patience=2, max_steps=cap)
    early.update(metric=f'avg@{samples}', samples_per_question=samples,
                 rule=f'Stop after 2 consecutive validations not strictly better than best avg@{samples}; cap {cap}')
    if dataset == 'lcb':
        early.update(clip_abs=None, clip_ratio=None, clip_guard_enabled=False)
    else:
        early['clip_guard_enabled'] = True
    protocol.pop('teacher_group_map', None)
    protocol.update(layer_mapping=MAPPING, layer_map=mapping, mapping_group_size=1, group_constraint=False,
                    monotone_constraint=True, mapping_rule=RULE, full_layer_constraint=True,
                    nearest_full_tie_break='shallower', attention_types_preserved=False,
                    attention_layer_pairs=pairs, cross_attention_mapping_allowed=True,
                    formula='h_s[i] + P_i(h_t[layer_map[i]])', early_stop=early,
                    steps_per_epoch=epoch_steps, eval_steps=eval_steps, checkpoint_steps=eval_steps,
                    total_evaluations=1+len(eval_steps), eval_samples=samples, eval_seed_stride=samples,
                    eval_seed=f'1234 + {samples}*index + sample_index', validation_before_training=True,
                    initial_eval_step=0, eval_selection=f'Best validation avg@{samples}; never select on test',
                    eval_coverage=f'{protocol["eval_questions"]} questions x{samples} at 0 and {eval_steps}',
                    wandb_project='platour',
                    wandb_run_name=f'platour_qwen35_35b_to_4b_{dataset}_{MAPPING}_grpo_lr1e5_fmtpen_g8_bs64_16k_earlystop_valavg{samples}',
                    source_execution_root=str(ROOT), test_samples=8 if dataset == 'lcb' else None)
    if dataset == 'lcb':
        protocol['eval_metric'] = 'avg@16 = pass-all-tests validation responses / 1600; pass@16 separately'
    dataset_info = json.loads((base/'dataset.json').read_text())
    dataset_info.update(train_prompts_sha256=hashlib.sha256((base/'train_prompts.json').read_bytes()).hexdigest(),
                        eval_prompts_sha256=hashlib.sha256((base/'eval_prompts.json').read_bytes()).hexdigest())
    save(base/'dataset.json', dataset_info)
    save(base/'mapping.json', dict(layer_mapping=MAPPING, rule=RULE, teacher_layers_0based=mapping,
                                   teacher_layers_1based=[j+1 for j in mapping], attention_layer_pairs=pairs))
    save(base/'protocol.json', protocol)
    return protocol
