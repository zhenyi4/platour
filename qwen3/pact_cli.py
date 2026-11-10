"""Full Qwen3 PACT-Check using checksum-pinned archived IT inputs by default."""

import argparse
import json
from pathlib import Path
import sys

from platour_it import DEFAULT_ROOT, load_inputs, release, supported_experiments


BACKBONES = {
    'sender': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
    'receiver': ('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
}


def resolve_backbone(role, local, *, cache_dir=None, local_files_only=False):
    if local is not None:
        path = Path(local).expanduser().resolve()
        if not (path/'config.json').is_file():
            raise ValueError('Local backbone must contain config.json: ' + str(path))
        return path
    from huggingface_hub import snapshot_download
    repo_id, revision = BACKBONES[role]
    return Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir,
        local_files_only=local_files_only, allow_patterns=['*.safetensors', '*.json',
        '*.model', 'tokenizer*', 'vocab.*', 'merges.txt', 'chat_template*', '*.tiktoken']))


def prepare_inputs(experiment, *, it_data_root=DEFAULT_ROOT, data_root=None,
                   cache_dir=None, local_files_only=False):
    """Align published TRAIN/VAL to the fixed TEST template without touching questions."""
    from platour_data import DATA, ensure_file, experiment_files, registry
    from pact_check import PACTCheck, _fingerprint
    test, donors, config, audit = load_inputs(experiment, it_data_root,
        cache_dir=cache_dir, local_files_only=local_files_only)
    template = release()['full_pact_templates'][str(experiment)]
    prefix, suffix = template['prefix_ids'], template['suffix_ids']
    source_suffix = template['source_suffix_ids']
    if any(not isinstance(ids, list) or not ids or
           any(type(token) is not int or token < 0 for token in ids)
           for ids in (prefix, suffix, source_suffix)) or len(suffix) < 2:
        raise ValueError('Invalid pinned full PACT template')
    for row in test:
        tokens = row['prompt_token_ids']
        if (len(tokens) <= len(prefix)+len(suffix) or tokens[:len(prefix)] != prefix
                or tokens[-len(suffix):] != suffix):
            raise ValueError('Frozen TEST differs from the pinned PACT template')
    table, splits, sources = registry(), {}, {}
    for name in experiment_files(experiment, 'train'):
        part = Path(name).stem
        path = ensure_file(name, (DATA if data_root is None else data_root)/name,
                           cache_dir=cache_dir, local_files_only=local_files_only)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        sources[part] = dict(path=name, sha256=table['files'][name]['sha256'])
        aligned = []
        for row in rows:
            tokens = row['prompt_token_ids']
            if (len(tokens) <= len(prefix)+len(source_suffix) or tokens[:len(prefix)] != prefix
                    or tokens[-len(source_suffix):] != source_suffix):
                raise ValueError('Published TRAIN/VAL differs from the pinned source template')
            aligned.append(dict(row, prompt_token_ids=tokens[:-len(source_suffix)] + suffix))
        splits[part] = aligned
    train, validation = splits['train'], splits['validation']
    validation_policy = 'published_train_validation_splits'
    if experiment >= 16 and ({tuple(r['prompt_token_ids']) for r in validation} &
                             {tuple(r['prompt_token_ids']) for r in test}):
        raise ValueError('Published GRPO validation contains evaluation prompts; install the fixed HF splits before PACT-Check')
    # Permit repeated signed rows, retaining indices and all TEST denominators.
    # Cross-split prompt overlap and duplicate indices remain errors.
    PACTCheck(None, train, validation, test, evaluator=None, allow_duplicate_prompts=True)
    if any(not isinstance(row.get('gold'), str) for row in train+validation+test):
        raise ValueError('All splits need independent gold labels')
    audit = dict(audit, full_pact_data={
        'sources': sources, 'source_repository': table['repositories']['qwen3'],
        'validation_policy': validation_policy,
        'template': template, 'template_suffix_replaced': source_suffix != suffix,
        'split_counts': {k: len(v) for k, v in [('train', train), ('validation', validation), ('test', test)]},
        'split_sha256': {k: _fingerprint(v) for k, v in [('train', train), ('validation', validation), ('test', test)]},
        'duplicate_prompt_rows': {k: len(v)-len({tuple(r['prompt_token_ids']) for r in v})
                                 for k, v in [('train', train), ('validation', validation), ('test', test)]},
        'dt_sampling': 'distinct TRAIN prompts only; seed 42',
        'test_policy': 'all archived rows with original indices and fixed effective donors'})
    return train, validation, test, donors, config, audit, prefix, suffix[:-1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from qwen3.parallel import maybe_launch, dist
    if maybe_launch(['pact-check', *argv]):
        return
    from pact_check import CONTROLS, selected_controls
    p = argparse.ArgumentParser(description=__doc__, epilog=
        'Default: run DT, IT, SC, TR. Use --controls to select any subset. '
        'Use --it-source custom for custom Qwen3 data.')
    p.add_argument('--controls', nargs='+', choices=CONTROLS, default=list(CONTROLS))
    from qwen3.parallel import add_arguments
    add_arguments(p)
    p.add_argument('--experiment', type=int, choices=supported_experiments(), required=True,
                   help='Released 8B→1.7B experiments with complete archived IT pairings')
    p.add_argument('--it-data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--it-profile', choices=('archived-it-v1',), default='archived-it-v1')
    p.add_argument('--data-root', type=Path, help='Published data root containing qwen3/<experiment>/')
    p.add_argument('--output', type=Path)
    for role in ('sender', 'receiver', 'bridge'):
        p.add_argument('--'+role, type=Path, help='Optional matching local released model/checkpoint')
    p.add_argument('--backend', choices=('vllm',), default='vllm')
    p.add_argument('--vllm-python', type=Path,
                             help='Optional vLLM interpreter override; defaults to current project Python')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Verify all full-suite inputs without loading models')
    p.add_argument('--control-steps', type=int, help=argparse.SUPPRESS)
    p.add_argument('--grpo-smoke-steps', type=int,
                   help='SC/TR GRPO only: initial VAL, N training steps, no further VAL or TEST')
    p.add_argument('--batch-size', type=int,
                   help='Effective training batch; default follows the published recipe (SFT: 64)')
    p.add_argument('--micro-batch', type=int,
                   help='Actual SFT micro-batch; defaults to ARC-C: 8, GSM8K: 4; effective batch is unchanged')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Extra recovery checkpoint interval; 0 saves only at selection-eligible VAL/final steps')
    p.add_argument('--control-objective', choices=('sft', 'grpo'),
                   help='Default: SFT for 07–12, GRPO for 16–18')
    p.add_argument('--tr-method', choices=('full', 'lora'), default='full',
                   help='TR training method (default: full)')
    p.add_argument('--tr-lora-rank', type=int, default=128)
    p.add_argument('--dt-samples', type=int, default=500)
    a = p.parse_args(argv)
    if not a.prepare_only:
        dist().init()
    try:
        a.controls = selected_controls(a.controls)
    except ValueError as error:
        p.error(str(error))
    if a.control_steps is not None:
        p.error('Qwen3 SC/TR train for 3 epochs; remove --control-steps')
    if not a.prepare_only and a.output is None:
        p.error('Full PACT-Check requires --output')
    if any(n < 1 for n in (a.tr_lora_rank, a.dt_samples)) or (a.batch_size is not None and a.batch_size < 1):
        p.error('Training and calibration budgets must be positive')
    if (a.micro_batch is not None and a.micro_batch < 1) or a.checkpoint_every < 0:
        p.error('Micro-batch must be positive and checkpoint interval nonnegative')
    prepared = prepare_inputs(a.experiment, it_data_root=a.it_data_root, data_root=a.data_root,
                              cache_dir=a.cache_dir, local_files_only=a.local_files_only)
    train, validation, test, donors, config, audit, a.prefix_ids, a.suffix_ids = prepared
    a.control_objective = a.control_objective or ('grpo' if a.experiment >= 16 else 'sft')
    if a.grpo_smoke_steps is not None and (a.grpo_smoke_steps < 1 or
            a.control_objective != 'grpo' or not set(a.controls) <= {'SC', 'TR'}):
        p.error('--grpo-smoke-steps requires positive steps, GRPO, and only SC/TR controls')
    if a.batch_size is None:
        from training_protocol import qwen3_batch_size
        a.batch_size = qwen3_batch_size(a.control_objective, config['task'])
    if a.micro_batch is None:
        from training_protocol import qwen3_micro_batch
        a.micro_batch = qwen3_micro_batch(a.control_objective, config['task'])
    if set(a.controls) & {'SC', 'TR'} and a.control_objective == 'sft' and any(not r.get('response_token_ids') for r in train+validation):
        p.error('SFT controls require response_token_ids for every TRAIN/VAL row')
    if 'DT' in a.controls and len({tuple(row['prompt_token_ids']) for row in train}) < a.dt_samples:
        p.error('DT sample count exceeds the number of distinct TRAIN prompts')
    a.task, a.mapping, a.samples, a.response_cap = (config[k] for k in ('task', 'mapping', 'samples', 'cap'))
    a.semantic_group_field = None
    print(json.dumps(dict(experiment=a.experiment, controls=a.controls,
        questions=len(test), responses_per_setting=0 if a.grpo_smoke_steps else len(test)*a.samples,
        test_performed=a.grpo_smoke_steps is None, smoke_only=a.grpo_smoke_steps is not None,
        sampling=config['sampling'],
        mapping=a.mapping, cap=a.response_cap, control_objective=a.control_objective,
        control_epochs=None if a.grpo_smoke_steps else 3, batch_size=a.batch_size,
        micro_batch=a.micro_batch, checkpoint_every=a.checkpoint_every,
        control_training_steps={label: a.grpo_smoke_steps or 3 * ((len(train) + a.batch_size - 1) // a.batch_size)
                                for label in ('SC', 'TR') if label in a.controls},
        control_validation_metric='heldout_nll' if a.control_objective == 'sft' else 'accuracy',
        control_validation_schedule='initial' if a.grpo_smoke_steps else 'epoch' if a.control_objective == 'sft' else 'half_epoch',
        data_audit=audit), ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    import torch
    from platour_exps import ensure_weight, require_vllm_python, verify_weight, weight_record
    from pact_check import _run_qwen3
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; full PACT-Check has not started')
    require_vllm_python(a.vllm_python)
    if a.local_files_only and a.bridge is None:
        p.error('--local-files-only execution requires --bridge with the released checkpoint')
    a.bridge = a.bridge or ensure_weight(a.experiment)
    _, _, size, digest = weight_record(a.experiment)
    verify_weight(a.bridge, size, digest)
    for role in ('sender', 'receiver'):
        setattr(a, role, resolve_backbone(role, getattr(a, role), cache_dir=a.cache_dir,
                                         local_files_only=a.local_files_only))
    return _run_qwen3(a, train, validation, test, donors, fixed_config=config,
                      fixed_it_audit=audit, allow_duplicate_prompts=True)


if __name__ == '__main__':
    main()
