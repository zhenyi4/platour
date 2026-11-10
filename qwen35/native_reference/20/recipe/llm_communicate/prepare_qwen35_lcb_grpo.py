"""Protocol and prompt files for 35B -> 4B bridge GRPO on LiveCodeBench release_v5 (group 2 of the plan).

Model-side settings (layer map, bridge, vLLM, sampling, GRPO hyper-parameters, activation checkpointing, 16K cap)
are taken from qwen35_bridge_opd_core.prepare() for the 35B DAPO-16K run, generated into a scratch directory, so
they cannot drift from the code. Only the task-specific fields are replaced:
  data      LCB release_v5: 780 train / 100 validation (latest dates), official generic prompt, non-thinking
  reward    official LiveCodeBench run_test (pinned commit), 1 iff every public+private test passes, judged in one
            disposable no-network container per response (lcb_score_worker.py / code_judge.py)
  schedule  fixed budget, `--epochs` (default 4), validation 100x4 before training and after every epoch,
            full checkpoint after each validation, no early stopping
Options: --learning-rate, --format-penalty (code rule: a response cut at the length cap gets reward 0; validation is
never penalised), --wandb-run-suffix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from qwen35_bridge_opd_core import ROOT, prepare as core_prepare, quarter_epoch_schedule, save
import code_judge
import code_prompts

DIFFICULTY = {'easy': 0, 'medium': 1, 'hard': 2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lr_tag(lr):
    mantissa, exponent = f'{lr:.0e}'.split('e')
    return f'lr{mantissa}e{abs(int(exponent))}'


def build(base, epochs, learning_rate, format_penalty, suffix):
    from transformers import AutoTokenizer
    base = Path(base).resolve()
    if (base/'protocol.json').exists():
        raise SystemExit(f'{base}/protocol.json exists; protocols are immutable, use a new --root')
    with tempfile.TemporaryDirectory(dir='/opt/dlami/nvme' if Path('/opt/dlami/nvme').exists() else None) as scratch:
        template = core_prepare(Path(scratch)/'dapo_template', 'grpo', 'Qwen3.5-35B-A3B', 'relative_depth_groups4',
            gpu_count=8, dataset='dapo', max_response_length=16384, epochs=1, evaluations_per_epoch=1)
        mapping = json.loads((Path(scratch)/'dapo_template/mapping.json').read_text())
    assert template['teacher'] == 'Qwen3.5-35B-A3B' and template['world_size'] == 8 and template['method'] == 'grpo'
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-4B', local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-35B-A3B', local_files_only=True)
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    split = json.loads((ROOT/'data/livecodebench/split.json').read_text())
    problems = code_prompts.lcb_problems()
    # Validation order: easy first (the 6-question smoke validation must not be all-hard), then date, key.
    val_keys = sorted(split['validation'], key=lambda k: (DIFFICULTY[problems[k]['difficulty']],
                                                          problems[k]['contest_date'], k))
    train = code_prompts.lcb_rows(split['train'], tokenizer, problems)
    validation = code_prompts.lcb_rows(val_keys, tokenizer, problems)
    for rows in (train, validation):
        for row in rows:
            assert code_prompts.encode(teacher_tokenizer, row['messages']) == row['prompt_token_ids']
            assert 2 <= len(row['prompt_token_ids']) <= template['max_prompt_length']
    assert not {r['key'] for r in train} & {r['key'] for r in validation}
    assert not {r['key'] for r in train+validation} & set(split['test'])
    steps, eval_steps = quarter_epoch_schedule(len(train), template['batch_size'], epochs, 1)
    commit = subprocess.check_output(['git', '-C', str(ROOT/'third_party/livecodebench'), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    image = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', code_judge.IMAGE],
                                    text=True).strip()
    protocol = dict(template)
    for key in ('val_metric', 'format_penalty'):
        protocol.pop(key, None)
    protocol.update(dataset='lcb', learning_rate=learning_rate,
        examples_per_epoch=len(train), epochs=epochs, steps_per_epoch=steps, evaluations_per_epoch=1,
        total_steps=steps*epochs, eval_steps=eval_steps, checkpoint_steps=eval_steps,
        total_evaluations=1+len(eval_steps), eval_questions=len(validation), eval_samples=4,
        validation_before_training=True, initial_eval_step=0,
        reward=('Official LiveCodeBench run_test (pinned commit), 1 iff every public+private test passes '
                '(6 s per test, 4 GiB, 1 CPU, no network, read-only, one container per response); '
                'code = content between the last two ``` lines (official extraction)'),
        objective='GRPO binary LCB pass reward; group-normalized advantage',
        score_worker='lcb_score_worker.py',
        code_judge=dict(code_judge.DEFAULTS, image=code_judge.IMAGE, image_id=image, livecodebench_commit=commit,
                        workers='PLATOUR_CODE_JUDGE_WORKERS, default 64'),
        data_source='livecodebench/code_generation_lite@0fe84c3 release_v5 (test.jsonl..test5.jsonl, 880 problems)',
        split_rule=split['rule'], train_dates=split['train_dates'], validation_dates=split['validation_dates'],
        source_sha256=sha(ROOT/'data/livecodebench/split.json'),
        eval_source_sha256=sha(ROOT/'data/livecodebench/problems.jsonl'),
        prompt_format='Official LiveCodeBench generic prompt (SYSTEM_MESSAGE_GENERIC + generic question template)',
        enable_thinking=False, eval_metric='avg@4 = validation responses passing all tests / 400; pass@4 separately',
        eval_coverage='LCB release_v5 validation100 x4 before training and after every epoch',
        eval_selection='Fixed budget, no early stopping; full bridge/optimizer/rank-RNG checkpoint after every epoch eval',
        data_order=f'Fresh torch.randperm({len(train)}), seed 42 + zero_based_epoch',
        validation_order='easy, medium, hard; then contest date, key (smoke uses the first 6)',
        wandb_project='hidden-bridge-lcb-grpo',
        wandb_run_name=(f'qwen35_35b_to_4b_lcb_relative_depth_groups4_constant_gate1_grpo_g8_bs64_16k_'
                        f'{lr_tag(learning_rate)}_{epochs}epoch_eval{1+len(eval_steps)}_vllm_8gpu{suffix}'))
    protocol.pop('vllm_rollout_tuning', None)
    if template.get('vllm_rollout_tuning'):
        protocol['vllm_rollout_tuning'] = template['vllm_rollout_tuning']
    if format_penalty:
        protocol.update(format_penalty=True, reward=protocol['reward']+
            '; format penalty: a response cut at the length cap gets reward 0 (validation unpenalised)')
    save(base/'train_prompts.json', train)
    save(base/'eval_prompts.json', validation)
    save(base/'dataset.json', dict(train=len(train), validation=len(validation),
        train_prompts_sha256=sha(base/'train_prompts.json'), eval_prompts_sha256=sha(base/'eval_prompts.json'),
        maximum_train_prompt_tokens=max(len(r['prompt_token_ids']) for r in train),
        maximum_eval_prompt_tokens=max(len(r['prompt_token_ids']) for r in validation),
        validation_difficulty={d: sum(r['difficulty'] == d for r in validation) for d in DIFFICULTY},
        train_difficulty={d: sum(r['difficulty'] == d for r in train) for d in DIFFICULTY}))
    save(base/'mapping.json', mapping)
    assert mapping['teacher_layers_0based'] == protocol['layer_map']
    save(base/'protocol.json', protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--format-penalty', action='store_true')
    parser.add_argument('--wandb-run-suffix', default='')
    args = parser.parse_args()
    protocol = build(args.root, args.epochs, args.learning_rate, args.format_penalty, args.wandb_run_suffix)
    print(json.dumps(dict(prepared=True, root=str(args.root.resolve()), total_steps=protocol['total_steps'],
        eval_steps=protocol['eval_steps'], wandb_run_name=protocol['wandb_run_name'])), flush=True)


if __name__ == '__main__':
    main()
