"""User-supplied public data and model locations; no site-specific defaults."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .runner import evaluate, evaluate_vllm, load_bridge, prepare_grpo_holdout, read_rows, train
from .checkpoints import atomic_json, rank


ROOT = Path(__file__).resolve().parents[1]


def select(identifier: int) -> dict:
    catalog = json.loads((Path(__file__).with_name("experiments.json")).read_text())
    try:
        entry = next(item for item in catalog["experiments"] if item["id"] == identifier)
    except StopIteration as error:
        raise ValueError(f"Unknown experiment {identifier}") from error
    sender = "235b" if entry["sender"].startswith("Qwen3-235B") else "8b"
    config = {**catalog[entry["method"]], **entry,
              "receiver": "Qwen3-1.7B",
              "tag": f"{identifier:02d}_qwen3_{sender}_to_1p7b_{entry['method']}_{entry['task']}"}
    if entry['method'] == 'sft' and entry['task'] in (
            'arc_c', 'mmlu_redux', 'chemistry_no_cot'):
        config['eval_sampling'] = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1}
    if entry['method'] == 'grpo':
        manifest_path = ROOT / 'data/qwen3' / config['tag'] / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if manifest.get('validation_split', {}).get('policy') == 'published_train_holdout_v2':
            config.update(published_grpo_validation=True,
                          original_train_examples=config['train_examples'],
                          train_examples=manifest['counts']['train'],
                          validation_examples=manifest['counts']['validation'])
            steps = (config['train_examples'] + config['batch_size'] - 1) // config['batch_size']
            config.update(steps_per_epoch=steps, total_steps=steps*3,
                          eval_steps=sorted({(i*steps+1)//2 for i in range(1, 7)}))
    return config


def audit_inputs(config: dict, paths: dict[str, Path]) -> None:
    from platour_data import ensure_file, is_lfs_pointer, registry
    manifest = json.loads((ROOT / 'data' / 'qwen3' / config['tag'] / 'manifest.json').read_text())
    if manifest['experiment_id'] != config['id'] or manifest['method'] != config['method']:
        raise ValueError('Input manifest does not match this Qwen3 experiment')
    for split, path in paths.items():
        if not Path(path).exists() or is_lfs_pointer(path):
            name = 'qwen3/' + config['tag'] + '/' + split + '.jsonl'
            if registry()['files'][name]['sha256'] != manifest['sha256'][split]:
                raise ValueError('Custom manifest does not identify the published data: ' + split)
            ensure_file(name, path)
        with Path(path).open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('Materialize the signed input data before training/evaluation')
            stream.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != manifest['sha256'][split]:
            raise ValueError('Original signed prompt/answer split differs: ' + split)


def audit_published_weight(config: dict, path: Path) -> None:
    from safetensors import safe_open

    with Path(path).open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Materialize the inference checkpoint from Git LFS first')
    with safe_open(path, framework='pt', device='cpu') as weights:
        metadata = weights.metadata() or {}
        if (len(weights.keys()) != 28 or
                (config['id'] <= 17 and metadata.get('experiment') != str(config['id'])) or
                any(key in metadata for key in ('selected_step', 'selected_checkpoint_step'))):
            raise ValueError('Published Qwen3 weight identity or private selection metadata differs')


def _verified_bridge(path: Path, marker: Path | None = None, *, expected_sha=None,
                     expected_step=None) -> tuple[dict, str]:
    """Bridge projections of a selected checkpoint and the SHA-256 of its file."""
    import torch
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError('Selected bridge checkpoint is missing')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if marker is not None:
        record = json.loads(marker.read_text()) if marker.is_file() else {}
        if (not record.get('complete') or
                expected_step is not None and record.get('step') != expected_step):
            raise ValueError('Selected checkpoint complete marker is missing or differs')
        expected_sha = record.get('sha256', {}).get(path.name)
    if not expected_sha or digest != expected_sha:
        raise ValueError('Selected checkpoint bridge SHA differs from its signed marker')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or 'P' not in state:
        raise ValueError('Selected checkpoint has no bridge projection state')
    return state['P'], digest


def _final_sft_step(run):
    budgets = []
    terminal = run / 'complete.json'
    if terminal.is_file():
        record = json.loads(terminal.read_text())
        if record.get('complete') is not True:
            raise ValueError('Final SFT checkpoint requires complete training')
        budgets.append(record.get('steps'))
    portable, native = run / 'training_protocol.json', run / 'protocol.json'
    if portable.is_file():
        budgets.append(json.loads(portable.read_text()).get('total_steps'))
    elif native.is_file():
        config = json.loads(native.read_text()).get('config', {})
        budgets.append(3 if config.get('smoke') else config.get('total_steps'))
    if (not budgets or any(type(step) is not int or step < 1 for step in budgets)
            or len(set(budgets)) != 1):
        raise ValueError('Final SFT checkpoint requires a known, consistent training budget')
    return budgets[0]


def selected_run_state(run: Path) -> tuple[dict, dict]:
    """Recorded selection, or final signed SFT checkpoint for native runs without a selection."""
    run = Path(run).absolute()
    chosen = run / 'selection.json'
    if chosen.is_file():
        selection = json.loads(chosen.read_text())
        if selection.get('test_used_for_selection') is not False:
            raise ValueError('Selection must not use TEST')
        step = selection['selected_step']
        if selection['policy'] == 'final_sft_checkpoint_v1' and step != _final_sft_step(run):
            raise ValueError('Final SFT checkpoint is missing or training is incomplete')
        expected_sha = None
        if 'selected_checkpoint' in selection:
            # Native GRPO keeps evaluation snapshots, with checkpoint.json and
            # a completed validation summary rather than optimizer commit markers.
            path, marker = Path(selection['selected_checkpoint']), None
            if not path.resolve().is_relative_to(run.resolve()):
                raise ValueError('Selected checkpoint belongs to another run')
            metadata = json.loads((path.parent / 'checkpoint.json').read_text())
            summary = json.loads((path.parent / 'summary.json').read_text())
            state_sha = selection.get('selected_bridge_state_sha256')
            if (path.parent.name != f'step-{step:04d}' or metadata.get('step') != step or
                    summary.get('complete') is not True or summary.get('evaluation_role') != 'validation' or
                    not state_sha or metadata.get('bridge_state_sha256') != state_sha or
                    summary.get('bridge_state_sha256') != state_sha):
                raise ValueError('Selected native checkpoint validation identity differs')
            expected_sha = metadata.get('bridge_sha256')
        else:
            directory = run / 'checkpoints' / f"step-{selection['selected_step']:04d}"
            path, marker = directory / 'bridge.pt', directory / 'complete.json'
        state, digest = _verified_bridge(path, marker, expected_sha=expected_sha, expected_step=step)
        return state, {
            'source_run': str(run), 'selected_step': selection['selected_step'],
            'selection_policy': selection['policy'], 'selected_checkpoint_sha256': digest}
    candidates = []
    for marker in sorted((run / 'checkpoints').glob('step-[0-9][0-9][0-9][0-9]/complete.json')):
        record = json.loads(marker.read_text())
        step = record.get('step')
        nll = (record.get('metric') or {}).get('heldout_nll')
        evaluation = run / 'evaluation' / f'step-{step:04d}.json'
        if nll is None and evaluation.is_file():
            nll = json.loads(evaluation.read_text()).get('heldout_nll')
        if record.get('complete') and isinstance(nll, float):
            candidates.append((nll, step, marker))
    if not candidates:
        raise FileNotFoundError('Training run has no recorded selection or signed SFT checkpoint with held-out NLL')
    nll, step, marker = max(candidates, key=lambda row: row[1])
    if step != _final_sft_step(run):
        raise ValueError('Final SFT checkpoint is missing or training is incomplete')
    state, digest = _verified_bridge(marker.parent / 'bridge.pt', marker, expected_step=step)
    return state, {
        'source_run': str(run), 'selected_step': step, 'validation_heldout_nll': nll,
        'selection_policy': 'final_sft_checkpoint_v1', 'selected_checkpoint_sha256': digest}


def portable_training_config(config: dict, *, micro_batch: int | None = None) -> dict:
    """Preserve portable defaults; permit an explicit SFT memory override."""
    config = dict(config)
    if config['method'] == 'sft' and config['task'] == 'gsm8k':
        config['micro_batch'] = min(config['micro_batch'], 4)
    if micro_batch is not None:
        if (config['method'] != 'sft' or type(micro_batch) is not int or
                not 1 <= micro_batch <= config['batch_size']):
            raise ValueError('--micro-batch requires portable SFT and a size between 1 and the effective batch')
        config['micro_batch'] = micro_batch
    return config


def main() -> None:
    from .parallel import maybe_launch
    if maybe_launch(sys.argv[1:]):
        return
    parser = argparse.ArgumentParser(description="Public Qwen3 bridge experiment runner")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--experiment", type=int, required=True, choices=range(1, 19))
        cmd.add_argument("--sender", required=True,
                         help="Local backbone directory or, for evaluation, HF repository ID")
        cmd.add_argument("--receiver", required=True,
                         help="Local backbone directory or, for evaluation, HF repository ID")
        cmd.add_argument("--output", type=Path, required=True)
        if name == "train":
            from .parallel import add_arguments
            add_arguments(cmd)
            cmd.add_argument("--train-data", type=Path, help="Defaults to the published signed train split")
            cmd.add_argument("--validation-data", type=Path, help="Defaults to the signed SFT or published GRPO validation split")
            cmd.add_argument("--test-data", type=Path,
                             help="Signed final test: SFT uses the final checkpoint; GRPO uses validation-best "
                                  "(GRPO: defaults to published TEST and its dataset validation policy; SFT: optional)")
            cmd.add_argument('--test-checkpoint', choices=('last',),
                             default='last',
                             help='SFT test point: final training checkpoint; VAL loss is monitoring only')
            cmd.add_argument("--resume", type=Path)
            cmd.add_argument('--micro-batch', type=int,
                             help='Explicit portable SFT activation-memory override; effective batch and epochs stay fixed')
            cmd.add_argument('--train-backend',
                             choices=('portable', 'historical-sft', 'historical-grpo'),
                             default='portable',
                             help='Native SFT or frozen native-vLLM GRPO (eight GPUs)')
            cmd.add_argument('--smoke', action='store_true',
                             help='Bounded native SFT/GRPO structural check in a fresh output directory')
            cmd.add_argument('--eval-backend', choices=('vllm',), default='vllm',
                             help='GRPO validation and the final SFT/GRPO test; SFT held-out NLL uses HF')
            cmd.add_argument('--vllm-python', type=Path,
                             help='Optional vLLM interpreter override; defaults to current project Python')
            cmd.add_argument('--rollout-worker-mode', choices=('persistent', 'restart'),
                             help='Portable GRPO: reuse a sleeping receiver engine (default) or restart per use')
        else:
            from .eval_parallel import add_arguments
            add_arguments(cmd)
            cmd.add_argument("--test-data", type=Path, help="Defaults to the published signed test split")
            cmd.add_argument("--weights", help="Local checkpoint or HF repository ID; defaults to the verified release")
            cmd.add_argument("--run", type=Path,
                             help="Test the recorded checkpoint (new SFT: final; GRPO: validation-best) of this training output")
            cmd.add_argument('--backend', choices=('vllm',), default='vllm')
            cmd.add_argument('--vllm-python', type=Path,
                             help='Optional vLLM interpreter override; defaults to current project Python')
            cmd.add_argument('--eval-batch-size', type=int, default=32)
            cmd.add_argument('--vllm-memory-fraction', type=float, default=.25)
            cmd.add_argument('--parity-report', action='store_true',
                             help='Optionally record HF/vLLM log-probability differences without a pass threshold')
    args = parser.parse_args()
    from .eval_parallel import configure_worker, launch, shard_rows
    configure_worker()
    if args.action == 'evaluate':
        if args.n_gpu is not None:
            launch(args)
            return
        if args.gpus_per_replica is not None:
            parser.error('--gpus-per-replica requires --n-gpu')
    config = select(args.experiment)
    data_root = ROOT / 'data/qwen3' / config['tag']
    if args.action == 'evaluate':
        args.test_data = args.test_data or data_root / 'test.jsonl'
    else:
        args.train_data = args.train_data or data_root / 'train.jsonl'
        if config['method'] == 'sft' or config.get('published_grpo_validation'):
            args.validation_data = args.validation_data or data_root / 'validation.jsonl'
        if config['method'] == 'grpo':
            args.test_data = args.test_data or data_root / 'test.jsonl'
    historical_sft = args.action == 'train' and args.train_backend == 'historical-sft'
    historical_grpo = args.action == 'train' and args.train_backend == 'historical-grpo'
    if (args.action == 'train' and args.micro_batch is not None and
            (args.train_backend != 'portable' or config['method'] != 'sft')):
        raise ValueError('--micro-batch is only available for portable SFT')
    if args.action == 'train' and args.train_backend == 'portable':
        config = portable_training_config(config, micro_batch=args.micro_batch)
    if args.action == 'train' and args.rollout_worker_mode is not None:
        if args.train_backend != 'portable' or config['method'] != 'grpo':
            raise ValueError('--rollout-worker-mode is only available for portable GRPO')
        config['rollout_worker_mode'] = args.rollout_worker_mode
    micro_override = os.environ.get('PLATOUR_SFT_FP8_SMOKE_MICRO_BATCH')
    if micro_override is not None and (not historical_sft or not args.smoke or
            not config['sender'].startswith('Qwen3-235B') or
            micro_override not in ('1', '2', '4') or
            int(micro_override) >= config['micro_batch']):
        raise ValueError('Reduced native FP8 micro-batch is smoke-only; preserve global batch 64')
    if historical_grpo:
        if config['method'] != 'grpo':
            raise ValueError('Native GRPO requires a signed GRPO recipe')
        if args.vllm_python is not None and args.vllm_python.resolve() != Path(sys.executable).resolve():
            raise ValueError('Native GRPO needs vLLM in the same Python as the training backend')
    if historical_sft:
        if config['method'] != 'sft':
            raise ValueError('Historical SFT requires a signed SFT recipe')
        if int(os.environ.get('WORLD_SIZE', '1')) != 4 or not all(
                key in os.environ for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
            raise RuntimeError('Historical SFT needs torchrun with four ranks on eight GPUs')
    elif int(os.environ.get('WORLD_SIZE', '1')) != 1:
        if args.action != 'train':
            raise RuntimeError('Evaluation is one process; it places models on visible GPUs automatically')
        if config['sender'].startswith('Qwen3-235B') and 'PLATOUR_GPU_GROUP' not in os.environ:
            raise ValueError('Portable data parallel replicates the sender per rank; use '
                             '--train-backend historical-sft/historical-grpo for the 235B sender')
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        local = int(os.environ['LOCAL_RANK'])
        timeout = timedelta(hours=float(os.environ.get('PLATOUR_DIST_TIMEOUT_HOURS', '24')))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            if not dist.is_initialized():
                dist.init_process_group('nccl', device_id=torch.device('cuda', local),
                                        timeout=timeout)
            args.portable_device = f'cuda:{local}'
        else:
            if not dist.is_initialized():
                dist.init_process_group('gloo', timeout=timeout)
            args.portable_device = 'cpu'
    if args.action == 'train' and args.smoke and not (historical_sft or historical_grpo):
        raise ValueError('--smoke requires a historical training backend')
    if historical_grpo and args.eval_backend != 'vllm':
        raise ValueError('Frozen GRPO always validates with native vLLM')
    if ((args.action == 'evaluate' and args.backend == 'vllm') or
            (args.action == 'train' and args.eval_backend == 'vllm' and
             (config['method'] == 'grpo' or
              args.test_data is not None))):
        from platour_exps import require_vllm_python
        require_vllm_python(args.vllm_python)
    if args.action == 'train':
        if config['method'] == 'sft':
            if args.validation_data is None:
                parser.error('SFT needs --validation-data (its signed held-out split)')
            audit_inputs(config, {'train': args.train_data, 'validation': args.validation_data,
                                  **({'test': args.test_data} if args.test_data else {})})
            config['test_checkpoint'] = args.test_checkpoint
            config.update(eval_backend=args.eval_backend,
                          vllm_python=str(args.vllm_python) if args.vllm_python else None)
        else:
            if args.test_data is None or (args.validation_data is not None and not config.get('published_grpo_validation')):
                parser.error('GRPO needs signed TEST and the validation policy of its pinned dataset')
            audit_inputs(config, {'train': args.train_data, 'test': args.test_data,
                **({'validation': args.validation_data} if config.get('published_grpo_validation') else {})})
        checkpoint = None
    else:
        audit_inputs(config, {'test': args.test_data})
        if args.run is not None:
            if args.weights is not None:
                parser.error('Choose either --weights or a training --run, not both')
            checkpoint = None
        else:
            from platour_exps import resolve_weight_input
            checkpoint = resolve_weight_input(args.experiment, args.weights)
        if checkpoint is not None:
            audit_published_weight(config, checkpoint)
    if historical_sft:
        if args.resume is None and args.output.exists() and int(os.environ['RANK']) == 0:
            raise FileExistsError('Historical SFT requires a fresh output root')
        if args.resume is not None and (not args.output.is_dir() or
                                        args.resume.parent.absolute() != args.output.absolute() / 'checkpoints' or
                                        not (args.resume / 'complete.json').is_file()):
            raise ValueError('Historical SFT resume needs a signed step in this output root')
        if config['sender'].startswith('Qwen3-235B'):
            from importlib.util import find_spec
            if find_spec('vllm') is None:
                raise RuntimeError('Historical FP8 expert training requires vLLM 0.17.1 '
                                   'in the four-rank HF training Python')
    if args.action == 'train':
        examples = read_rows(args.train_data)
        if len(examples) != config['train_examples']:
            raise ValueError('The training split does not match this experiment')
        if config['method'] == 'sft':
            validation = read_rows(args.validation_data)
            final_test = read_rows(args.test_data) if args.test_data else None
            split = None
        else:
            final_test = read_rows(args.test_data)
            published_validation = read_rows(args.validation_data) if config.get('published_grpo_validation') else None
            if historical_grpo:
                from .native_grpo import audit_resume, execute, prepare
                if args.resume is None:
                    prepare(config, args.sender, args.receiver, examples, final_test,
                            args.output, smoke=args.smoke,
                            **({'validation_rows': published_validation} if published_validation is not None else {}))
                    execute(args.output)
                else:
                    root = args.output.absolute()
                    saved = args.resume.absolute()
                    expected = root / ('smoke' if args.smoke else 'train') / 'checkpoints'
                    if (saved.parent != expected or saved.is_symlink() or
                            not saved.name.startswith('step-') or len(saved.name) != 9 or
                            not saved.name[5:].isdigit()):
                        raise ValueError('Native GRPO resume requires a signed step in this output')
                    original = json.loads((root / 'original_protocol.json').read_text())
                    if (original['teacher'] != config['sender'] or
                            original['student'] != config['receiver'] or
                            Path(args.sender).resolve().name != config['sender'] or
                            Path(args.receiver).resolve().name != config['receiver']):
                        raise ValueError('Native GRPO resume model or recipe differs')
                    current = json.loads((root / 'protocol.json').read_text())
                    if current['eval_samples'] != config.get('validation_samples', config['eval_samples']):
                        raise ValueError('Resume validation sampling differs; use a fresh run for the new avg@k')
                    if (current['teacher_checkpoint'] != str(Path(args.sender).resolve()) or
                            current['student_checkpoint'] != str(Path(args.receiver).resolve()) or
                            current['smoke_only'] != args.smoke):
                        raise ValueError('Native GRPO resume backbone or training mode differs')
                    step = int(saved.name[5:])
                    audit_resume(root, step)
                    execute(root, resume_step=step)
                return
            examples, validation, config, split = prepare_grpo_holdout(
                examples, final_test, config,
                **({"validation_rows": published_validation} if published_validation is not None else {}))
            config.update(eval_backend=args.eval_backend,
                          vllm_python=str(args.vllm_python) if args.vllm_python else None)
    if historical_sft:
        from .aligned_sft import train_aligned
        original_micro = config['micro_batch']
        micro = int(micro_override) if micro_override is not None else original_micro
        config = {**config, 'smoke': args.smoke, 'micro_batch': micro,
                  'gradient_accumulation_steps': 64 // (4 * micro)}
        if micro_override is not None:
            config['smoke_fp8_memory_override'] = {
                'original_micro_batch': original_micro, 'actual_micro_batch': micro,
                'original_gradient_accumulation': 64 // (4 * original_micro),
                'actual_gradient_accumulation': 64 // (4 * micro),
                'global_batch_size_unchanged': 64,
                'reason': '235B FP8 first native padded backward exceeded per-GPU memory'}
        train_aligned(config, args.sender, args.receiver, examples, validation,
                      args.output, micro_batch=micro, resume=args.resume, final_test=final_test)
        return
    if args.action == 'evaluate':
        from platour_exps import resolve_model_input
        args.sender = resolve_model_input(args.sender)
        args.receiver = resolve_model_input(args.receiver)
    bridge, tokenizer = load_bridge(args.sender, args.receiver,
                                     config["layer_mapping"], config.get("initialization", "zero"),
                                     **({'device_map': args.portable_device}
                                        if getattr(args, 'portable_device', None) and
                                        'PLATOUR_GPU_GROUP' not in os.environ else {}))
    expected_depth = 94 if config["sender"].startswith("Qwen3-235B") else 36
    if (bridge.sender.config.num_hidden_layers != expected_depth or
            bridge.receiver.config.num_hidden_layers != 28 or
            bridge.sender.config.hidden_size != 4096 or
            bridge.receiver.config.hidden_size != 2048):
        raise ValueError("Downloaded public sender or receiver differs from this experiment")
    if args.action == "train":
        train(config, bridge, tokenizer, examples, validation, args.output,
              resume=args.resume, final_test=final_test, split=split)
    else:
        if args.output.exists():
            raise FileExistsError('Use a fresh evaluation directory')
        provenance = None
        if checkpoint is None:
            state, provenance = selected_run_state(args.run)
            bridge.P.load_state_dict(state, strict=True)
        else:
            bridge.load_public(checkpoint)
        rows = shard_rows(read_rows(args.test_data))
        if args.backend == 'vllm':
            config['parity_report'] = args.parity_report
            result = evaluate_vllm(bridge, tokenizer, rows, config,
                count=config['eval_samples'], cap=config['response_cap'],
                output=args.output, batch_size=args.eval_batch_size,
                memory_fraction=args.vllm_memory_fraction,
                worker_python=str(args.vllm_python) if args.vllm_python else None)
        else:
            result = evaluate(bridge, tokenizer, rows, config,
                              count=config['eval_samples'], cap=config['response_cap'])
            args.output.mkdir(parents=True)
        if rank() == 0:
            atomic_json({"experiment": args.experiment, **result,
                         "question_indices": [row["index"] for row in rows],
                         "checkpoint_source": ('selected_checkpoint_of_training_run' if provenance
                                               else 'published_release'),
                         "selection": provenance},
                        args.output / "summary.json")
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
