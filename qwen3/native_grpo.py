"""SHA-checked, train-holdout native Qwen3 GRPO from the frozen fixed branch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import native_grpo_validation as policy


ASSETS = Path(__file__).with_name('native_reference')
PROTOCOLS = Path(__file__).with_name('native_protocols.json')
HASHES = Path(__file__).with_name('native_source_sha256.json')
FP8_KERNEL_REVISION = '8c178950e8710e26a2210e4e909cd02dc16c8715'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle():
    expected = json.loads(HASHES.read_text())
    if len(expected) != 222 or not all(name.startswith(('base/', 'overlays/'))
                                        and '..' not in Path(name).parts for name in expected):
        raise ValueError('Frozen native GRPO bundle manifest differs')
    actual = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise ValueError('Frozen native GRPO source or licenses are missing or unexpected')
    for name, digest in expected.items():
        if sha256(ASSETS / name) != digest:
            raise ValueError('Frozen native GRPO source checksum differs: ' + name)
    return expected


def zero_native_13_core_source(source):
    """Change only ID13's fresh projection init in a private, SHA-recorded copy."""
    import ast

    old = ("    bound = 1e-4 / math.sqrt(bridge.teacher.config.hidden_size)\n"
           "    rng = torch.Generator(device=bridge.student.device).manual_seed(42)\n"
           "    with torch.no_grad():\n"
           "        for p in bridge.bridge_parameters():\n"
           "            p.uniform_(-bound, bound, generator=rng)\n"
           "    bridge.initialization = 'random'\n"
           "    return dict(kind='small random uniform', seed=42, bound=bound,\n"
           "                scale=1e-4, constant_gate=1., sft_checkpoint_loaded=False)")
    new = ("    with torch.no_grad():\n"
           "        for p in bridge.bridge_parameters():\n"
           "            p.zero_()\n"
           "    bridge.initialization = 'zero'\n"
           "    return dict(kind='exact-zero linear bridge', seed=42,\n"
           "                projection_initialization='exact zero', constant_gate=1.,\n"
           "                sft_checkpoint_loaded=False)")
    if source.count(old) != 1:
        raise ValueError('Original ID13 initialization source differs; refuse an unverified patch')
    active = source.replace(old, new, 1)
    ast.parse(active)
    return active


def zero_native_13_trainer_source(source):
    """Ensure the ID13 checkpoint records zero rather than the frozen random recipe."""
    import ast

    old = "config.update(initialization=f'small random uniform +/-1e-4/sqrt({bridge.teacher.config.hidden_size})',constant_gate=1.)"
    new = "config.update(initialization='exact-zero linear bridge',constant_gate=1.)"
    if source.count(old) != 1:
        raise ValueError('Original ID13 checkpoint metadata differs; refuse an unverified patch')
    active = source.replace(old, new, 1)
    ast.parse(active)
    return active


def _training_metric_rows(path):
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if position != len(lines) - 1 or raw.endswith(b'\n'):
                raise ValueError('Committed GRPO metric line is corrupt') from None
            break
    return rows



CHEMISTRY_SCORER = 'recipe/llm_communicate/eval_qwen3_17b_chemistry.py'


def strict_chemistry_source(text):
    """Update only the private native evaluator; archived source remains intact."""
    old = "        value=text.split('<answer>')[-1].split('</answer>')[0].strip()"
    new = "        matches=re.findall(r'<answer>(.*?)</answer>',text,flags=re.I|re.S)\n        value=matches[-1].strip() if matches else None"
    if text.count(old) != 1:
        raise ValueError('Native Chemistry scorer source differs')
    return text.replace(old, new).replace(
        "        # Match SDPO's last <answer> extraction, including its incomplete-tag\n        # behavior; never recover a letter from arbitrary reasoning prose.",
        "        # Require a complete final answer field; missing fields are incorrect.")


def verify_chemistry_scorer(root, protocol):
    if (protocol.get('chemistry_answer_field_required') is not True or
            sha256(Path(root) / 'reference_source' / CHEMISTRY_SCORER) != protocol.get('chemistry_scorer_sha256')):
        raise ValueError('Native Chemistry scoring policy differs; prepare a fresh run')


def audit_resume(root, step):
    """Reject incomplete GRPO optimizer, rank RNG, validation or source identity."""
    import torch

    root = Path(root).absolute()
    protocol = json.loads((root / 'protocol.json').read_text())
    verify_chemistry_scorer(root, protocol)
    folder = root / ('smoke' if protocol['smoke_only'] else 'train')
    boundaries = (1,) if protocol['smoke_only'] else tuple(protocol['checkpoint_steps'])
    if (type(step) is not int or step not in boundaries or
            not 0 < step < (2 if protocol['smoke_only'] else protocol['total_steps']) or
            (folder / 'complete.json').exists() or (root / 'selection.json').exists()):
        raise ValueError('Resume needs an unfinished run and a signed checkpoint boundary')
    if not (root / 'native_process.json').is_file():
        raise FileNotFoundError('No original native process identity for this run')
    prior = json.loads((root / 'native_process.json').read_text())
    if (type(prior.get('pid')) is not int or prior['pid'] <= 1 or
            prior.get('smoke') != protocol['smoke_only']):
        raise ValueError('Prior native run identity differs')
    for receipt in (root / 'native_process.json',
                    *root.glob('native_attempts/attempt-*/native_process.json')):
        pid = json.loads(receipt.read_text()).get('pid')
        if type(pid) is not int or pid <= 1:
            raise ValueError('Prior native process group identity differs')
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError('Old native training process group is still alive')
    verify_bundle()
    patch = json.loads((root / 'native_resume_source_patch.json').read_text())
    identifier = patch['experiment']
    relative = Path(patch['source'])
    if (identifier not in range(13, 19) or relative.is_absolute() or
            '..' in relative.parts or
            relative.as_posix() != 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'):
        raise ValueError('Unsupported GRPO resume source identity')
    original = ASSETS / 'overlays' / f'{identifier:02d}' / relative
    text = original.read_text()
    if protocol.get('smoke_decoder_gpu_placement'):
        if identifier != 15 or not protocol['smoke_only']:
            raise ValueError('Secondary decoder placement is only supported by ID15 smoke')
        for before, after in (
                ("assert physical[device_index]==protocol['decoder_gpu_ids'][local]",
                 "assert physical[device_index+1]==protocol['decoder_gpu_ids'][local]"),
                ("gpu=physical[device_index],memory_fraction=protocol['vllm_memory_fraction']",
                 "gpu=protocol['decoder_gpu_ids'][local],memory_fraction=protocol['vllm_memory_fraction']")):
            if text.count(before) != 1:
                raise ValueError('Frozen smoke placement patch differs')
            text = text.replace(before, after, 1)
    source_protocol = json.loads(PROTOCOLS.read_text())[str(identifier)]
    variant = root / 'zero_bridge_initialization_patch.json'
    if protocol.get('zero_initialization_variant'):
        if (identifier != 13 or protocol.get('initialization') != 'exact-zero linear bridge' or
                protocol.get('historical_initialization') != source_protocol['initialization'] or
                not variant.is_file()):
            raise ValueError('Zero-initialized native bridge resume identity differs')
        marker = json.loads(variant.read_text())
        core_name = Path('recipe/llm_communicate/qwen35_bridge_opd_core.py')
        original_core = ASSETS / 'overlays/13' / core_name
        expected_core = zero_native_13_core_source(original_core.read_text())
        if (marker.get('experiment') != 13 or
                marker.get('original_core_sha256') != sha256(original_core) or
                marker.get('active_core_sha256') != hashlib.sha256(expected_core.encode()).hexdigest() or
                marker['active_core_sha256'] != sha256(root / 'reference_source' / core_name) or
                marker.get('original_trainer_sha256') != sha256(original) or
                not marker.get('not_identical_to_published_id13_training_initialization')):
            raise ValueError('Native ID13 zero bridge source or original identity differs')
        text = zero_native_13_trainer_source(text)
        if marker.get('active_trainer_before_resume_sha256') != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('Native ID13 zero bridge checkpoint metadata differs')
    elif (variant.exists() or protocol.get('initialization') != source_protocol.get('initialization')):
        raise ValueError('Historical GRPO bridge initialization differs from its signed source')
    expected_active = resume_source_text(text, identifier)
    if (patch['original_scientific_sha256'] != sha256(original) or
            patch.get('before_resume_sha256') != hashlib.sha256(text.encode()).hexdigest() or
            patch['active_sha256'] != hashlib.sha256(expected_active.encode()).hexdigest() or
            patch['active_sha256'] != sha256(root / 'reference_source' / relative)):
        raise ValueError('Private GRPO resume variant differs from the frozen original')
    if json.loads((root / 'original_protocol.json').read_text()) != source_protocol:
        raise ValueError('Frozen GRPO original scientific protocol differs')
    protected = ('teacher', 'student', 'method', 'dataset', 'layer_map', 'group_size',
                 'batch_size', 'learning_rate', 'train_sampling', 'eval_sampling',
                 'max_prompt_length', 'max_response_length', 'teacher_weight_precision',
                 'teacher_model_parallel_size', 'world_size', 'train_reward_policy')
    if any(protocol.get(key) != source_protocol.get(key) for key in protected):
        raise ValueError('Native GRPO reward, model or sampling recipe differs')
    if protocol.get('smoke_decoder_memory_override'):
        fraction = protocol['smoke_decoder_memory_override']
        if (not protocol['smoke_only'] or source_protocol.get('teacher_weight_precision') != 'fp8' or
                fraction.get('original_fraction') != source_protocol['vllm_memory_fraction'] or
                fraction.get('actual_fraction') != protocol.get('vllm_memory_fraction') or
                not 0.065 <= protocol['vllm_memory_fraction'] < source_protocol['vllm_memory_fraction']):
            raise ValueError('Native FP8 smoke decoder resource identity differs')
    elif protocol.get('vllm_memory_fraction') != source_protocol.get('vllm_memory_fraction'):
        raise ValueError('Native GRPO original decoder resource recipe differs')
    if sha256(root / 'validation_policy.py') != protocol['validation_policy_sha256']:
        raise ValueError('Native holdout policy differs')
    policy.verify_inputs(root, protocol)
    for name, expected in json.loads((root / 'reference_source_sha256.json').read_text()).items():
        path = root / 'reference_source' / name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(path) != expected:
            raise ValueError('Prepared native source differs: ' + name)
    checkpoint = folder / 'checkpoints' / f'step-{step:04d}'
    if checkpoint.is_symlink():
        raise ValueError('Do not resume through a linked checkpoint')
    initial = json.loads((folder / 'initialization.json').read_text())
    signed = json.loads((checkpoint / 'checkpoint.json').read_text())
    marker = json.loads((checkpoint / 'complete.json').read_text())
    if (signed.get('step') != step or marker.get('step') != step or
            marker.get('complete') is not True or
            marker.get('world_size') != protocol['world_size'] or
            signed.get('initial_bridge_state_sha256') != initial['bridge_state_sha256'] or
            marker.get('bridge_state_sha256') != signed.get('bridge_state_sha256') or
            sha256(checkpoint / 'bridge.pt') != signed.get('bridge_sha256')):
        raise ValueError('GRPO bridge checkpoint identity differs')
    hashes = marker.get('files')
    expected = ('bridge.pt', 'optimizer.pt',
                *(f'rng.rank-{rank:02d}.pt' for rank in range(protocol['world_size'])))
    if not isinstance(hashes, dict) or set(hashes) != set(expected) or any(
            sha256(checkpoint / name) != hashes[name] for name in expected):
        raise ValueError('GRPO optimizer or rank RNG file SHA differs')
    optimizer = torch.load(checkpoint / 'optimizer.pt', weights_only=True,
                           map_location='cpu', mmap=True)
    if (optimizer.get('step') != step or
            optimizer.get('initial_sha256') != initial['bridge_state_sha256'] or
            not isinstance(optimizer.get('optimizer'), dict)):
        raise ValueError('GRPO optimizer step or baseline differs')
    del optimizer
    for rank in range(protocol['world_size']):
        state = torch.load(checkpoint / f'rng.rank-{rank:02d}.pt',
                           weights_only=True, map_location='cpu')
        if not all(torch.is_tensor(state.get(k)) for k in ('cpu_rng', 'cuda_rng')):
            raise ValueError('Missing independent GRPO rank CPU/CUDA RNG')
    if not protocol['smoke_only']:
        for boundary in (0, *(n for n in protocol['eval_steps'] if n <= step)):
            record = json.loads((folder / 'evaluation' / f'step-{boundary:04d}' /
                                 'summary.json').read_text())
            if (not record.get('complete') or record.get('step') != boundary or
                    record.get('evaluation_role') != 'validation' or
                    record.get('dataset_sha256') != protocol['eval_prompts_sha256'] or
                    record.get('questions') != protocol['eval_questions'] or
                    record.get('responses') != protocol['eval_questions'] * protocol['eval_samples'] or
                    record.get('scored_responses') != record['responses'] or
                    record.get('gold_missing') or record.get('scoring_timeouts')):
                raise ValueError('GRPO historical held-out validation is incomplete')
            if (boundary == step and
                    record.get('bridge_state_sha256') != signed['bridge_state_sha256']):
                raise ValueError('GRPO validation and checkpoint weights differ')
    metrics = _training_metric_rows(folder / 'metrics.jsonl')
    if [row.get('step') for row in metrics[:step]] != list(range(1, step + 1)):
        raise ValueError('GRPO training metric prefix is not contiguous')
    if (step > len(metrics) or
            not any(row.get('grad_norm', 0) > 0 for row in metrics[:step])):
        raise ValueError('GRPO signed resume has no nonzero training gradient')
    return checkpoint


def archive_resume_tail(root, step):
    """Keep every failed artifact while removing stale RPC paths before replay."""
    root = Path(root).absolute()
    checkpoint = audit_resume(root, step)
    folder = checkpoint.parents[1]
    archive = folder / 'resume_archive' / f'after-step-{step:04d}-{time.time_ns()}'
    if archive.is_symlink():
        raise ValueError('Cannot archive GRPO failure evidence via a symlink')
    archive.mkdir(parents=True)
    moved = []
    for kind in ('rollouts', 'evaluation', 'checkpoints'):
        parent = folder / kind
        if not parent.exists():
            continue
        for path in sorted(parent.glob('step-*')):
            if path.is_symlink():
                raise ValueError('Cannot archive a linked GRPO step')
            try:
                index = int(path.name.split('-')[1])
            except (IndexError, ValueError) as error:
                raise ValueError('Unexpected GRPO step directory') from error
            if index > step:
                destination = archive / kind
                destination.mkdir(exist_ok=True)
                path.rename(destination / path.name)
                moved.append(f'{kind}/{path.name}')
    for path in (*folder.glob('failure.rank-*.json'), *folder.glob('resume.rank-*.json'),
                 *folder.glob('resume_rng.rank-*.json'),
                 *(folder / name for name in ('workers', 'score_worker',
                                              'vllm_handoff_parity', 'platour_audit',
                                              'worst_case_profile.json', 'intentional_interruption.json',
                                              'status.json', 'live.json'))):
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError('Cannot archive a linked GRPO worker artifact')
        path.rename(archive / path.name)
        moved.append(path.name)
    metric_file = folder / 'metrics.jsonl'
    metrics = _training_metric_rows(metric_file)
    shutil.copyfile(metric_file, archive / 'metrics-before-resume.jsonl')
    if len(metrics) > step or not metric_file.read_bytes().endswith(b'\n'):
        temporary = metric_file.with_suffix('.pending')
        if temporary.exists():
            raise FileExistsError('Refusing an existing GRPO pending metric rewrite')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in metrics[:step]))
        temporary.replace(metric_file)
    shutil.copyfile(root / 'phase.json', archive / 'phase-before-resume.json')
    policy.save(archive / 'archive.json', {'resume_step': step, 'moved': moved,
                                          'prior_metrics': len(metrics),
                                          'kept_metrics': step, 'checkpoint': str(checkpoint)})
    return archive


def sign_checkpoint(directory, step, world):
    """Mark the optimizer and all ranks' RNG complete after their barrier."""
    directory = Path(directory)
    if type(step) is not int or step < 1 or type(world) is not int or world < 1:
        raise ValueError('Signed GRPO checkpoint needs a positive step and rank count')
    metadata = json.loads((directory / 'checkpoint.json').read_text())
    if metadata.get('step') != step or (directory / 'complete.json').exists():
        raise ValueError('GRPO checkpoint identity changed or already signed')
    names = ('bridge.pt', 'optimizer.pt',
             *(f'rng.rank-{rank:02d}.pt' for rank in range(world)))
    files = {name: sha256(directory / name) for name in names}
    if files['bridge.pt'] != metadata['bridge_sha256']:
        raise ValueError('Native GRPO bridge changed before rank RNG was committed')
    policy.save(directory / 'complete.json', {
        'complete': True, 'step': step, 'world_size': world,
        'bridge_state_sha256': metadata['bridge_state_sha256'], 'files': files})


def restore_bridge_and_optimizer(directory, step, bridge, optimizer, initial_sha,
                                 weight_digest):
    """Restore native FP32 bridge/AdamW without modifying the frozen backbones."""
    import torch

    directory = Path(directory)
    identity = json.loads((directory / 'checkpoint.json').read_text())
    marker = json.loads((directory / 'complete.json').read_text())
    if (identity.get('step') != step or marker.get('step') != step or
            marker.get('complete') is not True or
            identity.get('initial_bridge_state_sha256') != initial_sha or
            sha256(directory / 'bridge.pt') != identity.get('bridge_sha256') or
            sha256(directory / 'optimizer.pt') != marker.get('files', {}).get('optimizer.pt')):
        raise ValueError('GRPO model or optimizer identity differs from the signed step')
    state = torch.load(directory / 'bridge.pt', map_location='cpu', weights_only=True, mmap=True)
    bridge.maps.load_state_dict(state['P'], strict=True)
    del state
    if weight_digest(bridge) != identity['bridge_state_sha256']:
        raise ValueError('Restored GRPO bridge differs from the signed FP32 weights')
    saved = torch.load(directory / 'optimizer.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved.get('step') != step or saved.get('initial_sha256') != initial_sha:
        raise ValueError('Restored GRPO optimizer or original baseline differs')
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    if (len(optimizer.state) != len(optimizer.param_groups[0]['params']) or
            any(int(value['step']) != step for value in optimizer.state.values())):
        raise ValueError('Resumed GRPO AdamW parameter-step history is incomplete')
    return identity['bridge_state_sha256']


def resume_source_text(source, identifier):
    """Add private resume/checkpoint I/O without editing the frozen math or rollout."""
    import ast

    if identifier not in range(13, 19):
        raise ValueError('Only the six signed Qwen3 GRPO experiments support this patch')
    text = source

    def patch(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f'Frozen GRPO resume patch matched {text.count(old)} times: {old[:85]!r}')
        text = text.replace(old, new, 1)

    patch("parser.add_argument('--objective',choices=['opd','grpo'],default='opd')\n    args=parser.parse_args()",
          "parser.add_argument('--objective',choices=['opd','grpo'],default='opd')\n"
          "    parser.add_argument('--resume-step',type=int)\n    args=parser.parse_args()")
    patch("    run_dir.mkdir(parents=True,exist_ok=True)\n"
          "    if (run_dir/'metrics.jsonl').exists() or (run_dir/'complete.json').exists():\n"
          "        raise RuntimeError('Fresh training refuses preexisting metrics/checkpoints')",
          "    run_dir.mkdir(parents=True,exist_ok=True)\n"
          "    resume_step=args.resume_step or 0\n"
          "    resume_directory=run_dir/'checkpoints'/f'step-{resume_step:04d}' if resume_step else None\n"
          "    if (run_dir/'complete.json').exists() or "
          "(run_dir/'metrics.jsonl').exists() and not resume_step:\n"
          "        raise RuntimeError('Native GRPO requires a fresh or explicitly audited resume run')")
    if identifier != 13:
        patch("    resume_step=0 if args.smoke else protocol.get('resume_step',0)\n"
              "    resume_directory=Path(protocol['resume_from_checkpoint']) if resume_step else None\n"
              "    if resume_step:\n"
              "        assert protocol['bridge_recipe']=='qwen235_17b_chemistry_hidden_grpo'\n"
              "        assert resume_step%protocol['steps_per_epoch']==0,'Continuation starts at an epoch boundary'\n"
              "        state=torch.load(resume_directory/'bridge.pt',map_location='cpu',weights_only=True,mmap=True)\n"
              "        bridge.maps.load_state_dict(state['P']);del state\n"
              "        init=dict(kind='Resume actual bridge and AdamW state',resume_step=resume_step,\n"
              "            checkpoint=str(resume_directory),optimizer_resumed=True,bridge_reinitialized=False)\n"
              "    else:\n"
              "        init=initialize(bridge)",
              "    init=initialize(bridge)")
        patch("    if resume_step:\n"
              "        assert initial_sha==protocol['resume_bridge_state_sha256'],'Resumed bridge differs from parent final weights'\n", '')
        patch("    if resume_step:\n"
              "        saved=torch.load(resume_directory/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)\n"
              "        assert saved['step']==resume_step\n"
              "        optimizer.load_state_dict(saved['optimizer']);del saved\n"
              "        assert len(optimizer.state)==len(params)\n"
              "        assert all(int(s['step'])==resume_step for s in optimizer.state.values())\n"
              "        assert all(g['lr']==learning_rate for g in optimizer.param_groups)\n"
              "        save(run_dir/f'resume.rank-{rank:02d}.json',dict(passed=True,step=resume_step,\n"
              "            bridge_state_sha256=initial_sha,optimizer_step=resume_step,\n"
              "            optimizer_tensors=len(optimizer.state),weights_restored=True,optimizer_restored=True))\n", '')
    restored = (
        "    if resume_step:\n"
        "        from qwen3.native_grpo import restore_bridge_and_optimizer\n"
        "        restored=restore_bridge_and_optimizer(resume_directory,resume_step,bridge,optimizer,\n"
        "            initial_sha,weight_digest)\n"
        "        hashes=[None]*world\n"
        "        dist.all_gather_object(hashes,restored)\n"
        "        assert len(set(hashes))==1\n"
        "        save(run_dir/f'resume.rank-{rank:02d}.json',dict(passed=True,step=resume_step,\n"
        "            bridge_state_sha256=restored,optimizer_step=resume_step,\n"
        "            weights_restored=True,optimizer_restored=True))\n")
    patch("        torch.optim.AdamW(params,lr=learning_rate,weight_decay=0.,betas=(.9,.999)))\n"
          "    student_path=Path(protocol.get('student_checkpoint',ROOT/'data/models/Qwen3.5-4B'))",
          "        torch.optim.AdamW(params,lr=learning_rate,weight_decay=0.,betas=(.9,.999)))\n"
          + restored + "    student_path=Path(protocol.get('student_checkpoint',ROOT/'data/models/Qwen3.5-4B'))")
    original_step = 'resume_step' if identifier != 13 else '0'
    patch("    if rank==0:\n"
          f"        checkpoint(bridge,optimizer,run_dir/'initial',{original_step},initial_sha)\n"
          "        save(run_dir/'initialization.json',dict(init,bridge_state_sha256=initial_sha,\n"
          "            replica_hashes=initial_hashes,layer_map=bridge.layer_map))\n"
          "        save(run_dir/'runtime.json',runtime)",
          "    if rank==0:\n"
          "        if resume_step:\n"
          "            assert json.loads((run_dir/'initialization.json').read_text())['bridge_state_sha256']==initial_sha\n"
          "            save(run_dir/f'runtime.resume-step-{resume_step:04d}.json',runtime)\n"
          "        else:\n"
          "            checkpoint(bridge,optimizer,run_dir/'initial',0,initial_sha)\n"
          "            save(run_dir/'initialization.json',dict(init,bridge_state_sha256=initial_sha,\n"
          "                replica_hashes=initial_hashes,layer_map=bridge.layer_map))\n"
          "            save(run_dir/'runtime.json',runtime)")
    patch("    if not args.smoke and protocol['validation_before_training']:",
          "    if not args.smoke and not resume_step and protocol['validation_before_training']:")
    if identifier == 13:
        patch("        started=time.monotonic()\n        steps=2 if args.smoke else protocol['total_steps']",
              "        if resume_step:\n"
              "            rng=torch.load(resume_directory/f'rng.rank-{rank:02d}.pt',\n"
              "                map_location='cpu',weights_only=True)\n"
              "            torch.set_rng_state(rng['cpu_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])\n"
              "        started=time.monotonic()\n        steps=2 if args.smoke else protocol['total_steps']")
        patch("        for step in range(1,steps+1):",
              "        for step in range(resume_step+1,steps+1):")
        patch("        if not args.smoke:\n            expected_visited=[i for step_index in range(steps)\n"
              "                for i in orders[step_index//epoch_steps][(step_index%epoch_steps)*batch_size:\n"
              "                                                          (step_index%epoch_steps+1)*batch_size]]\n"
              "            assert visited==expected_visited\n"
              "            if protocol.get('max_train_steps') is None:\n"
              "                assert len(visited)==examples*epochs",
              "        if not args.smoke:\n"
              "            expected_visited=[i for step_index in range(resume_step,steps)\n"
              "                for i in orders[step_index//epoch_steps][(step_index%epoch_steps)*batch_size:\n"
              "                                                          (step_index%epoch_steps+1)*batch_size]]\n"
              "            assert visited==expected_visited")
    else:
        patch("                assert len(visited)==examples*(epochs-resume_step//epoch_steps)",
              "                assert len(visited)==len(expected_visited)")
    patch("        nonzero_updates=0",
          "        nonzero_updates=(sum(json.loads(line).get('grad_norm',0)>0\n"
          "            for line in (run_dir/'metrics.jsonl').read_text().splitlines()) if resume_step else 0)")
    patch("            if (not args.smoke and save_now) or (args.smoke and step==steps):",
          "            if (not args.smoke and save_now) or (args.smoke and\n"
          "                    (step==steps or protocol.get('smoke_resume_checkpoints'))):")
    patch("                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),\n"
          "                    destination/f'rng.rank-{rank:02d}.pt')",
          "                temporary=destination/f'rng.rank-{rank:02d}.pt.pending'\n"
          "                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),\n"
          "                    temporary)\n"
          "                temporary.replace(destination/f'rng.rank-{rank:02d}.pt')\n"
          "                dist.barrier()\n"
          "                if rank==0:\n"
          "                    from qwen3.native_grpo import sign_checkpoint\n"
          "                    sign_checkpoint(destination,step,world)\n"
          "                dist.barrier()\n"
          "                if args.smoke and not resume_step and (\n"
          "                    os.environ.get('PLATOUR_GRPO_SMOKE_INTERRUPT_AFTER_SIGNED_STEP')==str(step)):\n"
          "                    if rank==0:save(run_dir/'intentional_interruption.json',\n"
          "                        dict(step=step,signed_checkpoint=str(destination)))\n"
          "                    dist.barrier()\n"
          "                    raise RuntimeError('Controlled GRPO smoke stopped after a signed step')")
    ast.parse(text)
    return text


def prepare(config, sender, receiver, train_rows, test_rows, output, *, smoke=False,
            validation_rows=None):
    """Freeze original signed rows and full source before allocating a GPU."""
    output = Path(output).absolute()
    sender, receiver = Path(sender).resolve(), Path(receiver).resolve()
    if config['method'] != 'grpo' or sender.parent != receiver.parent:
        raise ValueError('Native GRPO needs a GRPO recipe and one shared model root')
    originals = json.loads(PROTOCOLS.read_text())
    original = originals[str(config['id'])]
    if (sender.name != original['teacher'] or receiver.name != original['student'] or
            len(train_rows) != config['train_examples'] or not test_rows or
            original['group_size'] != 8 or original['method'] != 'grpo'):
        raise ValueError('Frozen GRPO recipe, models or signed input count differs')
    placement = os.environ.get('PLATOUR_NATIVE_VLLM_GPU_PLACEMENT')
    if placement is not None and (placement != 'secondary' or not smoke or
            config['id'] != 15 or original.get('teacher_model_parallel_size') != 2 or
            original.get('teacher_weight_precision') != 'fp8'):
        raise ValueError('Secondary vLLM placement is only for ID15 FP8 two-step smoke')
    memory_override = os.environ.get('PLATOUR_NATIVE_VLLM_FRACTION')
    if memory_override is not None:
        fraction = float(memory_override)
        if (not smoke or original.get('teacher_weight_precision') != 'fp8' or
                not 0.065 <= fraction < original['vllm_memory_fraction']):
            raise ValueError('Lower vLLM memory fraction is FP8 smoke-only; keep the original formal recipe')
    if validation_rows is None and any('original_train_index' in row for row in train_rows):
        raise ValueError('Published TRAIN requires its explicit validation split; refusing to resplit')
    verify_bundle()
    if output.exists():
        raise FileExistsError('Native GRPO requires a fresh run directory')
    output.mkdir(parents=True)
    policy.save(output / 'train_prompts.json', train_rows)
    policy.save(output / 'eval_prompts.json', test_rows)
    protocol = dict(original)
    for key in ('resume_from_checkpoint', 'resume_step', 'reuse_completed_smoke_root',
                'max_train_steps', 'selected_step', 'source_checkpoint_sha256'):
        protocol.pop(key, None)
    protocol.update(teacher_checkpoint=str(sender), student_checkpoint=str(receiver),
                    wandb_mode='disabled', wandb_enabled=False,
                    verify_source_hashes=False, smoke_only=smoke,
                    smoke_resume_checkpoints=smoke,
                    execution_backend='bundled frozen native GRPO',
                    train_prompts_sha256=sha256(output / 'train_prompts.json'),
                    eval_prompts_sha256=sha256(output / 'eval_prompts.json'))
    if memory_override is not None:
        protocol['smoke_decoder_memory_override'] = {
            'original_fraction': original['vllm_memory_fraction'],
            'actual_fraction': fraction, 'reason': 'FP8 teacher backward headroom'}
        protocol['vllm_memory_fraction'] = fraction
    protocol["validation_samples"] = config.get("validation_samples", original["eval_samples"])
    policy.configure(output, protocol, seed=42, validation_rows=validation_rows)
    shutil.copyfile(Path(policy.__file__), output / 'validation_policy.py')
    protocol['validation_policy_sha256'] = sha256(output / 'validation_policy.py')
    parallel = protocol['teacher_model_parallel_size']
    gpu_ids = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(',')))
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or 8 // parallel != original['world_size']:
        raise ValueError('Native GRPO requires eight distinct GPUs in the historical rank layout')
    protocol['gpu_ids'] = gpu_ids
    protocol['decoder_gpu_ids'] = gpu_ids[::parallel]
    if placement == 'secondary':
        protocol['smoke_decoder_gpu_placement'] = {
            'original_gpu_ids': gpu_ids[::parallel],
            'actual_gpu_ids': gpu_ids[parallel - 1::parallel],
            'reason': 'FP8 long-answer teacher backward headroom on the primary GPU'}
        protocol['decoder_gpu_ids'] = gpu_ids[parallel - 1::parallel]
    if parallel > 1:
        protocol['teacher_gpu_groups'] = [gpu_ids[i:i + parallel]
                                          for i in range(0, 8, parallel)]
        if 'teacher_gpu_pairs' in protocol:
            protocol['teacher_gpu_pairs'] = protocol['teacher_gpu_groups']
    cache = os.environ.get('HF_HUB_CACHE')
    if cache is not None:
        protocol['fp8_kernel_cache'] = str(Path(cache).parent)
    frozen = output / 'reference_source'
    shutil.copytree(ASSETS / 'base', frozen)
    shutil.copytree(ASSETS / 'overlays' / f"{config['id']:02d}", frozen,
                    dirs_exist_ok=True)
    if config['id'] == 13:
        core_relative = Path('recipe/llm_communicate/qwen35_bridge_opd_core.py')
        core = frozen / core_relative
        original_core = ASSETS / 'overlays/13' / core_relative
        core.write_text(zero_native_13_core_source(core.read_text()))
        trainer_relative = Path('recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py')
        active_trainer = frozen / trainer_relative
        active_trainer.write_text(zero_native_13_trainer_source(active_trainer.read_text()))
        protocol['historical_initialization'] = original['initialization']
        protocol['initialization'] = 'exact-zero linear bridge'
        protocol['zero_initialization_variant'] = True
        policy.save(output / 'zero_bridge_initialization_patch.json', {
            'experiment': 13, 'original_core_sha256': sha256(original_core),
            'active_core_sha256': sha256(core),
            'original_trainer_sha256': sha256(ASSETS / 'overlays/13' / trainer_relative),
            'active_trainer_before_resume_sha256': sha256(active_trainer),
            'not_identical_to_published_id13_training_initialization': True})
    if placement == 'secondary':
        relative = Path('recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py')
        trainer = frozen / relative
        original_text = trainer.read_text()
        substitutions = (
            ("assert physical[device_index]==protocol['decoder_gpu_ids'][local]",
             "assert physical[device_index+1]==protocol['decoder_gpu_ids'][local]"),
            ("gpu=physical[device_index],memory_fraction=protocol['vllm_memory_fraction']",
             "gpu=protocol['decoder_gpu_ids'][local],memory_fraction=protocol['vllm_memory_fraction']"))
        modified = original_text
        for before, after in substitutions:
            if modified.count(before) != 1:
                raise ValueError('The frozen GPU placement source differs; no smoke patch applied')
            modified = modified.replace(before, after)
        trainer.write_text(modified)
        policy.save(output / 'smoke_source_patch.json', {
            'source': relative.as_posix(), 'original_sha256': sha256(ASSETS / 'overlays/15' / relative),
            'active_sha256': sha256(trainer), 'placement': 'teacher secondary GPU',
            'substitutions': len(substitutions), 'scientific_data_reward_loss_unchanged': True})
    relative = Path('recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py')
    trainer = frozen / relative
    baseline_sha = sha256(ASSETS / 'overlays' / f"{config['id']:02d}" / relative)
    before_resume_sha = sha256(trainer)
    trainer.write_text(resume_source_text(trainer.read_text(), config['id']))
    policy.save(output / 'native_resume_source_patch.json', {
        'experiment': config['id'], 'source': relative.as_posix(),
        'original_scientific_sha256': baseline_sha, 'before_resume_sha256': before_resume_sha,
        'active_sha256': sha256(trainer), 'frozen_train_reward_and_loss_unchanged': True,
        'checkpoint_strategy': 'formal original validation boundaries; every signed smoke update'})
    if placement == 'secondary':
        marker = json.loads((output / 'smoke_source_patch.json').read_text())
        marker['before_resume_sha256'] = marker['active_sha256']
        marker['active_sha256'] = sha256(trainer)
        policy.save(output / 'smoke_source_patch.json', marker)
    (frozen / 'data').mkdir(exist_ok=True)
    (frozen / 'data/models').symlink_to(sender.parent, target_is_directory=True)
    for package in ('third_party/verl/verl', 'third_party/verl/verl/utils',
                    'third_party/verl/verl/utils/reward_score', 'third_party/c2c/rosetta',
                    'third_party/c2c/rosetta/model', 'third_party/c2c/rosetta/utils',
                    'third_party/c2c/rosetta/train'):
        init = frozen / package / '__init__.py'
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text('"""Bundled native GRPO dependency namespace."""\n')
    scorer = frozen / CHEMISTRY_SCORER
    scorer.write_text(strict_chemistry_source(scorer.read_text()))
    protocol.update(chemistry_answer_field_required=True, chemistry_scorer_sha256=sha256(scorer))
    policy.save(output / 'protocol.json', protocol)
    policy.save(output / 'original_protocol.json', original)
    policy.save(output / 'reference_source_sha256.json', {
        str(p.relative_to(frozen)): sha256(p) for p in frozen.rglob('*.py')})
    policy.save(output / 'reference_audit.json', {
        'experiment': config['id'], 'backend': 'bundled frozen native GRPO',
        'simplified_hf_trainer_used': False, 'external_checkout_required': False,
        'signed_train_count': len(train_rows), 'signed_test_count': len(test_rows),
        'original_train_sha256': protocol['train_prompts_sha256'],
        'source_files': len(list(frozen.rglob('*.py')))})
    return output


def worker(root, *, resume_step=None):
    root = Path(root).absolute()
    frozen = root / 'reference_source'
    sys.path[:0] = [str(frozen / part) for part in
                    ('recipe/llm_communicate', 'scripts', 'third_party/verl', 'third_party/c2c')]
    from transformers.integrations import finegrained_fp8 as fp8
    if not hasattr(fp8, 'FP8Expert') and hasattr(fp8, 'FP8Experts'):
        fp8.FP8Expert = fp8.FP8Experts
    from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING
    _HUB_KERNEL_MAPPING['finegrained-fp8'] = dict(
        repo_id='kernels-community/finegrained-fp8', revision=FP8_KERNEL_REVISION)
    protocol = json.loads((root / 'protocol.json').read_text())
    verify_chemistry_scorer(root, protocol)
    if protocol.get('teacher_weight_precision') == 'fp8' and hasattr(fp8, 'load_finegrained_fp8_kernel'):
        fp8.load_finegrained_fp8_kernel()
    import train_qwen35_bridge_opd_vllm as trainer
    if sha256(root / 'validation_policy.py') != protocol['validation_policy_sha256']:
        raise ValueError('Frozen holdout policy changed')
    sys.path.insert(0, str(root))
    import validation_policy
    validation_policy.install(trainer, root)
    sys.argv = [str(frozen / 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'),
                '--root', str(root), '--objective', 'grpo']
    if protocol['smoke_only']:
        sys.argv += ['--smoke', '--stress']
    if resume_step is not None:
        sys.argv += ['--resume-step', str(resume_step)]
    trainer.main()


def execute(root, *, resume_step=None):
    root = Path(root).absolute()
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha256(root / 'validation_policy.py') != protocol['validation_policy_sha256']:
        raise ValueError('Frozen holdout policy changed')
    policy.verify_inputs(root, protocol)
    for name, digest in json.loads((root / 'reference_source_sha256.json').read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(root / 'reference_source' / name) != digest:
            raise ValueError('Prepared GRPO source changed: ' + name)
    if resume_step is not None:
        if os.environ.get('PLATOUR_GRPO_SMOKE_INTERRUPT_AFTER_SIGNED_STEP'):
            raise ValueError('Do not carry the controlled smoke interruption into a resumed run')
        archive_resume_tail(root, resume_step)
    else:
        interruption = os.environ.get('PLATOUR_GRPO_SMOKE_INTERRUPT_AFTER_SIGNED_STEP')
        if interruption is not None and (not protocol['smoke_only'] or interruption != '1'):
            raise ValueError('Only one fresh signed smoke step may trigger a controlled interruption')
    environment = dict(os.environ, WANDB_MODE='disabled', WANDB_DISABLED='true',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PLATOUR_BRIDGE_PYTHON=sys.executable,
                       PLATOUR_GENERATION_PYTHON=sys.executable,
                       PLATOUR_SCORING_PYTHON=os.environ.get('PLATOUR_SCORING_PYTHON', sys.executable),
                       PYTHONPATH=os.pathsep.join(str(root / 'reference_source' / part)
                           for part in ('scripts', 'recipe/llm_communicate',
                                        'third_party/verl', 'third_party/c2c')))
    port = int(environment.get('PLATOUR_NATIVE_MASTER_PORT', '39116'))
    if not 1024 <= port <= 65535:
        raise ValueError('Native GRPO rendezvous port must be in 1024..65535')
    environment['MASTER_ADDR'] = '127.0.0.1'
    environment['MASTER_PORT'] = str(port)
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--nnodes=1',
               f"--nproc-per-node={protocol['world_size']}",
               '--master-addr=127.0.0.1', f'--master-port={port}', '--max-restarts=0',
               '-m', 'qwen3.native_grpo', 'worker', '--root', str(root)]
    if resume_step is not None:
        command += ['--resume-step', str(resume_step)]
    environment['PYTHONPATH'] += os.pathsep + str(Path(__file__).resolve().parents[1])
    if os.environ.get('PYTHONPATH'):
        environment['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']
    policy.save(root / 'phase.json', {'phase': 'smoke' if protocol['smoke_only'] else 'train',
                                      'time': time.time(), 'world_size': protocol['world_size'],
                                      'resumed_from_step': resume_step})
    try:
        from platour_exps import run_native_training
        folder = root / ('smoke' if protocol['smoke_only'] else 'train')
        run_native_training(command, cwd=root / 'reference_source', environment=environment,
                            output=root, failure_directory=folder, smoke=protocol['smoke_only'],
                            resume_step=resume_step, allow_smoke_resume=True)
        completed = json.loads((folder / 'complete.json').read_text())
        if (not completed.get('complete') or not completed.get('replica_exact_weights') or
                completed.get('nonzero_gradient_updates', 0) < 1 or
                completed.get('smoke') != protocol['smoke_only']):
            raise RuntimeError('Native GRPO structural completion differs')
        if protocol['smoke_only']:
            profile = json.loads((folder / 'worst_case_profile.json').read_text())
            if not profile['passed'] or not profile['vllm_resident']:
                raise RuntimeError('Native GRPO worst-case profile failed')
            quality = json.loads((folder / 'evaluation/step-0002/summary.json').read_text())
            if (not quality.get('complete') or quality['questions'] != 6 or
                    quality['responses'] != 12 or quality.get('scored_responses') != 12 or
                    quality.get('gold_missing') or quality.get('scoring_timeouts') or
                    quality['accuracy'] <= 0 or quality['clipping_rate'] >= 1):
                raise RuntimeError('Native GRPO smoke scoring or coverage incomplete')
        else:
            if completed['steps'] != protocol['total_steps']:
                raise RuntimeError('Native GRPO total updates differ')
            selection = json.loads((root / 'selection.json').read_text())
            tested = json.loads((root / 'test/summary.json').read_text())
            if (selection['test_used_for_selection'] or not tested['complete'] or
                    tested['selected_step'] != selection['selected_step']):
                raise RuntimeError('Validation-best final test did not complete')
        policy.save(root / 'phase.json', {'phase': 'complete', 'time': time.time()})
    except BaseException as error:
        policy.save(root / 'phase.json', {'phase': 'failed', 'error': repr(error),
                                          'time': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('worker', 'execute'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-step', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, resume_step=args.resume_step)
    else:
        execute(args.root, resume_step=args.resume_step)


if __name__ == '__main__':
    main()
