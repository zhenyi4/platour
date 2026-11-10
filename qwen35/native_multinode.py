"""Portable, fail-closed two-host 20/25 launcher based on the original native worker.

`plan` and `prepare` never initialize CUDA. `run` requires separate explicit
invocations on both authorized eight-GPU hosts, a shared absolute output path,
and live NCCL/shared-storage proof before any model training can start.
"""

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from uuid import uuid4

from .catalog import select
from .checkpoints import atomic_json, sha256
from .cli import DATA, inputs
from .native_deepcoder import verify_bundle as verify_code_bundle
from .native_math20 import verify_bundle as verify_math_bundle
from .scoring import judge_service_info
from .judge_service import ORIGINAL_IMAGE


TEMPLATES = Path(__file__).with_name('native_multinode_protocols.json')
SOURCES = Path(__file__).with_name('native_reference')
TOPOLOGY = dict(nnodes=2, gpus_per_node=8, local_world_size=2, world_size=4,
                total_gpus=16, teacher_model_parallel_size=4)
CONFIG_SOURCE = 'recipe/llm_communicate/qwen35_multinode_config.py'
TRAINER_SOURCE = 'recipe/llm_communicate/train_qwen35_bridge_opd_vllm.py'
QUALIFICATION_SOURCE = 'recipe/llm_communicate/formal_qualification.py'
SHIMS = {'recipe/llm_communicate/training_policy.py':
             'from qwen35.native_multinode_training_policy import validation_update, training_seed, common_policy\n',
         'recipe/llm_communicate/shared_judge.py':
             'from qwen35.native_multinode_judge import DeepCoderScorer, service_info, SERVICE\n'}
CODE_SUPPORT = ('native_multinode_training_policy.py', 'native_multinode_judge.py',
                'native_multinode_qualification.py', 'scoring.py', 'judge_service.py',
                '_sandbox.py')


FP8_KERNEL = ('kernels-community/finegrained-fp8', '8c178950e8710e26a2210e4e909cd02dc16c8715')


def fp8_kernel_cache(cache):
    """Complete pinned FP8 expert-kernel snapshot for the offline native ranks.

    The ranks run with ``HF_HUB_OFFLINE=1``. kernels 0.16 then resolves the kernel only
    from a *complete* cached snapshot, while an online load stores just this machine's
    build variant, so fetch the whole pinned revision once (with Hub access) into the
    run's signed ``fp8_kernel_cache`` and point ``KERNELS_CACHE`` at it.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    repo, revision = FP8_KERNEL
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo, repo_type='kernel', revision=revision, cache_dir=cache,
                          local_files_only=True)
    except LocalEntryNotFoundError:
        try:
            snapshot_download(repo, repo_type='kernel', revision=revision, cache_dir=cache)
        except Exception as error:
            raise RuntimeError(f'Pinned FP8 expert kernel {repo}@{revision} is not cached in '
                               f'{cache}; start this stage once with Hugging Face Hub access') from error
    return str(cache)


def _support_sources(identifier):
    return ({name: sha256(Path(__file__).with_name(name)) for name in CODE_SUPPORT}
            if identifier == 25 else {})


def _expected_sources(identifier, originals):
    config = (SOURCES / str(identifier) / CONFIG_SOURCE).read_text()
    variants = {CONFIG_SOURCE: hashlib.sha256(_private_config(config).encode()).hexdigest()}
    trainer = (SOURCES / str(identifier) / TRAINER_SOURCE).read_text()
    body = _private_math_trainer(trainer) if identifier == 20 else _private_code_trainer(trainer)
    variants[TRAINER_SOURCE] = hashlib.sha256(body.encode()).hexdigest()
    if identifier == 25:
        original = (SOURCES / '25' / QUALIFICATION_SOURCE).read_text()
        variants[QUALIFICATION_SOURCE] = hashlib.sha256(
            _private_qualification(original).encode()).hexdigest()
        variants.update({name: hashlib.sha256(body.encode()).hexdigest()
                         for name, body in SHIMS.items()})
    return {**originals, **variants}


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Original two-host science source differs from the audited port')
    return text.replace(old, new, 1)


def _private_config(text):
    text = _replace_once(text, 'ORIGINAL = ROOT.parents[2]', 'ORIGINAL = ROOT.parent')
    text = _replace_once(text,
        "digest(ORIGINAL/manifest['launcher']) != manifest['launcher_sha256']",
        "digest(Path(manifest['launcher'])) != manifest['launcher_sha256']")
    return text


def _private_qualification(text):
    left = 'def verify_evidence(original,protocol=None):'
    right = '\n\ndef verify_live(root,protocol,rank,initial_sha):'
    if text.count(left) != 1 or text.count(right) != 1:
        raise ValueError('Original qualification code changed')
    start, finish = text.index(left), text.index(right)
    body = ('def verify_evidence(original,protocol=None):\n'
            '    from qwen35.native_multinode_qualification import verify_evidence as verified\n'
            '    return verified(original,protocol)\n')
    return text[:start] + body + text[finish:]


def _private_math_trainer(text):
    """Port ID25's per-step signed rolling I/O to new ID20 private copies."""
    import ast

    text = _replace_once(text,
        "def checkpoint_complete(directory, world):\n"
        "    return all((directory/name).exists() for name in ('checkpoint.json','bridge.pt','optimizer.pt')) and \\\n"
        "        all((directory/f'rng.rank-{r:02d}.pt').exists() for r in range(world))",
        "def checkpoint_content_sha(path):\n"
        "    with Path(path).open('rb') as stream:\n"
        "        return hashlib.file_digest(stream,'sha256').hexdigest()\n\n\n"
        "def checkpoint_complete(directory, world, require_marker=False):\n"
        "    names=('checkpoint.json','bridge.pt','optimizer.pt',*(f'rng.rank-{r:02d}.pt' for r in range(world)))\n"
        "    if not all((directory/name).is_file() and (directory/name).stat().st_size>0 for name in names):return False\n"
        "    if not require_marker:return True\n"
        "    marker=directory/'complete.json'\n"
        "    if not marker.is_file():return False\n"
        "    try:status=json.loads(marker.read_text());meta=json.loads((directory/'checkpoint.json').read_text())\n"
        "    except (ValueError,OSError):return False\n"
        "    return (status.get('complete') is True and status.get('step')==meta.get('step')\n"
        "        and status.get('world_size')==world and status.get('bridge_state_sha256')==meta.get('bridge_state_sha256')\n"
        "        and meta.get('bridge_sha256')==checkpoint_content_sha(directory/'bridge.pt')\n"
        "        and status.get('optimizer_sha256')==checkpoint_content_sha(directory/'optimizer.pt')\n"
        "        and status.get('rank_rng_sha256')=={str(r):checkpoint_content_sha(directory/f'rng.rank-{r:02d}.pt') for r in range(world)})")
    text = _replace_once(text,
        "def evaluation_summaries(run_dir, upto):",
        "def retire_previous_checkpoint(run_dir, step, protected, world):\n"
        "    current=run_dir/'checkpoints'/f'step-{step:04d}'\n"
        "    if not checkpoint_complete(current,world,require_marker=True):\n"
        "        raise RuntimeError('New native checkpoint incomplete; preserve its predecessors')\n"
        "    # Keep n/n-1 plus protected steps; sweep every older signed step (crash-safe).\n"
        "    for target in sorted((run_dir/'checkpoints').glob('step-[0-9][0-9][0-9][0-9]')):\n"
        "        previous=int(target.name[5:])\n"
        "        if previous<=0 or previous>step-2 or previous in protected:continue\n"
        "        if target.is_symlink() or not checkpoint_complete(target,world,require_marker=True):\n"
        "            if previous==step-2:raise RuntimeError('Previous native checkpoint is not signed; refuse deletion')\n"
        "            continue\n"
        "        shutil.rmtree(target)\n\n\n"
        "def evaluation_summaries(run_dir, upto):")
    text = _replace_once(text,
        "    assert checkpoint_complete(directory,world), f'Incomplete checkpoint: {directory}'",
        "    protocol=json.loads((run_dir.parent/'protocol.json').read_text())\n"
        "    assert checkpoint_complete(directory,world,require_marker=protocol.get('per_step_checkpoints',False)), f'Incomplete checkpoint: {directory}'")
    text = _replace_once(text,
        "assert resume<protocol['total_steps'] and resume in (protocol.get('checkpoint_steps') or [])",
        "assert 0<resume<protocol['total_steps'] and (protocol.get('per_step_checkpoints',False) or resume in (protocol.get('checkpoint_steps') or []))")
    text = _replace_once(text,
        "            checkpoint_steps=protocol.get('checkpoint_steps')\n"
        "            save_now=(step in checkpoint_steps if checkpoint_steps is not None else step%epoch_steps==0)\n"
        "            if (not args.smoke and save_now) or (args.smoke and step==steps):\n"
        "                destination=run_dir/'checkpoints'/f'step-{step:04d}' if checkpoint_steps is not None else run_dir/f'epoch-{epoch+1}'\n"
        "                if rank==0:\n"
        "                    checkpoint(bridge,optimizer,destination,step,initial_sha)\n"
        "                dist.barrier()\n"
        "                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),\n"
        "                    destination/f'rng.rank-{rank:02d}.pt')",
        "            checkpoint_steps=protocol.get('checkpoint_steps')\n"
        "            per_step=bool(protocol.get('per_step_checkpoints',False)) and not args.smoke\n"
        "            save_now=per_step or (step in checkpoint_steps if checkpoint_steps is not None else step%epoch_steps==0)\n"
        "            if (not args.smoke and save_now) or (args.smoke and step==steps):\n"
        "                destination=run_dir/'checkpoints'/f'step-{step:04d}' if checkpoint_steps is not None else run_dir/f'epoch-{epoch+1}'\n"
        "                if rank==0:checkpoint(bridge,optimizer,destination,step,initial_sha)\n"
        "                dist.barrier()\n"
        "                temporary=destination/f'rng.rank-{rank:02d}.pt.tmp'\n"
        "                torch.save(dict(cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)\n"
        "                temporary.replace(destination/f'rng.rank-{rank:02d}.pt')\n"
        "                if per_step:\n"
        "                    dist.barrier()\n"
        "                    if rank==0:\n"
        "                        if not checkpoint_complete(destination,world):raise RuntimeError('Incomplete four-rank optimizer/RNG checkpoint')\n"
        "                        save(destination/'complete.json',dict(complete=True,step=step,world_size=world,\n"
        "                            bridge_state_sha256=weight_digest(bridge),\n"
        "                            optimizer_sha256=checkpoint_content_sha(destination/'optimizer.pt'),\n"
        "                            rank_rng_sha256={str(r):checkpoint_content_sha(destination/f'rng.rank-{r:02d}.pt') for r in range(world)}))\n"
        "                        retire_previous_checkpoint(run_dir,step,{0,*checkpoint_steps},world)\n"
        "                    dist.barrier()")
    ast.parse(text)
    return text


def _private_code_trainer(text):
    import ast

    text = _replace_once(text,
        "    previous_step=step-1\n"
        "    if previous_step<=0 or previous_step in protected_steps:\n"
        "        return False\n"
        "    previous=run_dir/'checkpoints'/f'step-{previous_step:04d}'\n"
        "    if previous.exists():\n"
        "        if not checkpoint_complete(previous,world,require_marker=True):\n"
        "            raise RuntimeError(f'Previous checkpoint incomplete; refusing deletion: {previous}')\n"
        "        shutil.rmtree(previous)\n"
        "        return True\n"
        "    return False\n",
        "    # Keep n/n-1 plus protected steps; sweep every older signed step (crash-safe).\n"
        "    retired=False\n"
        "    for previous in sorted((run_dir/'checkpoints').glob('step-[0-9][0-9][0-9][0-9]')):\n"
        "        previous_step=int(previous.name[5:])\n"
        "        if previous_step<=0 or previous_step>step-2 or previous_step in protected_steps:\n"
        "            continue\n"
        "        if previous.is_symlink() or not checkpoint_complete(previous,world,require_marker=True):\n"
        "            if previous_step==step-2:\n"
        "                raise RuntimeError(f'Previous checkpoint incomplete; refusing deletion: {previous}')\n"
        "            continue\n"
        "        shutil.rmtree(previous)\n"
        "        retired=True\n"
        "    return retired\n")
    text = _replace_once(text, 'def checkpoint_complete(directory, world, require_marker=False):',
        "def checkpoint_content_sha(path):\n"
        "    with Path(path).open('rb') as stream:\n"
        "        return hashlib.file_digest(stream,'sha256').hexdigest()\n\n\n"
        'def checkpoint_complete(directory, world, require_marker=False):')
    text = _replace_once(text,
        "            and status.get('bridge_state_sha256')==meta.get('bridge_state_sha256'))",
        "            and status.get('bridge_state_sha256')==meta.get('bridge_state_sha256')\n"
        "            and meta.get('bridge_sha256')==checkpoint_content_sha(directory/'bridge.pt')\n"
        "            and status.get('optimizer_sha256')==checkpoint_content_sha(directory/'optimizer.pt')\n"
        "            and status.get('rank_rng_sha256')=={str(r):checkpoint_content_sha(directory/f'rng.rank-{r:02d}.pt') for r in range(world)})")
    text = _replace_once(text,
        "                        save(destination/'complete.json',dict(complete=True,step=step,world_size=world,\n"
        "                            bridge_state_sha256=weight_digest(bridge),optimizer_and_four_rank_rng_saved=True))",
        "                        save(destination/'complete.json',dict(complete=True,step=step,world_size=world,\n"
        "                            bridge_state_sha256=weight_digest(bridge),optimizer_and_four_rank_rng_saved=True,\n"
        "                            optimizer_sha256=checkpoint_content_sha(destination/'optimizer.pt'),\n"
        "                            rank_rng_sha256={str(r):checkpoint_content_sha(destination/f'rng.rank-{r:02d}.pt') for r in range(world)}))")
    ast.parse(text)
    return text


def prepare(identifier, sender, receiver, output, *, data_root=DATA, service=None):
    """Stage only byte-checked original science plus public signed prompt tokens."""
    if identifier not in (20, 25):
        raise ValueError('Only original two-host IDs 20 and 25 are supported')
    output, sender, receiver = Path(output).absolute(), Path(sender).resolve(), Path(receiver).resolve()
    case = select(identifier)
    if (output.exists() or sender.name != case.sender or receiver.name != 'Qwen3.5-4B' or
            sender.parent != receiver.parent or
            not all((path / 'config.json').is_file() for path in (sender, receiver))):
        raise ValueError('Fresh shared root and matching public FP8 sender/4B receiver required')
    signed = inputs(case, data_root)
    if identifier == 20:
        from .native_math20 import audit as verify_original_inputs
        verify_original_inputs(case, signed, data_root)
    else:
        from .native_deepcoder import audit as verify_original_inputs
        verify_original_inputs(case, signed, data_root)
    originals = verify_math_bundle() if identifier == 20 else verify_code_bundle()
    template = json.loads(TEMPLATES.read_text())[str(identifier)]
    protocol = template['protocol'].copy()
    if (protocol.get('world_size') != 4 or protocol.get('nnodes') != 2 or
            protocol.get('gpus_per_node') != 8 or protocol.get('total_steps') != case.steps or
            protocol.get('eval_samples') != case.val_samples or
            protocol.get('initialization') != 'zero' or
            len(signed['train']) != case.train_questions or
            len(signed['validation']) != case.val_questions):
        raise ValueError('Original native recipe or signed data differs')
    judge = None
    if identifier == 25:
        if service is None:
            raise ValueError('Original 25 needs the pinned isolated CPU Docker judge service')
        judge = judge_service_info(service)
        if ('25' not in judge.get('supported_experiments', ()) or
                judge.get('image_id') != ORIGINAL_IMAGE or
                judge.get('checker_sha256') is None or
                judge.get('case_proof_sha256') is None or not judge.get('case_root')):
            raise RuntimeError('Original 25 has no approved Docker/checker/case service')
        verify_original_inputs(case, signed, data_root, case_root=judge['case_root'],
                               verify_cases=True)
    output.mkdir(parents=True)
    source = output / 'reference_source'
    shutil.copytree(SOURCES / str(identifier), source)
    config = source / CONFIG_SOURCE
    config.write_text(_private_config(config.read_text()))
    trainer = source / TRAINER_SOURCE
    trainer.write_text(_private_math_trainer(trainer.read_text()) if identifier == 20
                       else _private_code_trainer(trainer.read_text()))
    active = {CONFIG_SOURCE: sha256(config), TRAINER_SOURCE: sha256(trainer)}
    if identifier == 25:
        qualified = source / QUALIFICATION_SOURCE
        qualified.write_text(_private_qualification(qualified.read_text()))
        active[QUALIFICATION_SOURCE] = sha256(qualified)
        for name, body in SHIMS.items():
            destination = source / name
            destination.write_text(body)
            active[name] = sha256(destination)
    model_root = source / 'data/models'
    model_root.parent.mkdir(parents=True, exist_ok=True)
    model_root.symlink_to(sender.parent, target_is_directory=True)
    manifest = {'files': {**originals, **active},
                'support_source_sha256': _support_sources(identifier),
                'launcher': str(Path(__file__).resolve()),
                'launcher_sha256': sha256(Path(__file__)),
                'original_bundle_sha256': sha256(
                    Path(__file__).with_name('native_math20_source_sha256.json' if identifier == 20
                                             else 'native_deepcoder_source_sha256.json'))}
    atomic_json(output / 'source_manifest.json', manifest)
    atomic_json(output / 'train_prompts.json', signed['train'])
    atomic_json(output / 'eval_prompts.json', signed['validation'])
    protocol.update(source_execution_root=str(source),
                    source_manifest_sha256=sha256(output / 'source_manifest.json'),
                    train_prompts_sha256=sha256(output / 'train_prompts.json'),
                    eval_prompts_sha256=sha256(output / 'eval_prompts.json'),
                    teacher_checkpoint=str(sender), student_checkpoint=str(receiver),
                    fp8_kernel_cache=str(output / 'runtime_cache/fp8_kernel'),
                    wandb_mode='disabled', wandb_enabled=False,
                    per_step_checkpoints=True, non_validation_checkpoint_retention=2,
                    launch_provenance='original signed science with public case/token inputs',
                    original_protocol_sha256=template['original_protocol_sha256'])
    if identifier == 25:
        protocol.update(judge_service_root=str(Path(service).resolve()),
                        judge_service_spec_id=judge['spec_id'],
                        code_judge=dict(protocol.get('code_judge', {}),
                                        image=judge['image'], image_id=judge['image_id']))
        protocol.pop('qualification_evidence', None)
    from .native_selection import contract
    protocol.update(contract())
    atomic_json(output / 'protocol.json', protocol)
    atomic_json(output / 'prepare.json', {'complete': True, 'experiment': identifier,
        'source_manifest_sha256': protocol['source_manifest_sha256'],
        'protocol_sha256': sha256(output / 'protocol.json'),
        'train_rows': len(signed['train']), 'validation_rows': len(signed['validation']),
        'no_gpu_process_started': True, 'two_host_probe_required': True,
        'judge_qualification_required': identifier == 25})
    return output


def verify_prepared(root, identifier):
    root = Path(root).absolute()
    proof = json.loads((root / 'prepare.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    manifest = json.loads((root / 'source_manifest.json').read_text())
    originals = verify_math_bundle() if identifier == 20 else verify_code_bundle()
    source_record = Path(__file__).with_name('native_math20_source_sha256.json'
                                             if identifier == 20 else 'native_deepcoder_source_sha256.json')
    if (proof.get('complete') is not True or proof.get('experiment') != identifier or
            proof.get('protocol_sha256') != sha256(root / 'protocol.json') or
            proof.get('source_manifest_sha256') != sha256(root / 'source_manifest.json') or
            protocol.get('source_manifest_sha256') != proof['source_manifest_sha256'] or
            protocol.get('source_execution_root') != str(root / 'reference_source') or
            protocol.get('world_size') != 4 or protocol.get('total_steps') != select(identifier).steps or
            manifest.get('launcher') != str(Path(__file__).resolve()) or
            manifest.get('launcher_sha256') != sha256(Path(__file__)) or
            manifest.get('original_bundle_sha256') != sha256(source_record) or
            manifest.get('support_source_sha256') != _support_sources(identifier) or
            manifest.get('files') != _expected_sources(identifier, originals)):
        raise RuntimeError('Prepared native two-host science, data or launcher SHA differs')
    source = root / 'reference_source'
    for name, expected in manifest['files'].items():
        file = source / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or file.is_symlink() or
                not file.is_file() or sha256(file) != expected):
            raise RuntimeError('Private native science source changed: ' + name)
    for split, file in (('train', 'train_prompts.json'), ('eval', 'eval_prompts.json')):
        if sha256(root / file) != protocol[split + '_prompts_sha256']:
            raise RuntimeError('Signed original native two-host input changed: ' + file)
    if identifier == 25:
        judge = judge_service_info(protocol['judge_service_root'])
        if (judge['spec_id'] != protocol['judge_service_spec_id'] or
                judge['image_id'] != ORIGINAL_IMAGE or '25' not in judge['supported_experiments']):
            raise RuntimeError('Original isolated ID25 Docker judge changed')
    from .native_selection import verify
    verify(protocol)
    return protocol


def torchrun_command(root, node_rank, master_addr, master_port, attempt, *, python=None):
    """Return exactly two local TP4 ranks per eight-GPU host; no shell evaluation."""
    if (node_rank not in (0, 1) or not 1024 <= master_port <= 65535 or
            not attempt or not all(c.isalnum() or c in '_-' for c in attempt)):
        raise ValueError('Invalid two-host rank, port or attempt identity')
    try:
        address = ipaddress.ip_address(master_addr)
    except ValueError as error:
        if not master_addr or master_addr.lower() == 'localhost' or any(
                char in master_addr for char in ('/', ' ', '\n', '<', '>')):
            raise ValueError('Specify reachable node-0 address on both hosts') from error
    else:
        if address.is_loopback or address.is_unspecified or address.is_link_local:
            raise ValueError('Two-host rendezvous must not use localhost/loopback')
    root = Path(root).absolute()
    control = root / 'invocations' / attempt / 'control.json'
    return [str(python or sys.executable), '-B', '-m', 'torch.distributed.run',
            '--nnodes=2', '--nproc-per-node=2', f'--node-rank={node_rank}',
            f'--master-addr={master_addr}', f'--master-port={master_port}',
            '--max-restarts=0', '-m', 'qwen35.native_multinode', 'worker',
            '--root', str(root), '--control', str(control)]


def _wait(path, folder, seconds):
    deadline = time.monotonic() + seconds
    while not Path(path).is_file():
        failures = list(Path(folder).glob('failure-node-*.json'))
        if failures:
            raise RuntimeError('Other node failed; preserve its native failure receipt')
        if time.monotonic() > deadline:
            raise TimeoutError('Two-host shared-file rendezvous timed out')
        time.sleep(.5)
    return json.loads(Path(path).read_text())


def _check_local_gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    if visible != '0,1,2,3,4,5,6,7':
        raise ValueError('Native TP4 requires eight node-local GPUs in physical order')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
                           '--format=csv,noheader,nounits'], capture_output=True,
                          text=True, check=True, timeout=20).stdout.splitlines()
    if len(rows) != 8 or any(int(row.split(',')[1]) < 140000 or int(row.split(',')[2]) >= 512
                             for row in rows):
        raise RuntimeError('Eight idle node-local GPUs with at least 140 GB each are required; '
                           'no prior owner is stopped')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'], capture_output=True,
                          text=True, check=True, timeout=20).stdout
    if apps.strip():
        raise RuntimeError('An existing GPU owner is active; never interrupt another job')


def _interface(node_rank, master_addr):
    address = socket.gethostbyname(master_addr)
    if node_rank == 0:
        rows = subprocess.run(['ip', '-o', '-4', 'addr', 'show'],
                              capture_output=True, text=True, check=True, timeout=15).stdout.splitlines()
        matches = [row.split()[1].split('@')[0] for row in rows
                   if 'inet ' + address + '/' in row]
        device = matches[0] if len(matches) == 1 else None
    else:
        fields = subprocess.run(['ip', '-4', 'route', 'get', address],
                                capture_output=True, text=True, check=True,
                                timeout=15).stdout.split()
        device = fields[fields.index('dev') + 1] if 'dev' in fields else None
    if not device or device == 'lo':
        raise RuntimeError('Two-host Gloo/NCCL must use the reachable non-loopback interface')
    return device


def _checkpoint_step(root, identifier, protocol):
    base = Path(root) / 'train/checkpoints'
    values = []
    for directory in base.glob('step-[0-9][0-9][0-9][0-9]'):
        if directory.is_symlink():
            raise ValueError('Original native checkpoint cannot redirect through a symlink')
        step = int(directory.name[5:])
        if (not 0 < step < protocol['total_steps'] or
                not protocol.get('per_step_checkpoints') and
                step not in protocol['checkpoint_steps']):
            continue
        names = ['checkpoint.json', 'bridge.pt', 'optimizer.pt',
                 *(f'rng.rank-{i:02d}.pt' for i in range(4))]
        if protocol.get('per_step_checkpoints'):
            names.append('complete.json')
        if any(not (directory / name).is_file() or (directory / name).is_symlink() or
               (directory / name).stat().st_size == 0 for name in names):
            continue
        try:
            metadata = json.loads((directory / 'checkpoint.json').read_text())
        except (ValueError, OSError):
            continue
        if (metadata.get('step') != step or
                metadata.get('bridge_sha256') != sha256(directory / 'bridge.pt')):
            continue
        if protocol.get('per_step_checkpoints'):
            try:
                done = json.loads((directory / 'complete.json').read_text())
            except (ValueError, OSError):
                continue
            if (done.get('complete') is not True or done.get('step') != step or
                    done.get('world_size') != 4 or
                    done.get('bridge_state_sha256') != metadata.get('bridge_state_sha256') or
                    done.get('optimizer_sha256') != sha256(directory / 'optimizer.pt') or
                    done.get('rank_rng_sha256') != {
                        str(rank): sha256(directory / f'rng.rank-{rank:02d}.pt')
                        for rank in range(4)}):
                continue
        values.append(step)
    if not values:
        raise RuntimeError('No newest signed two-host optimizer/all-rank RNG checkpoint')
    return max(values)


def _control(root, identifier, mode, attempt, resume_step):
    protocol = verify_prepared(root, identifier)
    source = sha256(Path(root) / 'source_manifest.json')
    if mode in ('smoke', 'train'):
        probe = json.loads((Path(root) / 'probe_audit.json').read_text())
        if (not probe.get('passed') or not probe.get('nccl_validated') or
                probe.get('cpu_simulation') or probe.get('source_manifest_sha256') != source or
                len(set(probe.get('hosts', ()))) != 2):
            raise RuntimeError('Actual two-host NCCL/shared-storage probe missing')
    if mode == 'train':
        for path in (Path(root) / 'invocations').glob('*/node-*-process.json'):
            receipt = json.loads(path.read_text())
            pgid = receipt.get('pgid')
            if type(pgid) is not int or pgid <= 1:
                raise ValueError('Prior two-host native process-group identity differs')
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError('An original two-host rank/process group is still alive')
        if (Path(root) / 'train/complete.json').exists():
            raise FileExistsError('Finished native two-host TRAIN cannot be started again')
        if resume_step is not None:
            if resume_step != _checkpoint_step(root, identifier, protocol):
                raise ValueError('Must resume the newest complete two-host optimizer step')
        elif (Path(root) / 'train/metrics.jsonl').exists():
            raise FileExistsError('Existing native TRAIN requires an explicit signed resume step')
        if identifier == 25:
            from .native_multinode_qualification import verify_evidence
            evidence = verify_evidence(root, protocol if 'qualification_evidence' in protocol else None)
            if 'qualification_evidence' not in protocol:
                protocol['qualification_evidence'] = evidence
                atomic_json(Path(root) / 'protocol.json', protocol)
                receipt = json.loads((Path(root) / 'prepare.json').read_text())
                receipt['protocol_sha256'] = sha256(Path(root) / 'protocol.json')
                atomic_json(Path(root) / 'prepare.json', receipt)
    return {'mode': mode, 'root': str(Path(root).absolute()),
            'invocation': str(Path(root).absolute() / 'invocations' / attempt),
            'nonce': uuid4().hex, 'topology': TOPOLOGY,
            'options': {'teacher': '397b', 'dataset': protocol['dataset'], 'nnodes': 2,
                        'gpus_per_node': 8}, 'source_manifest_sha256': source,
            'cpu_probe': False, 'probe_mib': 64, 'timeout': 300,
            'resume_step': resume_step,
            'protocol_sha256': sha256(Path(root) / 'protocol.json')}


def worker(root, control):
    from importlib import import_module
    root = Path(root).absolute()
    receipt = json.loads(Path(control).read_text())
    protocol = verify_prepared(root, receipt['experiment'])
    if (receipt.get('root') != str(root) or
            receipt['protocol_sha256'] != sha256(root / 'protocol.json') or
            receipt['source_manifest_sha256'] != sha256(root / 'source_manifest.json')):
        raise RuntimeError('Torchrun node/rank control and prepared protocol differ')
    folder = root / 'reference_source/recipe/llm_communicate'
    sys.path[:0] = [str(folder), str(root / 'reference_source/scripts')]
    science = import_module('qwen35_multinode_worker')
    if receipt['mode'] != 'probe':
        from .native_selection import install
        install(import_module('train_qwen35_bridge_opd_vllm'))
    sys.argv = [str(folder / 'qwen35_multinode_worker.py'), '--control', str(control)]
    science.main()


def run(root, identifier, node_rank, master_addr, master_port, mode, attempt,
        *, bridge_python=None, generation_python=None, scoring_python=None,
        resume_step=None, wait_seconds=1800, max_seconds=604800):
    """Supervise only this node's newly owned torchrun group; never auto-stop peers."""
    bridge_python = Path(bridge_python or sys.executable)
    generation_python = Path(generation_python or bridge_python)
    scoring_python = Path(scoring_python or bridge_python)
    if mode not in ('probe', 'smoke', 'train') or identifier not in (20, 25):
        raise ValueError('Select a qualified original two-host probe/smoke/train mode')
    if (not 0 < wait_seconds <= 7200 or not 0 < max_seconds <= 604800 or
            not all(Path(p).is_file() for p in (bridge_python, generation_python, scoring_python))):
        raise ValueError('Installed runtime interpreters and bounded deadlines required')
    command = torchrun_command(root, node_rank, master_addr, master_port, attempt,
                               python=bridge_python)
    root = Path(root).absolute()
    _check_local_gpus()
    folder = root / 'invocations' / attempt
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / f'node-{node_rank}.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owned = None
    try:
        control_path = folder / 'control.json'
        kernel_cache = json.loads((root / 'protocol.json').read_text())['fp8_kernel_cache']
        if node_rank == 0:
            if control_path.exists():
                raise FileExistsError('Never reuse an existing two-host attempt')
            kernels = fp8_kernel_cache(kernel_cache)  # node 0 fetches the shared copy first
            value = _control(root, identifier, mode, attempt, resume_step)
            value['experiment'] = identifier
            atomic_json(control_path, value)
        else:
            value = _wait(control_path, folder, wait_seconds)
            kernels = fp8_kernel_cache(kernel_cache)
        if (value.get('experiment') != identifier or value.get('mode') != mode or
                value.get('resume_step') != resume_step or
                value.get('root') != str(root)):
            raise RuntimeError('Two-host original trainer controls disagree')
        source = root / 'reference_source'
        env = {key: item for key, item in os.environ.items()
               if key not in ('WANDB_API_KEY', 'HF_TOKEN')}
        env.update(WANDB_MODE='disabled', WANDB_DISABLED='true',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1',
                   KERNELS_CACHE=kernels, PLATOUR_BRIDGE_PYTHON=str(bridge_python),
                   PLATOUR_GENERATION_PYTHON=str(generation_python),
                   PLATOUR_SCORING_PYTHON=str(scoring_python),
                   PLATOUR_CODE_JUDGE_EXPERIMENT='25' if identifier == 25 else '')
        network = _interface(node_rank, master_addr)
        env['GLOO_SOCKET_IFNAME'] = env['NCCL_SOCKET_IFNAME'] = network
        paths = [str(source / 'recipe/llm_communicate'), str(source / 'scripts'),
                 str(source / 'third_party/verl'), str(Path(__file__).resolve().parents[1])]
        if identifier == 25:
            service_root = str(json.loads((root / 'protocol.json').read_text())['judge_service_root'])
            info = judge_service_info(service_root)
            checker = Path(info['checker_root']).resolve()
            original_lcb = checker.parent.parent
            if (checker.name != 'evaluation' or checker.parent.name != 'lcb_runner' or
                    not (original_lcb / 'lcb_runner/utils/extraction_utils.py').is_file()):
                raise RuntimeError('Original LiveCodeBench extraction source missing on this node')
            paths.append(str(original_lcb))
            env['PLATOUR_CODE_JUDGE_SERVICE'] = service_root
        env['PYTHONPATH'] = os.pathsep.join((*paths, os.environ.get('PYTHONPATH', '')))
        with (folder / f'node-{node_rank}.log').open('x') as log:
            owned = subprocess.Popen(command, cwd=source, env=env,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(folder / f'node-{node_rank}-process.json', {
                'pid': owned.pid, 'pgid': owned.pid, 'mode': mode,
                'root': str(root), 'started': time.time()})
            deadline = time.monotonic() + max_seconds
            while owned.poll() is None:
                if list(folder.glob('failure-node-*.json')):
                    raise RuntimeError('Other node failed; stop only this owned worker group')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Owned native two-host worker exceeded its deadline')
                time.sleep(2)
            if owned.returncode:
                raise RuntimeError(f'Native rank process failed: see node-{node_rank}.log')
        atomic_json(folder / f'node-{node_rank}-finished.json',
                    {'complete': True, 'nonce': value['nonce']})
        if node_rank == 0:
            for index in range(2):
                other = _wait(folder / f'node-{index}-finished.json', folder, wait_seconds)
                if other.get('nonce') != value['nonce']:
                    raise RuntimeError('Peer rank result identity differs')
            if mode == 'probe':
                proof = json.loads((folder / 'probe_result.json').read_text())
                if (not proof.get('passed') or not proof.get('nccl_validated') or
                        len(set(proof.get('hosts', ()))) != 2):
                    raise RuntimeError('Actual two-host NCCL/shared-storage probe failed')
                atomic_json(root / 'probe_audit.json', proof)
            else:
                completed = json.loads((root / ('smoke' if mode == 'smoke' else 'train') /
                                        'complete.json').read_text())
                if (not completed.get('complete') or completed.get('world_size') != 4 or
                        completed.get('nonzero_gradient_updates', 0) < 1):
                    raise RuntimeError('Original native two-host model/optimizer proof incomplete')
                if mode == 'smoke':
                    profiles = [json.loads((root / 'smoke' /
                        f'worst_case_profile.rank-{index:02d}.json').read_text())
                        for index in range(4)]
                    if any(not item.get('passed') or not item.get('vllm_resident') or
                           item.get('response_tokens') != 16384 for item in profiles):
                        raise RuntimeError('Four-rank resident 16K smoke proof incomplete')
                    atomic_json(root / 'smoke_audit.json', {'passed': True,
                        'structural': completed, 'worst_cases': profiles,
                        'source_manifest_sha256': value['source_manifest_sha256']})
            atomic_json(folder / 'stage_complete.json', {'complete': True,
                        'experiment': identifier, 'mode': mode,
                        'nonce': value['nonce']})
        else:
            _wait(folder / 'stage_complete.json', folder, wait_seconds)
        return folder / 'stage_complete.json'
    except BaseException as error:
        atomic_json(folder / f'failure-node-{node_rank}.json',
                    {'error': repr(error), 'time': time.time()})
        if owned is not None and owned.poll() is None:
            os.killpg(owned.pid, signal.SIGTERM)
            try:
                owned.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(owned.pid, signal.SIGKILL)
                owned.wait(timeout=30)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare', 'run', 'worker'))
    parser.add_argument('--experiment', type=int, choices=(20, 25))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sender', type=Path)
    parser.add_argument('--receiver', type=Path)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--service', type=Path)
    parser.add_argument('--mode', choices=('probe', 'smoke', 'train'), default='probe')
    parser.add_argument('--node-rank', type=int, choices=(0, 1))
    parser.add_argument('--master-addr')
    parser.add_argument('--master-port', type=int, default=29500)
    parser.add_argument('--attempt', default='first')
    parser.add_argument('--bridge-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--vllm-python', type=Path, help='Optional override; defaults to --bridge-python')
    parser.add_argument('--scoring-python', type=Path, help='Optional override; defaults to --bridge-python')
    parser.add_argument('--resume-step', type=int)
    parser.add_argument('--resume-latest', action='store_true')
    parser.add_argument('--control', type=Path)
    args = parser.parse_args()
    if args.action == 'worker':
        if args.control is None:
            parser.error('Worker needs its signed --control')
        worker(args.root, args.control)
        return
    if args.experiment is None:
        parser.error('Select original two-host experiment 20 or 25')
    if args.action == 'prepare':
        if args.sender is None or args.receiver is None:
            parser.error('Preparation needs public sender and 4B receiver backbones')
        prepared = prepare(args.experiment, args.sender, args.receiver,
                           args.root, data_root=args.data_root, service=args.service)
        print(json.dumps({'prepared': True, 'root': str(prepared),
                          'gpu_started': False, 'experiment': args.experiment}))
        return
    if args.master_addr is None:
        parser.error('Provide the reachable node-0 --master-addr')
    if args.action == 'plan':
        print(json.dumps({'experiment': args.experiment, 'requires_two_hosts': True,
            'gpu_started': False, 'prepared_root_required': str(args.root.absolute()),
            'commands': [torchrun_command(args.root, node, args.master_addr,
                                          args.master_port, args.attempt,
                                          python=args.bridge_python) for node in (0, 1)]}, indent=2))
        return
    if args.node_rank is None:
        parser.error('Run needs --node-rank')
    if args.resume_latest:
        if args.resume_step is not None or args.mode != 'train':
            parser.error('--resume-latest needs original formal TRAIN and no explicit step')
        protocol = verify_prepared(args.root, args.experiment)
        args.resume_step = _checkpoint_step(args.root, args.experiment, protocol)
    run(args.root, args.experiment, args.node_rank, args.master_addr,
        args.master_port, args.mode, args.attempt, bridge_python=args.bridge_python,
        generation_python=args.vllm_python, scoring_python=args.scoring_python,
        resume_step=args.resume_step)


if __name__ == '__main__':
    main()
