"""One experiment-number entry point for Qwen3 and Qwen3.5 references."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


MODEL_DIR = Path(__file__).resolve().parent / 'models'


def require_vllm_python(python: Path | None = None) -> str:
    """Check the same pinned native decode runtime for both model families."""
    executable = str(python or sys.executable)
    try:
        check = subprocess.run([executable, '-B', '-c',
            'import importlib.metadata as m; print(m.version("vllm"))'],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError('Cannot start the vLLM worker Python; check the active environment '
                           'or the optional --vllm-python override') from error
    if check.returncode or check.stdout.strip() != '0.17.1':
        raise RuntimeError('Training and evaluation require vLLM 0.17.1 in the current Python. '
                           'Install with python scripts/install_environment.py and activate it; '
                           '--vllm-python is an optional interpreter override')
    return executable


def run_native_training(command, *, cwd, environment, output, failure_directory,
                        smoke, max_seconds=None, grace_seconds=30, poll_seconds=2,
                        resume_step=None, allow_smoke_resume=False):
    """Bound a newly owned POSIX process group; stop on signed rank failure."""
    if os.name != 'posix':
        raise RuntimeError('Native distributed training supervision requires POSIX')
    raw_limit = (max_seconds if max_seconds is not None else
                 environment.get('PLATOUR_NATIVE_TIMEOUT_SECONDS',
                                 10800 if smoke else 604800))
    limit = float(raw_limit)
    if (not math.isfinite(limit) or limit <= 0 or
            not math.isfinite(grace_seconds) or grace_seconds < 0 or
            not math.isfinite(poll_seconds) or poll_seconds <= 0):
        raise ValueError('Native training time limits must be finite and positive')
    output, failure_directory = Path(output).absolute(), Path(failure_directory).absolute()
    if not failure_directory.is_relative_to(output):
        raise ValueError('Failure evidence must belong to this private output root')
    if resume_step is None:
        if list(failure_directory.glob('failure.rank-*.json')):
            raise FileExistsError('Stale rank failure evidence; use a new run')
        if (output / 'native_process.json').exists() or (output / 'train.log').exists():
            raise FileExistsError('Native launch identity already exists; use a new run')
        receipt_root = output
        old_failures = {}
    else:
        if type(resume_step) is not int or resume_step < 1 or (smoke and not allow_smoke_resume):
            raise ValueError('Only an explicitly audited run resumes from a positive signed step')
        prior = output / 'native_process.json'
        if not prior.is_file() or not (output / 'train.log').is_file():
            raise FileNotFoundError('Cannot resume without this run’s original launcher identity')
        history = json.loads(prior.read_text())
        if history.get('smoke') != smoke:
            raise ValueError('Prior native run smoke/formal identity differs')
        old_pid = history['pid']
        if type(old_pid) is not int or old_pid <= 1:
            raise ValueError('Prior native process group identity is invalid')
        try:
            os.killpg(old_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Prior native process group is still alive; do not overlap training')
        attempts = output / 'native_attempts'
        if attempts.is_symlink():
            raise ValueError('Native attempts directory may not redirect outside this run')
        attempts.mkdir(exist_ok=True)
        number = 2
        while (attempts / f'attempt-{number:04d}').exists():
            number += 1
        receipt_root = attempts / f'attempt-{number:04d}'
        receipt_root.mkdir()
        old_failures = {path.name: (path.stat().st_ino, path.stat().st_mtime_ns)
                        for path in failure_directory.glob('failure.rank-*.json')}

    def record(name, payload):
        destination = receipt_root / name
        temporary = destination.with_suffix('.pending')
        temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
        temporary.replace(destination)

    def group_alive(pid):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def stop_owned_group(process, reason):
        record('native_termination.json', {
            'reason': reason, 'pid': process.pid, 'pgid': process.pid,
            'time': time.time(), 'scope': 'only the new native child session'})
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_alive(process.pid):
                break
            time.sleep(min(.1, max(0., deadline - time.monotonic())))
        if group_alive(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=15)

    def interrupted(signum, frame):
        raise InterruptedError('Native supervisor received a termination signal')

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    started = time.monotonic()
    try:
        with (receipt_root / 'train.log').open('x') as logfile:
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
                start_new_session=True)
            record('native_process.json', {
                'pid': process.pid, 'pgid': process.pid, 'smoke': bool(smoke),
                'resume_step': resume_step, 'max_seconds': limit,
                'grace_seconds': grace_seconds, 'time': time.time(),
                'new_session': True, 'failure_directory': str(failure_directory)})
            while True:
                failures = sorted(path for path in failure_directory.glob('failure.rank-*.json')
                                  if (path.stat().st_ino, path.stat().st_mtime_ns) !=
                                  old_failures.get(path.name))
                if failures:
                    raise RuntimeError('Native rank failure: ' + failures[0].name)
                status = process.poll()
                if status is not None:
                    if status:
                        raise subprocess.CalledProcessError(status, command)
                    if group_alive(process.pid):
                        raise RuntimeError('Native root exited but descendants are still alive')
                    return subprocess.CompletedProcess(command, status)
                if time.monotonic() - started >= limit:
                    raise TimeoutError('Native training exceeded its explicit runtime limit')
                time.sleep(min(poll_seconds, max(0., limit - (time.monotonic() - started))))
    except BaseException as error:
        if process is not None:
            stop_owned_group(process, repr(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def weight_record(experiment: int) -> tuple[str, str, int, str]:
    if type(experiment) is not int or not 1 <= experiment <= 30:
        raise ValueError('Experiment must be between 1 and 30')
    manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
    entries = manifest['files']
    repositories = manifest.get('repositories')
    if repositories is not None and set(repositories) != {f'{i:02d}' for i in range(1, 31)}:
        raise ValueError('Published per-experiment repositories are incomplete')
    repo_id = (repositories[f'{experiment:02d}'] if repositories is not None
               else manifest.get('repo_id'))
    if (not isinstance(repo_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id)
            or set(entries) != {f'{i:02d}' for i in range(1, 31)}):
        raise ValueError('Published model manifest is incomplete')
    filename, size, digest = entries[f'{experiment:02d}']
    standard_filename = {23: 'model.safetensors', 29: 'model.safetensors',
                         24: 'adapter_model.safetensors', 30: 'adapter_model.safetensors'}.get(experiment)
    if (not isinstance(filename, str) or
            not (re.fullmatch(rf'{experiment:02d}_[a-z0-9_]+\.safetensors', filename)
                 or standard_filename is not None and filename == standard_filename) or
            type(size) is not int or size <= 0 or
            not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('Invalid published model identity')
    return repo_id, filename, size, digest


def verify_weight(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size != size:
        raise ValueError(f'Checkpoint size differs from the published model manifest: {path.name}')
    actual = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError(f'Checkpoint SHA-256 differs from the published model manifest: {path.name}')


def release_config(experiment: int, *, repo_id: str | None = None) -> dict | None:
    """Read the actual model configuration before fetching its referenced weights."""
    manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
    if 'repositories' not in manifest:
        return None  # Compatibility with earlier single-repository releases.
    identifier = f'{experiment:02d}'
    published_repo, filename, size, digest = weight_record(experiment)
    repo_id = repo_id or published_repo
    config_filename = 'adapter_config.json' if filename == 'adapter_model.safetensors' else 'config.json'
    config_sha = manifest.get('configs', {}).get(identifier)
    if not isinstance(config_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', config_sha):
        raise ValueError('Missing pinned public model configuration SHA')
    folder = MODEL_DIR / '.cache' / 'platour-configs' / identifier / config_sha
    if repo_id != published_repo:
        folder = folder / hashlib.sha256(repo_id.encode()).hexdigest()
    path = folder / config_filename
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        fetched = Path(hf_hub_download(repo_id=repo_id, filename=config_filename, local_dir=folder))
        if fetched != path:
            raise RuntimeError('Downloaded model configuration location differs')
    if hashlib.sha256(path.read_bytes()).hexdigest() != config_sha:
        raise ValueError('Public model configuration SHA differs')
    config = json.loads(path.read_text())
    if filename == 'model.safetensors':
        if config.get('model_type') != 'qwen3_5_text' or config.get('tie_word_embeddings') is not False:
            raise ValueError('Full release must preserve the independent text-model embeddings and head')
        return config
    if filename == 'adapter_model.safetensors':
        if (config.get('peft_type') != 'LORA' or config.get('r') != 128 or
                config.get('lora_alpha') != 128 or config.get('base_model_name_or_path') != 'Qwen/Qwen3.5-4B'):
            raise ValueError('PEFT release configuration differs from the selected receiver adapter')
        return config
    if (config.get('schema_version') != 1 or config.get('experiment_id') != experiment or
            config.get('weight') != {'filename': filename, 'size': size, 'sha256': digest}):
        raise ValueError('Model configuration does not match this published experiment')
    return config


def ensure_weight(experiment: int, *, repo_id: str | None = None) -> Path:
    published_repo, filename, size, digest = weight_record(experiment)
    repo_id = repo_id or published_repo
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Expected a Hugging Face repository ID: namespace/model')
    config = release_config(experiment, repo_id=repo_id)
    # The config is part of normal loading, not a synthetic download-count request.
    if config is not None and 'weight' in config:
        filename = config['weight']['filename']
    folder = (MODEL_DIR / '.cache' / 'platour-weights' / f'{experiment:02d}'
              if filename in ('model.safetensors', 'adapter_model.safetensors') else MODEL_DIR)
    path = folder / filename
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        try:
            fetched = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                           local_dir=folder))
        except Exception as error:
            raise RuntimeError(f'Model {filename} is not yet accessible at '
                               f'https://huggingface.co/{repo_id}') from error
        if fetched != path:
            raise RuntimeError('Downloaded checkpoint location differs from the model manifest')
    verify_weight(path, size, digest)
    return path


def resolve_weight_input(experiment: int, value: str | Path | None = None) -> Path:
    """Resolve a local checkpoint or a Hub ID for the selected release experiment."""
    if value is None:
        return ensure_weight(experiment)
    raw = os.fspath(value)
    path = Path(raw).expanduser()
    # Existing files and explicit local paths take precedence over Hub IDs.
    if (not path.is_file() and not raw.startswith(('/', './', '../', '~'))
            and not raw.endswith('.safetensors')
            and re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', raw)):
        return ensure_weight(experiment, repo_id=raw)
    return path


def resolve_model_input(value: str | Path) -> Path:
    """Make an evaluation backbone available locally for HF and vLLM loaders."""
    raw = os.fspath(value)
    path = Path(raw).expanduser()
    if path.is_dir():
        return path
    if (raw.startswith(('/', './', '../', '~')) or
            not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', raw)):
        raise FileNotFoundError(f'Base model directory does not exist: {path}')
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo_id=raw, allow_patterns=[
        '*.safetensors', '*.json', '*.model', 'tokenizer*', 'vocab.*',
        'merges.txt', 'chat_template*', '*.tiktoken']))
    if not (snapshot / 'config.json').is_file():
        raise ValueError(f'Hugging Face backbone has no config.json: {raw}')
    return snapshot


def main() -> None:
    from qwen3.parallel import maybe_launch
    if maybe_launch(sys.argv[1:]):
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'it-data':
        from platour_it import main as it_data
        it_data(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'download-data':
        from platour_data import main as download_data
        download_data(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'pact-check':
        from pact_check import main as run_pact_check
        run_pact_check(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        parser = argparse.ArgumentParser(description='Download a checksum-verified model')
        parser.add_argument('--experiment', type=int, choices=range(1, 31), required=True)
        args = parser.parse_args(sys.argv[2:])
        print(ensure_weight(args.experiment))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate', 'init-random', 'download', 'download-data'))
    parser.add_argument('--experiment', type=int, required=True)
    detailed_help = '--experiment' in sys.argv[1:] and any(
        flag in sys.argv[1:] for flag in ('-h', '--help'))
    arguments = ([value for value in sys.argv[1:] if value not in ('-h', '--help')]
                 if detailed_help else None)
    args, _ = parser.parse_known_args(arguments)
    if 1 <= args.experiment <= 18:
        if args.action == 'init-random':
            parser.error('init-random applies to Qwen3.5 experiments 22 and 28')
        from qwen3.cli import main as run
    elif 19 <= args.experiment <= 30:
        from qwen35.cli import main as run
    else:
        parser.error('experiment must be between 1 and 30')
    run()


if __name__ == '__main__':
    main()
