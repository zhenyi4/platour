"""Install or verify the shared PLatour HF/vLLM environment (Linux x86_64)."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / 'requirements/unified-runtime.txt'
OVERRIDES = ROOT / 'requirements/unified-overrides.txt'


def dependency_audit(distributions=None):
    """Check installed mandatory requirements; allow exactly the tested override."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
    packages = {canonicalize_name(d.metadata['Name']): d
                for d in (metadata.distributions() if distributions is None else distributions)}
    errors, accepted = [], []
    for name, dist in sorted(packages.items()):
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            target = packages.get(canonicalize_name(req.name))
            if target is not None and (not req.specifier or req.specifier.contains(target.version)):
                continue
            issue = dict(package=name, version=dist.version, requirement=raw,
                         installed=None if target is None else target.version)
            known = (name == 'vllm' and Version(dist.version) == Version('0.17.1')
                     and canonicalize_name(req.name) == 'transformers'
                     and str(req.specifier) == '<5,>=4.56.0' and req.url is None
                     and target is not None and Version(target.version) == Version('5.17.0'))
            (accepted if known else errors).append(issue)
    if errors:
        raise RuntimeError('Unexpected dependency conflicts: ' + json.dumps(errors, sort_keys=True))
    return dict(dependencies_checked=len(packages), accepted_metadata_conflicts=accepted)


def pinned_requirements(path=REQUIREMENTS):
    from packaging.requirements import Requirement
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-r '):
            yield from pinned_requirements(Path(path).parent / line[3:].strip())
        else:
            yield Requirement(line)


def verify(*, gpu):
    import torch
    if (sys.version_info[:2] != (3, 11) or platform.system() != 'Linux'
            or platform.machine() != 'x86_64' or not torch._C._GLIBCXX_USE_CXX11_ABI
            or not (torch.version.cuda or '').startswith('12.')):
        raise RuntimeError('Shared runtime requires Linux x86_64, Python 3.11, CUDA 12 Torch and CXX11 ABI')
    versions = {}
    for req in pinned_requirements():
        installed = metadata.version(req.name)
        if req.specifier and not req.specifier.contains(installed):
            raise RuntimeError(f'Shared runtime pin differs: {req}; installed {installed}')
        versions[req.name] = installed
    if versions['causal-conv1d'].split('+')[0] != '1.6.2.post1':
        raise RuntimeError('Shared runtime requires the pinned causal-conv1d wheel')
    report = dependency_audit()
    command = [sys.executable, '-B', str(ROOT / 'scripts/install_qwen35_extras.py'), '--check-only']
    if not gpu:
        command.append('--imports-only')
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, '-B', '-c', 'import vllm'], check=True)
    if gpu:
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check_vllm_runtime.py')], check=True)
    return dict(complete=True, python=sys.executable, versions=versions,
                gpu_kernels_checked=gpu, **report)


def find_uv():
    for candidate in (shutil.which('uv'), Path(sys.executable).parent / 'uv',
                      Path.home() / '.local/bin/uv', Path.home() / '.cargo/bin/uv'):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).absolute())
    raise RuntimeError('Install uv first (python -m pip install uv), or pass --uv /path/to/uv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, help='New environment directory (default: checkout/.venv)')
    parser.add_argument('--python', default='3.11', help='Python 3.11 interpreter for uv venv')
    parser.add_argument('--uv', type=Path, help='Explicit uv executable')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without creating an environment')
    parser.add_argument('--check-only', action='store_true', help='Check --venv, or the current Python if omitted')
    parser.add_argument('--imports-only', action='store_true', help='Skip GPU kernels; does not certify GPU execution')
    parser.add_argument('--audit-only', action='store_true', help='Audit dependencies in the current Python')
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(dependency_audit(), indent=2))
        return
    if args.check_only:
        if args.venv is not None:
            command = [str(args.venv.expanduser().absolute() / 'bin/python'), '-B', str(Path(__file__).resolve()), '--check-only']
            if args.imports_only:
                command.append('--imports-only')
            if args.dry_run:
                print(json.dumps({'commands': [command]}, indent=2))
            else:
                subprocess.run(command, check=True)
        elif args.dry_run:
            print(json.dumps({'check_python': sys.executable, 'gpu_kernels': not args.imports_only}))
        else:
            print(json.dumps(verify(gpu=not args.imports_only), indent=2))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('The pinned CUDA wheel supports Linux x86_64')
    target = (args.venv or ROOT / '.venv').expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Preserve the existing environment: {target}; use --check-only or a new --venv')
    uv = str(args.uv.expanduser().absolute()) if args.uv else find_uv()
    python = target / 'bin/python'
    commands = [[uv, 'venv', '--python', args.python, str(target)],
                [uv, 'pip', 'install', '--python', str(python), '--override', str(OVERRIDES),
                 '-e', str(ROOT), '-r', str(REQUIREMENTS)],
                [str(python), '-B', str(Path(__file__).resolve()), '--check-only']]
    if args.imports_only:
        commands[-1].append('--imports-only')
    print(json.dumps(dict(environment=str(target), commands=commands), indent=2), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    for command in commands:
        subprocess.run(command, env=env, check=True)
    # Preserve partial installations after failure; sign only a verified install.
    (target / 'platour-runtime.json').write_text(json.dumps(dict(complete=True,
        created_at=datetime.now(timezone.utc).isoformat(), checkout=str(ROOT),
        python=str(python), gpu_kernels_checked=not args.imports_only), indent=2) + '\n')
    print(f'Shared environment ready: source {shlex.quote(str(target / "bin/activate"))}')


if __name__ == '__main__':
    main()
