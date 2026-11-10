"""Check Qwen3.5 kernels or complete an existing runtime; new installs use install_environment.py."""
import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / 'requirements/qwen35-extras.txt'
CORE = ('torch', 'transformers', 'huggingface-hub', 'tokenizers', 'fla-core',
        'tilelang', 'kernels', 'accelerate', 'pyarrow', 'safetensors', 'math-verify', 'triton')
WHEEL = 'causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
# Digest published with the upstream v1.6.2.post1 release assets.
WHEEL_SHA256 = '6ef3667660ef4404f9f833bb6988ab46e5024aec29af0f0575c195e307acf4ed'
WHEEL_URL = ('https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/'
             + WHEEL.replace('+', '%2B') + '#sha256=' + WHEEL_SHA256)


def versions():
    return {item.metadata['Name']: item.version for item in metadata.distributions()}


def find_uv():
    candidates = (shutil.which('uv'), Path(sys.executable).parent / 'uv',
                  Path.home() / '.local/bin/uv', Path.home() / '.cargo/bin/uv')
    for candidate in candidates:
        if candidate is not None:
            path = Path(candidate)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.absolute())
    raise RuntimeError('uv not found on PATH, beside Python, in ~/.local/bin or ~/.cargo/bin')


def active_users():
    """Refuse in-place package changes while this venv has other Python jobs."""
    python = os.path.abspath(sys.executable)
    active = []
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        if int(entry.parent.name) == os.getpid():
            continue
        try:
            args = entry.read_bytes().split(b'\0')
            executable = os.fsdecode(args[0])
            if executable and os.path.abspath(executable) == python:
                active.append(int(entry.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(active)


def verify(gpu):
    import torch
    modules = ('transformers', 'huggingface_hub', 'tokenizers', 'fla', 'peft', 'datasets',
               'pandas', 'pyarrow', 'wandb', 'tilelang', 'kernels', 'accelerate',
               'ninja', 'packaging', 'causal_conv1d', 'causal_conv1d_cuda')
    for module in modules:
        importlib.import_module(module)
    import training_protocol
    if Path(training_protocol.__file__).resolve() != ROOT / 'training_protocol.py':
        raise RuntimeError('Editable install does not point to this checkout')
    report = {'imports_complete': True, 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'gpu_kernel_checked': False}
    if gpu:
        from causal_conv1d import causal_conv1d_fn
        from torch.nn import functional as F
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: imports passed but GPU kernel check cannot run')
        # Tiny forward/backward check of the installed extension against PyTorch.
        torch.manual_seed(42)
        x = torch.randn(2, 64, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y = causal_conv1d_fn(x, w, activation='silu')
        expected = F.silu(F.conv1d(x, w.unsqueeze(1), padding=3, groups=64)[..., :32])
        torch.testing.assert_close(y, expected, atol=.05, rtol=.05)
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        if not all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, w)):
            raise RuntimeError('causal-conv1d backward returned invalid gradients')
        report['gpu_kernel_checked'] = True
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Inspect versions and plan, without installation')
    parser.add_argument('--check-only', action='store_true', help='Verify imports and CUDA kernel only')
    parser.add_argument('--imports-only', action='store_true', help='Skip the CUDA kernel check')
    args = parser.parse_args()
    if args.check_only:
        verify(not args.imports_only)
        return
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Run with the project training virtualenv Python')
    uv = find_uv()
    import torch
    if (sys.version_info[:2] != (3, 11) or platform.system() != 'Linux' or
            platform.machine() != 'x86_64' or torch.__version__.split('+')[0] != '2.10.0' or
            not (torch.version.cuda or '').startswith('12.') or not torch._C._GLIBCXX_USE_CXX11_ABI):
        raise RuntimeError('Pinned wheel requires Linux x86_64, Python 3.11, Torch 2.10 CUDA 12, CXX11 ABI')
    pins = {name: metadata.version(name) for name in CORE}
    try:
        pins['vllm'] = metadata.version('vllm')
    except metadata.PackageNotFoundError as error:
        raise RuntimeError('The default runtime now includes vLLM. Create a fresh shared '
                           'environment with scripts/install_environment.py') from error
    if pins['vllm'] != '0.17.1' or pins['transformers'] != '5.17.0':
        raise RuntimeError('Use scripts/install_environment.py for the pinned shared runtime; '
                           'this supplemental installer preserves existing core versions')
    pins.update({name: value for name, value in versions().items() if name.lower().startswith('nvidia-')})
    requested = [line for line in REQUIREMENTS.read_text().splitlines() if line and not line.startswith('#')]
    processes = active_users()
    print(json.dumps({'python': sys.executable, 'uv': uv, 'preserved_versions': pins,
        'supplemental_requirements': requested, 'causal_conv1d_wheel': WHEEL,
        'other_venv_processes': processes, 'vllm_shares_python': 'vllm' in pins}, indent=2), flush=True)
    if args.dry_run:
        return
    if processes:
        raise RuntimeError(f'This training venv is in use by PIDs {processes}; finish or stop those jobs first')
    output = ROOT / 'outputs' / ('qwen35-env-install-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'-{os.getpid()}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'before.json').write_text(json.dumps(versions(), indent=2))
    constraints = output / 'preserve-core.txt'
    constraints.write_text(''.join(f'{name}=={value}\n' for name, value in sorted(pins.items())))
    command = [uv, 'pip', 'install', '--python', sys.executable]
    if pins.get('vllm') == '0.17.1' and pins['transformers'] == '5.17.0':
        command += ['--override', str(ROOT / 'requirements/unified-overrides.txt')]
    subprocess.run([*command, '--constraint', str(constraints), '-r', str(REQUIREMENTS),
                    'causal-conv1d @ ' + WHEEL_URL], check=True)
    # Refresh editable module registration after new top-level modules were added.
    subprocess.run([*command, '--no-deps', '--no-build-isolation', '-e', str(ROOT)], check=True)
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/install_environment.py'),
                    '--audit-only'], check=True)
    (output / 'after.json').write_text(json.dumps(versions(), indent=2))
    for name, expected in pins.items():
        if metadata.version(name) != expected:
            raise RuntimeError(f'Core package unexpectedly changed: {name}')
    # Import in a fresh process after installation; don't reuse old loaded libraries.
    check = [sys.executable, '-B', str(Path(__file__).resolve()), '--check-only']
    if args.imports_only:
        check.append('--imports-only')
    subprocess.run(check, check=True)
    (output / 'complete.json').write_text(json.dumps({'complete': True,
        'gpu_kernel_checked': not args.imports_only, 'core_versions_preserved': pins}))
    print(f'Environment completed; receipt: {output / "complete.json"}')


if __name__ == '__main__':
    main()
