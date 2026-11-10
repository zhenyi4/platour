"""Install private CUDA 12.8 compiler/headers for Hopper FlashInfer JIT kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
# NVIDIA redistrib_12.8.1.json: compiler, runtime headers/library, and CCCL.
# https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json
PACKAGES = {
    'cuda_nvcc': ('12.8.93', '9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3'),
    'cuda_cudart': ('12.8.90', '8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68'),
    'cuda_cccl': ('12.8.90', '0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b'),
}
PROBE = '''#include <cuda/ptx>
#include <cstdint>
__global__ void check_tensormap(void* map) {
  namespace ptx = cuda::ptx;
  ptx::tensormap_replace_global_dim(ptx::space_global, map, ptx::n32_t<1>{}, uint32_t(16));
  ptx::fence_proxy_tensormap_generic(ptx::sem_release, ptx::scope_cta);
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_cta, map, ptx::n32_t<128>{});
}
'''


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive(name, cache):
    version, digest = PACKAGES[name]
    filename = f'{name}-linux-x86_64-{version}-archive.tar.xz'
    path = cache / filename
    if path.exists():
        if sha256(path) != digest:
            raise RuntimeError(f'Cached archive SHA differs: {path}')
        return path
    url = BASE + f'{name}/linux-x86_64/' + filename
    pending = path.with_suffix('.pending')
    print(f'Downloading {name} {version}', flush=True)
    with urlopen(url, timeout=60) as source, pending.open('wb') as dest:
        shutil.copyfileobj(source, dest, length=1024 * 1024)
    if sha256(pending) != digest:
        raise RuntimeError(f'NVIDIA archive SHA differs: {pending}')
    pending.replace(path)
    return path


def compiler_probe(target):
    nvcc = target / 'bin/nvcc'
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    if 'release 12.8' not in version:
        raise RuntimeError('Expected private CUDA 12.8 compiler')
    with TemporaryDirectory(prefix='platour-cuda-probe-') as folder:
        source, obj = Path(folder)/'probe.cu', Path(folder)/'probe.o'
        source.write_text(PROBE)
        subprocess.run([str(nvcc), '-std=c++20', '-arch=sm_90a',
            '-I'+str(target/'include'), '-I'+str(target/'include/cccl'),
            '-c', str(source), '-o', str(obj)], check=True)
    return version


def environment(target):
    # FlashInfer 0.6.4 does not rewrite an existing build.ninja when CUDA_HOME
    # changes. Keep its generated sources/build rules private to this toolchain.
    return (
        'export CUDA_HOME=' + shlex.quote(str(target)) + '\n'
        'export CUDA_PATH="$CUDA_HOME"\n'
        'export CUDACXX="$CUDA_HOME/bin/nvcc"\n'
        'export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"\n'
        'export FLASHINFER_WORKSPACE_BASE="$CUDA_HOME/flashinfer-workspace"\n'
        'export PATH="$CUDA_HOME/bin:$PATH"\n'
        'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/cuda-12.8')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('These NVIDIA archives are for Linux x86_64')
    target = args.output.absolute()
    if args.dry_run:
        print(json.dumps({'target': str(target), 'packages': PACKAGES,
                          'system_cuda_modified': False}, indent=2))
        return
    if not shutil.which('g++'):
        raise RuntimeError('A host C++ compiler (g++) is required for CUDA JIT')
    marker = target/'complete.json'
    if target.exists():
        if not marker.is_file():
            raise FileExistsError(f'Use a fresh toolkit directory: {target}')
        record = json.loads(marker.read_text())
        if (record.get('packages') != {k: list(v) for k, v in PACKAGES.items()} or
                record.get('nvcc_sha256') != sha256(target/'bin/nvcc')):
            raise RuntimeError('Private CUDA toolkit receipt differs')
        compiler_probe(target)
        (target/'env.sh').write_text(environment(target))
        print(f'Private CUDA toolchain verified: {target / "env.sh"}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    cache = target.parent/'cuda-downloads'
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='.cuda-12.8-', dir=target.parent) as staging:
        stage = Path(staging)
        merged = stage/'toolkit'
        merged.mkdir()
        for name in PACKAGES:
            downloaded = archive(name, cache)
            unpacked = stage/name
            unpacked.mkdir()
            with tarfile.open(downloaded) as handle:
                handle.extractall(unpacked, filter='data')
            roots = list(unpacked.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError(f'Unexpected NVIDIA archive layout: {downloaded}')
            for entry in roots[0].iterdir():
                if entry.is_dir():
                    shutil.copytree(entry, merged/entry.name, dirs_exist_ok=True, symlinks=True)
                else:
                    licenses = merged/'component-metadata'/name
                    licenses.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry, licenses/entry.name)
        if (merged/'lib').is_dir() and not (merged/'lib64').exists():
            (merged/'lib64').symlink_to('lib', target_is_directory=True)
        version = compiler_probe(merged)
        (merged/'env.sh').write_text(environment(target))
        (merged/'complete.json').write_text(json.dumps({'complete': True,
            'packages': PACKAGES, 'nvcc_sha256': sha256(merged/'bin/nvcc'),
            'compiler_version': version, 'hopper_ptx_api_compile_passed': True,
            'gpu_execution_checked': False}, indent=2))
        merged.rename(target)
    print(f'Private CUDA toolchain installed: source {target / "env.sh"}')


if __name__ == '__main__':
    main()
