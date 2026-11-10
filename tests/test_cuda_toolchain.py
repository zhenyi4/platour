"""Private CUDA installation integrity and publication checks; no network/GPU."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('cuda_installer',
    Path(__file__).resolve().parents[1]/'scripts/install_cuda_toolchain.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class ToolchainInstallTests(unittest.TestCase):
    def test_download_and_cached_archive_require_exact_sha(self):
        data = b'archive payload'
        with TemporaryDirectory() as tmp, \
             patch.dict(installer.PACKAGES, {'test': ('1', hashlib.sha256(data).hexdigest())}), \
             patch.object(installer, 'urlopen', return_value=io.BytesIO(data)) as fetch:
            path = installer.archive('test', Path(tmp))
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(installer.archive('test', Path(tmp)), path)
            self.assertEqual(fetch.call_count, 1)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'SHA differs'):
                installer.archive('test', Path(tmp))

    def test_bad_download_is_never_published(self):
        with TemporaryDirectory() as tmp, \
             patch.dict(installer.PACKAGES, {'test': ('1', '0'*64)}), \
             patch.object(installer, 'urlopen', return_value=io.BytesIO(b'wrong')):
            with self.assertRaisesRegex(RuntimeError, 'SHA differs'):
                installer.archive('test', Path(tmp))
            self.assertFalse(list(Path(tmp).glob('*.tar.xz')))

    def test_install_is_staged_and_existing_install_is_verified(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            payload = folder/'archive.tar.xz'
            with tarfile.open(payload, 'w:xz') as archive:
                for name in ('bin/nvcc', 'include/cuda.h', 'lib/libcudart.so', 'LICENSE'):
                    info = tarfile.TarInfo('component/'+name)
                    info.size = 3
                    archive.addfile(info, io.BytesIO(b'abc'))
            target = folder/'private toolkit'
            with patch('sys.argv', ['install', '--output', str(target)]), \
                 patch.object(installer, 'archive', return_value=payload), \
                 patch.object(installer.shutil, 'which', return_value='/usr/bin/g++'), \
                 patch.object(installer, 'compiler_probe', side_effect=RuntimeError('compile failed')):
                with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                    installer.main()
                self.assertFalse(target.exists())
            with patch('sys.argv', ['install', '--output', str(target)]), \
                 patch.object(installer, 'archive', return_value=payload) as fetch, \
                 patch.object(installer.shutil, 'which', return_value='/usr/bin/g++'), \
                 patch.object(installer, 'compiler_probe', return_value='release 12.8'):
                installer.main()
                self.assertTrue((target/'lib64').is_dir())
                self.assertIn('FLASHINFER_NVCC', (target/'env.sh').read_text())
                self.assertTrue(json.loads((target/'complete.json').read_text())['complete'])
                (target/'env.sh').write_text('export CUDA_HOME=/old/toolkit\n')
                installer.main()
                env = (target/'env.sh').read_text()
                self.assertIn('FLASHINFER_WORKSPACE_BASE="$CUDA_HOME/flashinfer-workspace"', env)
                self.assertNotIn('/old/toolkit', env)
                self.assertEqual(fetch.call_count, len(installer.PACKAGES))
                (target/'bin/nvcc').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'receipt differs'):
                    installer.main()
