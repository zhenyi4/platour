"""Shared-runtime installation, dependency audit and interpreter routing contracts."""
import ast
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import install_environment as installer


def distribution(name, version, *requirements):
    return SimpleNamespace(metadata={'Name': name}, version=version, requires=requirements)


class UnifiedEnvironmentTests(unittest.TestCase):
    def run_installer(self, *args):
        output = StringIO()
        with patch.object(sys, 'argv', ['install_environment.py', *map(str, args)]), redirect_stdout(output):
            installer.main()
        return output.getvalue()

    def test_audit_accepts_only_the_exact_tested_metadata_conflict(self):
        for parent, child, bound, allowed in (
                ('0.17.1', '5.17.0', '<5,>=4.56.0', True),
                ('0.17.2', '5.17.0', '<5,>=4.56.0', False),
                ('0.17.1', '5.18.0', '<5,>=4.56.0', False),
                ('0.17.1', '5.17.0', '<4', False)):
            packages = [distribution('vllm', parent, 'transformers' + bound),
                        distribution('transformers', child)]
            with self.subTest(parent=parent, child=child, bound=bound):
                if allowed:
                    report = installer.dependency_audit(packages)
                    self.assertEqual(len(report['accepted_metadata_conflicts']), 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Unexpected dependency'):
                        installer.dependency_audit(packages)

    def test_audit_rejects_unrelated_missing_or_incompatible_dependencies(self):
        packages = [distribution('vllm', '0.17.1', 'transformers<5,>=4.56.0'),
                    distribution('transformers', '5.17.0')]
        for extra in (distribution('consumer', '1', 'missing>=1'),
                      distribution('consumer', '1', 'transformers<4')):
            with self.subTest(requirements=extra.requires), self.assertRaisesRegex(RuntimeError, 'consumer'):
                installer.dependency_audit([*packages, extra])
        self.assertEqual(installer.dependency_audit([
            distribution('consumer', '1', 'missing; extra == "optional"')])['dependencies_checked'], 1)

    def test_install_dry_run_keeps_paths_as_arguments_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installer.subprocess, 'run') as run:
            target = Path(folder) / 'new environment'
            result = json.loads(self.run_installer('--venv', target, '--uv', '/tools with spaces/uv', '--dry-run'))
            self.assertFalse(target.exists())
            run.assert_not_called()
            commands = result['commands']
            self.assertEqual(commands[0][-1], str(target))
            self.assertEqual(commands[1][0], '/tools with spaces/uv')
            self.assertIn('--override', commands[1])
            self.assertIn(str(installer.REQUIREMENTS), commands[1])
            self.assertEqual(commands[2][0], str(target / 'bin/python'))

    def test_existing_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installer.subprocess, 'run') as run:
            marker = Path(folder) / 'keep'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                self.run_installer('--venv', folder, '--uv', '/uv')
            self.assertEqual(marker.read_text(), 'original')
            run.assert_not_called()

    def test_receipt_requires_successful_install_and_verification(self):
        for fail_at in (1, 2, None):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'environment'
                calls = []
                def run(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(kwargs['env']['PATH'].split(':')[0], str(target / 'bin'))
                    if len(calls) == 1:
                        target.mkdir()
                    elif len(calls) - 1 == fail_at:
                        raise subprocess.CalledProcessError(1, command)
                with patch.object(installer.subprocess, 'run', side_effect=run):
                    if fail_at is None:
                        self.run_installer('--venv', target, '--uv', '/uv', '--imports-only')
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            self.run_installer('--venv', target, '--uv', '/uv')
                receipt = target / 'platour-runtime.json'
                self.assertEqual(receipt.exists(), fail_at is None)
                if receipt.exists():
                    self.assertFalse(json.loads(receipt.read_text())['gpu_kernels_checked'])
                    self.assertIn('--imports-only', calls[-1])

    def test_check_target_uses_selected_python_without_installing(self):
        with patch.object(installer.subprocess, 'run') as run:
            self.run_installer('--venv', '/selected env', '--check-only', '--imports-only')
            command = run.call_args.args[0]
            self.assertEqual(command[0], '/selected env/bin/python')
            self.assertIn('--check-only', command)
            self.assertIn('--imports-only', command)
            self.assertNotIn('--venv', command)

    def test_supplemental_installer_rejects_old_split_stack_before_mutation(self):
        from scripts import install_qwen35_extras as extras
        for vllm, transformers in ((None, '5.17.0'), ('0.17.1', '5.3.0')):
            def version(name):
                if name == 'vllm' and vllm is None:
                    raise extras.metadata.PackageNotFoundError(name)
                return vllm if name == 'vllm' else transformers if name == 'transformers' else '1'
            with self.subTest(vllm=vllm, transformers=transformers), \
                 patch.object(sys, 'argv', ['install_qwen35_extras.py']), \
                 patch.object(extras, 'find_uv', return_value='/uv'), \
                 patch.object(extras.metadata, 'version', side_effect=version), \
                 patch.object(extras.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'scripts/install_environment.py'):
                    extras.main()
                run.assert_not_called()

    def test_all_release_card_examples_use_shared_setup_and_valid_python(self):
        from qwen35.release_card import usage_sections
        from scripts.prepare_hf_release import experiment_records
        records = experiment_records()
        self.assertEqual(len(records), 30)
        for record in records:
            with self.subTest(experiment=record['id']):
                card = usage_sections(record, standard_format=record['id'] in (23, 24, 29, 30))
                self.assertIn('python3.11 scripts/install_environment.py', card)
                self.assertNotIn('.venv-vllm', card)
                self.assertNotIn('vllm_python=', card)
                self.assertNotIn('--vllm-python "', card)
                for snippet in re.findall(r'```python\n(.*?)```', card, re.S):
                    ast.parse(snippet)

    def test_native_cli_defaults_and_explicit_interpreters(self):
        from qwen35.cli import main
        for identifier, module, extras in (
                (19, 'native_training', ['--sender', '35B']),
                (21, 'native_training', ['--sender', '4B']),
                (22, 'native_random', ['--statistics-root', 'statistics']),
                (23, 'native_full', []), (24, 'native_lora', [])):
            for explicit in (False, True):
                with self.subTest(identifier=identifier, explicit=explicit):
                    argv = ['platour', 'train', '--experiment', str(identifier),
                            '--receiver', '4B', '--output', 'new-run', '--smoke',
                            '--train-backend', 'historical-native', *extras]
                    if explicit:
                        argv += ['--vllm-python', '/generation/python', '--scoring-python', '/scoring/python']
                    with patch.object(sys, 'argv', argv), patch('platour_exps.require_vllm_python'), \
                         patch('qwen35.cli.inputs', return_value={'train': [], 'validation': []}), \
                         patch(f'qwen35.{module}.prepare'), patch(f'qwen35.{module}.execute') as execute:
                        main()
                    options = execute.call_args.kwargs
                    self.assertEqual(options['generation_python'], Path('/generation/python' if explicit else sys.executable))
                    self.assertEqual(options['scoring_python'], Path('/scoring/python' if explicit else sys.executable))

    def test_native_launchers_pass_shared_python_to_all_roles(self):
        class StopBeforeGPU(Exception):
            pass
        for name in ('native_training', 'native_random', 'native_full', 'native_lora'):
            module = import_module('qwen35.' + name)
            with self.subTest(module=name), tempfile.TemporaryDirectory() as folder:
                protocol = dict(gpu_ids=list(range(8)), world_size=8, smoke_only=True)
                with patch.object(module, 'verify_prepared', return_value=protocol), \
                     patch.object(module, 'original_math_checker', return_value='/checker'), \
                     patch('platour_exps.run_native_training', side_effect=StopBeforeGPU) as run:
                    with self.assertRaises(StopBeforeGPU):
                        module.execute(folder)
                environment = run.call_args.kwargs['environment']
                for role in ('BRIDGE', 'GENERATION', 'SCORING'):
                    self.assertEqual(environment[f'PLATOUR_{role}_PYTHON'], sys.executable)

    def test_multinode_cli_defaults_all_roles_to_current_python(self):
        from qwen35.cli import main
        argv = ['platour', 'train', '--experiment', '20', '--sender', '397B', '--receiver', '4B',
                '--output', 'new-run', '--train-backend', 'native-multinode',
                '--native-stage', 'probe', '--node-rank', '0', '--master-addr', '192.0.2.42']
        with patch.object(sys, 'argv', argv), patch('platour_exps.require_vllm_python'), \
             patch('qwen35.native_multinode.run') as run:
            main()
        for name in ('bridge_python', 'generation_python', 'scoring_python'):
            self.assertEqual(run.call_args.kwargs[name], Path(sys.executable))


if __name__ == '__main__':
    unittest.main()
