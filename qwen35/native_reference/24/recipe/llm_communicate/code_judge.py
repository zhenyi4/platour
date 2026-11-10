"""Host side of the code judge: code extraction, prompts' judge specs, and one disposable Docker container per
response (no network, read-only root, all capabilities dropped, unprivileged user, memory/CPU/PID limits).

A response is correct iff every test passes. Infrastructure failures raise (never silently counted as wrong);
programs killed for exceeding the container memory limit count as wrong (error_code -6; official codes: -1 compile,
-2 wrong answer, -3 time limit, -4 runtime error; ours: -5 no code block, -6 out of memory).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'lcb-checker:local'
CHECKER = ROOT/'third_party/livecodebench/lcb_runner/evaluation'
WORKER = ROOT/'recipe/llm_communicate/code_sandbox_worker.py'
SAMPLES = ROOT/'data/livecodebench/samples'
LABEL = 'platour.code-judge'
DEFAULTS = dict(lcb_test_timeout=6, program_timeout=10, memory='4g', cpus='1', pids=64)
sys.path.insert(0, str(ROOT/'third_party/livecodebench'))


def extract_code(text):
    """Official LiveCodeBench extraction (lcb_runner.utils.extraction_utils.extract_code, generic style):
    the content between the last two lines containing ``` ; empty string if fewer than two."""
    from lcb_runner.utils.extraction_utils import extract_code as official
    return official(text, None)


def humaneval_program(spec, code):
    # The prompt's own import lines are prepended (e.g. `from typing import List`); nothing else is added.
    return '\n'.join(spec['prompt_imports']+[code, '', spec['test'], f"check({spec['entry_point']})", ''])


def mbpp_program(spec, code):
    # Solution first: some setup code builds objects of classes the solution defines (e.g. task 367).
    return '\n'.join([code, '', spec.get('test_setup_code') or '', *spec['test_list'], ''])


def request_for(spec, code, settings):
    if spec['kind'] == 'lcb':
        return dict(kind='lcb', key=spec['key'], code=code, timeout=settings['lcb_test_timeout']), spec['tests']
    program = humaneval_program(spec, code) if spec['kind'] == 'humaneval' else mbpp_program(spec, code)
    return dict(kind='program', program=program, timeout=settings['program_timeout']), 1


def judge(spec, response, settings=None):
    """Judge one response; returns dict(correct, code, results, metadata, error_code, seconds)."""
    settings = dict(DEFAULTS, **(settings or {}))
    code = extract_code(response)
    started = time.monotonic()
    if not code.strip():
        return dict(correct=False, code=code, no_code=True, results=[], metadata=dict(error='no code block'),
                    error_code=-5, seconds=0.)
    request, tests = request_for(spec, code, settings)
    per_test = settings['lcb_test_timeout'] if spec['kind'] == 'lcb' else settings['program_timeout']
    deadline = (per_test+1)*tests+30
    name = f'platour-judge-{uuid4().hex[:12]}'
    command = ['docker', 'run', '--ulimit', 'core=0', '--name', name, '--label', f'{LABEL}={os.getpid()}', '-i', '--network=none',
        '--read-only', '--user', '65534:65534', '--cap-drop=ALL', '--security-opt=no-new-privileges',
        f"--pids-limit={settings['pids']}", f"--memory={settings['memory']}", f"--memory-swap={settings['memory']}",
        f"--cpus={settings['cpus']}", '--tmpfs=/tmp:rw,noexec,nosuid,size=64m',
        '-v', f'{CHECKER}:/checker:ro', '-v', f'{WORKER}:/worker.py:ro', '-v', f'{SAMPLES}:/samples:ro',
        IMAGE, 'python', '/worker.py']
    try:
        run = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, timeout=deadline)
    except subprocess.TimeoutExpired as error:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=60)
        raise RuntimeError(f'judge container exceeded its global deadline ({deadline}s)') from error
    inspection = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                                text=True, capture_output=True, timeout=60)
    subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=60)
    state = json.loads(inspection.stdout) if inspection.returncode == 0 else {}
    if run.returncode:
        if state.get('OOMKilled'):
            # -6: container OOM. Not -4, which the official checker uses for runtime errors.
            result = dict(results=[-6], metadata=dict(error_code=-6, memory_limit_exceeded=True,
                error_message=f"exceeded the {settings['memory']} container memory limit"))
        elif (run.returncode == 139 and state.get('ExitCode') == 139 and not state.get('Error')
              and 'File "<string>"' in run.stderr):
            result = dict(results=[-4], metadata=dict(error_code=-4, contestant_process_crashed=True,
                error_message='Runtime Error: SIGSEGV while executing generated code',
                stderr_tail=run.stderr[-2000:]))
        else:
            raise RuntimeError(f'judge infrastructure failure exit={run.returncode} state={state}: {run.stderr[-2000:]}')
    else:
        result = json.loads(run.stdout.strip().splitlines()[-1])
    results = result['results']
    correct = bool(results) and all(x > 0 for x in results)
    if correct and len(results) != tests:
        raise RuntimeError(f'Judge reported success for {len(results)} tests; expected {tests}')
    metadata = result.get('metadata') or {}
    error_code = 0 if correct else (metadata.get('error_code') if isinstance(metadata, dict) and 'error_code' in metadata
                                    else min(results) if results else -1)
    return dict(correct=correct, code=code, no_code=False, results=results, metadata=metadata,
                error_code=error_code, seconds=time.monotonic()-started)


def cleanup_exited():
    """Remove exited judge containers of any owner (left behind when a scorer was killed mid-request)."""
    ids = subprocess.run(['docker', 'ps', '-aq', '--filter', f'label={LABEL}', '--filter', 'status=exited'],
                         text=True, capture_output=True, timeout=60).stdout.split()
    if ids:
        subprocess.run(['docker', 'rm', '-f', *ids], capture_output=True, timeout=120)
    return len(ids)


def cleanup(owner=None):
    """Remove leftover judge containers (of one owner pid, or all)."""
    selector = f'label={LABEL}' if owner is None else f'label={LABEL}={owner}'
    ids = subprocess.run(['docker', 'ps', '-aq', '--filter', selector], text=True, capture_output=True,
                         timeout=60).stdout.split()
    if ids:
        subprocess.run(['docker', 'rm', '-f', *ids], capture_output=True, timeout=120)
    return len(ids)
