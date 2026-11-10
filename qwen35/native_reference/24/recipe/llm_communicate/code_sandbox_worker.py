"""Runs inside the disposable judge container (no network, read-only root, unprivileged user).

stdin: {"kind": "lcb", "key": ..., "code": ..., "timeout": 6}
         -> pinned official LiveCodeBench run_test on /samples/<key>.json (public + private tests)
       {"kind": "program", "program": ..., "timeout": 10}
         -> run a complete Python program (HumanEval / MBPP: solution + asserts) in a child interpreter
stdout: one JSON object with "results" (list of ints, >0 = pass) and "metadata".
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from uuid import uuid4


def run_program(request):
    reader, writer = os.pipe()
    token = uuid4().hex.encode()
    wrapper = ('import os, sys\n'
               '_write = os.write\n'
               '_fd = int(sys.argv[1])\n'
               '_source = sys.argv[2]\n'
               'exec(compile(_source, "<solution-and-tests>", "exec"), {"__name__": "__main__"})\n'
               f'_write(_fd, {token!r})\n')
    completed = False
    try:
        run = subprocess.run([sys.executable, '-I', '-c', wrapper, str(writer), request['program']],
            pass_fds=(writer,), cwd='/tmp', capture_output=True, text=True,
            timeout=request['timeout'], stdin=subprocess.DEVNULL)
        os.set_blocking(reader, False)
        try:
            completed = os.read(reader, len(token)+1) == token
        except BlockingIOError:
            completed = False
        passed = run.returncode == 0 and completed
        error = None if passed else (run.stderr[-1500:] or f'exit {run.returncode}; tests_completed={completed}')
        code = 1 if passed else (-4 if run.returncode == 0 else -2)
    except subprocess.TimeoutExpired:
        passed, error, code = False, 'timeout', -3
    finally:
        os.close(reader)
        os.close(writer)
    return dict(results=[code], metadata=dict(error=error, tests_completed=completed,
                                             judging_version='program_completion_pipe_v2'))


def main():
    request = json.load(sys.stdin)
    result = run_request(request)
    print(json.dumps(result, default=str))


def run_request(request):
    if request['kind'] == 'lcb':
        spec = importlib.util.spec_from_file_location('official_checker', '/checker/testing_util.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with open(f"/samples/{request['key']}.json") as handle:
            sample = json.load(handle)
        with contextlib.redirect_stdout(io.StringIO()):
            results, metadata = checker.run_test(sample, test=request['code'], timeout=request['timeout'])
        return dict(results=[int(x) for x in results], metadata=metadata)
    if request['kind'] == 'program':
        return run_program(request)
    raise ValueError(request['kind'])


if __name__ == '__main__':
    main()
