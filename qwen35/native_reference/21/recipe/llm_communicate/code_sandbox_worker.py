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
import subprocess
import sys

request = json.load(sys.stdin)
if request['kind'] == 'lcb':
    spec = importlib.util.spec_from_file_location('official_checker', '/checker/testing_util.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    with open(f"/samples/{request['key']}.json") as handle:
        sample = json.load(handle)
    with contextlib.redirect_stdout(io.StringIO()):
        results, metadata = checker.run_test(sample, test=request['code'], timeout=request['timeout'])
    print(json.dumps(dict(results=[int(x) for x in results], metadata=metadata), default=str))
elif request['kind'] == 'program':
    try:
        run = subprocess.run([sys.executable, '-I', '-c', request['program']], cwd='/tmp', capture_output=True,
                             text=True, timeout=request['timeout'], stdin=subprocess.DEVNULL)
        passed = run.returncode == 0
        error = None if passed else (run.stderr[-1500:] or f'exit {run.returncode}')
        code = 1 if passed else -2
    except subprocess.TimeoutExpired:
        passed, error, code = False, 'timeout', -3
    print(json.dumps(dict(results=[code], metadata=dict(error=error))))
else:
    raise ValueError(request['kind'])
