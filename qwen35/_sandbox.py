"""Disposable unprivileged Docker worker for the pinned LiveCodeBench checker."""

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
               'exec(compile(_source, "<solution-and-tests>", "exec"), '
               '{"__name__": "__main__"})\n'
               f'_write(_fd, {token!r})\n')
    completed = False
    try:
        result = subprocess.run([sys.executable, '-I', '-c', wrapper,
                                 str(writer), request['program']], pass_fds=(writer,),
                                cwd='/tmp', capture_output=True, text=True,
                                timeout=request['timeout'], stdin=subprocess.DEVNULL)
        os.set_blocking(reader, False)
        try:
            completed = os.read(reader, len(token) + 1) == token
        except BlockingIOError:
            completed = False
        code = 1 if result.returncode == 0 and completed else -2
    except subprocess.TimeoutExpired:
        code = -3
    finally:
        os.close(reader)
        os.close(writer)
    return {'results': [code]}


def main():
    request = json.load(sys.stdin)
    if request['kind'] == 'lcb':
        spec = importlib.util.spec_from_file_location('official_checker', '/checker/testing_util.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with open('/samples/' + request['key'] + '.json') as stream:
            sample = json.load(stream)
        with contextlib.redirect_stdout(io.StringIO()):
            results, _ = checker.run_test(sample, test=request['code'], timeout=request['timeout'])
        result = {'results': [int(value) for value in results]}
    elif request['kind'] == 'program':
        result = run_program(request)
    else:
        raise ValueError('Unsupported benchmark kind')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
