"""The original Answer/math-equivalence and sandboxed programming reward contracts."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

ANSWER = re.compile(r'Answer:\s*([^\n\r]+)', re.IGNORECASE)
ANSWER_LINE = re.compile(r'^Answer:\s*\S', re.IGNORECASE)
SAFE_KEY = re.compile(r'[a-z0-9_]+\Z')


def normalize_math_answer(answer):
    text = re.sub(r'^Answer\s*:\s*', '', str(answer).strip(), count=1,
                  flags=re.IGNORECASE).strip()
    if text.endswith('.') and not text.endswith(r'\.'):
        text = text[:-1].rstrip()
    for left, right in (('$$', '$$'), (r'\[', r'\]'), (r'\(', r'\)'), ('$', '$')):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            inside = text[len(left):-len(right)]
            if left not in inside and right not in inside:
                text = inside.strip()
                break
    for command in (r'\boxed{', r'\fbox{'):
        if text.startswith(command):
            depth = 1
            for end in range(len(command), len(text)):
                depth += (text[end] == '{') - (text[end] == '}')
                if depth == 0:
                    if end == len(text) - 1:
                        text = text[len(command):end].strip()
                    break
    return text.replace('<=', r'\le ').replace('>=', r'\ge ').replace('!=', r'\ne ')


def math_score(gold, response, *, clipped=False, strict_format=False):
    if clipped:
        return 0
    last = next((x.strip() for x in reversed(response.split('\n')) if x.strip()), '')
    matches = ANSWER.findall(response)
    if strict_format and not ANSWER_LINE.match(last):
        return 0
    if not matches:
        return 0
    actual = normalize_math_answer(last if strict_format else matches[-1])
    reference = normalize_math_answer(gold)
    if not actual or not reference:
        return 0
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig
    from math_verify.errors import TimeoutException
    config = [LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0)]
    for timeout in (5, 10, 20):
        try:
            expected = parse(r'\boxed{' + reference + '}', extraction_config=config,
                             fallback_mode='no_fallback', extraction_mode='first_match',
                             parsing_timeout=timeout, raise_on_error=True)
            output = parse(r'\boxed{' + actual + '}', extraction_config=config,
                           fallback_mode='no_fallback', extraction_mode='first_match',
                           parsing_timeout=timeout, raise_on_error=True)
            if expected and output:
                return int(verify(expected, output, strict=True, timeout_seconds=timeout,
                                  raise_on_error=True))
            return int(re.sub(r'\s+', '', actual) == re.sub(r'\s+', '', reference))
        except (TimeoutError, TimeoutException):
            if timeout == 20:
                raise RuntimeError('Unresolved original math-verify timeout') from None
    raise AssertionError('Unreachable')


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


@lru_cache(maxsize=8192)
def _case_sha(path, size, mtime_ns):
    return _digest(path)


def judge_service_info(root):
    """Reject stale/shared-service identities before delegating a judge call."""
    root = Path(root).resolve()
    info = json.loads((root / 'ready.json').read_text())
    if (not info.get('ready') or info.get('backend') != 'isolated_docker_shared_files' or
            not isinstance(info.get('spec_id'), str) or len(info['spec_id']) != 64 or
            not isinstance(info.get('instance'), str) or len(info['instance']) != 32 or
            not isinstance(info.get('heartbeat'), (int, float)) or
            not 0 <= time.time() - info['heartbeat'] < 15 or
            not (root / 'requests').is_dir() or not (root / 'replies').is_dir()):
        raise RuntimeError('Original isolated judge service is unavailable or stale')
    return info


def _shared_code_score(root, spec, response):
    root = Path(root).resolve()
    service = judge_service_info(root)
    experiment = os.environ.get('PLATOUR_CODE_JUDGE_EXPERIMENT')
    if not experiment or experiment not in service.get('supported_experiments', []):
        raise RuntimeError('Original code judge does not support this experiment')
    ident = uuid4().hex
    request = {'id': ident, 'spec_id': service['spec_id'],
               'instance': service['instance'], 'experiment': experiment,
               'spec': spec, 'response': response}
    incoming = root / 'requests' / (ident + '.json')
    pending = root / 'requests' / (ident + '.pending')
    with pending.open('x') as stream:
        json.dump(request, stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(incoming)
    reply = root / 'replies' / (ident + '.json')
    deadline = time.monotonic() + 3600
    while not reply.is_file():
        current = judge_service_info(root)
        if (current['instance'] != service['instance'] or
                current['spec_id'] != service['spec_id']):
            raise RuntimeError('Original code judge changed during scoring; resume checkpoint')
        if time.monotonic() > deadline:
            raise TimeoutError('Isolated code judge did not finish before its deadline')
        time.sleep(.2)
    result = json.loads(reply.read_text())
    if (result.get('id') != ident or result.get('spec_id') != service['spec_id'] or
            result.get('instance') != service['instance'] or not result.get('complete') or
            type(result.get('correct')) is not int or result['correct'] not in (0, 1)):
        raise RuntimeError('Isolated judge returned an incomplete or mismatched response')
    return result['correct']


def code_score(spec, response, *, case_dir, checker_dir, docker_image):
    """All tests must pass; never execute generated code on the host."""
    from lcb_runner.utils.extraction_utils import extract_code
    code = extract_code(response, None)
    if not code.strip():
        return 0
    kind = spec['kind']
    service = os.environ.get('PLATOUR_CODE_JUDGE_SERVICE')
    if service:
        if kind not in ('lcb', 'deepcoder', 'humaneval', 'mbpp'):
            raise ValueError('Unsupported original code judge task')
        return _shared_code_score(service, spec, response)
    if kind in ('lcb', 'deepcoder'):
        key = spec['key']
        if not SAFE_KEY.fullmatch(key):
            raise ValueError('Unsafe code-judge sample key')
        sample = (Path(case_dir) / (key + '.json')).resolve()
        if not sample.is_file() or not sample.is_relative_to(Path(case_dir).resolve()):
            raise FileNotFoundError('Official programming test cases must be obtained separately')
        if kind == 'deepcoder':
            stat = sample.stat()
            if _case_sha(str(sample), stat.st_size, stat.st_mtime_ns) != spec['sample_sha256']:
                raise RuntimeError('Original DeepCoder train/validation unit tests changed')
        request = {'kind': 'lcb', 'key': key, 'code': code, 'timeout': 6}
        expected = spec['tests']
    elif kind == 'humaneval':
        program = '\n'.join(spec['prompt_imports'] + [code, '', spec['test'],
                          f"check({spec['entry_point']})", ''])
        request, expected = {'kind': 'program', 'program': program, 'timeout': 10}, 1
    elif kind == 'mbpp':
        program = '\n'.join([code, '', spec.get('test_setup_code') or '', *spec['test_list'], ''])
        request, expected = {'kind': 'program', 'program': program, 'timeout': 10}, 1
    else:
        raise ValueError('Unsupported code judge task')
    if not Path(checker_dir, 'testing_util.py').is_file():
        raise FileNotFoundError('Pinned official LiveCodeBench checker directory required')
    container_name = 'qwen35-judge-' + uuid4().hex[:12]
    worker = Path(__file__).with_name('_sandbox.py')
    command = ['docker', 'run', '--rm', '--name', container_name, '-i', '--network=none',
               '--read-only', '--user', '65534:65534', '--cap-drop=ALL',
               '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=4g',
               '--memory-swap=4g', '--cpus=1', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m',
               '-v', f'{Path(checker_dir).resolve()}:/checker:ro',
               '-v', f'{worker.resolve()}:/worker.py:ro',
               '-v', f'{Path(case_dir).resolve()}:/samples:ro',
               docker_image, 'python', '/worker.py']
    timeout = (7 * expected + 30) if request['kind'] == 'lcb' else 41
    try:
        result = subprocess.run(command, input=json.dumps(request), capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True, timeout=60)
        raise RuntimeError('Judge container exceeded the approved global deadline') from None
    if result.returncode != 0:
        raise RuntimeError('Judge container failed; no answer may be silently scored wrong')
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        codes = payload['results']
    except (IndexError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError('Judge returned an invalid response') from error
    if not codes or (all(v > 0 for v in codes) and len(codes) != expected):
        raise RuntimeError('Official unit-test coverage incomplete')
    return int(len(codes) == expected and all(v > 0 for v in codes))
