"""Original DeepCoder pass-all-tests reward via the pinned CPU Docker judge service.

This module is imported only by the new two-host ID25 private science copy.
No generated code is ever evaluated in a host Python process.
"""

from functools import lru_cache
import json
import os
from pathlib import Path
import time

from .checkpoints import atomic_json, sha256
from .judge_service import ORIGINAL_IMAGE
from .scoring import code_score, judge_service_info


SERVICE = Path(os.environ.get('PLATOUR_CODE_JUDGE_SERVICE', '/nonexistent-judge-service'))


@lru_cache(maxsize=4)
def _extractor(checker_root):
    from lcb_runner.utils import extraction_utils as official
    from .judge_cases import PINNED
    expected = Path(checker_root).resolve().parent / 'utils/extraction_utils.py'
    if (expected.is_symlink() or Path(official.__file__).resolve() != expected.resolve() or
            sha256(expected) != json.loads(PINNED.read_text())['extractor_sha256']):
        raise RuntimeError('Original LiveCodeBench host extractor source differs')
    return official.extract_code


def _has_code(response, checker_root):
    return bool(_extractor(str(checker_root))(response, None).strip())


def service_info():
    info = judge_service_info(SERVICE)
    if ('25' not in info.get('supported_experiments', ()) or
            info.get('image_id') != ORIGINAL_IMAGE):
        raise RuntimeError('This original isolated code judge/image does not support experiment 25')
    return info


class DeepCoderScorer:
    def __init__(self, directory, timeout=7200):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError('Native two-host scorer directory may not redirect')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.protocol = next((p / 'protocol.json' for p in self.directory.parents
                              if (p / 'protocol.json').is_file()), None)
        if self.protocol is None:
            raise FileNotFoundError('Signed two-host DeepCoder training protocol is missing')
        self.root = self.protocol.parent.resolve()
        self.info = service_info()
        if (json.loads(self.protocol.read_text()).get('judge_service_spec_id') !=
                self.info['spec_id']):
            raise RuntimeError('Two-host native training/judge service identity differs')
        self.timeout = timeout
        atomic_json(self.directory / 'ready.json', {'cpu_only': True,
                    'service': str(SERVICE), 'spec_id': self.info['spec_id']})

    def call(self, request):
        if service_info()['spec_id'] != self.info['spec_id']:
            raise RuntimeError('Pinned DeepCoder judge service changed during a request')
        files = request.get('inputs')
        expected = request.get('expected_responses')
        destination = Path(request.get('result', '')).resolve()
        if (not isinstance(files, list) or type(expected) is not int or
                not 0 < expected <= 8192 or not destination.is_relative_to(self.root) or
                destination.exists()):
            raise ValueError('Native judge request requires a new in-run result and signed response budget')
        rows = []
        for filename in files:
            source = Path(filename)
            if (source.is_symlink() or not source.resolve().is_relative_to(self.root) or
                    not source.is_file()):
                raise ValueError('Original native rollout shard is outside the signed run')
            data = json.loads(source.read_text())
            if data.get('complete') is not True:
                raise RuntimeError('Incomplete native vLLM response shard')
            rows.extend(data['rows'])
        rows.sort(key=lambda row: (row['index'], row['sample_index']))
        if (len(rows) != expected or len({(r['index'], r['sample_index']) for r in rows}) != expected):
            raise RuntimeError('Native DeepCoder question/sample denominator differs')
        started = time.monotonic()
        scored = []
        for row in rows:
            if row['judge']['kind'] not in ('deepcoder', 'lcb', 'humaneval', 'mbpp'):
                raise ValueError('Unsupported original DeepCoder grading task')
            has_code = _has_code(row['response'], self.info['checker_root'])
            acc = code_score(row['judge'], row['response'], case_dir=self.info['case_root'],
                             checker_dir=self.info['checker_root'], docker_image=self.info['image'])
            clipped = row['finish_reason'] == 'length'
            penalized = bool(request.get('format_penalty')) and clipped
            scored.append(dict(row, acc=int(acc and not penalized),
                               score=float(acc and not penalized),
                               acc_unpenalized=acc, score_unpenalized=float(acc),
                               no_code=not has_code, format_ok=not clipped and has_code,
                               gold_missing=False, scoring_timed_out=False))
        n = len(scored)
        groups = {}
        for row in scored:
            groups.setdefault(row['index'], []).append(row['acc'])
        passed = sum(any(values) for values in groups.values())
        clipped = sum(row['finish_reason'] == 'length' for row in scored)
        summary = {'complete': True, 'responses': n, 'scored_responses': n,
                   'questions': len(groups), 'correct': sum(r['acc'] for r in scored),
                   'accuracy': sum(r['acc'] for r in scored) / n,
                   'passed_questions': passed, 'pass_at_k': passed / len(groups),
                   'clipping_rate': clipped / n, 'clipped': clipped,
                   'mean_tokens': sum(r['tokens'] for r in scored) / n,
                   'no_code_rate': sum(r['no_code'] for r in scored) / n,
                   'gold_missing': 0, 'scoring_timeouts': 0,
                   'judge_service_spec_id': self.info['spec_id'],
                   'judge_wall_seconds': time.monotonic() - started}
        if request.get('format_penalty'):
            summary.update(format_penalty=True,
                accuracy_unpenalized=sum(r['acc_unpenalized'] for r in scored) / n,
                format_violation_rate=sum(not r['format_ok'] for r in scored) / n,
                format_zeroed_rate=sum(r['acc_unpenalized'] and not r['acc'] for r in scored) / n)
        scored_path = destination.with_name('scored.jsonl')
        if scored_path.exists():
            raise FileExistsError('Never overwrite original scored native rollouts')
        scored_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in scored))
        atomic_json(destination, summary)
        return summary

    def close(self):
        pass
