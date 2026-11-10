"""Opt-in shared-files code judge: frozen ID26/29 Docker scoring on one host.

Run only on a machine with the original authorized image and checker. GPU clients
may share a filesystem, but generated programs execute only in disposable Docker
containers on the service host. Never use the host interpreter as a fallback.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from .checkpoints import atomic_json, sha256
from .cli import DATA, inputs, signed_rows
from .catalog import select

REFERENCE = Path(__file__).with_name('native_reference')
ORIGINAL_IMAGE = 'sha256:c620f366cb0a3a82143c15d5b367a9843539b2a8d2b501744f5e0ebd4d9d0b2e'
SOURCES = {'25': '25', '26': '25', '29': '23'}


def frozen_sources(checker, cases):
    from .native_deepcoder import verify_bundle as verify_25
    from .native_full import verify_bundle as verify_23
    verify_25()
    verify_23()
    third_party = str(checker.parents[1])
    if third_party not in sys.path:
        sys.path.insert(0, third_party)
    modules = {}
    for experiment, original in SOURCES.items():
        directory = REFERENCE / original / 'recipe/llm_communicate'
        source = directory / 'code_judge.py'
        worker = directory / 'code_sandbox_worker.py'
        if not source.is_file() or not worker.is_file():
            raise FileNotFoundError('Frozen original Docker scorer or worker is unavailable')
        name = 'platour_original_code_judge_' + experiment
        location = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(location)
        location.loader.exec_module(module)
        module.CHECKER = checker
        module.SAMPLES = cases
        module.WORKER = worker
        module.IMAGE = 'lcb-checker:local'
        modules[experiment] = module
    return modules


class JudgeService:
    def __init__(self, root, cases, checker, *, image='lcb-checker:local',
                 data_root=DATA, workers=4):
        self.root = Path(root).absolute()
        self.cases = Path(cases).resolve()
        self.checker = Path(checker).resolve()
        self.image = image
        if (self.root.exists() or workers < 1 or workers > 16 or
                not self.cases.is_dir() or
                not (self.checker / 'testing_util.py').is_file() or
                not (self.cases.parent / 'complete.json').is_file()):
            raise ValueError('Use a new private service root and verified official cases/checker')
        proof = json.loads((self.cases.parent / 'complete.json').read_text())
        self.checker_hash = sha256(self.checker / 'testing_util.py')
        extractor = self.checker.parent / 'utils/extraction_utils.py'
        from .judge_cases import PINNED
        pinned = json.loads(PINNED.read_text())
        if extractor.is_symlink() or not extractor.is_file():
            raise RuntimeError('Original LiveCodeBench code extractor is missing')
        self.extractor_hash = sha256(extractor)
        self.case_proof_sha = sha256(self.cases.parent / 'complete.json')
        if (not proof.get('complete') or not proof.get('per_question_sha256_verified') or
                proof.get('checker_sha256') != self.checker_hash or
                proof.get('extractor_sha256') != self.extractor_hash or
                self.extractor_hash != pinned['extractor_sha256'] or
                proof.get('training_validation_cases') != 18873 + 201 or
                proof.get('lcb_v6_cases') != 175):
            raise RuntimeError('Sealed code-judge cases/checker differ')
        inspect = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                 capture_output=True, text=True, timeout=30)
        self.image_id = inspect.stdout.strip() if inspect.returncode == 0 else ''
        if self.image_id != ORIGINAL_IMAGE:
            raise RuntimeError('Docker daemon or pinned historical code-judge image is unavailable')
        signed = inputs(select(26), data_root)
        self.deepcoder = {}
        for rows in signed.values():
            for row in rows:
                spec = row['judge']
                key = spec['key']
                if (key in self.deepcoder or spec['kind'] != 'deepcoder' or
                        not key.replace('_', '').isalnum()):
                    raise RuntimeError('Original DeepCoder unit-test identity differs')
                self.deepcoder[key] = (spec['sample_sha256'], spec['tests'])
        if len(self.deepcoder) != 18873 + 201:
            raise RuntimeError('Original DeepCoder TRAIN/VAL denominator differs')
        manifest = json.loads((Path(data_root) / 'manifest.json').read_text())['benchmark']
        self.tests = {}
        for suite in ('lcb', 'humaneval', 'mbpp'):
            path = Path(data_root) / 'deepcoder' / (suite + '.jsonl')
            if sha256(path) != manifest[suite]['sha256']:
                raise RuntimeError('Frozen original code test split differs: ' + suite)
            rows = signed_rows(path, expected=manifest[suite]['rows'])
            self.tests[suite] = {json.dumps(row['judge'], sort_keys=True) for row in rows}
        if len(self.tests['lcb']) != 175:
            raise RuntimeError('Original LCB v6 denominator differs')
        self.lcb = proof['lcb_case_sha256']
        for name, expected in self.lcb.items():
            if sha256(self.cases / name) != expected:
                raise RuntimeError('Original LCB v6 case SHA differs')
        self.modules = frozen_sources(self.checker, self.cases)
        source_hashes = {key: {'judge': sha256(Path(module.__file__)),
                                'worker': sha256(module.WORKER)}
                         for key, module in self.modules.items()}
        identity = {'backend': 'isolated_docker_shared_files',
                    'image': image, 'image_id': self.image_id,
                    'case_root': str(self.cases), 'checker_root': str(self.checker),
                    'checker_sha256': self.checker_hash,
                    'extractor_sha256': self.extractor_hash,
                    'case_proof_sha256': self.case_proof_sha,
                    'supported_experiments': list(SOURCES),
                    'frozen_judge_source_sha256': source_hashes}
        self.spec_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.info = dict(identity, spec_id=self.spec_id, instance=uuid4().hex,
                         pid=os.getpid())
        self.done = threading.Event()
        self.workers = workers

    def _heartbeat(self):
        while not self.done.is_set():
            atomic_json(self.root / 'ready.json',
                        dict(self.info, ready=True, heartbeat=time.time()))
            self.done.wait(2)
        atomic_json(self.root / 'ready.json',
                    dict(self.info, ready=False, heartbeat=time.time()))

    def score(self, request):
        if (not isinstance(request, dict) or request.get('spec_id') != self.spec_id or
                request.get('instance') != self.info['instance'] or
                request.get('experiment') not in SOURCES or
                not isinstance(request.get('response'), str) or
                len(request['response']) > 2_000_000):
            raise RuntimeError('Wrong judge identity, experiment or oversized response')
        current = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                  self.image], capture_output=True, text=True, timeout=20)
        if (current.returncode or current.stdout.strip() != self.image_id or
                sha256(self.checker / 'testing_util.py') != self.checker_hash or
                sha256(self.checker.parent / 'utils/extraction_utils.py') != self.extractor_hash or
                sha256(self.cases.parent / 'complete.json') != self.case_proof_sha):
            raise RuntimeError('Pinned isolated image/checker/extractor/case seal changed during judging')
        spec = request.get('spec')
        if not isinstance(spec, dict):
            raise ValueError('Missing signed original judge specification')
        kind = spec.get('kind')
        if kind == 'deepcoder':
            key = spec.get('key')
            if (not isinstance(key, str) or key not in self.deepcoder or
                    (spec.get('sample_sha256'), spec.get('tests')) != self.deepcoder[key]):
                raise RuntimeError('Original TRAIN/VAL case identity differs')
            file = self.cases / (key + '.json')
            if file.is_symlink() or sha256(file) != spec['sample_sha256']:
                raise RuntimeError('Original TRAIN/VAL case SHA differs')
            checked = {'kind': 'lcb', 'key': key, 'tests': spec['tests']}
        elif kind in self.tests:
            if json.dumps(spec, sort_keys=True) not in self.tests[kind]:
                raise RuntimeError('Only signed original code benchmark rows may be scored')
            if kind == 'lcb':
                key = spec['key']
                if (key + '.json' not in self.lcb or
                        sha256(self.cases / (key + '.json')) != self.lcb[key + '.json']):
                    raise RuntimeError('Official LCB v6 case SHA differs')
            checked = spec
        else:
            raise ValueError('Unknown original programming benchmark')
        score = self.modules[request['experiment']].judge(checked, request['response'])
        if not isinstance(score, dict) or type(score.get('correct')) is not bool:
            raise RuntimeError('Original isolated Docker scorer returned invalid verdict')
        return int(score['correct'])

    def _process(self, path):
        ident = path.stem
        result = {'id': ident, 'spec_id': self.spec_id,
                  'instance': self.info['instance'], 'complete': False}
        try:
            if len(ident) != 32 or any(ch not in '0123456789abcdef' for ch in ident):
                raise ValueError('Unsafe judge RPC identity')
            request = json.loads(path.read_text())
            if request.get('id') != ident:
                raise RuntimeError('Judge RPC filename and identity differ')
            result.update(correct=self.score(request), complete=True)
        except Exception as error:
            result['error'] = repr(error)
        atomic_json(self.root / 'replies' / path.name, result)
        path.rename(self.root / 'archive' / path.name)

    def run(self):
        self.root.mkdir(mode=0o700)
        for name in ('requests', 'active', 'replies', 'archive'):
            (self.root / name).mkdir(mode=0o700)
        worker = threading.Thread(target=self._heartbeat, daemon=True)
        previous_term = signal.signal(signal.SIGTERM, lambda *_: self.done.set())
        worker.start()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                pending = []
                while not self.done.is_set():
                    for path in (self.root / 'requests').glob('*.json'):
                        active = self.root / 'active' / path.name
                        path.rename(active)
                        pending.append(pool.submit(self._process, active))
                    pending = [task for task in pending if not task.done()]
                    self.done.wait(.25)
        finally:
            self.done.set()
            worker.join(timeout=5)
            signal.signal(signal.SIGTERM, previous_term)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--checker', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    service = JudgeService(args.root, args.cases, args.checker,
                           data_root=args.data_root, workers=args.workers)
    service.run()


if __name__ == '__main__':
    main()
