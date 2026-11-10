"""Local-only Docker sandbox service for shared-storage LCB jobs. CUDA and listening ports are unused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import uuid

import code_judge
import code_prompts
import lcb_score_worker
from qwen35_bridge_opd_core import ROOT, save
from qwen35_multinode_config import verify_sources
from shared_code_judge_client import SERVICE, ALLOWED_ROOTS, within


class Service:
    def __init__(self, directory=SERVICE, allowed_roots=ALLOWED_ROOTS, workers=64):
        self.directory = Path(directory).absolute()
        self.allowed_roots = Path(allowed_roots).absolute()
        self.workers = workers
        self.instance = uuid.uuid4().hex
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = None
        self.completed = 0
        self.failed = 0
        for path in (code_judge.WORKER,code_judge.CHECKER/'testing_util.py'):
            if not path.is_file():
                raise RuntimeError(f'Missing sandbox/checker file: {path}')
        if not code_judge.SAMPLES.is_dir():
            raise RuntimeError('Missing official LCB test samples')
        self.problems = code_prompts.lcb_problems()
        version = verify_sources()
        image = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',code_judge.IMAGE],text=True,timeout=30).strip()
        subprocess.check_output(['docker','version','--format','{{.Server.Version}}'],text=True,timeout=30)
        spec = dict(backend='shared_files_docker',source_manifest_sha256=version,image=code_judge.IMAGE,
                    image_id=image,judge_settings=code_judge.DEFAULTS,workers=workers)
        self.spec_id = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        self.info = dict(spec,spec_id=self.spec_id,instance=self.instance,pid=os.getpid(),
                         hostname=socket.gethostname(),started=time.time())
        code_judge.IMAGE=image

    def heartbeat(self):
        while not self.stop.is_set():
            with self.lock:
                status = dict(self.info,ready=True,heartbeat=time.time(),active_request=self.active,
                              completed_requests=self.completed,failed_requests=self.failed)
            save(self.directory/'ready.json',status)
            self.stop.wait(2)

    def request(self, path, pool):
        message = json.loads(Path(path).read_text())
        ident = message['id']
        if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident) or Path(path).stem!=ident:
            raise ValueError('Invalid shared request identity')
        if message['spec_id']!=self.spec_id or message['service_instance']!=self.instance:
            raise ValueError('Stale service request')
        if not time.time()<message['deadline']<=time.time()+7200:
            raise ValueError('Expired or invalid request deadline')
        root = within(message['root'],self.allowed_roots)
        protocol = json.loads((root/'protocol.json').read_text())
        if protocol.get('dataset')!='lcb' or protocol.get('judge_service_spec_id')!=self.spec_id:
            raise ValueError('Unsupported or mismatched run protocol')
        result = within(message['result'],root)
        if result.exists():
            raise ValueError('Scored result already exists; refuse overwrite')
        rows = []
        for filename in message['inputs']:
            payload = json.loads(within(filename,root).read_text())
            if not payload.get('complete'):
                raise ValueError('Incomplete generated input')
            rows.extend(payload['rows'])
        if not len(rows)==message['expected_responses'] or not 0<len(rows)<=8192:
            raise ValueError('Response count mismatch')
        rows.sort(key=lambda r:(r['index'],r['sample_index']))
        if len({(r['index'],r['sample_index']) for r in rows})!=len(rows):
            raise ValueError('Duplicate response identity')
        for row in rows:
            key = row['key']
            if key not in self.problems or row['judge'].get('kind')!='lcb':
                raise ValueError('Only known official LCB tasks are accepted')
            row['judge']=dict(kind='lcb',key=key,tests=self.problems[key]['tests'])
        started = time.monotonic()
        scored = list(pool.map(lcb_score_worker.judge_row,rows))
        penalty = message['format_penalty']
        if penalty:
            scored = [lcb_score_worker.apply_penalty(row) for row in scored]
        summary = lcb_score_worker.summarize(scored,penalty,time.monotonic()-started)
        summary.update(judge_service_spec_id=self.spec_id,judge_service_backend='shared_files_docker')
        result.parent.mkdir(parents=True,exist_ok=True)
        output = result.with_name('scored.jsonl')
        if output.exists():
            raise ValueError('Scored rows already exist; refuse overwrite')
        temporary = output.with_name(output.name+f'.{ident}.tmp')
        with temporary.open('w') as stream:
            for row in scored:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        temporary.replace(output)
        save(result,summary)
        return dict(complete=True,id=ident,spec_id=self.spec_id,service_instance=self.instance,
                    responses=len(scored),seconds=time.monotonic()-started,finished=time.time())

    def run(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        for name in ('requests','active','replies','archive'):(self.directory/name).mkdir(exist_ok=True)
        with (self.directory/'service.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for path in (self.directory/'active').glob('*.json'):
                save(self.directory/'replies'/path.name,dict(complete=False,id=path.stem,
                     error='Judge service interrupted previous active request'))
                path.rename(self.directory/'archive'/f'{path.stem}-interrupted.json')
            heartbeat = threading.Thread(target=self.heartbeat,daemon=True)
            heartbeat.start()
            print(json.dumps(dict(stage='ready',service=str(self.directory),workers=self.workers,
                                  spec_id=self.spec_id,uses_cuda=False,listening_ports=False)),flush=True)
            try:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    while not self.stop.is_set():
                        pending = sorted((self.directory/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime)
                        if not pending:
                            self.stop.wait(.5)
                            continue
                        source = pending[0]
                        active = self.directory/'active'/source.name
                        source.rename(active)
                        with self.lock:self.active=source.stem
                        try:
                            response = self.request(active,pool)
                            with self.lock:self.completed+=1
                        except Exception as error:
                            traceback.print_exc()
                            response = dict(complete=False,id=source.stem,error=repr(error),finished=time.time())
                            with self.lock:self.failed+=1
                        finally:
                            code_judge.cleanup(owner=os.getpid())
                        save(self.directory/'replies'/source.name,response)
                        active.rename(self.directory/'archive'/source.name)
                        with self.lock:self.active=None
                        print(json.dumps(dict(stage='scored' if response['complete'] else 'scoring_failed',
                                              id=source.stem,seconds=response.get('seconds'),error=response.get('error'))),flush=True)
            finally:
                self.stop.set()
                heartbeat.join(timeout=5)
                save(self.directory/'ready.json',dict(self.info,ready=False,heartbeat=time.time()))
                code_judge.cleanup(owner=os.getpid())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=SERVICE)
    parser.add_argument('--workers',type=int,default=64)
    args = parser.parse_args()
    if not 1<=args.workers<=64:
        parser.error('Use 1..64 sandbox workers')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Start this CPU-only service with CUDA_VISIBLE_DEVICES empty')
    lcb_score_worker.WORKERS=args.workers
    lcb_score_worker.SETTINGS={}
    service = Service(args.directory,workers=args.workers)
    def stopped(*_):service.stop.set()
    signal.signal(signal.SIGTERM,stopped)
    signal.signal(signal.SIGINT,stopped)
    service.run()


if __name__=='__main__':
    main()
