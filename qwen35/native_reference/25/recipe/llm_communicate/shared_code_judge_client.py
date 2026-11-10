"""Shared-file RPC to a local Docker code judge; no network endpoint or remote daemon access."""
import json
import os
from pathlib import Path
import time
import uuid

from qwen35_bridge_opd_core import save
from qwen35_multinode_config import ORIGINAL

SERVICE = ORIGINAL/'outputs/shared_code_judge_v3'
ALLOWED_ROOTS = ORIGINAL/'outputs/multinode_runs'


def within(path, prefix):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(prefix).resolve()):
        raise ValueError(f'Path outside permitted shared jobs: {path}')
    return resolved


def service_info(service=SERVICE, max_age=30):
    path = Path(service)/'ready.json'
    if not path.exists():
        raise RuntimeError(f'Local isolated code-judge service is not running: {path}')
    info = json.loads(path.read_text())
    if not info.get('ready') or time.time()-info['heartbeat']>max_age:
        raise RuntimeError('Local isolated code-judge service heartbeat is stale; do not train')
    if info.get('backend')!='shared_files_docker' or not info.get('spec_id'):
        raise RuntimeError('Unexpected judge-service backend')
    return info


class SharedCodeScorer:
    def __init__(self, directory, service=SERVICE, timeout=3600):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.service = Path(service).absolute()
        self.timeout = timeout
        candidates = [p for p in self.directory.parents if (p/'protocol.json').exists()]
        if not candidates:
            raise ValueError('Scorer directory has no saved run protocol')
        self.root = within(candidates[0],ALLOWED_ROOTS)
        protocol = json.loads((self.root/'protocol.json').read_text())
        self.info = service_info(self.service)
        if protocol.get('dataset')!='lcb' or protocol.get('judge_service_spec_id')!=self.info['spec_id']:
            raise ValueError('Run protocol and local judge-service identity do not match')
        save(self.directory/'ready.json',dict(cpu_only=True,remote_via_shared_files=True,
             service=str(self.service),spec_id=self.info['spec_id']))

    def call(self, request):
        info = service_info(self.service)
        if info['spec_id']!=self.info['spec_id']:
            raise RuntimeError('Judge service changed; refuse stale scoring')
        result = within(request['result'],self.root)
        if result.exists():
            raise ValueError(f'Refusing stale scorer result: {result}')
        inputs = [str(within(p,self.root)) for p in request['inputs']]
        ident = uuid.uuid4().hex
        expected = int(request['expected_responses'])
        if not 0<expected<=8192:
            raise ValueError('Invalid expected response count')
        message = dict(id=ident,root=str(self.root),inputs=inputs,result=str(result),
            expected_responses=expected,format_penalty=bool(request.get('format_penalty',False)),
            spec_id=info['spec_id'],service_instance=info['instance'],created=time.time(),
            deadline=time.time()+self.timeout)
        save(self.service/'requests'/f'{ident}.json',message)
        reply = self.service/'replies'/f'{ident}.json'
        deadline = time.monotonic()+self.timeout
        while not reply.exists():
            current = service_info(self.service)
            if current['spec_id']!=info['spec_id'] or current['instance']!=info['instance']:
                raise RuntimeError('Code-judge service restarted during a request; resume from checkpoint')
            if time.monotonic()>deadline:
                raise TimeoutError(f'Code-judge request timed out: {ident}')
            time.sleep(.5)
        response = json.loads(reply.read_text())
        if response.get('id')!=ident or not response.get('complete'):
            raise RuntimeError(f'Isolated code judging failed: {response.get("error", "bad reply identity")}')
        if response.get('spec_id')!=info['spec_id'] or response.get('service_instance')!=info['instance']:
            raise RuntimeError('Code-judge response identity mismatch')
        summary = json.loads(result.read_text())
        if not summary.get('complete') or summary.get('responses')!=expected:
            raise RuntimeError('Incomplete code-scoring output')
        return summary

    def close(self):
        pass
