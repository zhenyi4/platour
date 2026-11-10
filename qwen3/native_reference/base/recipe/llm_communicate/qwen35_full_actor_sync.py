"""Publish one actual full actor snapshot for all local vLLM replicas."""
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from qwen35_bridge_opd_core import save


def shared_full_actor_snapshot(actor,engine,digest):
    run=engine.directory.parents[2]
    directory=run/'actor_sync'
    directory.mkdir(exist_ok=True)
    ready=directory/(digest+'.json')
    rank=dist.get_rank() if dist.is_initialized() else 0
    cached=(json.loads(ready.read_text()) if ready.exists() else None)
    if rank==0 and (cached is None or not Path(cached['actor_weights']).exists()):
        # The initial checkpoint already has the exact native weights. Avoid
        # rewriting it, and preserve tied weights rather than duplicating them.
        initial=run/'initial/checkpoint.json'
        if initial.exists() and json.loads(initial.read_text())['bridge_state_sha256']==digest:
            path=run/'initial/bridge.pt'
        else:
            path=directory/(digest+'.pt')
            temporary=path.with_suffix('.pt.tmp')
            state={n:t.detach().cpu() for n,t in actor.bridge_state_tensors().items()}
            torch.save(state,temporary)
            temporary.replace(path)
        save(ready,dict(actor_weights=str(path),actor_state_sha256=digest))
    start=time.monotonic()
    while not ready.exists() or not Path(json.loads(ready.read_text())['actor_weights']).exists():
        if time.monotonic()-start>600:raise TimeoutError('Rank0 failed to publish full actor weights')
        time.sleep(.1)
    payload=json.loads(ready.read_text())
    assert payload['actor_state_sha256']==digest
    # All prior rollout/eval RPCs finished before the caller's update barrier.
    # Keep the current and most recent older sync files; resumable checkpoints
    # and evaluation weights live outside this directory and are untouched.
    retention=getattr(actor,'full_actor_snapshot_retention',None)
    if rank==0 and retention is not None:
        retain_full_actor_snapshots(directory,Path(payload['actor_weights']),retention)
    return payload['actor_weights'],payload['actor_state_sha256']


def retain_full_actor_snapshots(directory,current,retention):
    if retention<2:raise ValueError('Need current and previous full-actor snapshots')
    files=sorted(directory.glob('*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    protected={current} if current.parent==directory else set()
    for path in files:
        if len(protected)>=retention:break
        protected.add(path)
    for path in files:
        if path not in protected:
            # Delete only owned digest-named transient synchronization files.
            if len(path.stem)!=64 or any(c not in '0123456789abcdef' for c in path.stem):
                continue
            path.unlink()
