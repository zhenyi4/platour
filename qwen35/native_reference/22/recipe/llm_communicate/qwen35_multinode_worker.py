"""torchrun worker: lightweight communication/shared-storage probe or explicit bridge training."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import sys
import time

from qwen35_bridge_opd_core import save
from qwen35_multinode_config import digest, rank_layout, verify_sources


def probe(control):
    import torch
    import torch.distributed as dist
    layout = rank_layout(dict(control['topology'], teacher_key=control['options']['teacher']))
    rank, world = layout['rank'], layout['world']
    cpu = control['cpu_probe']
    device = torch.device('cpu') if cpu else torch.device('cuda', layout['primary'])
    if not cpu:
        torch.cuda.set_device(device)
        assert torch.cuda.get_device_capability(device) == (9, 0)
    dist.init_process_group('gloo' if cpu else 'nccl', timeout=timedelta(seconds=control['timeout']),
                            **({} if cpu else dict(device_id=device)))
    try:
        identity = [control['nonce'] if rank == 0 else None]
        dist.broadcast_object_list(identity, src=0)
        assert identity[0] == control['nonce']
        invocation = Path(control['invocation'])
        marker = invocation/'shared_markers'/f'rank-{rank:02d}.json'
        report = dict(rank=rank, node_rank=layout['node_rank'], local_rank=layout['local_rank'],
            primary_gpu=layout['primary'], teacher_local_devices=layout['devices'], hostname=socket.gethostname(),
            nonce=control['nonce'], source_manifest_sha256=verify_sources(),
            root=control['root'], torch_version=torch.__version__,
            nccl_socket_ifname=os.environ.get('NCCL_SOCKET_IFNAME'), gloo_socket_ifname=os.environ.get('GLOO_SOCKET_IFNAME'))
        if not cpu:
            report['capability'] = [9, 0]
        save(marker, report)
        dist.barrier()
        reports = [None]*world
        dist.all_gather_object(reports, report)
        for i, expected in enumerate(reports):
            actual = json.loads((invocation/'shared_markers'/f'rank-{i:02d}.json').read_text())
            assert actual == expected and actual['nonce'] == control['nonce']
        assert len({r['root'] for r in reports}) == 1
        assert len({r['source_manifest_sha256'] for r in reports}) == 1
        assert len({r['torch_version'] for r in reports}) == 1
        hostnames = {r['hostname'] for r in reports}
        if not cpu:
            assert len(hostnames) == control['topology']['nnodes'], 'Two different physical hosts required'
        values = torch.full((4,), float(rank+1), device=device)
        dist.all_reduce(values)
        expected = world*(world+1)/2
        assert torch.equal(values, torch.full_like(values, expected))
        count = control['probe_mib']*1024*1024//4
        payload = torch.empty(count, device=device, dtype=torch.float32)
        timings = []
        for iteration in range(7):
            payload.fill_(float(rank+1))
            if not cpu:
                torch.cuda.synchronize(device)
            started = time.monotonic()
            dist.all_reduce(payload)
            if not cpu:
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            assert payload[0].item() == expected and payload[-1].item() == expected
            if iteration >= 2:
                timings.append(elapsed)
        latency = torch.tensor(sum(timings)/len(timings), device=device, dtype=torch.float64)
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        if rank == 0:
            result = dict(passed=True, nccl_validated=not cpu, cpu_simulation=cpu, nonce=control['nonce'],
                topology=control['topology'], source_manifest_sha256=control['source_manifest_sha256'],
                hosts=sorted(hostnames), reports=reports, shared_storage_verified=True,
                probe_mib=control['probe_mib'], max_rank_mean_allreduce_seconds=float(latency),
                note='Collective payload latency, not NCCL bus-bandwidth or end-to-end training speed', finished=time.time())
            save(invocation/'probe_result.json', result)
            print(json.dumps({k:result[k] for k in ('passed','nccl_validated','cpu_simulation','hosts',
                                                   'probe_mib','max_rank_mean_allreduce_seconds')}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    assert control['source_manifest_sha256'] == verify_sources()
    if control['mode'] == 'probe':
        probe(control)
        return
    root = Path(control['root'])
    assert digest(root/'protocol.json') == control['protocol_sha256']
    protocol=json.loads((root/'protocol.json').read_text())
    for name,key in [('train_prompts.json','train_prompts_sha256'),('eval_prompts.json','eval_prompts_sha256')]:
        assert digest(root/name)==protocol[key], f'Prepared inputs changed: {name}'
    import train_qwen35_bridge_opd_vllm as trainer
    sys.argv = [str(Path(trainer.__file__)), '--root', str(root), '--objective', 'grpo']
    if control['mode'] == 'smoke':
        sys.argv += ['--smoke', '--stress']
    elif control['resume_step'] is not None:
        sys.argv += ['--resume-step', str(control['resume_step'])]
    trainer.main()


if __name__ == '__main__':
    main()
