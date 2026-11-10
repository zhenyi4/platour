"""Optional torchrun data parallelism for the portable Qwen3.5 trainer.

Every rank holds one replica on its local GPU, processes a disjoint slice of
the same global batch, and all-reduces gradients before the identical update.
A single process (no ``WORLD_SIZE``) keeps the original behaviour.
"""

from datetime import timedelta
import os


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def init():
    """Return ``(rank, world_size, local_rank)``; NCCL on GPUs, gloo on a CPU-only host."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world == 1:
        return 0, 1, 0
    import torch
    import torch.distributed as dist
    if not all(key in os.environ for key in ('RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')):
        raise RuntimeError('Multi-process portable training must be launched by torchrun')
    local = int(os.environ['LOCAL_RANK'])
    hours = float(os.environ.get('PLATOUR_DIST_TIMEOUT_HOURS', '24'))
    if not torch.cuda.is_available():
        if not dist.is_initialized():
            dist.init_process_group('gloo', timeout=timedelta(hours=hours))
        return dist.get_rank(), dist.get_world_size(), local
    if local >= torch.cuda.device_count():
        raise RuntimeError('Each torchrun rank needs its own visible local GPU')
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group('nccl', timeout=timedelta(hours=hours),
                                device_id=torch.device('cuda', local))
    return dist.get_rank(), dist.get_world_size(), local


def device(local):
    """The training device of this rank."""
    import torch
    return f'cuda:{local}' if torch.cuda.is_available() else 'cpu'


def rank():
    dist = _dist()
    return dist.get_rank() if dist else 0


def world():
    dist = _dist()
    return dist.get_world_size() if dist else 1


def barrier():
    dist = _dist()
    if dist:
        dist.barrier()


def broadcast(value):
    """Rank zero's Python value on every rank."""
    dist = _dist()
    if not dist:
        return value
    holder = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def gather(value):
    """All ranks' Python values in rank order (on every rank)."""
    dist = _dist()
    if not dist:
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def all_reduce_sum(values, device):
    """Sum a short list of floats across ranks."""
    dist = _dist()
    if not dist:
        return [float(value) for value in values]
    import torch
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def all_reduce_gradients(parameters):
    """Sum gradients so every rank applies the identical global update."""
    dist = _dist()
    if not dist:
        return
    import torch
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)


def local_slice(rows):
    """This rank's disjoint, deterministic part of a global list."""
    return rows[rank()::world()]


def on_rank_zero(function):
    """Run ``function`` once on rank zero; share its value or failure with every rank."""
    error = value = failure = None
    if rank() == 0:
        try:
            value = function()
        except BaseException as exc:  # propagate instead of leaving peers at a barrier
            failure, error = exc, f'{type(exc).__name__}: {exc}'
    status = broadcast((error, value))
    if status[0] is not None:
        if failure is not None:
            raise failure
        raise RuntimeError('Rank-zero training step failed: ' + status[0])
    return status[1]
