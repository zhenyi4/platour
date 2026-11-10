"""Crash-safe training states: commit weights, optimizer and RNG before validation."""

import hashlib
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def atomic_torch(value: object, destination: Path) -> None:
    temporary = destination.with_name(destination.name + f".{rank()}.pending")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json(value: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".pending")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def capture_rng(cuda_devices: tuple[int, ...] | None = None) -> dict:
    result = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if cuda_devices is None:
        result["accelerator"] = (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else [])
    else:
        if cuda_devices and not torch.cuda.is_available():
            raise RuntimeError('Cannot checkpoint native CUDA RNG without GPUs')
        result["cuda_devices"] = list(cuda_devices)
        result["accelerator"] = [torch.cuda.get_rng_state(device) for device in cuda_devices]
    return result


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    if 'cuda_devices' in value:
        if len(value['cuda_devices']) != len(value['accelerator']):
            raise RuntimeError('Signed CUDA device/RNG mapping differs')
        for device, state in zip(value['cuda_devices'], value['accelerator'], strict=True):
            torch.cuda.set_rng_state(state, device)
    elif value["accelerator"]:
        torch.cuda.set_rng_state_all(value["accelerator"])


def save_step(output: Path, step: int, bridge: torch.nn.Module,
              optimizer: torch.optim.Optimizer, scheduler: object | None = None, *,
              cuda_devices: tuple[int, ...] | None = None,
              record: dict | None = None, identity: str | None = None) -> Path:
    """A checkpoint is usable iff its complete marker exists after every rank's RNG."""
    # No rank may create this directory while peers still inspect resume state.
    barrier()
    directory = output / "checkpoints" / f"step-{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite signed step {step}")
    if rank() == 0:
        atomic_torch({"P": bridge.P.state_dict()}, directory / "bridge.pt")
        atomic_torch({"optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict() if scheduler is not None else None,
                      "step": step}, directory / "training.pt")
    atomic_torch(capture_rng(cuda_devices), directory / f"rng.rank-{rank():02d}.pt")
    barrier()
    if rank() == 0:
        required = [directory / "bridge.pt", directory / "training.pt"] + [
            directory / f"rng.rank-{i:02d}.pt" for i in range(world_size())]
        if not all(file.is_file() for file in required):
            raise RuntimeError("Checkpoint state is incomplete; do not validate or resume")
        hashes = {}
        for file in required:
            with file.open('rb') as stream:
                hashes[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        atomic_json({"complete": True, "step": step, "world_size": world_size(),
                     "identity": identity,
                     "sha256": hashes, "metric": record}, directory / "complete.json")
    barrier()
    return directory


def load_step(directory: Path, bridge: torch.nn.Module,
              optimizer: torch.optim.Optimizer, scheduler: object | None = None, *,
              expected_identity: str | None = None) -> int:
    marker = json.loads((directory / "complete.json").read_text())
    if not marker.get("complete"):
        raise RuntimeError("Checkpoint is incomplete")
    if expected_identity is not None and marker.get('identity') != expected_identity:
        raise RuntimeError('Checkpoint belongs to another training protocol or lacks resume identity')
    if marker["world_size"] != world_size():
        raise RuntimeError(f"Resume must use the same number of torchrun ranks as the checkpoint "
                           f"(saved with {marker['world_size']}, now {world_size()})")
    if 'sha256' not in marker:
        raise RuntimeError('Checkpoint lacks signed file hashes')
    required = {'bridge.pt', 'training.pt', *(f'rng.rank-{i:02d}.pt'
                                               for i in range(world_size()))}
    if set(marker['sha256']) != required:
        raise RuntimeError('Signed checkpoint file set differs')
    for name, expected in marker['sha256'].items():
        with (directory / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Signed checkpoint state SHA differs: ' + name)
    step = marker["step"]
    bridge.P.load_state_dict(torch.load(directory / "bridge.pt", weights_only=True,
                                        map_location="cpu")["P"])
    state = torch.load(directory / "training.pt", weights_only=True, map_location="cpu")
    if state["step"] != step:
        raise RuntimeError("Optimizer and bridge checkpoint steps differ")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise RuntimeError("Scheduler state is missing")
        scheduler.load_state_dict(state["scheduler"])
    restore_rng(torch.load(directory / f"rng.rank-{rank():02d}.pt",
                           weights_only=True, map_location="cpu"))
    barrier()
    return step
