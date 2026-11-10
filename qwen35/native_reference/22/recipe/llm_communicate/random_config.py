"""One 8GPU host, eight receiver replicas, native4B train-only Gaussian V and zero W."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent.parents[1]


def verify_sources():
    manifest = ROOT.parent / 'source_manifest.json'
    meta = json.loads(manifest.read_text())
    for relative, expected in meta['files'].items():
        path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Random4B frozen source/statistics changed: {relative}')
    with (ORIGINAL / meta['launcher']).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['launcher_sha256']:
            raise RuntimeError('Random4B one-host launcher changed')
    with manifest.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def initialize_zero(bridge):
    return bridge.initialize_for_grpo(seed=42)


def rank_layout(protocol, environment=None):
    env = os.environ if environment is None else environment
    rank, world = int(env['RANK']), int(env['WORLD_SIZE'])
    local, local_world = int(env['LOCAL_RANK']), int(env.get('LOCAL_WORLD_SIZE', world))
    if (protocol['nnodes'] != 1 or protocol['world_size'] != 8 or protocol['teacher_model_parallel_size'] != 1 or
            world != local_world or world != 8 or local != rank or int(env.get('GROUP_RANK', '0')) != 0):
        raise RuntimeError('Random-template bridge requires one host, 8 local single-GPU receiver replicas')
    return dict(rank=rank, world=world, local_rank=local, local_world=local_world,
                node_rank=0, primary=local, devices=[local], teacher='random4B_template')


def stress_prompt(row, protocol):
    source = row['prompt_token_ids']
    template = protocol['token_template']
    prefix, suffix = template['prefix'], template['suffix']
    body = source[prefix:-suffix]
    needed = protocol['max_prompt_length'] - prefix - suffix
    if not body or needed < 1:
        raise RuntimeError('Cannot build full 2048-token stress without changing template')
    return source[:prefix] + (body * ((needed + len(body) - 1) // len(body)))[:needed] + source[-suffix:]
