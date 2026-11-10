"""Train-only Qwen3.5-4B template Gaussian initialization for the random bridge."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .checkpoints import sha256


def template(rows):
    tokens = [row['prompt_token_ids'] for row in rows]
    if not tokens or not all(tokens):
        raise ValueError('Training prompts cannot be empty')
    first = tokens[0]
    smallest = min(map(len, tokens))
    prefix = next((i for i in range(smallest) if any(row[i] != first[i] for row in tokens)),
                  smallest)
    suffix = next((i - 1 for i in range(1, smallest + 1)
                   if any(row[-i] != first[-i] for row in tokens)), smallest)
    if not prefix or suffix < 2 or any(len(row) <= prefix + suffix for row in tokens):
        raise ValueError('No disjoint frozen public prompt template')
    fixed = first[:prefix] + first[-suffix:-1]
    for row in tokens:
        if row[:prefix] + row[-suffix:-1] != fixed:
            raise ValueError('Input prompt does not match TRAIN-only template')
    return prefix, suffix, fixed


def save_native_statistics(case, receiver, data_root, chosen, prefix, suffix,
                           fixed, mean, std, output):
    """Keep train-only moments private for the original ID22 native trainer."""
    model = Path(receiver).resolve()
    output = Path(output)
    if case.identifier != 22 or model.name != 'Qwen3.5-4B' or output.exists():
        raise ValueError('Native ID22 moments need a new root and the public 4B model')
    if ((prefix, suffix, len(fixed)) != (42, 22, 63) or
            len(chosen) != 500 or len({row['index'] for row in chosen}) != 500 or
            list(mean.shape) != [63, 32, 1] or mean.shape != std.shape or
            not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()) or
            not bool((std >= 0).all())):
        raise ValueError('Native train-only Gaussian statistics shape or scope differs')
    shards = sorted(model.glob('model.safetensors-*.safetensors'))
    if len(shards) != 2:
        raise ValueError('Native Qwen3.5-4B model must contain two original shards')
    model_hashes = {part.name: sha256(part) for part in shards}
    source_sha = sha256(Path(data_root) / case.task / 'train.jsonl')
    record = {'token_template': {'prefix': prefix, 'suffix': suffix,
                                 'template_token_ids': fixed},
              'selected_sample_indices': [row['index'] for row in chosen],
              'source_train_sha256': source_sha,
              'source_model_config_sha256': sha256(model / 'config.json'),
              'source_model_index_sha256': sha256(model / 'model.safetensors.index.json'),
              'source_model_shards_sha256': model_hashes,
              'expected_mean_std_shape': [63, 32, 1],
              'statistics_source_model': 'Qwen3.5-4B',
              'stats_sample_source': 'train_only'}
    state = {'mean': mean.detach().cpu().contiguous(),
             'std': std.detach().cpu().contiguous(),
             'statistics_source_model': 'Qwen3.5-4B',
             'source_model_config_sha256': record['source_model_config_sha256'],
             'source_model_index_sha256': record['source_model_index_sha256'],
             'source_model_shards_sha256': model_hashes,
             'source_train_sha256': source_sha,
             'sample_count': 500, 'sample_indices': record['selected_sample_indices'],
             'template_token_ids': fixed, 'layer_map': list(range(32)),
             'broadcast_shape': [63, 32, 2560], 'variance_correction': 0,
             'seed': 42}
    destination = output / case.task
    destination.mkdir(parents=True)
    pending = destination / 'statistics.pt.pending'
    torch.save(state, pending)
    pending.replace(destination / 'statistics.pt')
    (destination / 'template.json').write_text(json.dumps(record, sort_keys=True) + '\n')
    (destination / 'complete.json').write_text(json.dumps({
        'complete': True, 'excluded_validation_and_test': True,
        'sample_count': 500, 'statistics_sha256': sha256(destination / 'statistics.pt')},
        sort_keys=True) + '\n')
    return destination


def init(case, receiver, output, *, data_root=None, sample_count=500,
         native_statistics_root=None):
    from .cli import DATA, inputs
    from .model import load_text
    if data_root is None:
        data_root = DATA
    if case.method != 'random' or sample_count != 500:
        raise ValueError('Only the fixed 500 TRAIN-only prompt statistics are supported')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite signed random-template statistics')
    if native_statistics_root is not None:
        if case.identifier != 22 or Path(native_statistics_root).exists():
            raise ValueError('Native moment export needs ID22 and a fresh train-only statistics root')
    data = inputs(case, data_root)
    train, val = data['train'], data['validation']
    prefix, suffix, fixed = template(train)
    if (len(fixed), prefix, suffix) != ((63, 42, 22) if case.task == 'dapo'
                                       else (67, 45, 23)):
        raise RuntimeError('The original dataset template identity differs')
    if any(row['prompt_token_ids'][:prefix] + row['prompt_token_ids'][-suffix:-1] != fixed
           for row in val):
        raise ValueError('Train/val public template token IDs differ')
    chosen = sorted(train, key=lambda row: hashlib.sha256(
        f'42:{row["index"]}'.encode()).hexdigest())[:sample_count]
    chosen.sort(key=lambda row: row['index'])
    model = load_text(receiver).eval().requires_grad_(False)
    if model.config.num_hidden_layers != 32 or model.config.hidden_size != 2560:
        raise ValueError('Statistics need the original frozen Qwen3.5-4B text model')
    shape = (len(fixed), 32, 1)
    total = torch.zeros(shape, dtype=torch.float64)
    square = torch.zeros(shape, dtype=torch.float64)
    captures = {}
    positions = []
    hooks = []
    try:
        for depth, layer in enumerate(model.model.layers):
            def grab(_layer, args, kwargs, slot=depth):
                hidden = args[0] if args else kwargs['hidden_states']
                captures[slot] = hidden[0].index_select(
                    0, torch.as_tensor(positions, device=hidden.device)).detach().float().cpu()
            hooks.append(layer.register_forward_pre_hook(grab, with_kwargs=True))
        with torch.inference_mode():
            for row in chosen:
                ids = row['prompt_token_ids']
                positions[:] = [*range(prefix), *range(len(ids) - suffix, len(ids) - 1)]
                captures.clear()
                x = torch.tensor([ids], device=model.device)
                model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
                if set(captures) != set(range(32)):
                    raise RuntimeError('Missing native Qwen3.5-4B pre-RMSNorm layer activation')
                matrix = torch.stack([captures[i] for i in range(32)], dim=1).double()
                if tuple(matrix.shape) != (len(fixed), 32, 2560):
                    raise RuntimeError('Unexpected train-only hidden statistics size')
                total += matrix.mean(-1, keepdim=True)
                square += matrix.square().mean(-1, keepdim=True)
    finally:
        for handle in hooks:
            handle.remove()
    mean = (total / sample_count).float()
    std = (square / sample_count - (total / sample_count).square()).clamp_min(0).sqrt().float()
    generator = torch.Generator(device='cpu').manual_seed(42)
    virtual = mean + std * torch.randn((len(fixed), 32, 2560), generator=generator)
    native_stats = (save_native_statistics(case, receiver, data_root, chosen,
                                          prefix, suffix, fixed, mean, std,
                                          native_statistics_root)
                    if native_statistics_root is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.pending')
    save_file({'virtual_hidden': virtual.contiguous(),
               'template_token_ids': torch.tensor(fixed, dtype=torch.int64)}, tmp,
              metadata={'format': 'pt', 'task': case.task, 'sample_count': str(sample_count)})
    tmp.replace(output)
    identity = {'complete': True, 'task': case.task, 'selection_seed': 42,
                'sample_indices': [row['index'] for row in chosen],
                'train_sha256': sha256(Path(data_root) / case.task / 'train.jsonl'),
                'statistics_source': 'frozen Qwen3.5-4B; 500 train questions only',
                'checkpoint_sha256': sha256(output)}
    if native_stats is not None:
        identity['native_train_only_statistics_sha256'] = sha256(native_stats / 'statistics.pt')
    identity_path = output.with_suffix(output.suffix + '.manifest.json')
    identity_path.write_text(json.dumps(identity, sort_keys=True) + '\n')
    return identity
