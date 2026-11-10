"""Export text Full checkpoints and official-backbone PEFT adapters for Hub loading.

These exports contain inference state only. The separately pinned backbone used
for conversion is not asserted to be the backbone revision of a historical run.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import struct

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                   'chat_template.jinja', 'special_tokens_map.json', 'added_tokens.json')


def read_header(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 100_000_000:
            raise ValueError('Unexpected safetensors header size')
        return json.loads(stream.read(length))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def clone_file(source, destination):
    """Copy-on-write when available; never hardlink an original checkpoint."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    library = ctypes.CDLL(None, use_errno=True)
    method = getattr(library, 'clonefile', None)
    if method is not None:
        method.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        method.restype = ctypes.c_int
        if method(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 2_000_000_000:
        raise OSError('Insufficient disk space for a non-shared copy')
    shutil.copyfile(source, destination)


def text_config(base):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(base), local_files_only=True)
    config = getattr(config, 'text_config', config)
    if config.model_type != 'qwen3_5_text':
        raise ValueError('Expected a dense Qwen3.5 text configuration')
    config.architectures = ['Qwen3_5ForCausalLM']
    # The Full artifact explicitly stores both tensors. Preserve both values even
    # when the upstream initializer ties embeddings; fine tuning may untie them.
    config.tie_word_embeddings = False
    config._name_or_path = ''
    return config


def copy_tokenizer(base, output):
    base, output = Path(base), Path(output)
    if not (base / 'tokenizer.json').is_file() or not (base / 'tokenizer_config.json').is_file():
        raise FileNotFoundError('Base tokenizer files are required')
    for name in TOKENIZER_FILES:
        if (base / name).is_file():
            clone_file(base / name, output / name)


def export_full(checkpoint, base, output):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh export directory')
    source = read_header(checkpoint)
    if source.pop('__metadata__', {}):
        raise ValueError('Full input must have no operational checkpoint metadata')
    config = text_config(base)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    if set(source) != set(expected) or any(source[name]['shape'] != shape for name, shape in expected.items()):
        raise ValueError('Full tensor schema differs from the text model configuration')
    if {value['dtype'] for value in source.values()} != {'BF16'}:
        raise ValueError('Expected the original BF16 Full inference tensors')
    del model
    output.mkdir(parents=True)
    config.save_pretrained(output)
    generation = GenerationConfig.from_model_config(config)
    generation.pad_token_id = config.eos_token_id
    generation.save_pretrained(output)
    clone_file(checkpoint, output / 'model.safetensors')
    copy_tokenizer(base, output)
    if (Path(base) / 'LICENSE').is_file():
        clone_file(Path(base) / 'LICENSE', output / 'LICENSE')
    return {'format': 'transformers_text_model', 'tensor_count': len(source),
            'weight': 'model.safetensors', 'tensor_bytes_unchanged': True}


def adapter_name(name, *, official_multimodal=True):
    if not re.fullmatch(r'model\.layers\.\d+\..+\.lora_[AB]', name):
        raise ValueError('Unexpected custom LoRA tensor name: ' + name)
    prefix = 'base_model.model.'
    if official_multimodal:
        name = 'model.language_model.' + name[len('model.'):]
    return prefix + name + '.weight'


def custom_lora_name(name):
    """Map a standard PEFT text-decoder key back to the project actor schema."""
    match = re.fullmatch(r'base_model\.model\.model\.(?:language_model\.)?'
                         r'(layers\.\d+\..+\.lora_[AB])\.weight', name)
    if match is None:
        raise ValueError('Unexpected PEFT text-decoder tensor name: ' + name)
    return 'model.' + match.group(1)


def export_lora(checkpoint, base, output, *, base_model_id, revision):
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    import torch
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh export directory')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', base_model_id) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A public base-model ID and exact Hub revision are required')
    config = AutoConfig.from_pretrained(str(base), local_files_only=True)
    official_multimodal = config.model_type == 'qwen3_5'
    if not official_multimodal and config.model_type != 'qwen3_5_text':
        raise ValueError('Expected the official Qwen3.5 dense backbone')
    with torch.device('meta'):
        backbone = (AutoModelForImageTextToText.from_config(config) if official_multimodal
                    else AutoModelForCausalLM.from_config(config))
    tensors, targets = {}, set()
    with safe_open(checkpoint, framework='pt', device='cpu') as handle:
        names = list(handle.keys())
        if len(names) != 496:
            raise ValueError('Expected 248 decoder-linear LoRA pairs')
        for name in names:
            value = handle.get_tensor(name)
            target, suffix = name.rsplit('.', 1)
            if suffix == 'lora_A' and value.shape[0] != 128 or suffix == 'lora_B' and value.shape[1] != 128:
                raise ValueError('Expected rank-128 adapter tensors')
            if str(value.dtype) != 'torch.float32':
                raise ValueError('Expected original FP32 LoRA tensors')
            module_name = ('model.language_model.' + target[len('model.'):]) if official_multimodal else target
            module = backbone.get_submodule(module_name)
            expected_shape = ((128, module.in_features) if suffix == 'lora_A'
                              else (module.out_features, 128))
            if not isinstance(module, torch.nn.Linear) or tuple(value.shape) != expected_shape:
                raise ValueError('LoRA tensor shape differs from the backbone target: ' + name)
            tensors[adapter_name(name, official_multimodal=official_multimodal)] = value
            targets.add(module_name)
    if len(targets) != 248 or any(target + '.lora_A' not in names or target + '.lora_B' not in names
                                  for target in {name.rsplit('.', 1)[0] for name in names}):
        raise ValueError('LoRA pairs are incomplete')
    del backbone
    output.mkdir(parents=True)
    save_file(tensors, output / 'adapter_model.safetensors', metadata={'format': 'pt'})
    # Full module paths restrict adapters to the text decoder, excluding vision.
    adapter = LoraConfig(base_model_name_or_path=base_model_id, revision=revision,
        task_type='CAUSAL_LM', r=128, lora_alpha=128, lora_dropout=0., bias='none',
        target_modules=sorted(targets), inference_mode=True)
    adapter.save_pretrained(output)
    return {'format': 'peft_adapter', 'tensor_count': len(tensors),
            'target_count': len(targets), 'weight': 'adapter_model.safetensors',
            'tensor_values_unchanged': True, 'base_model_id': base_model_id,
            'base_revision': revision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['full', 'lora'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-model-id')
    parser.add_argument('--revision')
    args = parser.parse_args()
    if args.kind == 'full':
        result = export_full(args.checkpoint, args.base, args.output)
    else:
        if not args.base_model_id or not args.revision:
            parser.error('LoRA requires --base-model-id and --revision')
        result = export_lora(args.checkpoint, args.base, args.output,
            base_model_id=args.base_model_id, revision=args.revision)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
