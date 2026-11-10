"""Actual local from_pretrained checks; not a benchmark or training reproduction."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exact_loaded_tensors(state, source_path, *, name_map=lambda name: name):
    import torch
    from safetensors import safe_open
    with safe_open(source_path, framework='pt', device='cpu') as handle:
        for name in handle.keys():
            actual = state[name_map(name)].detach().reshape(-1)
            expected = handle.get_tensor(name).reshape(-1)
            if actual.shape != expected.shape:
                raise ValueError('Loaded tensor shape differs: ' + name)
            for start in range(0, actual.numel(), 4_000_000):
                end = start + 4_000_000
                if not torch.equal(actual[start:end].to(device='cpu'), expected[start:end].to(dtype=actual.dtype)):
                    raise ValueError('Loaded tensor values differ: ' + name)
        return len(handle.keys())


def inference(model, tokenizer):
    import torch
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply with exactly the word hello.'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt', return_dict=True)
    model.eval()
    with torch.inference_mode():
        logits = model(**prompt, logits_to_keep=1, use_cache=False).logits.detach().cpu()
        generated = model.generate(**prompt, max_new_tokens=4, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite logits')
    return prompt, logits, generated[:, prompt['input_ids'].shape[1]:].cpu()


def verify_full(directory, original):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model, loading = AutoModelForCausalLM.from_pretrained(str(directory), dtype=torch.float32,
        device_map='cpu', local_files_only=True, output_loading_info=True, attn_implementation='eager')
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Full model load has missing/unexpected/mismatched tensors: ' + str(loading))
    count = exact_loaded_tensors(model.state_dict(), original)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    prompt, logits, answer = inference(model, tokenizer)
    result = {'kind': 'full', 'loader': 'AutoModelForCausalLM.from_pretrained',
        'all_loaded_tensors_equal_original': True, 'tensors_verified': count,
        'loading_info': {key: sorted(loading.get(key, [])) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')},
        'finite_forward_logits': True, 'greedy_generated_token_ids': answer[0].tolist(),
        'greedy_generated_text': tokenizer.decode(answer[0], skip_special_tokens=True),
        'synthetic_prompt_only': True, 'load_dtype': 'float32', 'transformers': transformers.__version__}
    del model
    gc.collect()
    return result


def verify_lora(directory, original, base):
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from qwen35.model import LoRALinear
    from qwen35.hf_export import adapter_name
    backbone, loading = AutoModelForImageTextToText.from_pretrained(str(base), dtype=torch.float32,
        device_map='cpu', local_files_only=True, output_loading_info=True, attn_implementation='eager')
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Base model loading has missing/unexpected/mismatched tensors: ' + str(loading))
    model = PeftModel.from_pretrained(backbone, str(directory), is_trainable=False,
        autocast_adapter_dtype=False, local_files_only=True)
    def mapped(name):
        return adapter_name(name).removesuffix('.weight') + '.default.weight'
    count = exact_loaded_tensors(model.state_dict(), original, name_map=mapped)
    layers = [(name, module) for name, module in backbone.named_modules() if isinstance(module, LoraLayer)]
    if len(layers) != 248 or any(not name.startswith('model.language_model.layers.') for name, _ in layers):
        raise ValueError('Adapters must cover exactly the 248 text decoder targets')
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    prompt, peft_logits, answer = inference(model, tokenizer)
    # Reuse the identical frozen backbone and replace only its adapter modules by
    # the project's original forward implementation and original A/B matrices.
    replacements = []
    try:
        with safe_open(original, framework='pt', device='cpu') as handle:
            for name, module in layers:
                reference = LoRALinear(module.base_layer, rank=128, alpha=128)
                source_name = 'model.' + name[len('model.language_model.'):]
                with torch.no_grad():
                    reference.lora_A.copy_(handle.get_tensor(source_name + '.lora_A'))
                    reference.lora_B.copy_(handle.get_tensor(source_name + '.lora_B'))
                parent, attr = name.rsplit('.', 1)
                owner = backbone.get_submodule(parent)
                replacements.append((owner, attr, module))
                setattr(owner, attr, reference)
        with torch.inference_mode():
            reference_logits = backbone(**prompt, logits_to_keep=1, use_cache=False).logits.detach().cpu()
        difference = (peft_logits - reference_logits).abs().max().item()
        torch.testing.assert_close(peft_logits, reference_logits, rtol=1e-5, atol=1e-5)
    finally:
        for owner, attr, module in replacements:
            setattr(owner, attr, module)
    result = {'kind': 'lora', 'loader': 'PeftModel.from_pretrained',
        'base_loader': 'AutoModelForImageTextToText.from_pretrained',
        'all_loaded_adapter_tensors_equal_original': True, 'tensors_verified': count,
        'text_decoder_targets': len(layers), 'vision_adapter_targets': 0,
        'finite_forward_logits': True, 'original_project_forward_max_abs_logit_difference': difference,
        'original_project_forward_matches': True,
        'greedy_generated_token_ids': answer[0].tolist(),
        'greedy_generated_text': tokenizer.decode(answer[0], skip_special_tokens=True),
        'synthetic_prompt_only': True, 'load_dtype': 'float32',
        'transformers': transformers.__version__, 'peft': peft.__version__}
    del model, backbone, layers, replacements
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['full', 'lora'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Use a fresh report path')
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if args.kind == 'full':
        result = verify_full(args.directory, args.original)
    else:
        if args.base is None:
            parser.error('LoRA needs --base')
        result = verify_lora(args.directory, args.original, args.base)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
