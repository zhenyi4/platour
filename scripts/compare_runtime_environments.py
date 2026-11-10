"""Exercise real PLatour GPU paths and compare two Python runtime configurations.

Run the same model snapshots and seed in a reference HF environment (passing its
decoder Python) and a candidate combined environment (omit --decoder-python).
This is a bounded compatibility check, not a training or benchmark reproduction.
Large tensor artifacts belong on a scratch/data volume.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    ('What is 17 + 25? Answer briefly.', '42'),
    ('A box contains 6 bags with 7 marbles each. How many marbles are there? Answer briefly.', '42'),
    ('Solve 3*x + 5 = 26. Give the value of x.', '7'),
    ('Compute the sum of the integers from 1 through 20. Give a short answer.', '210'),
    ('Remember: red=3, blue=5, green=7. ' * 20 + 'What is red + blue + green? Answer briefly.', '15'),
    ('A rectangle has perimeter 30 and length 9. What is its area? Answer briefly.', '54'),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def versions():
    result = {'python': sys.version, 'executable': sys.executable}
    for package in ('torch', 'transformers', 'tokenizers', 'vllm', 'fla-core',
                    'flash-linear-attention', 'causal-conv1d', 'tilelang', 'triton'):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def exercise(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    receiver = args.models / ('Qwen3-1.7B' if args.family == 'qwen3' else 'Qwen3.5-4B')
    sender = args.models / (args.sender or ('Qwen3-8B' if args.family == 'qwen3' else 'Qwen3.5-35B-A3B'))
    tokenizer = AutoTokenizer.from_pretrained(receiver, local_files_only=True)
    rows = []
    for i, (question, answer) in enumerate(QUESTIONS):
        ids = tokenizer.apply_chat_template([dict(role='user', content=question)],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        rows.append(dict(index=i, question=question, gold=answer, prompt_token_ids=ids,
                         response_token_ids=tokenizer.encode(answer, add_special_tokens=False)))
    save(args.output/'inputs.json', rows)
    print('Loading', sender, receiver, flush=True)
    if args.family == 'qwen3':
        from qwen3.bridge import Bridge
        teacher = AutoModelForCausalLM.from_pretrained(sender, dtype=torch.bfloat16,
            device_map='cuda:1', attn_implementation='sdpa', local_files_only=True)
        student = AutoModelForCausalLM.from_pretrained(receiver, dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa', local_files_only=True)
        actor = Bridge(teacher, student, 'last_aligned')
        parameters = list(actor.P.parameters())
    else:
        from qwen35.model import Actor, load_text
        from qwen35.catalog import EXPERIMENTS
        case = next(x for x in EXPERIMENTS if x.identifier == (21 if sender == receiver else 19))
        teacher = load_text(sender, device='cuda:1')
        student = load_text(receiver, device='cuda:0')
        actor = Actor(case, student, teacher)
        parameters = list(actor.trainable())
    # Identical nonzero initialization exercises the transferred memory, not just native decoding.
    generator = torch.Generator(device='cpu').manual_seed(123)
    with torch.no_grad():
        for parameter in parameters:
            bound = 1e-3 / math.sqrt(parameter.shape[-1])
            value = torch.empty(parameter.shape).uniform_(-bound, bound, generator=generator)
            parameter.copy_(value.to(parameter.device))
    report = dict(family=args.family, versions=versions(), sender=str(sender), receiver=str(receiver),
                  decoder_python=args.decoder_python or sys.executable,
                  same_environment=args.decoder_python is None,
                  gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
                  trainable_parameters=sum(p.numel() for p in parameters), steps=[])
    tensors = {}
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        row = rows[step]
        logprobs = actor.response_log_probs(row['prompt_token_ids'], row['response_token_ids'])
        loss = -logprobs.mean()
        loss.backward()
        gradients = [p.grad for p in parameters]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert norm > 0 and torch.isfinite(norm)
        tensors[f'step{step}.logprobs'] = logprobs.detach().cpu()
        for i, gradient in enumerate(gradients):
            tensors[f'step{step}.gradient.{i}'] = gradient.detach().cpu().clone()
        optimizer.step()
        report['steps'].append(dict(loss=loss.item(), gradient_norm=norm.item()))
        print('Training step', step, report['steps'][-1], flush=True)
    for i, parameter in enumerate(parameters):
        tensors[f'updated.{i}'] = parameter.detach().cpu().clone()
    report['frozen_backbones_have_no_gradients'] = all(
        p.grad is None for model in (teacher, student) for p in model.parameters())
    assert report['frozen_backbones_have_no_gradients']
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    torch.cuda.empty_cache()
    save_file(tensors, args.output/'training-tensors.safetensors')
    report['training_tensor_hashes'] = {name: digest(value) for name, value in tensors.items()}
    del tensors
    save(args.output/'training.json', report)
    print('Starting native decoder', report['decoder_python'], flush=True)
    if args.family == 'qwen3':
        from qwen3.vllm_decode import VLLMDecoder, check_parity
        with VLLMDecoder(actor, args.output, max_model_len=1024, memory_fraction=.25,
                         max_num_seqs=8, max_num_batched_tokens=1024,
                         worker_python=args.decoder_python) as decoder:
            report['hf_vllm_parity'] = check_parity(actor, decoder, rows)
            greedy = decoder.generate(rows, count=1, cap=64, parity=True,
                sampling=dict(temperature=0., top_p=1., top_k=-1, min_p=0.,
                              presence_penalty=0., frequency_penalty=0., repetition_penalty=1.))
            sampled = decoder.generate(rows, count=2, cap=64, parity=True,
                sampling=dict(temperature=1., top_p=.95, top_k=20, min_p=0.,
                              presence_penalty=0., frequency_penalty=0., repetition_penalty=1.))
            report['generation'] = dict(greedy=greedy, sampled=sampled)
    else:
        from qwen35.vllm_decode import VLLMDecoder
        weight = args.output/'bridge.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in actor.inference_state().items()}, weight)
        with VLLMDecoder(case, receiver, weight, args.output, actor=actor, memory_fraction=.30,
                         max_model_len=1024, max_num_seqs=8, worker_python=args.decoder_python,
                         weight_source='inference') as decoder:
            sampled = decoder.generate(rows, count=2, cap=64, batch_size=2, training=True)
            report['generation'] = dict(sampled=sampled)
            errors = []
            with torch.no_grad():
                for row, answer in zip(rows[:2], sampled[::2][:2], strict=True):
                    ids = answer['response_token_ids'][:8]
                    expected = actor.response_log_probs(row['prompt_token_ids'], ids).float().cpu()
                    actual = torch.tensor(answer['log_probs'][:len(ids)])
                    errors.extend((expected-actual).abs().tolist())
            report['hf_vllm_parity'] = dict(mean_abs_logprob_error=sum(errors)/len(errors),
                                          max_abs_logprob_error=max(errors), compared_tokens=len(errors))
    report.update(complete=True, elapsed_seconds=time.monotonic()-started,
                  max_gpu_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)])
    save(args.output/'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('generation', 'training_tensor_hashes')}, indent=2))


def compare(args):
    import torch
    from safetensors import safe_open
    left = json.loads((args.reference/'result.json').read_text())
    right = json.loads((args.candidate/'result.json').read_text())
    assert left['complete'] and right['complete'] and left['family'] == right['family']
    assert left['sender'] == right['sender'] and left['receiver'] == right['receiver']
    assert (args.reference/'inputs.json').read_bytes() == (args.candidate/'inputs.json').read_bytes()
    tensor_results = {}
    with safe_open(args.reference/'training-tensors.safetensors', framework='pt') as a, \
         safe_open(args.candidate/'training-tensors.safetensors', framework='pt') as b:
        assert set(a.keys()) == set(b.keys())
        for name in a.keys():
            x, y = a.get_tensor(name), b.get_tensor(name)
            difference = (x.float()-y.float()).abs()
            tensor_results[name] = dict(equal=torch.equal(x,y), max_abs=difference.max().item(),
                                       mean_abs=difference.mean().item())
    generations = {}
    for mode, first in left['generation'].items():
        second = right['generation'][mode]
        assert len(first) == len(second)
        matches, errors, common_tokens = [], [], 0
        for a,b in zip(first,second,strict=True):
            assert (a['index'],a['sample_index'],a['sampling_seed']) == (b['index'],b['sample_index'],b['sampling_seed'])
            matches.append(a['response_token_ids'] == b['response_token_ids'])
            # Logprobs are comparable only on the identical generated prefix.
            for token_a,token_b,lp_a,lp_b in zip(a['response_token_ids'],b['response_token_ids'],a['log_probs'],b['log_probs']):
                if token_a != token_b:
                    break
                errors.append(abs(lp_a-lp_b)); common_tokens += 1
        generations[mode] = dict(responses=len(matches), identical_responses=sum(matches),
            compared_prefix_tokens=common_tokens, mean_abs_logprob_error=sum(errors)/len(errors) if errors else None,
            max_abs_logprob_error=max(errors) if errors else None)
    result = dict(family=left['family'], reference=str(args.reference), candidate=str(args.candidate),
        training_tensors=len(tensor_results), identical_training_tensors=sum(x['equal'] for x in tensor_results.values()),
        max_training_tensor_abs_error=max(x['max_abs'] for x in tensor_results.values()),
        training_steps=dict(reference=left['steps'],candidate=right['steps']),
        generation=generations, hf_vllm_parity=dict(reference=left['hf_vllm_parity'],candidate=right['hf_vllm_parity']),
        tensor_details=tensor_results, scope='two teacher-forced optimizer updates and short native rollouts; no benchmark or full GRPO reproduction')
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k != 'tensor_details'},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('run')
    run.add_argument('--family',choices=['qwen3','qwen35'],required=True)
    run.add_argument('--models',type=Path,required=True)
    run.add_argument('--sender')
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--decoder-python')
    diff = commands.add_parser('compare')
    diff.add_argument('--reference',type=Path,required=True)
    diff.add_argument('--candidate',type=Path,required=True)
    diff.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    exercise(args) if args.command == 'run' else compare(args)


if __name__ == '__main__':
    main()
