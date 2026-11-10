"""Evaluate the held-out-validation-selected native LoRA adapter, never merged full-training weights."""
import argparse
from pathlib import Path

from lora4b_core import LoRADecoder, adapter_digest, base_digest, generate
from lora4b_train import read, sha
from qwen35_bridge_opd_core import ROOT, EVAL_SAMPLING, save
from qwen35_native_actor_grpo import load_bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    import os
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read(args.root / 'protocol.json')
    directory = args.root / f'train/checkpoints/step-{args.checkpoint_step:04d}'
    metadata = read(directory / 'checkpoint.json')
    if not metadata['complete'] or sha(directory / 'adapter.pt') != metadata['adapter_sha256']:
        raise RuntimeError('Selected LoRA test checkpoint is incomplete or changed')
    actor, _ = load_bridge(ROOT, protocol, directory / 'adapter.pt')
    actor.eval()
    if adapter_digest(actor) != metadata['adapter_state_sha256'] or base_digest(actor) != metadata['frozen_base_sha256']:
        raise RuntimeError('LoRA test must load the selected adapter and original frozen base exactly')
    plan = read(args.output / 'plan.json')
    context = 34816 if plan['dataset'] == 'dapo' else 18432
    decoder = LoRADecoder(args.output / 'generation' / f'rank-{args.rank:02d}', args.rank,
                         context=context, protocol=args.root / 'protocol.json')
    try:
        for benchmark in plan['benchmarks']:
            root = args.output / benchmark
            config = read(root / 'protocol.json')
            prompts = read(root / 'prompts.json')
            samples = config['samples_per_question']
            requests = [dict(row, sample_index=j, sampling_seed=1234 + samples * row['index'] + j)
                        for row in prompts[args.rank::8] for j in range(samples)]
            target = root / 'generation' / f'rank-{args.rank:02d}'
            generate(actor, decoder, requests, EVAL_SAMPLING, config['max_new_tokens'], target)
            save(target / 'complete.json', dict(complete=True, responses=len(requests),
                 adapter_state_sha256=metadata['adapter_state_sha256'], checkpoint_step=args.checkpoint_step))
    finally:
        decoder.close()


if __name__ == '__main__':
    main()
