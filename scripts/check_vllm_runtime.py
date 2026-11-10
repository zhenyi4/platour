"""Run a small real FA2 decode kernel before starting expensive training.

Run this script with the generation environment's Python and the same CUDA
visibility/library environment as the intended decoder. No models are loaded.
This checks a common failure point, not the complete model-specific runtime.
"""
import argparse
import importlib.metadata
import json
import sys


def check(device_index=0):
    version = importlib.metadata.version('vllm')
    if version != '0.17.1':
        raise RuntimeError(f'Expected vLLM 0.17.1; found {version}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is visible to the generation Python')
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    torch.cuda.set_device(device_index)
    device = torch.device('cuda', device_index)
    generator = torch.Generator(device=device).manual_seed(42)
    q = torch.randn(1, 16, 128, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(32, 8, 128, generator=generator, device=device, dtype=torch.bfloat16)
    v = torch.randn(32, 8, 128, generator=generator, device=device, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 32], device=device, dtype=torch.int32)
    output = flash_attn_varlen_func(q, k, v, max_seqlen_q=1, cu_seqlens_q=cu_q,
        max_seqlen_k=32, cu_seqlens_k=cu_k, causal=True, fa_version=2)
    torch.cuda.synchronize(device)
    # A single final-position query attends to all keys; compare against FP32 GQA.
    keys = k.float().repeat_interleave(2, dim=1)
    values = v.float().repeat_interleave(2, dim=1)
    attention = torch.einsum('qhd,khd->qhk', q.float(), keys) / (128 ** .5)
    reference = torch.einsum('qhk,khd->qhd', attention.softmax(-1), values)
    torch.testing.assert_close(output.float(), reference, atol=.02, rtol=.02)
    return dict(complete=True, vllm=version, torch=torch.__version__,
                torch_cuda=torch.version.cuda,
                capability=list(torch.cuda.get_device_capability(device)),
                check='flash_attention_2_bfloat16_gqa_decode',
                max_abs_error=float((output.float() - reference).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0, help='Index within CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    try:
        result = check(args.device)
    except Exception as error:
        print(json.dumps(dict(complete=False, error=str(error))), file=sys.stderr)
        print('Check the decoder Python, GPU visibility and driver/toolkit compatibility. '
              'For unsupported PTX, consult the NVIDIA forward-compatibility guide linked '
              'in README.md; changing Torch alone may not change the vLLM wheel kernels.',
              file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
