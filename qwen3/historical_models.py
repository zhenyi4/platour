"""Original dense/FP8 Qwen3 SFT loading, without source-machine imports."""
from importlib.metadata import PackageNotFoundError, version
import platform
import os
import types
from unittest.mock import patch

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .bridge import Bridge
from .checkpoints import atomic_json


def configure_math():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def fp8_device_map(primary, secondary):
    if primary == secondary:
        raise ValueError('Historical 235B-FP8 sender requires two distinct GPUs')
    placement = {'model.embed_tokens': primary, 'model.rotary_emb': primary,
                 'model.norm': secondary, 'lm_head': secondary}
    placement.update({f'model.layers.{i}': primary if i < 47 else secondary for i in range(94)})
    return placement


def runtime_versions():
    libraries = {}
    for package in ('torch', 'transformers', 'vllm', 'triton', 'accelerate', 'safetensors'):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            if package != 'vllm':
                raise
            libraries[package] = 'separate-generation-runtime'
    return dict(python=platform.python_version(), **libraries)


def _expert_class():
    try:
        from transformers.integrations.finegrained_fp8 import FP8Expert
    except ImportError:
        from transformers.integrations.finegrained_fp8 import FP8Experts as FP8Expert
    return FP8Expert


def _accelerate_experts(teacher):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    expert_class = _expert_class()

    def forward(module, hidden_states, top_k_index, top_k_weights):
        if torch.is_grad_enabled():
            raise RuntimeError('Frozen FP8 sender must execute under no_grad')
        with torch.cuda.device(hidden_states.device):
            return fused_experts(hidden_states.contiguous(), module.gate_up_proj, module.down_proj,
                top_k_weights.float().contiguous(), top_k_index.int().contiguous(),
                quant_config=fp8_w8a8_moe_quant_config(module.gate_up_proj_scale_inv,
                    module.down_proj_scale_inv, block_shape=module.block_size))
    count = 0
    for module in teacher.modules():
        if isinstance(module, expert_class):
            if module.gate_up_proj.dtype != torch.float8_e4m3fn or module.down_proj.dtype != torch.float8_e4m3fn:
                raise ValueError('Historical expert weights must stay FP8 E4M3')
            module.forward = types.MethodType(forward, module)
            count += 1
    if count != 94:
        raise ValueError(f'Expected 94 FP8 expert modules, got {count}')
    return count


FP8_KERNEL_REVISION = '8c178950e8710e26a2210e4e909cd02dc16c8715'


def _load_fp8(sender, primary, secondary):
    if version('vllm') != '0.17.1':
        raise RuntimeError('Historical frozen FP8 expert kernels require vLLM 0.17.1')
    from transformers.integrations import finegrained_fp8 as fp8
    from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING
    _HUB_KERNEL_MAPPING['finegrained-fp8'] = dict(repo_id='kernels-community/finegrained-fp8', revision=FP8_KERNEL_REVISION)
    os.environ.setdefault('TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR', '1')
    if hasattr(fp8, 'load_finegrained_fp8_kernel'):
        # Fail before allocating the large sender if the immutable reference
        # kernel is absent. HF_HUB_CACHE/KERNELS_CACHE must point at its cache.
        fp8.load_finegrained_fp8_kernel()
    # Retain the legacy local-kernel flag for older Transformers; current
    # builds use the explicit pinned reference above. Avoid the redundant
    # model-sized allocator warmup reservation.
    fp8._quantization_kernel = False
    placement = fp8_device_map(primary, secondary)
    with patch('transformers.modeling_utils.caching_allocator_warmup', lambda *a, **k: None):
        teacher = AutoModelForCausalLM.from_pretrained(sender, dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True, device_map=placement).eval().requires_grad_(False)
    scales = 0
    for name, parameter in teacher.named_parameters():
        if name.endswith('_scale_inv'):
            parameter.data = parameter.data.float()
            scales += 1
    if not scales:
        raise ValueError('No FP8 scale parameters found in the sender')
    modules = [m for m in teacher.modules() if isinstance(m, _expert_class())]
    if len(modules) != 94:
        raise ValueError('Sender FP8 expert structure differs from the historical model')
    references = []
    with torch.no_grad():
        for module in (modules[0], modules[-1]):
            device = module.gate_up_proj.device
            x = torch.randn(4, 4096, device=device, dtype=torch.bfloat16)
            indices = torch.arange(8, device=device)[None].expand(4, -1).contiguous()
            weights = torch.full((4, 8), 1 / 8, device=device, dtype=torch.float32)
            references.append((module, x, indices, weights, module(x, indices, weights)))
    count = _accelerate_experts(teacher)
    parity = []
    with torch.no_grad():
        for module, x, indices, weights, reference in references:
            actual = module(x, indices, weights)
            relative = float((actual.float() - reference.float()).norm() / reference.float().norm())
            passed = bool(torch.isfinite(actual).all()) and relative < .03
            parity.append(dict(device=str(x.device), relative_l2=relative, passed=passed))
    report = dict(teacher_placement=placement, fp8_kernel_revision=FP8_KERNEL_REVISION,
        fp8_expert_modules=count,
        fp8_scale_parameters=scales, scale_dtype='float32', expert_weights='float8_e4m3fn',
        expert_backend='vLLM fused_experts Triton, dynamic block activation quantization',
        fp8_kernel_parity=parity)
    if not all(p['passed'] for p in parity):
        raise RuntimeError(f'Frozen FP8 expert parity failed: {report}')
    return teacher, report


def load_historical_bridge(sender, receiver, mapping, initialization, *,
                           sender_family, primary=0, secondary=1, audit_path=None):
    """Four training replicas call with (0,1)/(2,3)/(4,5)/(6,7).

    The FP8 sender shares its primary GPU with the receiver; the dense sender
    lives entirely on the secondary GPU. No automatic device-map fallback.
    """
    if initialization != 'zero':
        raise ValueError('New bridge projections require zero initialization')
    if primary == secondary or torch.cuda.device_count() <= max(primary, secondary):
        raise ValueError('Historical paired-GPU loading requires two visible GPUs per replica')
    is_fp8 = sender_family == 'Qwen3-235B-A22B-FP8'
    if not is_fp8 and sender_family != 'Qwen3-8B':
        raise ValueError('Unsupported historical Qwen3 sender')
    source_config = AutoConfig.from_pretrained(sender, local_files_only=True)
    target_config = AutoConfig.from_pretrained(receiver, local_files_only=True)
    if (source_config.num_hidden_layers, source_config.hidden_size) != (94 if is_fp8 else 36, 4096):
        raise ValueError('Downloaded sender differs from the experiment')
    if (target_config.num_hidden_layers, target_config.hidden_size) != (28, 2048):
        raise ValueError('Downloaded receiver differs from the experiment')
    if is_fp8 and getattr(source_config, 'quantization_config', {}).get('quant_method') != 'fp8':
        raise ValueError('235B sender must be the native FP8 checkpoint')
    configure_math()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_device(primary)
    target = AutoModelForCausalLM.from_pretrained(receiver, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, device_map={'': primary})
    if is_fp8:
        source, report = _load_fp8(sender, primary, secondary)
    else:
        source = AutoModelForCausalLM.from_pretrained(sender, dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True, device_map={'': secondary})
        report = dict(teacher_placement={'': secondary}, teacher_precision='bfloat16')
    bridge = Bridge(source, target, mapping, initialization,
                    prefill_layout='fp8_prefix' if is_fp8 else 'dense_prompt').eval()
    tokenizer = AutoTokenizer.from_pretrained(receiver, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError('Historical receiver tokenizer requires a pad token')
    bridge.pad_token_id = tokenizer.pad_token_id
    teacher_tokenizer = AutoTokenizer.from_pretrained(sender, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or tokenizer.chat_template != teacher_tokenizer.chat_template:
        raise ValueError('Sender/receiver tokenizer or chat template differs')
    bridge.runtime_audit = dict(report, versions=runtime_versions(), receiver_device=primary,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        sender_path=str(sender), receiver_path=str(receiver),
        sender_config=source_config.to_dict(), receiver_config=target_config.to_dict(),
        prefill_layout=bridge.prefill_layout, sdpa_backend='math', tf32=False,
        bf16_reduced_precision_reduction=False)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(bridge.runtime_audit, audit_path)
    return bridge, tokenizer
