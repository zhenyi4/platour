"""Original FP8 GLM-5.2 prompt sender -> frozen BF16 Qwen3.5-4B receiver.

Eight GPUs hold one sender with contiguous layer partitions. Transfer uses
character-aligned BF16 block inputs; receiver decoding uses its native vLLM cache.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch

from gemma4_qwen35_math_bridge import GemmaQwenHiddenBridge


def sender_device_map(devices, layer_split, depth=78):
    if len(devices)!=len(layer_split) or len(set(devices))!=len(devices):
        raise ValueError('Need one distinct device for each sender layer partition')
    if not devices or any(n<=0 for n in layer_split) or sum(layer_split)!=depth:
        raise ValueError('Partitions must cover exactly the sender decoder depth')
    placement={'model.embed_tokens':devices[0], 'model.rotary_emb':devices[0],
               'model.norm':devices[-1], 'lm_head':devices[-1]}
    offset=0
    for device,count in zip(devices,layer_split,strict=True):
        placement.update({f'model.layers.{j}':device for j in range(offset,offset+count)})
        offset+=count
    return placement


class GlmQwenHiddenBridge(GemmaQwenHiddenBridge):
    """Reuse the tested cross-tokenizer prompt-only injection/gradient contract."""
    def __init__(self,*args,teacher_feature_cache_bytes=4*2**30,**kwargs):
        super().__init__(*args,**kwargs)
        if teacher_feature_cache_bytes<0:
            raise ValueError('Feature cache size must be nonnegative')
        self.teacher_feature_cache_bytes=teacher_feature_cache_bytes
        self._teacher_feature_cache=OrderedDict()
        self._teacher_feature_cache_size=0
        self.teacher_feature_cache_hits=0
        self.teacher_feature_cache_misses=0

    @torch.no_grad()
    def _features(self,prompt):
        # Cache only native frozen sender states, never receiver states or W(h_t).
        # W may change every update without invalidating these tensors.
        assert not self.teacher.training
        row=self._alignment(prompt)
        if not self.teacher_feature_cache_bytes or row.get('synthetic_stress_only'):
            return super()._features(prompt)
        key=tuple(prompt)
        if key in self._teacher_feature_cache:
            cpu_features,cpu_mask,size=self._teacher_feature_cache.pop(key)
            self._teacher_feature_cache[key]=(cpu_features,cpu_mask,size)
            self.teacher_feature_cache_hits+=1
            return (tuple(h.to(self.student.device) for h in cpu_features),
                    cpu_mask.to(self.student.device))
        self.teacher_feature_cache_misses+=1
        features,mask=super()._features(prompt)
        size=sum(h.numel()*h.element_size() for h in features)+mask.numel()*mask.element_size()
        if size<=self.teacher_feature_cache_bytes:
            while self._teacher_feature_cache_size+size>self.teacher_feature_cache_bytes:
                _,(_,_,old_size)=self._teacher_feature_cache.popitem(last=False)
                self._teacher_feature_cache_size-=old_size
            # Rollout runs under inference_mode. Save ordinary detached tensors
            # so subsequent training can safely use them to compute W gradients.
            with torch.inference_mode(False),torch.no_grad():
                cpu_features=tuple(h.detach().to('cpu').clone() for h in features)
                cpu_mask=mask.detach().to('cpu').clone()
            self._teacher_feature_cache[key]=(cpu_features,cpu_mask,size)
            self._teacher_feature_cache_size+=size
        return features,mask

    def metadata(self):
        return dict(super().metadata(),sender='zai-org/GLM-5.2-FP8',
                    teacher_precision='fp8',teacher_hidden_transfer_dtype='bfloat16',
                    sender_attention='DeepSeek Sparse Attention with cross-layer index sharing',
                    sender_mtp_used=False,
                    teacher_feature_cache='CPU LRU of frozen BF16 block inputs; independent of projector state',
                    teacher_feature_cache_bytes=self.teacher_feature_cache_bytes)


def load_bridge(run_root, protocol, devices):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear
    from qwen35_122b_4b_math_bridge import load_fp8_kernel, promote_fp8_scales
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student

    assert protocol['teacher_family']=='glm52' and protocol['method']=='grpo'
    assert len(devices)==8 and devices==list(range(devices[0],devices[0]+8))
    assert devices[0]==torch.cuda.current_device()
    run_root=Path(run_root)
    teacher_path=Path(protocol['teacher_checkpoint'])
    # Refuse a partial 756GB download before allocating any GPU model weights.
    expected=protocol['teacher_expected_files']
    incomplete=[name for name,size in expected.items()
                if not (teacher_path/name).is_file() or (teacher_path/name).stat().st_size!=size]
    if incomplete:
        raise RuntimeError(f'GLM download incomplete ({len(incomplete)} files); finish download first')
    audit=json.loads((run_root/'baseline_comparison.json').read_text())
    fingerprints={run_root/name:digest for name,digest in audit['data_sha256'].items()}
    fingerprints[run_root/'token_alignments.json']=audit['token_alignments_sha256']
    fingerprints.update({Path(name):digest for name,digest in protocol['model_tokenizer_sha256'].items()})
    for path,digest in fingerprints.items():
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, f'Prepared asset changed: {path}'

    config=AutoConfig.from_pretrained(teacher_path,local_files_only=True)
    assert config.model_type=='glm_moe_dsa' and config.num_hidden_layers==78 and config.hidden_size==6144
    assert config.vocab_size==154880
    quant=config.quantization_config
    assert quant['quant_method']=='fp8' and quant['weight_block_size']==[128,128]
    assert quant['activation_scheme']=='dynamic'
    runtime=accelerated_models()
    load_fp8_kernel()  # Existing pinned Triton kernels, available offline.
    student=load_student(Path(protocol['student_checkpoint']))
    placement=sender_device_map(devices,protocol['teacher_layer_split'])
    teacher,loading=AutoModelForCausalLM.from_pretrained(teacher_path,dtype=torch.bfloat16,
        attn_implementation='sdpa',experts_implementation='grouped_mm',
        local_files_only=True,device_map=placement,output_loading_info=True)
    assert not loading.get('missing_keys'), loading
    assert not loading.get('mismatched_keys') and not loading.get('error_msgs'), loading
    assert all(k.startswith('model.layers.78.') for k in loading.get('unexpected_keys',[])), loading
    # The official checkpoint includes an MTP block at index78. HF's native
    # causal decoder deliberately loads only its 78 ordinary blocks (indices0–77).
    teacher.eval().requires_grad_(False)
    teacher.set_experts_implementation('grouped_mm')
    scale_count=promote_fp8_scales(teacher)
    assert not any(p.is_meta for p in teacher.parameters())
    assert {p.device.index for p in teacher.parameters()}==set(devices)
    assert {p.device.index for p in student.parameters()}=={devices[0]}
    assert len(teacher.model.layers)==78
    experts=[m for m in teacher.modules() if isinstance(m,FP8Experts)]
    linears=[m for m in teacher.modules() if isinstance(m,FP8Linear)]
    assert len(experts)==75 and linears  # First three GLM blocks use dense MLPs.
    assert all(m.gate_up_proj.dtype==m.down_proj.dtype==torch.float8_e4m3fn for m in experts)
    assert all(m.gate_up_proj_scale_inv.dtype==m.down_proj_scale_inv.dtype==torch.float32 for m in experts)
    assert all(m.weight.dtype==torch.float8_e4m3fn for m in linears)
    alignments=json.loads((run_root/'token_alignments.json').read_text())
    bridge=GlmQwenHiddenBridge(student,teacher,alignments,
        max_teacher_prompt_length=protocol['max_teacher_prompt_length'],
        teacher_input_norm=protocol['teacher_input_norm'],
        teacher_input_norm_eps=protocol['teacher_input_norm_eps'],
        projection_init=protocol['projection_init'],
        teacher_feature_cache_bytes=protocol.get('teacher_feature_cache_bytes',4*2**30)).eval()
    assert bridge.layer_map==protocol['layer_map']
    assert sum(p.numel() for p in bridge.bridge_parameters())==protocol['trainable_parameters']
    allocation={device:sum(p.numel()*p.element_size() for p in teacher.parameters()
                          if p.device.index==device) for device in devices}
    runtime.update(teacher='zai-org/GLM-5.2-FP8',teacher_class=type(teacher).__name__,
        teacher_devices=devices,teacher_layer_split=protocol['teacher_layer_split'],
        teacher_device_map=placement,teacher_tensor_parallel_size=1,
        teacher_model_parallel_size=8,receiver_replicas=1,
        teacher_parameter_bytes_by_device=allocation,
        teacher_parameter_dtypes=sorted({str(p.dtype) for p in teacher.parameters()}),
        teacher_precision='official FP8 E4M3 block128 with dynamic activations; no CPU offload',
        teacher_experts='FP8 grouped_mm (pinned finegrained-fp8 Triton)',
        fp8_expert_modules=len(experts),fp8_linear_modules=len(linears),
        fp8_scale_parameters=scale_count,fp8_scale_cast='BF16/FP16 scales widened to FP32; weights remain FP8',
        teacher_hidden_transfer_dtype='bfloat16',teacher_attention='native DSA with index sharing, SDPA',
        teacher_mtp_used=False,different_tokenizers=True,
        token_alignment=bridge.metadata()['tokenizer_alignment'])
    return bridge,runtime
