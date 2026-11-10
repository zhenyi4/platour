"""Frozen model-parallel 122B/397B encoder; BF16 4B decoder."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from qwen35_direct_math_bridge import accelerated_models
from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge
from qwen35_student import load_student

FP8_KERNEL_REVISION='8c178950e8710e26a2210e4e909cd02dc16c8715'


def load_fp8_kernel():
    # Use an immutable revision so kernel loading works without Hub version
    # resolution during the offline training subprocesses.
    from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
    _HUB_KERNEL_MAPPING['finegrained-fp8']={
        'repo_id':'kernels-community/finegrained-fp8','revision':FP8_KERNEL_REVISION}
    return load_finegrained_fp8_kernel()


def promote_fp8_scales(teacher):
    # Exact widening of the official checkpoint's BF16 scales for FP8 kernels.
    count=0
    for name,p in teacher.named_parameters():
        if name.endswith('_scale_inv'):
            assert p.dtype in (torch.bfloat16,torch.float16,torch.float32)
            p.data=p.data.to(torch.float32)
            count+=1
    assert count>0
    return count


def relative_depth_layer_map(student_layers,teacher_layers,group_size=4):
    """Nearest normalized group depth, including both depth endpoints.

    Round half up with integer arithmetic; preserve each block's position
    inside the hybrid attention group. Layer numbers are zero based.
    """
    if group_size<1 or student_layers%group_size or teacher_layers%group_size:
        raise ValueError('Both depths must contain complete attention groups')
    students,teachers=student_layers//group_size,teacher_layers//group_size
    if students<2 or teachers<students:
        raise ValueError('Need at least two student groups and a deeper teacher')
    groups=[(2*i*(teachers-1)+(students-1))//(2*(students-1))
            for i in range(students)]
    return [group_size*j+k for j in groups for k in range(group_size)]


class PairedLastAlignedBridge(LastAlignedResidualBridge):
    def __init__(self,student,teacher,*,layer_mapping='last_aligned',lm_chunk_tokens=128):
        super().__init__(student,teacher,lm_chunk_tokens=lm_chunk_tokens)
        if layer_mapping=='relative_depth_groups4':
            self.layer_map=relative_depth_layer_map(student.config.num_hidden_layers,
                                                   teacher.config.num_hidden_layers)
            self.assignment=[[layer] for layer in self.layer_map]
        elif layer_mapping!='last_aligned':
            raise ValueError(f'Unknown layer mapping: {layer_mapping}')
        self.layer_mapping=layer_mapping
        assert all(student.config.layer_types[i]==teacher.config.layer_types[j]
                   for i,j in enumerate(self.layer_map))

    def metadata(self):
        result=dict(super().metadata(),layer_mapping=self.layer_mapping)
        if self.layer_mapping=='relative_depth_groups4':
            result.update(group_size=4,teacher_group_map=[j//4 for j in self.layer_map[::4]],
                mapping_rule='round_half_up(student_group*(teacher_groups-1)/(student_groups-1)); endpoints included')
        return result

    def teacher_features(self,hidden):
        # Teacher block inputs live on both GPUs; student projections and their
        # gradients always live on the decoder GPU. No teacher gradients.
        return tuple(h.to(self.student.device) for h in super().teacher_features(hidden))


class PairedFP8Bridge(PairedLastAlignedBridge):
    def teacher_features(self,hidden):
        # FP8 is a weight/compute format. Transfer the resulting activations in
        # BF16; never cast the quantized teacher's weight tensors to BF16.
        return tuple(h.to(device=self.student.device,dtype=torch.bfloat16)
                     for h in super().teacher_features(hidden))

    def metadata(self):
        return dict(super().metadata(),teacher_precision='fp8',
                    teacher_hidden_transfer_dtype='bfloat16')


def teacher_device_map(primary,secondary,split=16):
    assert primary!=secondary and split==16
    result={'model.visual':secondary,'model.language_model.embed_tokens':primary,
        'model.language_model.rotary_emb':primary,'model.language_model.norm':secondary,
        'lm_head':secondary}
    result.update({f'model.language_model.layers.{i}':primary if i<split else secondary
                   for i in range(48)})
    return result


def partitioned_teacher_device_map(devices,layer_split):
    """Contiguous decoder blocks, with embeddings and final norm at endpoints."""
    if len(devices)!=len(layer_split) or len(set(devices))!=len(devices):
        raise ValueError('One distinct device per layer partition is required')
    if not devices or any(n<=0 for n in layer_split):
        raise ValueError('Teacher partitions must be nonempty')
    result={'model.visual':devices[-1],'model.language_model.embed_tokens':devices[0],
        'model.language_model.rotary_emb':devices[0],'model.language_model.norm':devices[-1],
        'lm_head':devices[-1]}
    offset=0
    for device,count in zip(devices,layer_split,strict=True):
        result.update({f'model.language_model.layers.{i}':device for i in range(offset,offset+count)})
        offset+=count
    return result


def load_bridge(root,primary,secondary,layer_mapping='last_aligned',teacher_precision='bf16',
                *,teacher_name='Qwen3.5-122B-A10B',devices=None,layer_split=None):
    assert primary==torch.cuda.current_device() and secondary==primary+1
    assert teacher_precision in ('bf16','fp8')
    if teacher_name=='Qwen3.5-122B-A10B':
        teacher_layers,teacher_width=48,3072
        devices=[primary,secondary] if devices is None else list(devices)
        layer_split=[16,32] if layer_split is None else list(layer_split)
        assert devices==[primary,secondary] and layer_split==[16,32]
    else:
        assert teacher_name=='Qwen3.5-397B-A17B' and teacher_precision=='fp8'
        assert layer_mapping=='relative_depth_groups4'
        teacher_layers,teacher_width=60,4096
        devices=list(devices)
        layer_split=list(layer_split)
        assert devices==list(range(primary,primary+4)) and layer_split==[12,16,16,16]
    assert sum(layer_split)==teacher_layers
    runtime=accelerated_models()
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as moe
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
    moe.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    moe.torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule
    student=load_student(root/'data/models/Qwen3.5-4B')
    placement=partitioned_teacher_device_map(devices,layer_split)
    teacher_path=root/'data/models'/(teacher_name+'-FP8' if teacher_precision=='fp8' else teacher_name)
    if teacher_precision=='fp8':
        quant=json.loads((teacher_path/'config.json').read_text())['quantization_config']
        assert quant['quant_method']=='fp8' and quant['weight_block_size']==[128,128]
        assert quant['activation_scheme']=='dynamic'
        load_fp8_kernel()  # Already downloaded, also available offline.
    full=AutoModelForImageTextToText.from_pretrained(teacher_path,
        dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm',
        local_files_only=True,device_map=placement)
    with torch.device('meta'):
        teacher=AutoModelForCausalLM.from_config(full.config.text_config)
    teacher.model=full.model.language_model
    teacher.lm_head=full.lm_head
    teacher.generation_config=full.generation_config
    if teacher_precision=='fp8':
        teacher.hf_quantizer=full.hf_quantizer
        teacher.is_quantized=full.is_quantized
    teacher.set_experts_implementation('grouped_mm')
    teacher.eval().requires_grad_(False)
    if teacher_precision=='fp8':
        fp8_scale_count=promote_fp8_scales(teacher)
    del full
    torch.cuda.empty_cache()
    assert type(teacher).__name__=='Qwen3_5MoeForCausalLM'
    assert not any(p.is_meta for p in teacher.parameters())
    assert {p.device.index for p in teacher.parameters()}==set(devices)
    assert {p.device.index for p in student.parameters()}=={primary}
    assert teacher.config.num_hidden_layers==teacher_layers and student.config.num_hidden_layers==32
    assert teacher.config.hidden_size==teacher_width and student.config.hidden_size==2560
    assert teacher.config.layer_types[-32:]==student.config.layer_types
    bridge_class=PairedFP8Bridge if teacher_precision=='fp8' else PairedLastAlignedBridge
    bridge=bridge_class(student,teacher,layer_mapping=layer_mapping).eval()
    expected=(relative_depth_layer_map(32,teacher_layers) if layer_mapping=='relative_depth_groups4'
              else list(range(teacher_layers-32,teacher_layers)))
    assert bridge.layer_map==expected
    assert sum(p.numel() for p in bridge.bridge_parameters())==32*teacher_width*2560
    assert all(torch.count_nonzero(p)==0 for p in bridge.bridge_parameters())
    runtime.update(teacher_layers=teacher_layers,student_layers=32,layer_mapping=layer_mapping,
        layer_map=bridge.layer_map,teacher_class=type(teacher).__name__,teacher_experts='grouped_mm',
        teacher_grouped_mm='torch.nn.functional.grouped_mm',
        teacher_linear_prefill='FLA chunk_gated_delta_rule',student_class=type(student).__name__,
        teacher_tensor_parallel_size=1,teacher_model_parallel_size=len(devices),
        teacher_layer_split=layer_split,teacher_device_map=placement,
        student_device=primary,teacher_devices=devices,
        teacher_precision='original BF16, no quantization or CPU offload')
    if teacher_precision=='fp8':
        from transformers.integrations.finegrained_fp8 import FP8Experts,FP8Linear
        experts=[m for m in teacher.modules() if isinstance(m,FP8Experts)]
        linears=[m for m in teacher.modules() if isinstance(m,FP8Linear)]
        assert len(experts)==teacher_layers and linears
        assert all(m.gate_up_proj.dtype==m.down_proj.dtype==torch.float8_e4m3fn for m in experts)
        assert all(m.gate_up_proj_scale_inv.dtype==m.down_proj_scale_inv.dtype==torch.float32 for m in experts)
        assert all(m.weight.dtype==torch.float8_e4m3fn for m in linears)
        assert not any(p.requires_grad for p in teacher.parameters())
        runtime.update(teacher_precision='official FP8 E4M3 block128 weights with dynamic activation quantization; no CPU offload',
            quantization={'teacher':'official FP8 block128, dynamic activations','student':None},
            teacher_checkpoint=str(teacher_path),teacher_hidden_transfer_dtype='bfloat16',
            teacher_experts='FP8 grouped_mm (Triton finegrained-fp8)',
            teacher_grouped_mm='finegrained-fp8 grouped_matmul',
            fp8_expert_modules=len(experts),fp8_linear_modules=len(linears),
            fp8_scale_parameters=fp8_scale_count,
            fp8_scale_cast='checkpoint BF16 scales widened exactly to FP32; weights remain FP8',
            teacher_parameter_dtypes=sorted({str(p.dtype) for p in teacher.parameters()}))
    return bridge,runtime
