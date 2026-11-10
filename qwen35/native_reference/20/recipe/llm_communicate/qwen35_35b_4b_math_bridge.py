"""Frozen native 35B-A3B encoder -> 4B decoder, last 32 block inputs."""
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from qwen35_direct_math_bridge import accelerated_models, cache_check
from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge
from qwen35_student import load_student


class RelativeDepthResidualBridge(LastAlignedResidualBridge):
    """Use the same normalized four-layer group mapping as the 122B run."""
    def __init__(self, student, teacher, *, lm_chunk_tokens=128):
        from qwen35_122b_4b_math_bridge import relative_depth_layer_map
        super().__init__(student, teacher, lm_chunk_tokens=lm_chunk_tokens)
        self.layer_map = relative_depth_layer_map(student.config.num_hidden_layers,
                                                  teacher.config.num_hidden_layers)
        self.assignment = [[layer] for layer in self.layer_map]
        assert all(student.config.layer_types[i] == teacher.config.layer_types[j]
                   for i, j in enumerate(self.layer_map))

    def metadata(self):
        return dict(super().metadata(), layer_mapping='relative_depth_groups4',
            group_size=4, teacher_group_map=[j//4 for j in self.layer_map[::4]],
            mapping_rule='round_half_up(student_group*(teacher_groups-1)/(student_groups-1)); endpoints included')


class MonotoneFullResidualBridge(LastAlignedResidualBridge):
    def __init__(self, student, teacher, *, lm_chunk_tokens=128):
        from formal_config import layer_map_b
        super().__init__(student, teacher, lm_chunk_tokens=lm_chunk_tokens)
        self.layer_map = layer_map_b(student.config.layer_types, teacher.config.layer_types)
        self.assignment = [[j] for j in self.layer_map]

    def metadata(self):
        from formal_config import MAPPING, RULE
        return dict(super().metadata(), layer_mapping=MAPPING, mapping_rule=RULE,
                    group_size=1, group_constraint=False, monotone_constraint=True,
                    full_layer_constraint=True, nearest_full_tie_break='shallower',
                    attention_types_preserved=False)


def load_teacher(path):
    full = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='sdpa',
        experts_implementation='grouped_mm', local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(full.config.text_config)
    model.model = full.model.language_model
    model.lm_head = full.lm_head
    model.generation_config = full.generation_config
    assert type(model).__name__ == 'Qwen3_5MoeForCausalLM'
    assert not any(p.is_meta for p in model.parameters())
    model.set_experts_implementation('grouped_mm')
    return model.to('cuda').eval().requires_grad_(False)


def load_bridge(root,*,layer_mapping='last_aligned',mapping=None,selection=None):
    runtime = accelerated_models()
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as moe
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    moe.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    moe.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    student = load_student(root / 'data/models/Qwen3.5-4B')
    teacher = load_teacher(root / 'data/models/Qwen3.5-35B-A3B')
    assert student.config.num_hidden_layers == 32
    assert teacher.config.num_hidden_layers == 40
    assert student.config.layer_types == teacher.config.layer_types[-32:]
    assert student.config.hidden_size == 2560 and teacher.config.hidden_size == 2048
    if layer_mapping=='mse_per_student_shared_cosine_argmax':
        from qwen35_mse_argmax_math_bridge import MSEArgmaxBridge
        assert mapping is not None and selection is not None
        bridge=MSEArgmaxBridge(student,teacher,mapping,selection).eval()
        assert bridge.layer_map==mapping and bridge.assignment==[[j] for j in mapping]
    elif layer_mapping=='relative_depth_groups4':
        assert mapping is None and selection is None
        bridge = RelativeDepthResidualBridge(student, teacher).eval()
        assert [j//4 for j in bridge.layer_map[::4]] == [0,1,3,4,5,6,8,9]
    elif layer_mapping=='equal_width_full_to_full_monotone_b':
        assert mapping is None and selection is None
        bridge = MonotoneFullResidualBridge(student, teacher).eval()
    else:
        assert layer_mapping=='last_aligned' and mapping is None and selection is None
        bridge = LastAlignedResidualBridge(student, teacher).eval()
        assert bridge.layer_map == list(range(8, 40))
        assert bridge.assignment == [[i] for i in range(8, 40)]
    assert sum(p.numel() for p in bridge.bridge_parameters()) == 167772160
    assert all(torch.count_nonzero(p) == 0 for p in bridge.bridge_parameters())
    assert teacher.config._experts_implementation == 'grouped_mm'
    runtime.update(teacher_layers=40, student_layers=32,
                   layer_mapping=layer_mapping, layer_map=bridge.layer_map,
                   teacher_class=type(teacher).__name__, teacher_experts='grouped_mm',
                   teacher_grouped_mm='torch.nn.functional.grouped_mm',
                   teacher_linear_prefill='FLA chunk_gated_delta_rule',
                   student_class=type(student).__name__)
    if selection is not None:
        runtime.update(mse_selection=selection,attention_types_preserved=False,
                       teacher_layer_numbers_1based=[j+1 for j in mapping])
    if layer_mapping=='relative_depth_groups4':
        runtime.update(attention_types_preserved=True, mapping_group_size=4,
                       teacher_group_map=[j//4 for j in bridge.layer_map[::4]])
    return bridge, runtime


@torch.no_grad()
def experts_check(teacher):
    """Compare the native grouped GEMMs with the original expert implementation."""
    block = teacher.model.layers[0].mlp
    generator = torch.Generator(device='cuda').manual_seed(42)
    hidden = torch.randn(64, teacher.config.hidden_size, device='cuda',
                         dtype=torch.bfloat16, generator=generator)
    _, weights, selected = block.gate(hidden)
    config = block.experts.config
    initial = config._experts_implementation
    try:
        config._experts_implementation_internal = 'eager'
        reference = block.experts(hidden, selected, weights)
        config._experts_implementation_internal = 'grouped_mm'
        actual = block.experts(hidden, selected, weights)
    finally:
        config._experts_implementation_internal = initial
    error = actual.float() - reference.float()
    relative = float(error.norm() / reference.float().norm().clamp_min(1e-12))
    report = dict(passed=relative < .02, tokens=64, experts=teacher.config.num_experts,
                  active_experts_per_token=teacher.config.num_experts_per_tok,
                  dtype='bfloat16', implementation='grouped_mm', reference='native eager',
                  relative_l2=relative, maximum_abs_error=float(error.abs().max()))
    assert report['passed'], report
    return report
