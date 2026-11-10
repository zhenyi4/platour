"""Reuse audited node-local FP8 placement with the selected monotone full-to-full mapping."""
import torch

from qwen35_122b_4b_math_bridge import PairedFP8Bridge, load_bridge as load_models
from qwen35_multinode_config import MAPPING, RULE, layer_map_b


class MonotoneFP8Bridge(PairedFP8Bridge):
    def __init__(self, student, teacher, *, lm_chunk_tokens=128):
        super().__init__(student, teacher, layer_mapping='relative_depth_groups4', lm_chunk_tokens=lm_chunk_tokens)
        self.layer_map = layer_map_b(student.config.layer_types, teacher.config.layer_types)
        self.assignment = [[j] for j in self.layer_map]
        self.layer_mapping = MAPPING

    def metadata(self):
        return dict(super().metadata(), layer_mapping=MAPPING, mapping_rule=RULE,
            group_size=1, group_constraint=False, monotone_constraint=True, full_layer_constraint=True,
            nearest_full_tie_break='shallower', attention_types_preserved=False,
            attention_layer_pairs=[dict(student_layer=i, teacher_layer=j,
                student_attention=self.student.config.layer_types[i], teacher_attention=self.teacher.config.layer_types[j])
                for i, j in enumerate(self.layer_map)])


def load_bridge(root, primary, secondary, *, layer_mapping, teacher_precision, teacher_name, devices, layer_split):
    assert layer_mapping == MAPPING and teacher_precision == 'fp8'
    previous, runtime = load_models(root, primary, secondary, layer_mapping='relative_depth_groups4',
        teacher_precision='fp8', teacher_name=teacher_name, devices=devices, layer_split=layer_split)
    student, teacher = previous.student, previous.teacher
    del previous
    torch.cuda.empty_cache()
    bridge = MonotoneFP8Bridge(student, teacher).eval()
    assert all(torch.count_nonzero(p).item() == 0 for p in bridge.bridge_parameters())
    runtime.pop('teacher_group_map', None)
    runtime.update(layer_mapping=MAPPING, layer_map=bridge.layer_map, mapping_rule=RULE,
                   mapping_group_size=1, monotone_constraint=True, full_layer_constraint=True,
                   attention_types_preserved=False)
    return bridge, runtime
