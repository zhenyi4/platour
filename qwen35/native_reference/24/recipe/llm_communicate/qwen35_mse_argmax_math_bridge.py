"""Independent per-student MSE argmax assignment, with repeats and decreases."""
import json
from pathlib import Path

import torch

from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge


class MSEArgmaxBridge(LastAlignedResidualBridge):
    def __init__(self, student, teacher, mapping, selection, *, lm_chunk_tokens=128):
        super().__init__(student, teacher, lm_chunk_tokens=lm_chunk_tokens)
        self.configure_mapping(mapping, selection)

    def configure_mapping(self, mapping, selection):
        assert len(mapping) == self.student.config.num_hidden_layers
        assert all(isinstance(j, int) and 0 <= j < self.teacher.config.num_hidden_layers for j in mapping)
        self.layer_map = list(mapping)
        self.assignment = [[j] for j in self.layer_map]
        self.mse_selection = dict(selection)
        self.half = 'all32'
        self.audit_inactive()

    def audit_inactive(self):
        # Adapter for the verified six-rank trainer: all interfaces are active.
        assert all(p.requires_grad for p in self.maps.parameters())
        expected = self.student.config.num_hidden_layers * self.student.config.hidden_size * self.teacher.config.hidden_size
        assert sum(p.numel() for p in self.bridge_parameters()) == expected
        assert all(not p.requires_grad for model in (self.student, self.teacher) for p in model.parameters())

    def metadata(self):
        return dict(super().metadata(), layer_mapping='mse_per_student_shared_cosine_argmax',
            boundary='Independent per-student argmax; repeats, decreases and attention-type changes permitted',
            mse_selection=self.mse_selection, teacher_layer_numbers_1based=[j+1 for j in self.layer_map])


def install_loader(base):
    """Install the same explicit mapping for profiling, training and evaluation."""
    import qwen35_35b_4b_math_bridge as original
    spec = json.loads((Path(base) / 'mapping.json').read_text())
    mapping = spec['teacher_layers_0based']
    original_loader = original.load_bridge

    def load(root):
        bridge, runtime = original_loader(root)
        bridge.__class__ = MSEArgmaxBridge
        bridge.configure_mapping(mapping, spec['selection'])
        assert all(torch.count_nonzero(p) == 0 for p in bridge.bridge_parameters())
        runtime.update(layer_mapping='mse_per_student_shared_cosine_argmax', layer_map=list(mapping),
            mse_selection=spec['selection'], attention_types_preserved=False,
            teacher_layer_numbers_1based=spec['teacher_layers_1based'])
        return bridge, runtime

    original.load_bridge = load
