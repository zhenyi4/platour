"""Trainable V/W template bridge initialized only from native4B training hidden statistics."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F
from shared_subspace_residual_bridge import SharedSubspaceResidualBridge


class RandomTemplateBridge(nn.Module):
    inject = SharedSubspaceResidualBridge.inject

    def __init__(self, student, statistics, template):
        super().__init__()
        self.student = student.eval().requires_grad_(False)
        self.layer_map = list(range(student.config.num_hidden_layers))
        self.template = template
        self.student_activation_checkpointing = False
        positions, layers, width = statistics['broadcast_shape']
        if layers != len(self.layer_map) or width != student.config.hidden_size:
            raise RuntimeError('Template dimensions differ from the receiver model')
        if statistics['statistics_source_model'] != 'Qwen3.5-4B':
            raise RuntimeError('Only native4B statistics can initialize this experiment')
        if list(statistics['mean'].shape) != [positions, layers, 1] or statistics['std'].shape != statistics['mean'].shape:
            raise RuntimeError('Invalid template Gaussian scalar moments')
        if positions != len(template['template_token_ids']):
            raise RuntimeError('Statistics and active template positions differ')
        self.register_buffer('initial_mean', statistics['mean'].float().to(student.device))
        self.register_buffer('initial_std', statistics['std'].float().to(student.device))
        self.register_buffer('template_token_ids', torch.tensor(template['template_token_ids'], device=student.device))
        self.virtual_hidden = nn.Parameter(torch.empty(positions, layers, width, dtype=torch.float32, device=student.device))
        self.maps = nn.ModuleList([nn.Linear(width, width, bias=False, dtype=torch.float32, device=student.device)
                                  for _ in self.layer_map])
        self.initialize_for_grpo()

    def initialize_for_grpo(self, seed=42):
        rng = torch.Generator(device='cpu').manual_seed(seed)
        value = self.initial_mean.cpu() + self.initial_std.cpu() * torch.randn(
            self.virtual_hidden.shape, generator=rng, dtype=torch.float32)
        with torch.no_grad():
            self.virtual_hidden.copy_(value)
            for projection in self.maps:
                projection.weight.zero_()
        return dict(kind='native4B train-only template Gaussian V and exact-zero W', seed=seed,
                    template_positions=self.virtual_hidden.shape[0], teacher_model_loaded=False,
                    source_statistics_model='Qwen3.5-4B', sft_checkpoint_loaded=False,
                    matrix_trainable=True, projection_initialization='zero', constant_gate=1.)

    def bridge_parameters(self):
        return (self.virtual_hidden, *self.maps.parameters())

    def bridge_state_tensors(self):
        return dict(virtual_hidden=self.virtual_hidden, template_token_ids=self.template_token_ids,
                    **{f'P.{key}': value for key, value in self.maps.state_dict().items()})

    def bridge_checkpoint_state(self):
        return dict(virtual_hidden=self.virtual_hidden.detach().cpu().clone(),
                    template_token_ids=self.template_token_ids.cpu().clone(),
                    P={key: value.detach().cpu().clone() for key, value in self.maps.state_dict().items()})

    def load_bridge_checkpoint(self, path):
        state = torch.load(path, weights_only=True, map_location='cpu')
        if set(state) != {'virtual_hidden', 'template_token_ids', 'P'} or not torch.equal(
                state['template_token_ids'], self.template_token_ids.cpu()) or state['virtual_hidden'].shape != self.virtual_hidden.shape:
            raise RuntimeError('Random bridge checkpoint does not match template/V provenance')
        if not torch.isfinite(state['virtual_hidden']).all():
            raise RuntimeError('Random V checkpoint contains nonfinite values')
        self.maps.load_state_dict(state['P'], strict=True)
        with torch.no_grad():
            self.virtual_hidden.copy_(state['virtual_hidden'])

    def features(self, ids, mask=None):
        mask = torch.ones_like(ids) if mask is None else mask
        if mask.shape != ids.shape:
            raise RuntimeError('Prompt mask dimensions differ')
        classes = torch.full_like(ids, -1)
        prefix, suffix = self.template['prefix'], self.template['suffix']
        for batch in range(len(ids)):
            valid = mask[batch].nonzero(as_tuple=False).flatten()
            if len(valid) <= prefix + suffix:
                raise RuntimeError('Prompt cannot separate template from question')
            selected = torch.cat((valid[:prefix], valid[-suffix:-1]))
            if not torch.equal(ids[batch, selected], self.template_token_ids):
                raise RuntimeError('Template token identities differ from train-only source')
            classes[batch, selected] = torch.arange(len(selected), device=ids.device)
        return tuple(classes for _ in self.layer_map)

    def residual(self, index, hidden, classes, injection_mask):
        length = injection_mask.shape[1]
        if classes.shape != injection_mask.shape or hidden.shape[1] < length:
            raise RuntimeError('Template class / injection shape differs')
        active = injection_mask.bool() & classes.ge(0)
        with torch.autocast(hidden.device.type, enabled=False):
            projected = self.maps[index](self.virtual_hidden[:, index])
        delta = torch.zeros_like(hidden[:, :length])
        delta[active] = projected[classes[active]].to(hidden.dtype)
        return F.pad(delta, (0, 0, 0, hidden.shape[1] - length)) if length < hidden.shape[1] else delta

    def prefill(self, ids, mask, extra=None):
        features = self.features(ids, mask)
        injection = mask.bool() & features[0].ge(0)
        if extra is not None:
            ids = torch.cat((ids, extra), 1)
            mask = torch.cat((mask, torch.ones_like(extra)), 1)
        with self.inject(features, injection):
            return self.student(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True, logits_to_keep=1)

    def response_hidden(self, prompt, response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        prefix = torch.tensor([prompt], device=self.student.device)
        features = self.features(prefix)
        ids = torch.tensor([prompt + response[:-1]], device=self.student.device)
        context = checkpoint_student_blocks(self) if torch.is_grad_enabled() and self.student_activation_checkpointing else nullcontext()
        with context, self.inject(features, features[0].ge(0)):
            return self.student.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=ids.device)[None], use_cache=False).last_hidden_state[:, len(prompt) - 1:]

    def metadata(self):
        return dict(algorithm='native4B_train_only_template_gaussian_virtual_bridge',
            virtual_hidden_shape=list(self.virtual_hidden.shape), layer_map=self.layer_map,
            template=self.template, statistics_source_model='Qwen3.5-4B', teacher_model_loaded=False,
            frozen_receiver=True, matrix_trainable=True, projector_trainable=True,
            initialization='4B mean/std Gaussian V, zero W', constant_gate=1.,
            projection_dtype='float32', input_dim=self.student.config.hidden_size,
            trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))


def load_bridge(root, protocol):
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    statistics_root = Path(protocol['statistics_root'])
    sys.path.insert(0, str(statistics_root))
    from template_statistics import verify_statistics
    statistics = verify_statistics(protocol['dataset'])
    path = statistics_root / protocol['dataset'] / 'statistics.pt'
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != protocol['statistics_sha256']:
            raise RuntimeError('Native4B train statistics differ from frozen protocol')
    runtime = accelerated_models()
    student = load_student(Path(root) / 'data/models/Qwen3.5-4B')
    assert student.config.num_hidden_layers == 32 and student.config.hidden_size == 2560
    bridge = RandomTemplateBridge(student, statistics, protocol['token_template']).eval()
    runtime.update(bridge.metadata(), teacher='random4B_template', student='Qwen3.5-4B',
                   bf16_receiver=True, fp8_enabled=False, source_statistics_sha256=protocol['statistics_sha256'])
    return bridge, runtime
