"""Teacher-free template hidden vectors initialized with sender moments."""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from qwen35_bridge_opd_core import save, weight_digest
from qwen35_template_embedding_tuning import template_classes
from qwen35_virtual_teacher_bridge import VirtualTeacherBridge


class TemplateVirtualTeacherBridge(VirtualTeacherBridge):
    def __init__(self, student, statistics, prefix_ids, suffix_ids):
        nn.Module.__init__(self)
        self.student = student.eval().requires_grad_(False)
        self.layer_map = list(range(student.config.num_hidden_layers))
        self.prefix_ids, self.suffix_ids = list(prefix_ids), list(suffix_ids)
        assert len(self.prefix_ids) == 3 and len(self.suffix_ids) == 25
        mu, sigma = statistics['mean'], statistics['std']
        assert mu.shape == sigma.shape == (27, len(self.layer_map), 1)
        assert torch.isfinite(mu).all() and torch.isfinite(sigma).all() and (sigma >= 0).all()
        assert statistics['broadcast_shape'][:2] == [27, len(self.layer_map)]
        width = statistics['broadcast_shape'][2]
        assert width > 0 and len(statistics['layer_map']) == len(self.layer_map)
        template_ids = self.prefix_ids + self.suffix_ids[:-1]
        assert template_ids == statistics['template_token_ids']
        self.register_buffer('template_token_ids', torch.tensor(template_ids, device=student.device))
        self.register_buffer('initial_mean', mu.float().to(student.device))
        self.register_buffer('initial_std', sigma.float().to(student.device))
        self.register_buffer('source_layer_map', torch.tensor(statistics['layer_map'], device=student.device))
        self.virtual_hidden = nn.Parameter(torch.empty(27, len(self.layer_map), width,
            dtype=torch.float32, device=student.device))
        self.maps = nn.ModuleList([nn.Linear(width, student.config.hidden_size,
            bias=False, dtype=torch.float32, device=student.device) for _ in self.layer_map])
        self.student_activation_checkpointing = False
        self.initialize_for_grpo()

    def initialize_for_grpo(self, seed=42):
        # CPU sampling makes initialization identical across ranks/devices.
        rng = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(tuple(self.virtual_hidden.shape), generator=rng, dtype=torch.float32)
        with torch.no_grad():
            value = self.initial_mean.cpu() + self.initial_std.cpu() * noise
            self.virtual_hidden.copy_(value)
            for p in self.maps:
                p.weight.zero_()
        return dict(kind='template-position/layer Gaussian fitted to sender; exact-zero linear W',
            seed=seed, virtual_hidden_shape=list(self.virtual_hidden.shape),
            statistics_shape=list(self.initial_mean.shape), statistics_pooling='questions and hidden channels',
            sender_mean_directions_copied=False, template_only=True, question_direct_delta_zero=True,
            projection_initialization='exact zero', teacher_model_loaded=False, constant_gate=1.)

    def bridge_state_tensors(self):
        return dict(virtual_hidden=self.virtual_hidden, template_token_ids=self.template_token_ids,
            source_layer_map=self.source_layer_map,
            **{f'P.{k}': v for k, v in self.maps.state_dict().items()})

    def bridge_checkpoint_state(self):
        return dict(virtual_hidden=self.virtual_hidden.detach().cpu().clone(),
            template_token_ids=self.template_token_ids.cpu().clone(),
            source_layer_map=self.source_layer_map.cpu().clone(),
            P={k: v.detach().cpu().clone() for k, v in self.maps.state_dict().items()})

    def load_bridge_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert set(state) == {'virtual_hidden', 'template_token_ids', 'source_layer_map', 'P'}
        assert torch.equal(state['template_token_ids'], self.template_token_ids.cpu())
        assert torch.equal(state['source_layer_map'], self.source_layer_map.cpu())
        assert state['virtual_hidden'].shape == self.virtual_hidden.shape
        assert torch.isfinite(state['virtual_hidden']).all()
        self.maps.load_state_dict(state['P'], strict=True)
        with torch.no_grad():
            self.virtual_hidden.copy_(state['virtual_hidden'])

    def features(self, ids, mask=None):
        mask = torch.ones_like(ids) if mask is None else mask
        classes = template_classes(mask)
        active = classes.lt(27) & mask.bool()
        assert torch.equal(ids[active], self.template_token_ids[classes[active]]), 'Template token IDs changed'
        # Hooks only need integer position classes. Avoid materializing a dense
        # [batch,prompt,teacher_width] tensor for every layer.
        return tuple(classes for _ in self.layer_map)

    def residual(self, index, hidden, teacher_hidden, injection_mask):
        classes = teacher_hidden
        n = injection_mask.shape[1]
        assert classes.shape == injection_mask.shape and hidden.shape[1] >= n
        active = injection_mask.bool() & classes.lt(27)
        with torch.autocast(hidden.device.type, enabled=False):
            projected = self.maps[index](self.virtual_hidden[:, index])
            values = projected[classes[active]].to(hidden.dtype)
        delta = torch.zeros_like(hidden[:, :n])
        delta[active] = values
        return F.pad(delta, (0, 0, 0, hidden.shape[1] - n)) if n < hidden.shape[1] else delta

    def prefill(self, ids, mask, extra=None):
        features = self.features(ids, mask)
        injection = mask.bool() & features[0].lt(27)
        if extra is not None:
            ids = torch.cat([ids, extra], 1)
            mask = torch.cat([mask, torch.ones_like(extra)], 1)
        with self.inject(features, injection):
            return self.student(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True, logits_to_keep=1)

    def response_hidden(self, prompt, response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        prefix = torch.tensor([prompt], device=self.student.device)
        features = self.features(prefix)
        injection = features[0].lt(27)
        ids = torch.tensor([prompt + response[:-1]], device=self.student.device)
        context = checkpoint_student_blocks(self) if (torch.is_grad_enabled()
            and self.student_activation_checkpointing) else nullcontext()
        with context, self.inject(features, injection):
            return self.student.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
                use_cache=False).last_hidden_state[:, len(prompt) - 1:]

    def build_stress_prompt(self, source, length):
        body = source[3:-25]
        fill = length - 28
        assert fill > 0 and body
        return self.prefix_ids + (body * ((fill + len(body) - 1) // len(body)))[:fill] + self.suffix_ids

    @torch.no_grad()
    def training_statistics(self):
        deltas = torch.stack([p(self.virtual_hidden[:, i]) for i, p in enumerate(self.maps)], dim=1)
        return dict(virtual_hidden_norm=float(self.virtual_hidden.norm()),
            virtual_hidden_grad_norm=0. if self.virtual_hidden.grad is None else float(self.virtual_hidden.grad.norm()),
            projection_norm=sum(float(p.weight.square().sum()) for p in self.maps) ** .5,
            projection_grad_norm=sum(float(p.weight.grad.square().sum()) for p in self.maps if p.weight.grad is not None) ** .5,
            effective_delta_norm=float(deltas.norm()),
            effective_delta_mean_layer_rms=float(deltas.square().mean(-1).sqrt().mean()))

    def metadata(self):
        return dict(algorithm='teacher_free_template_virtual_hidden_bridge',
            formula='h_s[i,t] += W_i(V[template_position(t),i]); direct delta zero elsewhere',
            virtual_hidden_shape=list(self.virtual_hidden.shape),
            virtual_hidden_initialization='sender template-position/layer pooled Gaussian mean/std; CPU seed42',
            statistics_shape=list(self.initial_mean.shape), projection_initialization='zero',
            input_dim=self.maps[0].in_features, output_dim=self.student.config.hidden_size,
            layer_map=self.layer_map, source_teacher_layer_map=self.source_layer_map.tolist(),
            layer_mapping='independent virtual vector per template position and student layer',
            hidden_space='decoder block input before RMSNorm', injection='prefix3 + suffix24',
            constant_gate=1., bias=False, projection_dtype='float32', virtual_hidden_dtype='float32',
            matrix_trainable=True, projection_trainable=True, student_frozen=True,
            teacher_model_loaded=False, teacher_forward_calls=0, prefix_propagation=True,
            same_vector_all_questions=True, same_vector_all_tokens=False,
            question_direct_delta_zero=True, response_direct_delta_zero=True,
            last_prompt_token_injected=False, sender_mean_directions_copied=False,
            trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))


def load_bridge(root, protocol, checkpoint=None):
    import hashlib
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    runtime = accelerated_models()
    path = Path(protocol['virtual_hidden_statistics_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['virtual_hidden_statistics_sha256']
    stats = torch.load(path, map_location='cpu', weights_only=True)
    student = load_student(Path(protocol['student_checkpoint']))
    bridge = TemplateVirtualTeacherBridge(student, stats, protocol['template_prefix_ids'],
                                         protocol['template_suffix_ids']).eval()
    if checkpoint is not None:
        bridge.load_bridge_checkpoint(checkpoint)
    assert tuple(bridge.virtual_hidden.shape) == tuple(protocol['virtual_hidden_shape'])
    assert stats['layer_map'] == protocol['statistics_source_layer_map']
    assert sum(p.numel() for p in bridge.bridge_parameters()) == protocol['trainable_parameters']
    runtime.update(teacher_model_loaded=False, teacher_forward_calls=0,
        loaded_backbones=[protocol['student_checkpoint']], virtual_hidden_shape=list(bridge.virtual_hidden.shape),
        trainable_parameters=protocol['trainable_parameters'], identity_includes_virtual_hidden=True,
        checkpoint_state_keys=['P', 'virtual_hidden', 'template_token_ids', 'source_layer_map'])
    return bridge, runtime


def smoke_audit(bridge, engine, rows, directory, pad_id, eos):
    from qwen35_virtual_teacher_bridge import smoke_audit as base_audit
    from eval_qwen35_35b_4b_math_bridge_8gpu import prompt_tensors
    directory.mkdir(parents=True, exist_ok=True)
    ids, mask = prompt_tensors(rows[:2], pad_id, bridge.student.device)
    features = bridge.features(ids, mask)
    active = features[0].lt(27) & mask.bool()
    assert torch.equal(active.sum(-1), torch.full((2,), 27, device=ids.device))
    with torch.no_grad():
        # A nonzero W must still give exactly zero *direct* correction on all
        # question, padding, final-prompt, and response positions.
        bridge.maps[0].weight.zero_()
        bridge.maps[0].weight[0, 0] = 1.
        hidden = torch.zeros(ids.shape[0], ids.shape[1] + 3, bridge.student.config.hidden_size,
                             device=ids.device, dtype=torch.bfloat16)
        delta = bridge.residual(0, hidden, features[0], mask.bool())
        assert torch.count_nonzero(delta[:, :ids.shape[1]][~active]) == 0
        assert torch.count_nonzero(delta[:, ids.shape[1]:]) == 0
        expected = bridge.virtual_hidden[features[0][active], 0, 0].to(torch.bfloat16)
        assert torch.equal(delta[:, :ids.shape[1], 0][active], expected)
        bridge.maps[0].weight.zero_()
    base_audit(bridge, engine, rows, directory, pad_id, eos)
    save(directory / 'template_positions.json', dict(passed=True, active_positions_per_question=27,
        nonzero_projection_template_gather_exact=True, question_padding_final_prompt_response_direct_delta_zero=True,
        statistics_shape=list(bridge.initial_mean.shape), virtual_hidden_shape=list(bridge.virtual_hidden.shape),
        same_template_vectors_at_different_question_lengths=True, no_sender_loaded=True))
