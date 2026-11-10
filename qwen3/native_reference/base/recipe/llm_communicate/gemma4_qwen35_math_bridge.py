"""Gemma 4 prompt encoder -> native Qwen3.5 decoder, with distinct tokenizers.

Training and HF/vLLM prefill share the same prepared character-span alignment.
No sender logits or sender response tokens enter the GRPO objective.
"""
from contextlib import nullcontext
import math
import hashlib
from pathlib import Path
import json

import torch
from torch import nn
from torch.nn import functional as F

from direct_residual_chat_bridge import DirectResidualChatBridge


def equal_width_map(student_layers, teacher_layers):
    """Round half up, preserve endpoints; allow a shallower sender and repeats."""
    if student_layers < 2 or teacher_layers < 2:
        raise ValueError('Both models must have at least two layers')
    return [(2*i*(teacher_layers-1)+(student_layers-1))//(2*(student_layers-1))
            for i in range(student_layers)]


class GemmaQwenHiddenBridge(DirectResidualChatBridge):
    def __init__(self, student, teacher, alignments, *, max_teacher_prompt_length=4096,
                 teacher_input_norm='none', teacher_input_norm_eps=1e-6, projection_init='random'):
        # The original constructor assumes teacher depth >= student depth.
        nn.Module.__init__(self)
        self.student = student.requires_grad_(False).eval()
        self.teacher = teacher.requires_grad_(False).eval()
        self.variant = 'teacher_residual'
        if projection_init not in ('random', 'zero'):
            raise ValueError('Use random or zero projection initialization')
        self.initialization = projection_init
        if teacher_input_norm not in ('none', 'rmsnorm') or teacher_input_norm_eps <= 0:
            raise ValueError('Use none or rmsnorm and a positive epsilon')
        self.teacher_input_norm = teacher_input_norm
        self.teacher_input_norm_eps = teacher_input_norm_eps
        self.lambda_lm, self.lm_chunk_tokens, self.compute_mse = 1., 128, False
        self.layer_map = equal_width_map(student.config.num_hidden_layers,
                                         teacher.config.num_hidden_layers)
        self.assignment = [[j] for j in self.layer_map]
        self.maps = nn.ModuleList([nn.Linear(teacher.config.hidden_size, student.config.hidden_size,
            bias=False, device=student.device, dtype=torch.float32) for _ in self.layer_map])
        self.max_teacher_prompt_length = max_teacher_prompt_length
        self.alignments = {}
        for row in alignments:
            key = tuple(row['student_prompt_ids'])
            if key in self.alignments and self.alignments[key] != row:
                # Train/eval examples can have the same question and tokenization.
                old = self.alignments[key]
                fields = ('teacher_prompt_ids', 'teacher_indices', 'injection_mask')
                if any(old[field] != row[field] for field in fields):
                    raise ValueError('Conflicting alignment for one student prompt')
            self.alignments[key] = row

    def bridge_parameters(self):
        return tuple(self.maps.parameters())

    def residual(self, index, hidden, teacher_hidden, injection_mask):
        # Shared by gradient computation, HF rollout prefill and vLLM cache export.
        if self.teacher_input_norm == 'rmsnorm':
            with torch.autocast(hidden.device.type, enabled=False):
                teacher_hidden = F.rms_norm(teacher_hidden.float(),
                    (teacher_hidden.shape[-1],), weight=None, eps=self.teacher_input_norm_eps)
        return super().residual(index, hidden, teacher_hidden, injection_mask)

    def initialize_for_grpo(self, seed=42):
        bound = 1e-4 / math.sqrt(self.teacher.config.hidden_size)
        generator = torch.Generator(device=self.student.device).manual_seed(seed)
        with torch.no_grad():
            for projection in self.maps:
                if self.initialization == 'zero':
                    projection.weight.zero_()
                else:
                    projection.weight.uniform_(-bound, bound, generator=generator)
        if self.initialization == 'zero':
            return dict(initialization='all-zero projection matrices', scale=0.,
                        seed=seed, constant_gate=1., pretrained_bridge=False)
        return dict(initialization='small random uniform', bound=bound, scale=1e-4,
                    seed=seed, constant_gate=1., pretrained_bridge=False)

    def metadata(self):
        pairs = [dict(student_layer=i, teacher_layer=j,
                      student_attention=self.student.config.layer_types[i],
                      teacher_attention=self.teacher.config.layer_types[j])
                 for i, j in enumerate(self.layer_map)]
        return dict(super().metadata(), sender='google/gemma-4-26B-A4B-it',
            receiver='Qwen3.5-4B', layer_mapping='equal_width_layers',
            mapping_rule='round_half_up(i*(teacher_layers-1)/(student_layers-1)); endpoints included',
            group_size=1, group_constraint=False, cross_attention_mapping_allowed=True,
            attention_types_preserved=all(p['student_attention']==p['teacher_attention'] for p in pairs),
            attention_layer_pairs=pairs, constant_gate=1.,
            initialization=('all-zero projection matrices' if self.initialization=='zero' else
                            f'small random uniform +/-1e-4/sqrt({self.teacher.config.hidden_size}), seed42'),
            projection_init=self.initialization,
            teacher_hidden_transfer_dtype='bfloat16', tokenizer_alignment='last overlapping content token by character span',
            template_tokens_injected=False, response_injection=False,
            injection=('h_s + P(RMSNorm(h_t))' if self.teacher_input_norm=='rmsnorm' else 'h_s + P(h_t)'),
            teacher_input_norm=self.teacher_input_norm,
            teacher_input_norm_eps=self.teacher_input_norm_eps,
            teacher_input_norm_affine=False, teacher_input_norm_compute_dtype='float32',
            max_teacher_prompt_length=self.max_teacher_prompt_length)

    def _alignment(self, prompt):
        try:
            return self.alignments[tuple(prompt)]
        except KeyError:
            raise ValueError('Prompt has no prepared Gemma/Qwen alignment; prepare it before inference') from None

    def register_stress_prompt(self, prompt):
        """Synthetic exact-length allocation probe, never a training example."""
        key = tuple(prompt)
        if key in self.alignments:
            raise ValueError('Stress prompt collides with a real prepared prompt')
        source = max(self.alignments.values(), key=lambda r: len(r['teacher_prompt_ids']))
        n = self.max_teacher_prompt_length
        ids = source['teacher_prompt_ids']
        teacher = (ids*((n+len(ids)-1)//len(ids)))[:n]
        indices = [(i*(n-1))//(len(prompt)-1) for i in range(len(prompt)-1)]
        self.alignments[key] = dict(student_prompt_ids=list(prompt), teacher_prompt_ids=teacher,
            teacher_indices=indices, injection_mask=[True]*len(indices), synthetic_stress_only=True)
        return n

    @torch.no_grad()
    def _features(self, prompt):
        row = self._alignment(prompt)
        teacher_ids = row['teacher_prompt_ids']
        if len(teacher_ids) > self.max_teacher_prompt_length:
            raise ValueError('Teacher prompt exceeds the prepared token budget')
        device = self.teacher.model.embed_tokens.weight.device
        ids = torch.tensor([teacher_ids], device=device)
        captured, handles = {}, []
        indices = row['teacher_indices'] + [0]
        injection = row['injection_mask'] + [False]
        assert len(indices) == len(injection) == len(prompt)

        def capture(index):
            def hook(module, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                positions = torch.tensor(indices, device=hidden.device)
                # Explicit block inputs avoid depending on HF hidden_states conventions.
                captured[index] = hidden.index_select(1, positions).to(
                    device=self.student.device, dtype=torch.bfloat16).contiguous().clone()
            return hook
        try:
            for j in sorted(set(self.layer_map)):
                handles.append(self.teacher.model.layers[j].register_forward_pre_hook(capture(j), with_kwargs=True))
            self.teacher.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(len(teacher_ids), device=device)[None],
                use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        assert set(captured) == set(self.layer_map)
        features = tuple(captured[j] for j in self.layer_map)
        return features, torch.tensor([injection], dtype=torch.bool, device=self.student.device)

    def prefill(self, ids, mask, extra=None):
        features_by_row, masks = [], []
        for tokens, active in zip(ids, mask, strict=True):
            prompt = tokens[active.bool()].tolist()
            features, inject = self._features(prompt)
            padding = ids.shape[1] - len(prompt)
            assert bool(active[-1]) and not bool(active[:padding].any())
            features_by_row.append(tuple(F.pad(h, (0, 0, padding, 0)) for h in features))
            masks.append(F.pad(inject, (padding, 0)))
        features = tuple(torch.cat([row[i] for row in features_by_row], dim=0)
                         for i in range(len(self.layer_map)))
        inject = torch.cat(masks, dim=0)
        assert not bool(inject[:, -1].any()) and not bool(inject[~mask.bool()].any())
        if extra is not None:
            ids = torch.cat([ids, extra], dim=1)
            mask = torch.cat([mask, torch.ones_like(extra)], dim=1)
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        with self.inject(features, inject):
            return self.student(input_ids=ids, attention_mask=mask, position_ids=positions,
                                use_cache=True, logits_to_keep=1)

    def response_hidden(self, prompt, response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        features, inject = self._features(prompt)
        ids = torch.tensor([prompt+response[:-1]], device=self.student.device)
        context = (checkpoint_student_blocks(self) if torch.is_grad_enabled()
                   and getattr(self, 'student_activation_checkpointing', False) else nullcontext())
        with context, self.inject(features, inject):
            output = self.student.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=ids.device)[None], use_cache=False)
        return output.last_hidden_state[:, len(prompt)-1:]


def load_bridge(run_root, protocol, devices):
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    try:
        from transformers import Gemma4ForConditionalGeneration, Gemma4ForCausalLM
    except ImportError as error:
        raise RuntimeError('Use a Transformers environment supporting both Gemma4 and Qwen3.5') from error
    assert len(devices) in (1, 2) and devices[0] == torch.cuda.current_device()
    teacher_path = Path(protocol['teacher_checkpoint'])
    for name, size in protocol.get('teacher_expected_files', {}).items():
        path = teacher_path/name
        assert path.is_file() and path.stat().st_size==size, f'Teacher download incomplete: {path}'
    run_root = Path(run_root)
    audit = json.loads((run_root/'baseline_comparison.json').read_text())
    expected_files = {run_root/name:digest for name,digest in audit['data_sha256'].items()}
    expected_files[run_root/'token_alignments.json'] = audit['token_alignments_sha256']
    expected_files.update({Path(name):digest for name,digest in protocol['model_tokenizer_sha256'].items()})
    for path, expected in expected_files.items():
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==expected, f'Prepared asset changed: {path}'
    runtime = accelerated_models()
    student = load_student(Path(protocol['student_checkpoint']))
    teacher_config = json.loads((teacher_path/'config.json').read_text())['text_config']
    assert teacher_config['num_hidden_layers'] == 30 and teacher_config['hidden_size'] == 2816
    if len(devices) == 1:
        placement = {'': devices[0]}
    else:
        primary, secondary = devices
        placement = {'model.vision_tower': secondary, 'model.embed_vision': secondary,
            'model.language_model.embed_tokens': primary, 'model.language_model.rotary_emb': primary,
            'model.language_model.norm': secondary, 'lm_head': primary}
        placement.update({f'model.language_model.layers.{i}': primary if i<15 else secondary
                          for i in range(30)})
    full = Gemma4ForConditionalGeneration.from_pretrained(teacher_path, dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm',
        device_map=placement, local_files_only=True)
    with torch.device('meta'):
        teacher = Gemma4ForCausalLM(full.config.text_config)
    teacher.model, teacher.lm_head = full.model.language_model, full.lm_head
    teacher.generation_config = full.generation_config
    teacher.eval().requires_grad_(False)
    del full
    alignments = json.loads((Path(run_root)/'token_alignments.json').read_text())
    bridge = GemmaQwenHiddenBridge(student, teacher, alignments,
        max_teacher_prompt_length=protocol['max_teacher_prompt_length'],
        teacher_input_norm=protocol.get('teacher_input_norm', 'none'),
        teacher_input_norm_eps=protocol.get('teacher_input_norm_eps', 1e-6),
        projection_init=protocol.get('projection_init', 'random')).eval()
    assert bridge.layer_map == protocol['layer_map']
    assert sum(p.numel() for p in bridge.bridge_parameters()) == protocol['trainable_parameters']
    assert not any(p.is_meta for p in bridge.parameters())
    assert {p.device.index for p in student.parameters()} == {devices[0]}
    assert {p.device.index for p in teacher.parameters()} == set(devices)
    runtime.update(teacher='google/gemma-4-26B-A4B-it', teacher_dtype='bfloat16',
        teacher_devices=devices, teacher_experts='grouped_mm', different_tokenizers=True,
        token_alignment=bridge.metadata()['tokenizer_alignment'])
    return bridge, runtime
