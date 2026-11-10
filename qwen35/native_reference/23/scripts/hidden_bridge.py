"""Prompt-only linear hidden bridge; frozen decoders, differentiable student KV."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from full_vocab_kl import full_vocab_kl

L2_LOSSES = ("ce_hidden_l2", "ce_hidden_l2_separate")


def bridge_layer_map(student_layers, teacher_layers, mapping="last_aligned"):
    if not 1 <= student_layers <= teacher_layers:
        raise ValueError("Require 1 <= student layers <= teacher layers")
    if mapping == "last_aligned":
        return list(range(teacher_layers-student_layers, teacher_layers))
    if mapping == "penultimate_all":
        if teacher_layers < 2:
            raise ValueError("penultimate_all requires at least two teacher layers")
        return [teacher_layers-2]*student_layers
    if mapping == "last_two_halves":
        if student_layers % 2 or teacher_layers < 2:
            raise ValueError("last_two_halves requires even student layers and at least two teacher layers")
        half = student_layers // 2
        return [teacher_layers-2]*half + [teacher_layers-1]*half
    if mapping == "last_half_repeat2":
        if student_layers % 2:
            raise ValueError("last_half_repeat2 requires an even number of student layers")
        start = teacher_layers - student_layers // 2
        return [start + i // 2 for i in range(student_layers)]
    if mapping == "uniform_endpoints":
        if student_layers == 1:
            if teacher_layers != 1:
                raise ValueError("One student layer cannot align both teacher endpoints")
            return [0]
        return [round(i*(teacher_layers-1)/(student_layers-1)) for i in range(student_layers)]
    raise ValueError(f"Unknown layer mapping: {mapping}")


@dataclass
class HiddenBridgeOutput(CausalLMOutputWithPast):
    ce_loss: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None


def shared_hidden_mse(projectors, teacher_hidden, student_hidden, response_mask):
    """Raw squared L2, averaged over layers, actual response tokens and width.

    Only teacher states are detached. Projectors may be shared with prompt
    injection or independent; student prompt KV remains differentiable.
    """
    if not len(projectors) == len(teacher_hidden) == len(student_hidden):
        raise ValueError("Alignment layer counts differ")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("Alignment requires a nonempty boolean response mask")
    losses = []
    for projector, teacher, student in zip(projectors, teacher_hidden, student_hidden):
        if teacher.shape[:-1] != response_mask.shape or student.shape[:-1] != response_mask.shape:
            raise ValueError("Alignment hidden positions and response mask differ")
        projected = projector(teacher.detach()[response_mask])
        losses.append((projected.float()-student[response_mask].float()).square().mean())
    return torch.stack(losses).mean()


class InjectionGate(nn.Module):
    """FP32 gates, initialized to sigmoid(0)=0.5 without consuming RNG."""
    def __init__(self, mode, width, device):
        super().__init__()
        self.mode = mode
        if mode == "constant":
            self.register_buffer("value", torch.ones((), device=device, dtype=torch.float32))
        elif mode == "scalar":
            self.logit = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
        elif mode in ("token", "token_h"):
            self.weight = nn.Parameter(torch.zeros(1, width if mode == "token_h" else 2*width, device=device, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown gate: {mode}")

    def forward(self, hidden, delta):
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        if self.mode == "constant":
            return self.value.to(dtype)
        if self.mode == "scalar":
            return self.logit.to(dtype).sigmoid()
        def normalize(x):
            x = x.to(dtype)
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        inputs = normalize(hidden) if self.mode == "token_h" else torch.cat((normalize(hidden), normalize(delta)), dim=-1)
        # Keep small probabilities and their gradients in FP32 even when the
        # surrounding decoder runs under BF16 autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return torch.nn.functional.linear(inputs, self.weight.to(dtype), self.bias.to(dtype)).sigmoid()


class MLPProjector(nn.Module):
    """Two bias-free layers, teacher-width hidden, exact GELU, zero output."""
    def __init__(self, teacher_width, student_width, *, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(teacher_width, teacher_width, bias=False, device=device, dtype=dtype)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(teacher_width, student_width, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, hidden):
        return self.fc2(self.activation(self.fc1(hidden)))


def projection_output_layer(projection):
    return projection.fc2 if isinstance(projection, MLPProjector) else projection


class HiddenBridge(nn.Module):
    def __init__(self, student, teacher, *, loss_type="ce", kl_temperature=1.0, kl_chunk_tokens=64,
                 alignment_weight=1e-3, layer_mapping="last_aligned", gate_mode="none", projector_type="linear"):
        super().__init__()
        self.student = student.requires_grad_(False).eval()
        self.teacher = teacher.requires_grad_(False).eval()
        if loss_type not in ("ce", "full_vocab_kl", *L2_LOSSES):
            raise ValueError(f"Unknown loss: {loss_type}")
        if student.config.vocab_size != teacher.config.vocab_size:
            raise ValueError("Teacher/student output vocabularies differ")
        self.loss_type = loss_type
        self.kl_temperature = kl_temperature
        self.kl_chunk_tokens = kl_chunk_tokens
        if not math.isfinite(alignment_weight) or alignment_weight < 0:
            raise ValueError("alignment_weight must be finite and nonnegative")
        self.alignment_weight = alignment_weight
        ns, nt = student.config.num_hidden_layers, teacher.config.num_hidden_layers
        self.layer_mapping = layer_mapping
        self.layer_map = bridge_layer_map(ns, nt, layer_mapping)
        if projector_type not in ("linear", "mlp2"):
            raise ValueError(f"Unknown projector type: {projector_type}")
        self.projector_type = projector_type
        def make_projector():
            widths = (teacher.config.hidden_size, student.config.hidden_size)
            options = dict(device=student.device, dtype=student.dtype)
            return MLPProjector(*widths, **options) if projector_type == "mlp2" else nn.Linear(*widths, bias=False, **options)
        self.projectors = nn.ModuleList([make_projector() for _ in range(ns)])
        for projection in self.projectors:
            nn.init.zeros_(projection_output_layer(projection).weight)
        # Copy zero tensors into independent storage, without consuming RNG.
        self.answer_projectors = deepcopy(self.projectors) if loss_type == "ce_hidden_l2_separate" else None
        if gate_mode not in ("none", "constant", "scalar", "token", "token_h"):
            raise ValueError(f"Unknown gate mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.gates = nn.ModuleList([] if gate_mode == "none" else [
            InjectionGate(gate_mode, student.config.hidden_size, student.device) for _ in range(ns)])
        self.gate_statistics = None
        self.injection_statistics = None
        self.injection_samples = None

    def bridge_parameters(self):
        params = tuple(self.projectors.parameters())
        if self.answer_projectors is not None:
            params += tuple(self.answer_projectors.parameters())
        params += tuple(self.gates.parameters())
        return params

    def save_projectors(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.projectors.state_dict(), directory / "projectors.pt")
        (directory / "projector_config.json").write_text(json.dumps({
            "projector_type": self.projector_type,
            "teacher_width": self.teacher.config.hidden_size,
            "student_width": self.student.config.hidden_size,
            "hidden_width": self.teacher.config.hidden_size if self.projector_type == "mlp2" else None,
            "activation": "gelu" if self.projector_type == "mlp2" else None,
            "bias": False}) + "\n")
        (directory / "layer_mapping.json").write_text(json.dumps(
            {"layer_mapping": self.layer_mapping, "layer_map": self.layer_map}, indent=2)+"\n")
        (directory / "gate_config.json").write_text(json.dumps({"gate_mode": self.gate_mode})+"\n")
        if self.gate_mode != "none":
            torch.save(self.gates.state_dict(), directory / "gates.pt")
        if self.answer_projectors is not None:
            torch.save(self.answer_projectors.state_dict(), directory / "answer_projectors.pt")

    def load_projectors(self, checkpoint):
        checkpoint = Path(checkpoint)
        projector_config = checkpoint.with_name("projector_config.json")
        recorded_type = json.loads(projector_config.read_text())["projector_type"] if projector_config.exists() else "linear"
        if recorded_type != self.projector_type:
            raise ValueError("Checkpoint projector type does not match model")
        gate_config = checkpoint.with_name("gate_config.json")
        recorded_gate = json.loads(gate_config.read_text())["gate_mode"] if gate_config.exists() else "none"
        if recorded_gate != self.gate_mode:
            raise ValueError("Checkpoint gate mode does not match model")
        metadata = checkpoint.with_name("layer_mapping.json")
        if metadata.exists():
            recorded = json.loads(metadata.read_text())
            if recorded != {"layer_mapping": self.layer_mapping, "layer_map": self.layer_map}:
                raise ValueError("Checkpoint layer mapping does not match model")
        elif self.layer_mapping != "last_aligned":
            raise ValueError("Legacy checkpoint has no mapping metadata; only last_aligned is supported")
        self.projectors.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if self.gate_mode != "none":
            self.gates.load_state_dict(torch.load(checkpoint.with_name("gates.pt"), map_location="cpu", weights_only=True))
        if self.answer_projectors is not None:
            self.answer_projectors.load_state_dict(torch.load(
                checkpoint.with_name("answer_projectors.pt"), map_location="cpu", weights_only=True))

    @classmethod
    def from_paths(cls, student_path, teacher_path, checkpoint=None, **loss_options):
        if checkpoint is not None and "projector_type" not in loss_options:
            metadata = Path(checkpoint).with_name("projector_config.json")
            if metadata.exists():
                loss_options["projector_type"] = json.loads(metadata.read_text())["projector_type"]
        if checkpoint is not None and "gate_mode" not in loss_options:
            metadata = Path(checkpoint).with_name("gate_config.json")
            if metadata.exists():
                loss_options["gate_mode"] = json.loads(metadata.read_text())["gate_mode"]
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        student = AutoModelForCausalLM.from_pretrained(student_path, **kwargs).to("cuda")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_path, **kwargs).to("cuda")
        model = cls(student, teacher, **loss_options)
        if checkpoint is not None:
            model.load_projectors(checkpoint)
        return model

    def train(self, mode=True):
        super().train(mode)
        self.student.eval()
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_hidden(self, input_ids, attention_mask, position_ids):
        hidden, _ = self.teacher_prefix(input_ids, attention_mask, position_ids, use_cache=False)
        return hidden

    @torch.no_grad()
    def teacher_prefix(self, input_ids, attention_mask, position_ids, *, use_cache):
        out = self.teacher.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            output_hidden_states=True, use_cache=use_cache, return_dict=True,
        )
        # hidden_states[i] is block i INPUT. Do not use the final norm output.
        return tuple(out.hidden_states[i].detach() for i in self.layer_map), out.past_key_values

    @contextmanager
    def inject(self, teacher_hidden, injection_mask):
        handles = []

        def make_hook(index):
            def hook(module, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                projector = self.projectors[index]
                delta = projector(teacher_hidden[index].to(next(projector.parameters()).dtype))
                delta = delta * injection_mask.unsqueeze(-1).to(delta.dtype)
                # Supports a full teacher-forced reference pass in sanity checks.
                if delta.shape[1] < h.shape[1]:
                    delta = torch.nn.functional.pad(delta, (0, 0, 0, h.shape[1] - delta.shape[1]))
                if delta.shape != h.shape:
                    raise ValueError(f"Injection shape {delta.shape} != hidden {h.shape}")
                if self.gate_mode != "none":
                    gate = self.gates[index](h, delta)
                    if self.gate_statistics is not None:
                        values = gate.detach().expand(*h.shape[:-1], 1)[:, :injection_mask.shape[1], 0][injection_mask.bool()]
                        if values.numel():
                            summary = dict(total=values.sum(), count=values.numel(), minimum=values.min(), maximum=values.max())
                            previous = self.gate_statistics[index]
                            if previous is not None:
                                summary = dict(total=previous['total']+summary['total'], count=previous['count']+summary['count'],
                                               minimum=torch.minimum(previous['minimum'],summary['minimum']),
                                               maximum=torch.maximum(previous['maximum'],summary['maximum']))
                            self.gate_statistics[index] = summary
                    delta = (delta.to(gate.dtype)*gate).to(h.dtype)
                if self.injection_statistics is not None:
                    with torch.no_grad():
                        n = injection_mask.shape[1]
                        ratios = (delta[:, :n].float().norm(dim=-1) /
                                  h[:, :n].float().norm(dim=-1).clamp_min(1e-8))[injection_mask.bool()]
                        if self.injection_samples is not None:
                            self.injection_samples[index].append(ratios.detach())
                        if ratios.numel():
                            previous = self.injection_statistics[index]
                            summary = dict(total=ratios.sum(), count=ratios.numel(), maximum=ratios.max())
                            if previous is not None:
                                summary = dict(total=previous['total']+summary['total'], count=previous['count']+summary['count'],
                                               maximum=torch.maximum(previous['maximum'], summary['maximum']))
                            self.injection_statistics[index] = summary
                updated = h + delta
                if args:
                    return (updated, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": updated}
            return hook

        try:
            for index, layer in enumerate(self.student.model.layers):
                handles.append(layer.register_forward_pre_hook(make_hook(index), with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def prefill(self, input_ids, attention_mask, position_ids, injection_mask=None):
        if injection_mask is None:
            injection_mask = attention_mask.bool()
        hidden = self.teacher_hidden(input_ids, attention_mask, position_ids)
        with self.inject(hidden, injection_mask):
            output = self.student.model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True, return_dict=True,
            )
        return output.past_key_values

    def forward(self, batch, *, loss_type=None):
        objective = self.loss_type if loss_type is None else loss_type
        if objective not in ("ce", "full_vocab_kl", *L2_LOSSES):
            raise ValueError(f"Unknown loss: {objective}")
        if objective == "ce_hidden_l2_separate" and self.answer_projectors is None:
            raise ValueError("Independent answer projectors were not initialized")
        # Reuse Rosetta's two-section collator, including section-local padding,
        # max_length truncation, final prompt token and response label shifting.
        sections = batch["kv_cache_index"]
        if len(sections) != 2:
            raise ValueError("This prompt-only experiment expects exactly two sections")
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        if isinstance(input_ids, list):
            if len(input_ids) != 1:
                raise ValueError("Aligned/multiple-tokenizer inputs are unsupported")
            input_ids, attention_mask = input_ids[0], attention_mask[0]
        position_ids, labels = batch["position_ids"], batch["labels"]
        boundary = sections[0].shape[1]
        if not torch.all(labels[:, :boundary] == -100):
            raise ValueError("Bridge prefix unexpectedly contains response supervision")
        if not torch.all(labels[:, boundary] == -100) or not torch.all(labels[:, boundary + 1] >= 0):
            raise ValueError("First response token must follow the final prompt token")
        mask = (sections[0][..., 0] > 0) & attention_mask[:, :boundary].bool()
        if objective == "ce":
            cache = self.prefill(input_ids[:, :boundary], attention_mask[:, :boundary],
                                 position_ids[:, :boundary], mask)
            # Keep cache attached: response supervision must reach the prefix.
            return self.student(
                input_ids=input_ids[:, boundary:], attention_mask=attention_mask,
                position_ids=position_ids[:, boundary:], past_key_values=cache,
                labels=labels[:, boundary:], use_cache=True, return_dict=True,
            )

        teacher_hidden, teacher_cache = self.teacher_prefix(
            input_ids[:, :boundary], attention_mask[:, :boundary],
            position_ids[:, :boundary], use_cache=True,
        )
        with self.inject(teacher_hidden, mask):
            prefix = self.student.model(
                input_ids=input_ids[:, :boundary], attention_mask=attention_mask[:, :boundary],
                position_ids=position_ids[:, :boundary], use_cache=True, return_dict=True,
            )
        # Both models see exactly the same teacher-forced answer prefix.
        # Teacher response states provide targets ONLY; no response injection.
        with torch.no_grad():
            teacher_response = self.teacher.model(
                input_ids=input_ids[:, boundary:], attention_mask=attention_mask,
                position_ids=position_ids[:, boundary:], past_key_values=teacher_cache,
                use_cache=False, return_dict=True, output_hidden_states=objective in L2_LOSSES,
            )
        if objective in L2_LOSSES:
            student_response = self.student(
                input_ids=input_ids[:, boundary:], attention_mask=attention_mask,
                position_ids=position_ids[:, boundary:], past_key_values=prefix.past_key_values,
                labels=labels[:, boundary:], use_cache=True, return_dict=True, output_hidden_states=True,
            )
            # Actual answer token positions, NOT shifted next-token predictions.
            # This excludes the final prompt token and includes the last answer.
            response_mask = (labels[:, boundary:] != -100) & attention_mask[:, boundary:].bool()
            alignment = shared_hidden_mse(
                self.answer_projectors if objective == "ce_hidden_l2_separate" else self.projectors,
                tuple(teacher_response.hidden_states[i] for i in self.layer_map),
                student_response.hidden_states[:-1], response_mask,
            )
            return HiddenBridgeOutput(
                loss=student_response.loss+self.alignment_weight*alignment,
                ce_loss=student_response.loss, alignment_loss=alignment,
                logits=student_response.logits, past_key_values=student_response.past_key_values,
            )
        student_response = self.student.model(
            input_ids=input_ids[:, boundary:], attention_mask=attention_mask,
            position_ids=position_ids[:, boundary:], past_key_values=prefix.past_key_values,
            use_cache=True, return_dict=True,
        )
        # Position 0 of the response section is the final prompt token; its
        # distribution predicts the first answer token. Padding never contributes.
        response_mask = (labels[:, boundary+1:] != -100) & attention_mask[:, boundary+1:].bool()
        loss = full_vocab_kl(
            student_response.last_hidden_state[:, :-1], teacher_response.last_hidden_state[:, :-1],
            self.student.lm_head, self.teacher.lm_head, response_mask,
            temperature=self.kl_temperature, chunk_tokens=self.kl_chunk_tokens,
        )
        return CausalLMOutputWithPast(loss=loss, past_key_values=student_response.past_key_values)


def batch_to(batch, device="cuda"):
    return {key: [item.to(device) for item in value] if isinstance(value, list)
            else value.to(device) for key, value in batch.items()}


def cache_pairs(cache):
    return tuple((cache[i][0], cache[i][1]) for i in range(len(cache)))
