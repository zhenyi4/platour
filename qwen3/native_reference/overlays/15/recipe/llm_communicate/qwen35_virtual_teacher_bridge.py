"""Trainable, input-independent teacher replacement for the frozen Qwen3.5 actor."""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from qwen35_bridge_opd_core import ROOT, TRAIN_SAMPLING, save, weight_digest
from shared_subspace_residual_bridge import SharedSubspaceResidualBridge


class VirtualTeacherBridge(nn.Module):
    inject = SharedSubspaceResidualBridge.inject

    def __init__(self, student, width=3072, random_std=.02):
        super().__init__()
        self.student = student.eval().requires_grad_(False)
        self.layer_map = list(range(student.config.num_hidden_layers))
        self.random_std = random_std
        self.virtual_hidden = nn.Parameter(torch.empty(len(self.layer_map), width,
            dtype=torch.float32, device=student.device))
        self.maps = nn.ModuleList([nn.Linear(width, student.config.hidden_size,
            bias=False, dtype=torch.float32, device=student.device) for _ in self.layer_map])
        self.student_activation_checkpointing = False
        self.initialize_for_grpo()

    def initialize_for_grpo(self, seed=42):
        rng = torch.Generator(device=self.student.device).manual_seed(seed)
        with torch.no_grad():
            self.virtual_hidden.normal_(0., self.random_std, generator=rng)
            for projection in self.maps:
                projection.weight.zero_()
        return dict(kind='random virtual hidden + zero linear', seed=seed,
            virtual_hidden_distribution='normal', virtual_hidden_std=self.random_std,
            projection_initialization='exact zero', constant_gate=1.,
            sft_checkpoint_loaded=False, teacher_model_loaded=False)

    def bridge_parameters(self):
        return (self.virtual_hidden, *self.maps.parameters())

    def bridge_state_tensors(self):
        # The cache fingerprint must change when either the matrix or W changes.
        return {'virtual_hidden': self.virtual_hidden,
                **{f'P.{k}':v for k,v in self.maps.state_dict().items()}}

    def bridge_checkpoint_state(self):
        return {'virtual_hidden': self.virtual_hidden.detach().cpu().clone(),
                'P': {k:v.detach().cpu().clone() for k,v in self.maps.state_dict().items()}}

    def load_bridge_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert set(state) == {'P', 'virtual_hidden'}
        assert state['virtual_hidden'].shape == self.virtual_hidden.shape
        assert torch.isfinite(state['virtual_hidden']).all()
        self.maps.load_state_dict(state['P'], strict=True)
        with torch.no_grad():
            self.virtual_hidden.copy_(state['virtual_hidden'])

    def features(self, ids):
        return tuple(row[None, None].expand(ids.shape[0], ids.shape[1], -1)
                     for row in self.virtual_hidden)

    def residual(self, index, hidden, teacher_hidden, injection_mask):
        n = injection_mask.shape[1]
        assert teacher_hidden.shape[:2] == hidden[:, :n].shape[:2]
        # W(v) is identical for every question/token. Project once, broadcast
        # afterwards; autograd sums the token gradients into both W and v.
        with torch.autocast(hidden.device.type, enabled=False):
            delta = self.maps[index](self.virtual_hidden[index])
            delta = delta[None, None] * injection_mask[..., None]
        if n < hidden.shape[1]:
            delta = F.pad(delta, (0, 0, 0, hidden.shape[1]-n))
        return delta.to(hidden.dtype)

    def prefill(self, ids, mask, extra=None):
        features = self.features(ids)
        injection = mask.bool().clone()
        injection[:, -1] = False
        if extra is not None:
            ids = torch.cat([ids, extra], 1)
            mask = torch.cat([mask, torch.ones_like(extra)], 1)
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        with self.inject(features, injection):
            return self.student(input_ids=ids, attention_mask=mask,
                position_ids=positions, use_cache=True, logits_to_keep=1)

    def response_hidden(self, prompt, response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        prefix = torch.tensor([prompt], device=self.student.device)
        ids = torch.tensor([prompt+response[:-1]], device=self.student.device)
        injection = torch.ones_like(prefix, dtype=torch.bool)
        injection[:, -1] = False
        context = checkpoint_student_blocks(self) if (torch.is_grad_enabled()
            and self.student_activation_checkpointing) else nullcontext()
        with context, self.inject(self.features(prefix), injection):
            return self.student.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=ids.device)[None],
                use_cache=False).last_hidden_state[:, len(prompt)-1:]

    @torch.no_grad()
    def training_statistics(self):
        deltas = torch.stack([p(v) for p,v in zip(self.maps,self.virtual_hidden,strict=True)])
        return dict(virtual_hidden_norm=float(self.virtual_hidden.norm()),
            virtual_hidden_grad_norm=0. if self.virtual_hidden.grad is None else float(self.virtual_hidden.grad.norm()),
            projection_norm=sum(float(p.weight.square().sum()) for p in self.maps)**.5,
            projection_grad_norm=sum(float(p.weight.grad.square().sum()) for p in self.maps if p.weight.grad is not None)**.5,
            effective_delta_norm=float(deltas.norm()),
            effective_delta_mean_layer_rms=float(deltas.square().mean(-1).sqrt().mean()))

    def metadata(self):
        return dict(algorithm='teacher_free_trainable_hidden_bridge',
            formula='h_s[i,t] += W_i(virtual_hidden[i])',
            virtual_hidden_shape=list(self.virtual_hidden.shape),
            virtual_hidden_initialization=f'normal(0, {self.random_std}**2), seed42',
            projection_initialization='zero', input_dim=self.maps[0].in_features,
            output_dim=self.maps[0].out_features, layer_map=self.layer_map,
            layer_mapping='independent_virtual_vector_per_student_layer',
            hidden_space='decoder block input before RMSNorm', injection='prompt[:-1] only',
            constant_gate=1., bias=False, projection_dtype='float32', virtual_hidden_dtype='float32',
            matrix_trainable=True, projection_trainable=True, student_frozen=True,
            teacher_model_loaded=False, teacher_forward_calls=0,
            prefix_propagation=True, same_vector_all_tokens_and_questions=True,
            trainable_parameters=sum(p.numel() for p in self.bridge_parameters()))


def load_bridge(root, protocol, checkpoint=None):
    from qwen35_student import load_student
    from qwen35_direct_math_bridge import accelerated_models
    runtime = accelerated_models()
    student = load_student(Path(protocol['student_checkpoint']))
    assert student.config.num_hidden_layers == 32 and student.config.hidden_size == 2560
    bridge = VirtualTeacherBridge(student, protocol['virtual_hidden_shape'][1], protocol['virtual_hidden_std']).eval()
    if checkpoint is not None:
        bridge.load_bridge_checkpoint(checkpoint)
    runtime.update(teacher_model_loaded=False, teacher_forward_calls=0,
        loaded_backbones=[protocol['student_checkpoint']], virtual_hidden_shape=list(bridge.virtual_hidden.shape),
        trainable_parameters=sum(p.numel() for p in bridge.bridge_parameters()),
        checkpoint_state_keys=['P','virtual_hidden'], identity_includes_virtual_hidden=True)
    return bridge, runtime


def smoke_audit(bridge, engine, rows, directory, pad_id, eos):
    """Real-backbone gradient audit and HF/vLLM parity; restore fresh weights."""
    from qwen35_bridge_grpo_core import sampled_log_probs
    from train_qwen35_bridge_opd_vllm import generate
    initial = weight_digest(bridge)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    row = min(rows, key=lambda r:len(r['prompt_token_ids']))
    ids = torch.tensor([row['prompt_token_ids']], device=bridge.student.device)
    grads = []
    try:
        with torch.no_grad():
            original = bridge.student(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=True, logits_to_keep=1)
            modified = bridge.prefill(ids, torch.ones_like(ids))
            zero_error = float((original.logits-modified.logits).abs().max())
            assert zero_error == 0., zero_error
            response = [int(original.logits[0,-1].argmax())] * 4
            del original,modified
        optimizer = torch.optim.AdamW(bridge.bridge_parameters(), lr=1e-5, weight_decay=0.)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = -sampled_log_probs(bridge,row['prompt_token_ids'],response).mean()
            loss.backward()
            vector_grad = float(bridge.virtual_hidden.grad.norm())
            projection_grad = sum(float(p.weight.grad.square().sum()) for p in bridge.maps)**.5
            assert projection_grad > 0. and (vector_grad == 0. if step == 0 else vector_grad > 0.)
            assert all(p.grad is None for p in bridge.student.parameters())
            grads.append(dict(step=step+1, vector_grad_norm=vector_grad, projection_grad_norm=projection_grad))
            torch.nn.utils.clip_grad_norm_(bridge.bridge_parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in bridge.bridge_parameters())
        trained = weight_digest(bridge)
        with torch.no_grad():
            bridge.virtual_hidden[0,0].add_(1.)
        assert weight_digest(bridge) != trained
        with torch.no_grad():
            bridge.virtual_hidden[0,0].sub_(1.)
        # Checkpoint both trainable objects, including exact round trip.
        state = bridge.bridge_checkpoint_state()
        path = directory/'roundtrip.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(state,path)
        before = weight_digest(bridge)
        bridge.initialize_for_grpo()
        bridge.load_bridge_checkpoint(path)
        assert weight_digest(bridge) == before
        path.unlink()
        probes = [dict(**r,sample_index=0,sampling_seed=1234+4*r['index']) for r in rows[:3]]
        result,_ = generate(bridge,engine,probes,weight_digest(bridge),TRAIN_SAMPLING,8,
            directory/'handoff',pad_id,eos,return_logprobs=True)
        errors = []
        with torch.no_grad():
            for r in result['rows']:
                lp = sampled_log_probs(bridge,r['prompt_token_ids'],r['response_token_ids'])
                errors.extend((lp-torch.tensor(r['rollout_log_probs'],device=lp.device)).abs().tolist())
        parity = dict(mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors))
        assert parity['mean_abs_logprob_error'] < .05 and parity['max_abs_logprob_error'] < .2, parity
        save(directory/'audit.json',dict(passed=True,zero_projection_native_max_logit_error=zero_error,
            gradient_steps=grads,teacher_model_loaded=False,teacher_forward_calls=0,
            matrix_and_projection_checkpoint_roundtrip=True,matrix_changes_fingerprint=True,
            nonzero_bridge_hf_vllm_parity=parity,smoke_optimizer_discarded=True))
        del optimizer,state
    finally:
        bridge.initialize_for_grpo()
        for p in bridge.bridge_parameters():
            p.grad = None
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        assert weight_digest(bridge) == initial
        torch.cuda.empty_cache()
