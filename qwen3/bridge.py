"""Frozen-sender/frozen-receiver, prefix-only direct residual projection."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


def layer_map(sender_depth: int, receiver_depth: int, mapping: str) -> list[int]:
    if not 1 <= receiver_depth <= sender_depth:
        raise ValueError("Invalid sender and receiver depths")
    if mapping == "last_aligned":
        return list(range(sender_depth - receiver_depth, sender_depth))
    if mapping == "uniform_endpoints":
        if receiver_depth == 1:
            if sender_depth != 1:
                raise ValueError("One layer cannot align both endpoints")
            return [0]
        return [round(i * (sender_depth - 1) / (receiver_depth - 1)) for i in range(receiver_depth)]
    raise ValueError("Unsupported layer mapping")


class Bridge(nn.Module):
    """Apply a FP32 linear residual before each receiver decoder block."""

    def __init__(self, sender: nn.Module, receiver: nn.Module, mapping: str,
                 initialization: str = "zero", *, prefill_layout: str = "dense_prompt"):
        super().__init__()
        self.sender = sender.eval().requires_grad_(False)
        self.receiver = receiver.eval().requires_grad_(False)
        if prefill_layout not in {"dense_prompt", "fp8_prefix"}:
            raise ValueError("Unsupported Qwen3 prefill layout")
        self.prefill_layout = prefill_layout
        self._feature_provider = ContextVar(f'platour_features_{id(self)}', default=None)
        self.pad_token_id = receiver.config.pad_token_id
        self.mapping = mapping
        self.indices = layer_map(sender.config.num_hidden_layers,
                                 receiver.config.num_hidden_layers, mapping)
        self.P = nn.ModuleList([
            nn.Linear(sender.config.hidden_size, receiver.config.hidden_size,
                      bias=False, device=receiver.device, dtype=torch.float32)
            for _ in self.indices
        ])
        if initialization != "zero":
            raise ValueError("New bridge projections require zero initialization")
        with torch.random.fork_rng():
            torch.manual_seed(42)
            for projector in self.P:
                nn.init.zeros_(projector.weight)

    @contextmanager
    def use_features(self, provider):
        """Scope a PACT-Check sender intervention to one request, including vLLM prefill."""
        if not callable(provider):
            raise TypeError('Sender feature provider must be callable')
        token = self._feature_provider.set(provider)
        try:
            yield
        finally:
            self._feature_provider.reset(token)

    def _features(self, prompt_ids: list[int]) -> tuple[torch.Tensor, ...]:
        if len(prompt_ids) < 2:
            raise ValueError("Prompts must contain at least two tokens")
        provider = self._feature_provider.get()
        if provider is None:
            prefix = torch.tensor([prompt_ids[:-1]], device=self.sender.device)
            with torch.no_grad():
                output = self.sender.model(input_ids=prefix, use_cache=False,
                                           output_hidden_states=True)
            features = tuple(output.hidden_states[index] for index in self.indices)
        else:
            features = tuple(provider(prompt_ids))
        if (len(features) != len(self.P) or any(
                value.ndim != 3 or value.shape != (1, len(prompt_ids) - 1,
                                                    self.sender.config.hidden_size) or
                not torch.isfinite(value).all() for value in features)):
            raise ValueError('PACT sender features must match every prefix token and layer')
        return tuple(value.to(self.receiver.device).detach() for value in features)

    @contextmanager
    def inject(self, features: tuple[torch.Tensor, ...], count: int | torch.Tensor):
        injection_mask = count if isinstance(count, torch.Tensor) else None
        if injection_mask is not None:
            count = injection_mask.shape[1]
        if len(features) != len(self.P) or count != features[0].shape[1]:
            raise ValueError("Sender prefix and receiver prefix differ")
        handles = []
        try:
            for index, layer in enumerate(self.receiver.model.layers):
                def before(_layer, args, kwargs, i=index):
                    hidden = args[0] if args else kwargs["hidden_states"]
                    n = min(count, hidden.shape[1])
                    # Decoding after the first prefill sees a single token; no
                    # teacher representation is injected into response tokens.
                    if hidden.shape[1] < count or n == 0:
                        return args, kwargs
                    projected = self.P[i](features[i][:, :n].float())
                    if injection_mask is not None:
                        projected = projected * injection_mask[:, :n, None]
                    delta = F.pad(projected, (0, 0, 0, hidden.shape[1] - n)).to(
                        device=hidden.device, dtype=hidden.dtype)
                    if args:
                        return (hidden + delta, *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": hidden + delta}
                handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def response_log_probs(self, prompt: list[int], response: list[int]) -> torch.Tensor:
        if not response:
            raise ValueError("Empty response")
        ids = torch.tensor([prompt + response[:-1]], device=self.receiver.device)
        features = self._features(prompt)
        with self.inject(features, len(prompt) - 1):
            hidden = self.receiver.model(input_ids=ids, use_cache=False).last_hidden_state
        hidden = hidden[:, len(prompt) - 1:][0]
        labels = torch.tensor(response, device=hidden.device)
        pieces = []
        for start in range(0, len(labels), 32):
            logits = self.receiver.lm_head(hidden[start:start + 32]).float()
            tokens = labels[start:start + 32]
            pieces.append(logits.gather(1, tokens[:, None]).squeeze(1) - logits.logsumexp(-1))
        return torch.cat(pieces)

    def batch_response_loss(self, rows: list[dict], pad_token_id: int,
                            lm_chunk_tokens: int = 128) -> torch.Tensor:
        """Token-weighted CE over real right-padded SFT micro-batches."""
        from torch.utils.checkpoint import checkpoint

        if not rows or any(not row['prompt_token_ids'] or not row['response_token_ids']
                           for row in rows) or lm_chunk_tokens < 1:
            raise ValueError('SFT micro-batch needs prompts, answers and positive chunk size')
        width = max(len(row['prompt_token_ids']) + len(row['response_token_ids']) for row in rows)
        prefix_width = max(len(row['prompt_token_ids']) for row in rows)
        device = self.receiver.device
        ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        prefixes = torch.full((len(rows), prefix_width), pad_token_id, dtype=torch.long,
                              device=device)
        mask = torch.zeros_like(prefixes, dtype=torch.bool)
        for index, row in enumerate(rows):
            prompt, answer = row['prompt_token_ids'], row['response_token_ids']
            length, total = len(prompt), len(prompt) + len(answer)
            ids[index, :total] = torch.tensor(prompt + answer, device=device)
            attention[index, :total] = 1
            labels[index, length:total] = ids[index, length:total]
            prefixes[index, :length] = ids[index, :length]
            mask[index, :length] = True
        with torch.no_grad():
            hidden = self.sender.model(input_ids=prefixes.to(self.sender.device),
                attention_mask=mask.long().to(self.sender.device), use_cache=False,
                output_hidden_states=True).hidden_states
            features = tuple(hidden[i].detach().to(device).contiguous().clone()
                             for i in self.indices)
        injection = mask.clone()
        injection[torch.arange(len(rows), device=device), mask.long().sum(1) - 1] = False
        with self.inject(features, injection):
            hidden = self.receiver.model(input_ids=ids, attention_mask=attention,
                                         use_cache=False).last_hidden_state
        shifted = labels[:, 1:]
        active = shifted != -100
        predictors, targets = hidden[:, :-1][active], shifted[active]
        if not targets.numel():
            raise ValueError('SFT batch has no supervised response tokens')

        def chunk(states, tokens):
            return F.cross_entropy(self.receiver.lm_head(states).float(), tokens,
                                   reduction='sum')

        pieces = []
        for start in range(0, len(targets), lm_chunk_tokens):
            states = predictors[start:start + lm_chunk_tokens]
            tokens = targets[start:start + lm_chunk_tokens]
            pieces.append(checkpoint(chunk, states, tokens, use_reentrant=False)
                          if torch.is_grad_enabled() else chunk(states, tokens))
        return torch.stack(pieces).sum() / len(targets)

    @torch.no_grad()
    def prefix_cache_batch(self, prompts: list[list[int]], *, return_output: bool = False):
        """Export original right-padded fused prefix memory for native vLLM decode."""
        if not prompts or min(map(len, prompts)) < 2:
            raise ValueError("Prompts must contain at least two tokens")
        width = max(map(len, prompts))
        pad = self.pad_token_id
        if pad is None:
            pad = self.receiver.config.eos_token_id
            if isinstance(pad, list):
                pad = pad[0]
        ids = torch.full((len(prompts), width), pad, dtype=torch.long,
                         device=self.receiver.device)
        mask = torch.zeros_like(ids)
        for index, prompt in enumerate(prompts):
            ids[index, :len(prompt)] = torch.tensor(prompt, device=ids.device)
            mask[index, :len(prompt)] = 1
        if self._feature_provider.get() is not None:
            positions = (torch.arange(width, device=ids.device)[None].expand(len(prompts), -1)
                         if self.prefill_layout == 'fp8_prefix' else
                         (mask.cumsum(-1) - 1).clamp_min(0))
            slots = width - 1 if self.prefill_layout == 'fp8_prefix' else width
            injection = torch.zeros((len(prompts), slots), dtype=torch.bool, device=ids.device)
            supplied = [self._features(prompt) for prompt in prompts]
            features = tuple(torch.zeros((len(prompts), slots, self.sender.config.hidden_size),
                                         dtype=supplied[0][depth].dtype, device=ids.device)
                             for depth in range(len(self.P)))
            for batch, (prompt, row) in enumerate(zip(prompts, supplied, strict=True)):
                length = len(prompt) - 1
                injection[batch, :length] = True
                for depth, value in enumerate(row):
                    features[depth][batch, :length] = value[0]
        elif self.prefill_layout == "fp8_prefix":
            positions = torch.arange(width, device=ids.device)[None].expand(len(prompts), -1)
            injection = torch.zeros((len(prompts), width - 1), device=ids.device,
                                    dtype=torch.bool)
            for index, prompt in enumerate(prompts):
                injection[index, :len(prompt) - 1] = True
            hidden = self.sender.model(input_ids=ids[:, :-1].to(self.sender.device),
                attention_mask=injection.long().to(self.sender.device),
                position_ids=positions[:, :-1].to(self.sender.device),
                use_cache=False, output_hidden_states=True).hidden_states
            features = tuple(hidden[i].detach().to(ids.device).contiguous().clone()
                             for i in self.indices)
        else:
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            hidden = self.sender.model(input_ids=ids.to(self.sender.device),
                attention_mask=mask.to(self.sender.device),
                use_cache=False, output_hidden_states=True).hidden_states
            features = tuple(hidden[i].to(ids.device) for i in self.indices)
            injection = mask.bool()
            injection[torch.arange(len(prompts), device=ids.device), mask.sum(-1) - 1] = False
        with self.inject(features, injection):
            output = self.receiver.model(input_ids=ids, attention_mask=mask,
                                         position_ids=positions, use_cache=True)
        return output if return_output else output.past_key_values

    @torch.no_grad()
    def generate(self, prompt: list[int], *, seed: int, max_new_tokens: int,
                 temperature: float, top_p: float, top_k: int = -1) -> list[int]:
        features = self._features(prompt)
        ids = torch.tensor([prompt], device=self.receiver.device)
        options = dict(max_new_tokens=max_new_tokens,
                       do_sample=temperature > 0, pad_token_id=self.receiver.config.eos_token_id)
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                options["top_k"] = top_k
        device = ids.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if (
            device.type == "cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            with self.inject(features, len(prompt) - 1):
                output = self.receiver.generate(ids, **options)
        return output[0, len(prompt):].tolist()

    def load_public(self, filename: Path) -> None:
        with safe_open(filename, framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
            if metadata.get("layer_mapping") != self.mapping:
                raise ValueError("Checkpoint layer mapping mismatch")
            if metadata.get("layer_map_0_based") != __import__("json").dumps(self.indices, separators=(",", ":")):
                raise ValueError("Checkpoint sender layers mismatch")
            expected = {f"P.{i}.weight" for i in range(len(self.P))}
            if set(file.keys()) != expected:
                raise ValueError("Checkpoint tensor keys mismatch")
            for index, layer in enumerate(self.P):
                weight = file.get_tensor(f"P.{index}.weight")
                if weight.shape != layer.weight.shape:
                    raise ValueError("Checkpoint tensor shape mismatch")
                layer.weight.data.copy_(weight.to(layer.weight.device))
