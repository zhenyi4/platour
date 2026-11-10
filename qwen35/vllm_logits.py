"""Account for the HF-generated first response token in vLLM presence penalty."""

import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class HandoffPresencePenalty(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.requests = {}
        self.previous = None
        self.rows = self.tokens = self.penalties = None

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        def extract(params, prompt, outputs):
            token = (params.extra_args or {}).get('handoff_response_token')
            if token is None or params.presence_penalty == 0:
                return None
            return token, params.presence_penalty, outputs
        process_dict_updates(self.requests, batch_update, extract)
        active = tuple((row, token, penalty)
            for row, (token, penalty, outputs) in self.requests.items()
            if token not in outputs)
        if active != self.previous:
            self.previous = active
            self.rows = torch.tensor([r for r, t, p in active], device=self.device, dtype=torch.long)
            self.tokens = torch.tensor([t for r, t, p in active], device=self.device, dtype=torch.long)
            self.penalties = torch.tensor([p for r, t, p in active], device=self.device, dtype=torch.float32)

    def apply(self, logits):
        if self.previous:
            logits[self.rows, self.tokens] -= self.penalties
        return logits
