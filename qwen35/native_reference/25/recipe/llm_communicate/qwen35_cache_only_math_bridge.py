"""Teacher-conditioned prompt memory with an entirely native prompt hidden path."""
import contextlib

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge


class FunctionalLinearCacheLayer(LinearAttentionLayer):
    """Keep cache gradients without overwriting tensors saved by autograd."""
    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx,
                conv_kernel_size=conv_kernel_size)
        if self.has_previous_state[state_idx]:
            full = torch.cat((self.conv_states[state_idx], conv_states), dim=-1)
        else:
            full = conv_states
            self.has_previous_state[state_idx] = True
        width = self.conv_kernel_size[state_idx]
        saved = full[..., -width:]
        if saved.shape[-1] < width:
            saved = F.pad(saved, (width-saved.shape[-1], 0))
        self.conv_states[state_idx] = saved
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states


def functional_cache(config):
    cache = DynamicCache(config=config)
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            layer.__class__ = FunctionalLinearCacheLayer
            # Native single-token conv updates are in-place. This flag selects
            # the functional conv path; our cache still retains only last K.
            layer.record_past = True
    return cache


def positions(mask):
    return (mask.long().cumsum(-1)-1).clamp_min(0)


class CacheOnlyBridge(LastAlignedResidualBridge):
    def __init__(self, student, teacher, *, lm_chunk_tokens=128):
        super().__init__(student, teacher, lm_chunk_tokens=lm_chunk_tokens)
        self.configure_cache_only()

    def configure_cache_only(self):
        assert self.layer_map == list(range(self.teacher.config.num_hidden_layers-self.student.config.num_hidden_layers,
            self.teacher.config.num_hidden_layers))
        self.half = 'all32'
        self.audit_inactive()

    def audit_inactive(self):
        assert all(p.requires_grad for p in self.maps.parameters())
        expected = self.student.config.num_hidden_layers*self.student.config.hidden_size*self.teacher.config.hidden_size
        assert sum(p.numel() for p in self.bridge_parameters()) == expected
        assert all(not p.requires_grad for lm in (self.student,self.teacher) for p in lm.parameters())

    def metadata(self):
        return dict(super().metadata(), algorithm='last_aligned_cache_only_linear_bridge',
            injection='Cache branch only: KV from RMSNorm(h_native+P(h_teacher)); native prompt hidden unchanged',
            prompt_hidden_path='Entirely native frozen student; no bridge injection into any prompt block output',
            full_attention='Cache K/V computed from locally modified hidden exactly as in baseline',
            linear_attention='Entire native attention/state update from locally modified hidden, including Q/K/V and gates; discard output and retain conv/recurrent memory',
            response_path='Native decoder reads altered prompt memory and can propagate its effects across response layers',
            cache_updates='Functional conv/recurrent updates, including single-token training',
            hidden_space='decoder block input before RMSNorm', prefix_only=True,
            last_prefix_token_recomputed=True)

    @contextlib.contextmanager
    def inject(self, *args, **kwargs):
        raise RuntimeError('CacheOnlyBridge must use memory_prefill; main hidden injection is forbidden')
        yield

    def memory_prefill(self, ids, mask, *, return_hidden=False):
        """Compute native prompt block inputs once, then independently write each cache."""
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
        assert ids.shape == mask.shape and ids.shape[1] > 0
        assert bool(mask[:, -1].all()), 'Memory prefixes must be left padded'
        pos = positions(mask)
        with torch.no_grad():
            teacher = self.teacher.model(input_ids=ids, attention_mask=mask.long(),
                position_ids=pos, use_cache=False, output_hidden_states=True).hidden_states
            native = self.student.model(input_ids=ids, attention_mask=mask.long(),
                position_ids=pos, use_cache=False, output_hidden_states=True).hidden_states
        features = self.teacher_features(teacher)
        cos, sin = self.student.model.rotary_emb(native[0], pos[None].expand(3,-1,-1))
        cache = functional_cache(self.student.config)
        for i, block in enumerate(self.student.model.layers):
            hidden = native[i].detach()
            with torch.autocast(hidden.device.type, enabled=False):
                delta = self.maps[i](features[i].float()) * mask[...,None]
            changed = hidden + delta.to(hidden.dtype)
            conditioned = block.input_layernorm(changed)
            if block.block_type == 'full_attention':
                attn = block.self_attn
                shape = (*conditioned.shape[:-1],-1,attn.head_dim)
                key = attn.k_norm(attn.k_proj(conditioned).view(shape)).transpose(1,2)
                value = attn.v_proj(conditioned).view(shape).transpose(1,2)
                _, key = modeling.apply_rotary_pos_emb(key,key,cos,sin)
                cache.update(key,value,i)
            else:
                attn = block.linear_attn
                # Keep the entire current-layer state computation identical
                # to baseline; only its prompt output is discarded.
                attn(hidden_states=conditioned,cache_params=cache,attention_mask=mask.long())
        return (cache,native) if return_hidden else cache

    def native_memory_prefill(self, ids, mask):
        with torch.no_grad():
            cache = functional_cache(self.student.config)
            output = self.student.model(input_ids=ids,attention_mask=mask.long(),
                position_ids=positions(mask),past_key_values=cache,use_cache=True)
        return output.past_key_values

    def cached_prefill(self, ids, mask, extra=None):
        """All prompt tokens except the last form memory; recompute the last normally."""
        assert ids.shape[1] >= 2 and bool(mask[:,-1].all())
        cache = self.memory_prefill(ids[:,:-1],mask[:,:-1])
        tail = ids[:,-1:]
        if extra is not None:
            tail = torch.cat((tail,extra),dim=1)
        full_mask = torch.cat((mask[:,:-1],torch.ones_like(tail)),dim=1)
        pos = positions(full_mask)[:,-tail.shape[1]:]
        return self.student(input_ids=tail,attention_mask=full_mask,position_ids=pos,
            past_key_values=cache,use_cache=True,logits_to_keep=1)

    def pack_training(self, batch):
        counts = batch['prefix_mask'].long().sum(-1).tolist()
        replies = (batch['labels'] != -100).long().sum(-1).tolist()
        assert all(n>=2 for n in counts) and all(n>0 for n in replies)
        device = batch['input_ids'].device
        size = len(counts)
        width, continuation = max(n-1 for n in counts), max(replies)
        pad = self.student.generation_config.pad_token_id
        if pad is None: pad = self.student.config.pad_token_id
        if pad is None: pad = 0
        prefix_ids = torch.full((size,width),pad,dtype=torch.long,device=device)
        prefix_mask = torch.zeros_like(prefix_ids)
        decoder_ids = torch.full((size,continuation),pad,dtype=torch.long,device=device)
        decoder_mask = torch.zeros_like(decoder_ids)
        targets = torch.full_like(decoder_ids,-100)
        for i,(n,r) in enumerate(zip(counts,replies)):
            prefix_ids[i,width-(n-1):] = batch['input_ids'][i,:n-1]
            prefix_mask[i,width-(n-1):] = 1
            decoder_ids[i,:r] = batch['input_ids'][i,n-1:n+r-1]
            decoder_mask[i,:r] = 1
            targets[i,:r] = batch['labels'][i,n:n+r]
        mask = torch.cat((prefix_mask,decoder_mask),dim=1)
        return prefix_ids,prefix_mask,decoder_ids,mask,positions(mask)[:,width:],targets

    def continuation_loss(self, packed, cache):
        _,_,ids,mask,pos,targets = packed
        hidden = self.student.model(input_ids=ids,attention_mask=mask,position_ids=pos,
            past_key_values=cache,use_cache=True).last_hidden_state
        active = targets != -100
        predictors, answers = hidden[active], targets[active]
        def chunk(h,y):
            return F.cross_entropy(self.student.lm_head(h).float(),y,reduction='sum')
        pieces = []
        for offset in range(0,len(answers),self.lm_chunk_tokens):
            h = predictors[offset:offset+self.lm_chunk_tokens]
            y = answers[offset:offset+self.lm_chunk_tokens]
            pieces.append(checkpoint(chunk,h,y,use_reentrant=False) if torch.is_grad_enabled() else chunk(h,y))
        return torch.stack(pieces).sum()/len(answers)

    def chat_loss(self, batch, evaluate_lm=False):
        packed = self.pack_training(batch)
        cache = self.memory_prefill(packed[0],packed[1])
        lm = self.continuation_loss(packed,cache)
        return lm.new_zeros(()),lm,dict(lm_nll=lm,weighted_lm_loss=lm)

    def native_chat_loss(self,batch):
        packed = self.pack_training(batch)
        cache = self.native_memory_prefill(packed[0],packed[1])
        return self.continuation_loss(packed,cache)


def install_loader():
    import qwen35_35b_4b_math_bridge as original
    loader = original.load_bridge
    def load(root):
        bridge,runtime = loader(root)
        bridge.__class__ = CacheOnlyBridge
        bridge.configure_cache_only()
        runtime.update(bridge_mode='cache_only',prompt_hidden_path='native',
            memory_input='native_hidden_plus_teacher_projection',
            linear_control_gates='computed from locally modified hidden as in baseline',functional_cache_updates=True)
        return bridge,runtime
    original.load_bridge = load
    original.cache_check = cache_check


def prefill(bridge,ids,mask,extra=None):
    return bridge.cached_prefill(ids,mask,extra)


@torch.no_grad()
def cache_check(bridge,prompt):
    import eval_qwen35_35b_4b_math_bridge_8gpu as evaluation
    original = evaluation.prefill
    evaluation.prefill = prefill
    try:
        ids = torch.tensor([prompt],device=bridge.student.device)
        report = evaluation.batch_cache_check(bridge,ids,torch.ones_like(ids))
        return dict(report,last_prompt_token_injected=False,cache_only=True,
            prompt_hidden_path='native',response_injection=False)
    finally:
        evaluation.prefill = original
