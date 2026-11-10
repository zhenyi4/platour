"""Position-specific input embedding tuning on existing fixed template tokens."""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from qwen35_bridge_opd_core import save,weight_digest


def template_classes(mask,prefix=3,suffix=25):
    batch,n = mask.shape
    length = mask.sum(-1)
    assert bool((length>prefix+suffix).all())
    t = torch.arange(n,device=mask.device)[None]
    rel = t-(n-length)[:,None]
    assert torch.equal(mask.bool(),rel>=0), 'requires left padding'
    classes = torch.full((batch,n),prefix+suffix,dtype=torch.long,device=mask.device)
    classes = torch.where((rel>=0)&(rel<prefix),rel,classes)
    return torch.where(t>=n-suffix,prefix+t-(n-suffix),classes)


class TemplateEmbeddingTuning(nn.Module):
    def __init__(self,student,prefix_ids,suffix_ids):
        super().__init__()
        self.student = student.eval().requires_grad_(False)
        self.prefix_ids,self.suffix_ids = list(prefix_ids),list(suffix_ids)
        self.prefix,self.suffix = len(prefix_ids),len(suffix_ids)
        assert (self.prefix,self.suffix)==(3,25)
        self.layer_map = []  # Input embeddings only; no decoder-layer bridge.
        ids = torch.tensor(prefix_ids+suffix_ids[:-1],dtype=torch.long,device=student.device)
        self.register_buffer('template_token_ids',ids)
        self.template_embeddings = nn.Parameter(torch.empty(27,student.config.hidden_size,
            dtype=torch.float32,device=student.device))
        self.student_activation_checkpointing = False
        self.initialize_for_grpo()

    def initialize_for_grpo(self,seed=42):
        with torch.no_grad():
            native = self.student.get_input_embeddings()(self.template_token_ids)
            self.template_embeddings.copy_(native.float())
        return dict(kind='native 4B token embeddings copied into independent per-position FP32 parameters',
            positions=27,embedding_dim=self.student.config.hidden_size,seed=seed,
            random_initialization=False,sft_checkpoint_loaded=False,teacher_model_loaded=False,
            bridge_projection_present=False,last_prompt_token_tuned=False)

    def bridge_parameters(self):
        return (self.template_embeddings,)

    def bridge_state_tensors(self):
        return dict(template_embeddings=self.template_embeddings,template_token_ids=self.template_token_ids)

    def bridge_checkpoint_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.bridge_state_tensors().items()}

    def load_checkpoint(self,path):
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert set(state)=={'template_embeddings','template_token_ids'}
        assert torch.equal(state['template_token_ids'],self.template_token_ids.cpu())
        assert state['template_embeddings'].shape==self.template_embeddings.shape
        assert torch.isfinite(state['template_embeddings']).all()
        with torch.no_grad():
            self.template_embeddings.copy_(state['template_embeddings'])

    def input_embeddings(self,ids,mask,prompt_columns=None):
        """Replace active template positions; preserve native question/response lookup."""
        n = ids.shape[1] if prompt_columns is None else prompt_columns
        prompt_mask = mask[:,:n]
        classes = template_classes(prompt_mask,self.prefix,self.suffix)
        active = classes.lt(27) & prompt_mask.bool()
        # classes 0..26 are prefix3 + suffix24; final template27 and question28
        # both retain native embeddings. Repeated textual tokens have separate
        # trainable rows, preventing updates at question occurrences.
        indices = classes.clamp_max(26)
        expected_ids = self.template_token_ids[indices]
        assert torch.equal(ids[:,:n][active],expected_ids[active]), 'template token identities changed'
        native = self.student.get_input_embeddings()(ids)
        learned = self.template_embeddings[indices].to(native.dtype)
        prefix = torch.where(active[...,None],learned,native[:,:n])
        return torch.cat([prefix,native[:,n:]],1) if n<ids.shape[1] else prefix

    def prefill(self,ids,mask,extra=None):
        n = ids.shape[1]
        if extra is not None:
            ids = torch.cat([ids,extra],1)
            mask = torch.cat([mask,torch.ones_like(extra)],1)
        embeddings = self.input_embeddings(ids,mask,prompt_columns=n)
        return self.student(inputs_embeds=embeddings,attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=True,logits_to_keep=1)

    def response_hidden(self,prompt,response):
        from qwen35_bridge_grpo_core import checkpoint_student_blocks
        ids = torch.tensor([prompt+response[:-1]],device=self.student.device)
        mask = torch.ones_like(ids)
        embeddings = self.input_embeddings(ids,mask,prompt_columns=len(prompt))
        context = checkpoint_student_blocks(self) if (torch.is_grad_enabled()
            and self.student_activation_checkpointing) else nullcontext()
        with context:
            return self.student.model(inputs_embeds=embeddings,attention_mask=mask,
                position_ids=torch.arange(ids.shape[1],device=ids.device)[None],
                use_cache=False).last_hidden_state[:,len(prompt)-1:]

    def build_stress_prompt(self,source,length):
        body = source[self.prefix:-self.suffix]
        fill = length-self.prefix-self.suffix
        assert fill>0 and body
        return self.prefix_ids+(body*((fill+len(body)-1)//len(body)))[:fill]+self.suffix_ids

    @torch.no_grad()
    def training_statistics(self):
        native = self.student.get_input_embeddings()(self.template_token_ids).float()
        return dict(template_embedding_norm=float(self.template_embeddings.norm()),
            template_embedding_grad_norm=0. if self.template_embeddings.grad is None else float(self.template_embeddings.grad.norm()),
            template_embedding_delta_norm=float((self.template_embeddings-native).norm()))

    def metadata(self):
        return dict(algorithm='position_specific_template_input_embedding_tuning',
            formula='template input embedding = trainable E[position]; other token embeddings = native 4B',
            parameter_shape=list(self.template_embeddings.shape),trainable_parameters=self.template_embeddings.numel(),
            template_positions=27,prefix_tokens=3,suffix_tokens=25,last_prompt_token_tuned=False,
            initialization='native BF16 token embeddings widened exactly to FP32',
            per_position_parameters=True,shared_vocab_embedding_updated=False,
            question_embeddings_unchanged=True,response_embeddings_unchanged=True,
            padding_embeddings_unchanged=True,student_frozen=True,teacher_model_loaded=False,
            bridge_projection_present=False,hidden_space='model input embeddings, before layer0',
            prefix_propagation=True,parameter_dtype='float32',forward_dtype='bfloat16',
            template_token_ids=self.template_token_ids.tolist(),layer_map=[])


def load_bridge(root,protocol,checkpoint=None):
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    runtime = accelerated_models()
    student = load_student(Path(protocol['student_checkpoint']))
    assert student.config.hidden_size==2560 and student.config.num_hidden_layers==32
    tuner = TemplateEmbeddingTuning(student,protocol['template_prefix_ids'],protocol['template_suffix_ids']).eval()
    if checkpoint is not None:
        tuner.load_checkpoint(checkpoint)
    assert tuner.template_embeddings.numel()==69120
    runtime.update(teacher_model_loaded=False,teacher_forward_calls=0,bridge_projection_present=False,
        loaded_backbones=[protocol['student_checkpoint']],trainable_parameters=69120,
        parameter_shape=[27,2560],parameter_dtype='float32',model_input='inputs_embeds',
        inference='HF tuned-embedding prefill/first token; native vLLM continuation',
        shared_vocab_embedding_updated=False)
    return tuner,runtime


def smoke_audit(tuner,engine,rows,directory,pad_id,eos):
    from qwen35_bridge_grpo_core import sampled_log_probs
    from train_qwen35_bridge_opd_vllm import generate
    from eval_qwen35_35b_4b_math_bridge_8gpu import prompt_tensors
    initial = weight_digest(tuner)
    cpu_rng,cuda_rng = torch.get_rng_state(),torch.cuda.get_rng_state()
    directory.mkdir(parents=True,exist_ok=True)
    ids,mask = prompt_tensors(rows[:2],pad_id,tuner.student.device)
    try:
        with torch.no_grad():
            native = tuner.student(input_ids=ids,attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=True,logits_to_keep=1)
            modified = tuner.prefill(ids,mask)
            error = float((native.logits-modified.logits).abs().max())
            assert error==0.,error
            del native,modified
        row = min(rows,key=lambda r:len(r['prompt_token_ids']))
        prompt = torch.tensor([row['prompt_token_ids']],device=tuner.student.device)
        with torch.no_grad():
            out = tuner.prefill(prompt,torch.ones_like(prompt))
            response = [int(out.logits[0,-1].argmax())]*4
            del out
        optimizer = torch.optim.AdamW(tuner.bridge_parameters(),lr=1e-5,weight_decay=0.)
        gradients = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = -sampled_log_probs(tuner,row['prompt_token_ids'],response).mean()
            loss.backward()
            norm = float(tuner.template_embeddings.grad.norm())
            assert norm>0 and torch.isfinite(tuner.template_embeddings.grad).all()
            assert all(p.grad is None for p in tuner.student.parameters())
            gradients.append(norm)
            torch.nn.utils.clip_grad_norm_(tuner.bridge_parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        assert weight_digest(tuner)!=initial
        with torch.no_grad():
            original = tuner.student.get_input_embeddings()(ids)
            changed = tuner.input_embeddings(ids,mask)
            active = template_classes(mask).lt(27)&mask.bool()
            assert torch.equal(original[~active],changed[~active])
            bf16_changes = int(torch.count_nonzero(original[active]!=changed[active]))
            assert bf16_changes>0
            extra = torch.cat([ids,torch.ones((len(ids),3),device=ids.device,dtype=ids.dtype)],1)
            extended_mask = torch.cat([mask,torch.ones_like(extra[:,-3:])],1)
            extended = tuner.input_embeddings(extra,extended_mask,prompt_columns=ids.shape[1])
            assert torch.equal(extended[:,ids.shape[1]:],tuner.student.get_input_embeddings()(extra[:,ids.shape[1]:]))
        path = directory/'roundtrip.pt'
        torch.save(tuner.bridge_checkpoint_state(),path)
        trained = weight_digest(tuner)
        tuner.initialize_for_grpo()
        tuner.load_checkpoint(path)
        assert weight_digest(tuner)==trained
        path.unlink()
        probes = [dict(**r,sample_index=0,sampling_seed=1234+4*r['index']) for r in rows[:3]]
        result,_ = generate(tuner,engine,probes,trained,
            dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,repetition_penalty=1.,frequency_penalty=0.),
            8,directory/'handoff',pad_id,eos,return_logprobs=True)
        errors = []
        with torch.no_grad():
            for r in result['rows']:
                lp = sampled_log_probs(tuner,r['prompt_token_ids'],r['response_token_ids'])
                errors.extend((lp-torch.tensor(r['rollout_log_probs'],device=lp.device)).abs().tolist())
        parity = dict(mean_abs_logprob_error=sum(errors)/len(errors),max_abs_logprob_error=max(errors))
        assert parity['mean_abs_logprob_error']<.05 and parity['max_abs_logprob_error']<.2,parity
        save(directory/'audit.json',dict(passed=True,initial_native_max_logit_error=error,
            template_embedding_gradient_norms=gradients,changed_bf16_template_coordinates=bf16_changes,
            question_padding_last_prompt_embeddings_unchanged=True,response_embeddings_unchanged=True,
            checkpoint_roundtrip=True,teacher_model_loaded=False,bridge_projection_present=False,
            tuned_embedding_hf_vllm_parity=parity,audit_optimizer_discarded=True))
        del optimizer
    finally:
        tuner.initialize_for_grpo()
        tuner.template_embeddings.grad = None
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        assert weight_digest(tuner)==initial
        torch.cuda.empty_cache()
