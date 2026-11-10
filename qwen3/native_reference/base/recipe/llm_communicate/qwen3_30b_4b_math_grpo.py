"""Qwen3-30B-A3B encoder -> Qwen3-4B decoder, fresh MATH GRPO.

Shares the proven GRPO loss, distributed updates and native vLLM KV handoff
with the Qwen3.5 experiment. Tokenization and layer mapping are family-specific.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from qwen35_bridge_opd_core import ROOT, TRAIN_SAMPLING, QWEN3_EVAL_SAMPLING, save, quarter_epoch_schedule

STUDENT = ROOT/'data/models/Qwen3-4B'
TEACHER = ROOT/'data/models/Qwen3-30B-A3B'
BASE = ROOT/'outputs/llm_communicate/qwen3_30b_a3b_to_4b_math_grpo_uniform_endpoints_lr1e5_bs64_g8_cap4096_ep1_8gpu'
TRANSFER_MODES=('hidden','hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard','hidden_local_kv','sequential','post_prefill','c2c_soft')


def layer_map():
    from hidden_bridge import bridge_layer_map
    return bridge_layer_map(36, 48, 'uniform_endpoints')


def handoff_probe_rows(rows):
    """Override fixture seeds without mutating saved generation records."""
    return [{**row,'sampling_seed':42+i,'sample_index':0} for i,row in enumerate(rows[:2])]


def load_bridge(transfer_mode='hidden'):
    assert transfer_mode in TRANSFER_MODES
    import torch
    from transformers import AutoModelForCausalLM
    from direct_residual_chat_bridge import DirectResidualChatBridge
    options=dict(dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,
                 device_map={'':torch.cuda.current_device()})
    student=AutoModelForCausalLM.from_pretrained(STUDENT,**options).eval()
    teacher=AutoModelForCausalLM.from_pretrained(TEACHER,**options).eval()
    if transfer_mode=='hidden':
        bridge=DirectResidualChatBridge(student,teacher,variant='teacher_residual',initialization='zero')
        expected_parameters=188743680
        space='decoder block input before RMSNorm'
    elif transfer_mode in ('hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard'):
        from qwen3_hidden_grpo_ablations import Qwen3HiddenGRPOAblation
        bridge=Qwen3HiddenGRPOAblation(student,teacher,transfer_mode)
        expected_parameters=339738624 if transfer_mode in ('hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero') else 188743680
        space='decoder block input before RMSNorm'
    elif transfer_mode=='hidden_local_kv':
        from qwen3_local_hidden_kv_bridge import Qwen3LocalHiddenKVBridge
        bridge=Qwen3LocalHiddenKVBridge(student,teacher)
        expected_parameters=188743680
        space='native decoder block input before RMSNorm; local residual used only for K/V reconstruction'
    elif transfer_mode=='c2c_soft':
        from c2c_soft_grpo_bridge import Qwen3C2CSoftBridge,fuser_parameter_count
        bridge=Qwen3C2CSoftBridge(student,teacher)
        expected_parameters=36*fuser_parameter_count(512,1024,8)
        space='K post K-norm and RoPE; V post v_proj'
    else:
        from qwen3_kv_residual_bridge import Qwen3KVResidualBridge
        bridge=Qwen3KVResidualBridge(student,teacher,mode=transfer_mode,initialization='zero')
        expected_parameters=37748736
        space='K post K-norm and RoPE; V post v_proj'
    assert bridge.layer_map==layer_map()
    assert sum(p.numel() for p in bridge.bridge_parameters())==expected_parameters
    return bridge,dict(teacher='Qwen3-30B-A3B',student='Qwen3-4B',
        teacher_dtype='bfloat16',student_dtype='bfloat16',layer_map=bridge.layer_map,
        transfer_mode=transfer_mode,transfer_space=space,
        hidden_space=space if transfer_mode.startswith('hidden') else None,attention='all full attention',
        teacher_attention_backend='sdpa',student_attention_backend='sdpa',
        cuda_device=torch.cuda.current_device())


def verify_vllm_handoff(bridge,engine,rows,directory,pad,eos):
    """Compare nonzero-bridge full-KV decode with HF at BF16 tolerance.

    Exercise two prompt lengths and a padding-containing batched cache export.
    Restore the exact original projection bytes before the training smoke.
    """
    import torch
    from qwen35_bridge_opd_core import weight_digest
    from train_qwen35_bridge_opd_vllm import prefill,prompt_tensors
    from prepare_qwen35_bridge_vllm import cache_to_cpu,export_memory
    directory.mkdir(parents=True,exist_ok=True)
    memory=engine.directory/'memory'
    memory.mkdir(exist_ok=True)
    assert not list(memory.glob('*.safetensors'))
    chosen=handoff_probe_rows(rows)
    originals=[p.detach().cpu().clone() for p in bridge.bridge_parameters()]
    original_digest=weight_digest(bridge)
    try:
        with torch.inference_mode():
            probe=(bridge.handoff_probe_parameters() if hasattr(bridge,'handoff_probe_parameters')
                   else bridge.bridge_parameters())
            zero_probe=all(not bool(p.count_nonzero()) for p in probe)
            probe_scale=float(getattr(bridge,'handoff_probe_scale',1000.))
            if zero_probe:
                # A zero-init training ablation still needs a nontrivial handoff
                # fixture. Fill only the temporary probe; restore exact zeros below.
                generator=torch.Generator(device=bridge.student.device).manual_seed(42)
                for p in probe:
                    p.uniform_(-1e-4/(p.shape[-1]**.5),1e-4/(p.shape[-1]**.5),generator=generator)
                    p.mul_(probe_scale)
            else:
                for p in probe:p.mul_(probe_scale)
            ids,mask=prompt_tensors(chosen,pad,'cuda')
            # A very strong random probe can produce immediate EOS. That does
            # not exercise vLLM decode, so reduce only the temporary probe.
            for attempt in range(5):
                result=prefill(bridge,ids,mask)
                first=result.logits[:,-1].argmax(-1).tolist()
                if all(token not in eos for token in first):break
                del result
                for p in probe:p.mul_(.1)
                probe_scale*=.1
            else:
                raise AssertionError('No non-EOS handoff fixture after reducing temporary probe')
            digest=weight_digest(bridge)
            logits=result.logits[:,-1].float()
            first_logprobs=(logits.gather(1,torch.tensor(first,device=logits.device)[:,None]).squeeze(1)-logits.logsumexp(-1)).tolist()
            cpu_cache=cache_to_cpu(result.past_key_values)
            requests=[]
            for i,row in enumerate(chosen):
                transfer=export_memory(cpu_cache,row,digest,'full_hidden',memory,i)
                transfer['checkpoint_sha256']=digest
                requests.append(dict(**row,transfer=transfer,hf_first_token=first[i],
                    hf_first_logprob=first_logprobs[i],handoff_is_eos=first[i] in eos))
            cache=result.past_key_values
            expected=[[token] for token in first]
            finished=[token in eos for token in first]
            for _ in range(63):
                if all(finished):break
                new_ids=torch.tensor([[tokens[-1]] for tokens in expected],device=ids.device)
                mask=torch.cat([mask,torch.ones((len(chosen),1),device=mask.device,dtype=mask.dtype)],dim=1)
                positions=(mask.cumsum(-1)-1).clamp_min(0)[:,-1:]
                result=bridge.student(input_ids=new_ids,attention_mask=mask,position_ids=positions,
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=result.past_key_values
                tokens=result.logits[:,-1].argmax(-1).tolist()
                for i,token in enumerate(tokens):
                    if not finished[i]:
                        expected[i].append(token)
                        finished[i]=token in eos
            del result,cache,cpu_cache,ids,mask
        manifest=directory/'manifest.json'
        save(manifest,dict(rows=requests,tokens=64,bridge_state_sha256=digest,return_logprobs=True,
            sampling=dict(TRAIN_SAMPLING,temperature=0.)))
        actual=engine.call(dict(manifest=str(manifest),result=str(directory/'responses.json')))
        comparisons=[]
        from transformers import DynamicCache
        from safetensors.torch import load_file
        with torch.inference_mode():
            for row,request,want,got in zip(chosen,requests,expected,actual['rows'],strict=True):
                response=got['response_token_ids']
                reference=[request['hf_first_logprob']]
                greedy_agreement=[]
                if len(response)>1:
                    # Replay the identical vLLM response history against the
                    # original exported KV, avoiding autoregressive drift.
                    stored=load_file(memory/(request['transfer']['memory_key']+'.safetensors'))
                    cache=DynamicCache(config=bridge.student.config)
                    for i in range(len(bridge.layer_map)):
                        kv=stored[f'layer.{i}.kv'].to(bridge.student.device)
                        cache.update(kv[0].transpose(0,1)[None],kv[1].transpose(0,1)[None],i)
                    n=len(row['prompt_token_ids'])
                    ids=torch.tensor([response[:-1]],device=bridge.student.device)
                    result=bridge.student(input_ids=ids,past_key_values=cache,use_cache=True,
                        attention_mask=torch.ones((1,n+len(response)-1),device=ids.device,dtype=torch.long),
                        position_ids=torch.arange(n,n+len(response)-1,device=ids.device)[None])
                    logits=result.logits[0].float()
                    labels=torch.tensor(response[1:],device=ids.device)
                    reference.extend((logits.gather(1,labels[:,None]).squeeze(1)-logits.logsumexp(-1)).tolist())
                    greedy_agreement=(logits.argmax(-1)==labels).tolist()
                    del result,logits,cache,stored
                errors=[abs(a-b) for a,b in zip(reference,got['rollout_log_probs'],strict=True)]
                comparisons.append(dict(index=row['index'],prompt_tokens=len(row['prompt_token_ids']),
                    hf_token_ids=want,vllm_token_ids=response,identical=want==response,
                    same_history_logprob_mean_abs_error=sum(errors)/len(errors),
                    same_history_logprob_max_abs_error=max(errors),
                    same_history_greedy_agreement=sum(greedy_agreement)/max(1,len(greedy_agreement)),
                    native_vllm_decode_tokens_tested=max(0,len(response)-1),
                    passed=len(response)>1 and sum(errors)/len(errors)<.05 and max(errors)<.25))
        save(directory/'parity.json',dict(passed=all(r['passed'] for r in comparisons),
            nonzero_bridge=True,projection_scale_multiplier=probe_scale,
            zero_initialization_probe=zero_probe,left_padding_tested=True,
            tolerance=dict(mean_abs_logprob=.05,max_abs_logprob=.25),
            comparisons=comparisons,bridge_state_sha256=digest))
        assert all(r['passed'] for r in comparisons),f'HF/vLLM full-KV mismatch: {directory}/parity.json'
    finally:
        with torch.no_grad():
            for p,original in zip(bridge.bridge_parameters(),originals,strict=True):p.copy_(original)
        assert weight_digest(bridge)==original_digest
        for path in memory.glob('*.safetensors'):path.unlink()
        torch.cuda.empty_cache()


def verify_local_hidden_isolation(bridge,rows,directory,pad):
    """Real-model audit: one low projection cannot change higher prompt states."""
    import torch
    from qwen35_bridge_opd_core import weight_digest
    from train_qwen35_bridge_opd_vllm import prompt_tensors
    assert bridge.metadata()['transfer_mode']=='hidden_local_kv'
    original=[p.detach().cpu().clone() for p in bridge.bridge_parameters()]
    digest=weight_digest(bridge)
    ids,mask=prompt_tensors(rows[:2],pad,'cuda')
    def capture():
        hidden=[]
        hooks=[layer.register_forward_pre_hook(
            lambda mod,args,kwargs:hidden.append((args[0] if args else kwargs['hidden_states']).detach().cpu().clone()),
            with_kwargs=True) for layer in bridge.student.model.layers]
        try:cache=bridge._prefix(ids,mask)
        finally:
            for hook in hooks:hook.remove()
        return cache,hidden
    try:
        with torch.inference_mode():
            for p in bridge.bridge_parameters():p.zero_()
            reference,before=capture()
            bridge.maps[0].weight.copy_(original[0].to(bridge.student.device)*1000.)
            modified,after=capture()
            hidden_equal=[torch.equal(a,b) for a,b in zip(before,after,strict=True)]
            cache_equal=[torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
                         for a,b in zip(reference.layers,modified.layers,strict=True)]
            low_key=(reference.layers[0].keys-modified.layers[0].keys).float().abs()
            low_value=(reference.layers[0].values-modified.layers[0].values).float().abs()
            invalid=(~mask[:,:-1].bool())[:,None,:,None]
            padding_error=max(float((low_key*invalid).max()),float((low_value*invalid).max()))
            passed=(all(hidden_equal) and all(cache_equal[1:]) and
                    float(low_key.max())>0 and float(low_value.max())>0 and padding_error==0.)
            save(directory/'local_hidden_isolation.json',dict(passed=passed,
                perturbed_projection_layer_0based=0,all_other_projections_zero=True,
                native_prompt_hidden_exact_by_layer=hidden_equal,
                prefix_cache_exact_by_layer=cache_equal,
                higher_prefix_hidden_unchanged=all(hidden_equal[1:]),
                higher_prefix_cache_unchanged=all(cache_equal[1:]),
                low_key_max_abs_change=float(low_key.max()),low_value_max_abs_change=float(low_value.max()),
                padding_max_abs_change=padding_error,batch_size=len(ids),
                prompt_lengths=[len(row['prompt_token_ids']) for row in rows[:2]],
                decode_propagation_retained=True))
            assert passed,f'Local hidden isolation failed: {directory}/local_hidden_isolation.json'
    finally:
        with torch.no_grad():
            for p,value in zip(bridge.bridge_parameters(),original,strict=True):p.copy_(value)
        assert weight_digest(bridge)==digest
        torch.cuda.empty_cache()


def prepare(base=BASE, transfer_mode='hidden'):
    from transformers import AutoTokenizer
    base=Path(base).absolute()
    assert transfer_mode in TRANSFER_MODES
    if (base/'protocol.json').exists():
        protocol=json.loads((base/'protocol.json').read_text())
        assert protocol['teacher']=='Qwen3-30B-A3B' and protocol['student']=='Qwen3-4B'
        assert protocol['epochs']==1 and protocol['gpu_ids']==list(range(8))
        assert protocol['layer_map']==layer_map() and protocol['eval_sampling']==QWEN3_EVAL_SAMPLING
        assert protocol.get('transfer_mode','hidden')==transfer_mode
        for key in ('train','eval'):
            path=base/f'{key}_prompts.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol[f'{key}_prompts_sha256']
        if transfer_mode!='hidden' and not protocol['wandb_run_name'].startswith('ablation_'):
            protocol['wandb_run_name']='ablation_'+protocol['wandb_run_name']
            save(base/'protocol.json',protocol)
        if transfer_mode=='c2c_soft' and protocol.get('eval_gate_modes')!=['soft','hard']:
            protocol.update(eval_gate_modes=['soft','hard'],
                hard_eval_gate_rule='scalar gate_logit > 0; zero off; dynamic head weights remain sigmoid',
                evaluation_policy='Each scheduled evaluation runs soft and hard on identical questions/seeds; restore soft before training')
            save(base/'protocol.json',protocol)
        return protocol
    manifest=json.loads((ROOT/'outputs/model_downloads/qwen3_30b_a3b/manifest.json').read_text())
    assert manifest['repo']=='Qwen/Qwen3-30B-A3B'
    assert json.loads((ROOT/'outputs/model_downloads/qwen3_30b_a3b/state.json').read_text())['status']=='complete'
    assert all((TEACHER/name).stat().st_size==size for name,size in manifest['files'].items())
    tokenizer=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    teacher_tokenizer=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    assert tokenizer.get_vocab()==teacher_tokenizer.get_vocab()
    train_source=ROOT/'outputs/llm_communicate/qwen35_train_generation/inputs/math.jsonl'
    eval_source=ROOT/'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096/Qwen3.5-4B/prompts.json'
    train=[json.loads(line) for line in train_source.open()]
    evaluation=json.loads(eval_source.read_text())
    assert len(train)==7500 and len(evaluation)==500
    def tokenize(rows):
        result=[]
        for row in rows:
            messages=[dict(role='user',content=row['input_text'])]
            def ids(tok):
                value=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                               enable_thinking=False,return_dict=False)
                return value['input_ids'] if hasattr(value,'keys') else value
            prompt=ids(tokenizer)
            assert prompt==ids(teacher_tokenizer) and 2<=len(prompt)<=2048
            result.append(dict(index=row['index'],question=row['question'],gold=row['gold'],
                               input_text=row['input_text'],prompt_token_ids=prompt))
        assert [r['index'] for r in result]==list(range(len(rows)))
        return result
    train,evaluation=tokenize(train),tokenize(evaluation)
    assert not {tuple(r['prompt_token_ids']) for r in train}&{tuple(r['prompt_token_ids']) for r in evaluation}
    save(base/'train_prompts.json',train)
    save(base/'eval_prompts.json',evaluation)
    steps,eval_steps=quarter_epoch_schedule(7500,64,1)
    protocol=dict(model_family='qwen3',method='grpo',teacher='Qwen3-30B-A3B',student='Qwen3-4B',
        teacher_checkpoint=str(TEACHER),student_checkpoint=str(STUDENT),
        teacher_checkpoint_revision=manifest['revision'],teacher_weight_precision='bf16',vocab_size=151936,
        teacher_hidden_size=2048,student_hidden_size=2560,
        formula='h_s[i] + P_i(h_t[layer_map[i]])',layer_mapping='uniform_endpoints',layer_map=layer_map(),
        mapping_rule='round(i * 47 / 35), zero-based decoder block indices',mapping_group_size=1,
        monotone_constraint=True,attention_types_preserved=True,
        hidden_space='decoder block input before RMSNorm',injection='prompt[:-1] only',
        constant_gate=1.,initialization='uniform +/-1e-4/sqrt(2048), seed42; no SFT/pretrain checkpoint',
        frozen_backbones=True,projection_dtype='float32',trainable_parameters=188743680,
        dataset='math',objective='GRPO binary boxed MATH reward; group-normalized advantage',
        reward='VERL math-verify acc, 0/1; missing-gold groups have zero advantage',math_verify_boxed_only=True,
        optimizer='AdamW',learning_rate=1e-5,weight_decay=0.,betas=[.9,.999],grad_clip=1.,
        warmup_steps=0,scheduler='constant',epochs=1,examples_per_epoch=7500,batch_size=64,group_size=8,
        responses_per_full_batch=512,steps_per_epoch=steps,total_steps=steps,micro_batch=1,
        world_size=8,gpu_ids=list(range(8)),ppo_epochs=1,ppo_mini_batch_size_prompts=64,
        ppo_mini_batch_size_responses=512,loss_agg_mode='seq-mean-token-mean',clip_ratio=.2,clip_ratio_c=3.,
        norm_adv_by_std=True,advantage_std_correction=1,advantage_epsilon=1e-6,
        use_kl_loss=False,use_kl_in_reward=False,entropy_coefficient=0.,teacher_response_forward=False,
        rollout_correction='detached token-level IS, cap 2; native vLLM sampled log probabilities',
        old_policy='actor log_probs.detach(), one on-policy minibatch and one PPO epoch (VERL default)',
        loss_distribution='Unfiltered actor log probabilities at sampled response tokens',
        enable_thinking=False,max_prompt_length=2048,max_response_length=4096,
        train_sampling=TRAIN_SAMPLING,eval_sampling=QWEN3_EVAL_SAMPLING,
        sampling_source='https://huggingface.co/Qwen/Qwen3-4B#best-practices',
        eval_steps=eval_steps,eval_questions=500,eval_samples=4,total_evaluations=1+len(eval_steps),
        evaluations_per_epoch=4,checkpoint_steps=eval_steps,save_evaluation_bridges=False,
        validation_before_training=True,initial_eval_step=0,eval_seed='1234 + 4*index + sample_index',
        eval_coverage='MATH500 x4 before training, then every quarter epoch',
        eval_selection='Store bridge, optimizer and rank RNG at each quarterly evaluation',
        training_rollout_backend='HF bridge prefill + native vLLM full-KV decode',
        evaluation_backend='HF bridge prefill + native vLLM full-KV decode',
        vllm_tensor_parallel_size=1,vllm_replicas=8,vllm_memory_fraction=.18,
        vllm_max_model_len=6144,vllm_max_num_seqs=16,vllm_decode_batch_size=0,
        vllm_attention_backend='FLASH_ATTN',vllm_flash_attn_version=2,
        vllm_allocation_note='Full-attention KV needs more memory than Qwen3.5 hybrid; validate resident worst case',
        source_sha256=hashlib.sha256(train_source.read_bytes()).hexdigest(),
        eval_source_sha256=hashlib.sha256(eval_source.read_bytes()).hexdigest(),
        train_prompts_sha256=hashlib.sha256((base/'train_prompts.json').read_bytes()).hexdigest(),
        eval_prompts_sha256=hashlib.sha256((base/'eval_prompts.json').read_bytes()).hexdigest(),
        filter='None; retain clipped, incorrect and empty-gold training samples',resume=False,
        train_seed_formula='42 + 100000*zero_based_epoch + 1000*global_step + 8*question_index + sample_index',
        data_order='Fresh torch.randperm(7500), seed 42 + zero_based_epoch',
        wandb_project='hidden-bridge-math-grpo',
        wandb_run_name='qwen3_30b_a3b_to_4b_uniform_endpoints_constant_gate1_grpo_g8_bs64_4k_lr1e5_1epoch_eval5_vllm_8gpu')
    if transfer_mode in ('sequential','post_prefill','c2c_soft'):
        protocol.update(transfer_mode=transfer_mode,
            formula='K_s[i] + P_K_i(K_t[layer_map[i]]); V_s[i] + P_V_i(V_t[layer_map[i]])',
            hidden_space=None,transfer_space='K post K-norm and RoPE; V post v_proj',
            teacher_kv_width=512,student_kv_width=1024,trainable_parameters=37748736,
            query_transfer=False,
            prefix_propagation=transfer_mode=='sequential',
            transfer_timing='inside each prefix attention' if transfer_mode=='sequential' else 'after independent native prefix prefills',
            wandb_run_name=protocol['wandb_run_name'].replace('_uniform_endpoints_',f'_kv_{transfer_mode}_uniform_endpoints_'))
    if transfer_mode=='c2c_soft':
        from c2c_soft_grpo_bridge import C2C_REVISION,fuser_parameter_count
        protocol.update(trainable_parameters=36*fuser_parameter_count(512,1024,8),
            formula='KV_s + sigmoid(gate_logit) * sigmoid(head_logits(KV_s, KV_t)) * F(KV_s, KV_t)',
            constant_gate=None,gate_mode='deterministic sigmoid for rollout/update/eval',
            eval_gate_modes=['soft','hard'],
            hard_eval_gate_rule='scalar gate_logit > 0; zero off; dynamic head weights remain sigmoid',
            evaluation_policy='Each scheduled evaluation runs soft and hard on identical questions/seeds; restore soft before training',
            gumbel=False,dropout=0.,c2c_source_revision=C2C_REVISION,
            fuser=dict(hidden_dim=1024,intermediate_dim=1024,num_layers=3,
                       activation='GELU',normalization='RMSNorm eps=1e-6',
                       initial_output_scale=1e-4,initial_gate=.5,initial_head_weight=.5),
            initialization='standard internal MLP; output uniform +/-1e-4/sqrt(1024), bias zero; dynamic head/gate logits zero; seed42; no checkpoint',
            wandb_run_name=protocol['wandb_run_name'].replace('_kv_c2c_soft_','_c2c_soft_').replace('_constant_gate1_','_soft_gate_'))
    if transfer_mode in ('hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard'):
        zero=transfer_mode=='hidden_zero'
        linear=transfer_mode not in ('hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero')
        protocol.update(transfer_mode=transfer_mode,
            formula='h_s[i] + P_i(h_t[layer_map[i]])' if linear else
                'h_s[i] + W2_i(GELU(W1_i(h_t[layer_map[i]])))',
            projection=dict(type='linear' if linear else 'two_layer_mlp',input_dim=2048,output_dim=2560,
                             intermediate_dim=None if linear else 2048,activation=None if linear else 'GELU exact',bias=False),
            initialization='all-zero linear projection matrices; no checkpoint' if zero else
                 'linear uniform +/-1/sqrt(2048), seed42; scale=1; no checkpoint' if linear else
                'both MLP matrices uniform +/-1e-4/sqrt(2048), seed42; no checkpoint',
            trainable_parameters=188743680 if linear else 339738624,
            wandb_run_name=protocol['wandb_run_name'].replace('_uniform_endpoints_',f'_{transfer_mode}_uniform_endpoints_'))
    if transfer_mode in ('hidden_mlp_w2zero','hidden_factorized_w2zero'):
        nonlinear=transfer_mode=='hidden_mlp_w2zero'
        protocol['formula']='h_s[i] + W2_i(GELU(W1_i(h_t[layer_map[i]])))' if nonlinear else 'h_s[i] + W2_i(W1_i(h_t[layer_map[i]]))'
        protocol['projection'].update(activation='GELU exact' if nonlinear else None)
        protocol['initialization']='W1 Kaiming uniform a=sqrt(5), W2 all-zero, seed42; no checkpoint'
    if transfer_mode=='hidden_local_kv':
        protocol.update(transfer_mode=transfer_mode,
            formula='local_h = h_s_native + P(h_t_native); K/V = native_student_KV_heads(input_RMSNorm(local_h))',
            hidden_space='native decoder block input before RMSNorm',
            projection=dict(type='linear',input_dim=2048,output_dim=2560,
                            intermediate_dim=None,activation=None,bias=False),
            kv_reconstruction='native student input RMSNorm, k_proj/v_proj, K-norm and RoPE',
            native_prompt_residual_stream=True,prefix_propagation=False,decode_propagation=True,
            extra_trainable_kv_projection=False,
            wandb_run_name=protocol['wandb_run_name'].replace('_uniform_endpoints_','_hidden_local_kv_uniform_endpoints_'))
    if transfer_mode!='hidden':
        protocol['wandb_run_name']='ablation_'+protocol['wandb_run_name']
    save(base/'mapping.json',dict(teacher_layers_0based=layer_map(),teacher_layers_1based=[i+1 for i in layer_map()]))
    save(base/'protocol.json',protocol)
    source_files=[Path(__file__),ROOT/'scripts/direct_residual_chat_bridge.py']
    if transfer_mode in ('sequential','post_prefill','c2c_soft'):
        source_files.append(ROOT/'scripts/qwen3_kv_residual_bridge.py')
    if transfer_mode in ('hidden_zero','hidden_mlp','hidden_mlp_w2zero','hidden_factorized_w2zero','hidden_random_standard'):
        source_files.append(ROOT/'scripts/qwen3_hidden_grpo_ablations.py')
        source_files.append(Path(__file__).with_name('eval_qwen3_30b_4b_bridge_vllm.py'))
    if transfer_mode=='hidden_local_kv':
        source_files.extend([ROOT/'scripts/qwen3_local_hidden_kv_bridge.py',
            ROOT/'scripts/qwen3_kv_residual_bridge.py',
            Path(__file__).with_name('eval_qwen3_30b_4b_bridge_vllm.py')])
    if transfer_mode=='c2c_soft':
        source_files.append(ROOT/'scripts/c2c_soft_grpo_bridge.py')
        source_files.append(Path(__file__).with_name('eval_qwen3_30b_4b_bridge_vllm.py'))
    source_files.extend(Path(__file__).with_name(name) for name in (
        'train_qwen35_bridge_opd_vllm.py','qwen35_bridge_opd_core.py','qwen35_bridge_grpo_core.py',
        'qwen35_bridge_opd_vllm_worker.py','qwen35_hybrid_memory_connector.py',
        'qwen35_bridge_opd_score_worker.py','prepare_qwen35_bridge_vllm.py'))
    snapshot=base/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in source_files:(snapshot/path.name).write_bytes(path.read_bytes())
    save(base/'source_sha256.json',{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files})
    return protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--transfer-mode',choices=TRANSFER_MODES,default='hidden')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--queue',action='store_true')
    args=parser.parse_args()
    marker=args.transfer_mode if args.transfer_mode.startswith('hidden') else f'kv_{args.transfer_mode}'
    default_base=BASE if args.transfer_mode=='hidden' else BASE.with_name(BASE.name.replace(
        '_math_grpo_',f'_{marker}_math_grpo_'))
    base=(args.root or default_base).absolute()
    prepare(base,transfer_mode=args.transfer_mode)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,root=str(base))),flush=True)
        return
    if args.detach:
        python=Path(os.environ.get('PLATOUR_BRIDGE_PYTHON',ROOT/'.runtime/readme-check/bridge/bin/python')).absolute()
        command=[str(python),'-u',str(Path(__file__).absolute()),'--queue','--root',str(base),
                 '--transfer-mode',args.transfer_mode]
        if args.smoke_only:command+=['--smoke-only']
        with (base/'controller.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        save(base/'controller.json',dict(pid=process.pid,started=time.time(),command=command))
        print(json.dumps(dict(pid=process.pid,root=str(base))),flush=True)
    else:
        from run_qwen35_bridge_opd_vllm import queue
        queue(base,'grpo',smoke_only=args.smoke_only)


if __name__=='__main__':
    main()
