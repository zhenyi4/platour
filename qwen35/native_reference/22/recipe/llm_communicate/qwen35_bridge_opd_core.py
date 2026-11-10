"""On-policy full-vocabulary reverse KL for the last32 Qwen3.5 bridge."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
TRAIN_SAMPLING = dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
    presence_penalty=0., repetition_penalty=1., frequency_penalty=0.)
LEGACY_EVAL_SAMPLING = dict(temperature=1., top_p=1., top_k=40, min_p=0.,
    presence_penalty=2., repetition_penalty=1., frequency_penalty=0.)
GENERAL_SAMPLING = dict(temperature=1., top_p=.95, top_k=20, min_p=0.,
    presence_penalty=1.5, repetition_penalty=1., frequency_penalty=0.)
# Evaluation default. Thinking remains disabled; training is unchanged.
EVAL_SAMPLING = dict(GENERAL_SAMPLING)
BASE = ROOT / 'outputs/llm_communicate/qwen35_35b_to_4b_math_opd_constant1_lr1e5_6gpu_vllm'
GRPO_BASE = ROOT / 'outputs/llm_communicate/qwen35_35b_to_4b_math_grpo_constant1_lr1e5_bs64_g8_6gpu_vllm'
GRPO_122_BASE = ROOT / 'outputs/llm_communicate/qwen35_122b_to_4b_math_grpo_constant1_lr1e5_bs64_g8_8gpu_vllm'
GRPO_122_RELATIVE_BASE = ROOT / 'outputs/llm_communicate/qwen35_122b_to_4b_math_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_8gpu_vllm'
GRPO_122_FP8_RELATIVE_BASE = ROOT / 'outputs/llm_communicate/qwen35_122b_fp8_to_4b_math_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_8gpu_vllm'
GRPO_397_FP8_RELATIVE_BASE = ROOT / 'outputs/llm_communicate/qwen35_397b_fp8_to_4b_math_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_8gpu_vllm'
GRPO_35_ARGMAX_BASE = ROOT / 'outputs/llm_communicate/qwen35_35b_to_4b_math_grpo_mse20m_shared_argmax_constant1_lr1e5_bs64_g8_6gpu_vllm'
GRPO_35_RELATIVE_BASE = ROOT / 'outputs/llm_communicate/qwen35_35b_to_4b_math_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_6gpu_vllm'
MSE_ARGMAX_MAPPING = 'mse_per_student_shared_cosine_argmax'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def initialize(bridge):
    import torch
    bound = 1e-4 / math.sqrt(bridge.teacher.config.hidden_size)
    rng = torch.Generator(device=bridge.student.device).manual_seed(42)
    with torch.no_grad():
        for p in bridge.bridge_parameters():
            p.uniform_(-bound, bound, generator=rng)
    bridge.initialization = 'random'
    return dict(kind='small random uniform', seed=42, bound=bound,
                scale=1e-4, constant_gate=1., sft_checkpoint_loaded=False)


def weight_digest(bridge):
    """Hash actual ordered FP32 projection bytes, not an optimizer-step label."""
    result = hashlib.sha256()
    for name, value in (bridge.bridge_state_tensors() if hasattr(bridge,'bridge_state_tensors') else bridge.maps.state_dict()).items():
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def shared_cosine_argmax(matrix):
    """Independent row maxima, retaining repeats/decreases/type changes."""
    assert len(matrix)==32 and all(len(row)==40 for row in matrix)
    assert all(math.isfinite(value) for row in matrix for value in row)
    return [max(range(40),key=lambda j:(row[j],-j)) for row in matrix]


def mse_argmax_selection():
    path=ROOT/'outputs/llm_communicate/qwen35_35b_4b_fineweb20m_allpairs_halfep_mapping_then_122b_6gpu/mse_allpairs/statistics.json'
    data=json.loads(path.read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert data['complete'] and data['measured_pairs']==1280
    assert data['latent_dim']==2560 and data['train_tokens_per_pair']==19996672
    assert json.loads(path.with_name('complete.json').read_text())['statistics_sha256']==digest
    mapping=shared_cosine_argmax(data['matrices']['align_cosine'])
    previous=ROOT/'outputs/llm_communicate/qwen35_35b_to_4b_math_mse20m_shared_argmax_zero_epoch3_6gpu/mapping.json'
    spec=json.loads(previous.read_text())
    assert spec['teacher_layers_0based']==mapping
    assert spec['selection']['statistics_sha256']==digest
    assert spec['selection']['use_mse_projector_weights'] is False
    return mapping,dict(spec['selection'],previous_sft_mapping_path=str(previous),
                        previous_sft_mapping_sha256=hashlib.sha256(previous.read_bytes()).hexdigest())


def opd_loss(bridge, prompt, response, chunk_tokens=32):
    """All sampled next-token positions, including EOS or the last cap token.

    Both policies see the identical prompt and sampled response history. Only
    prompt[:-1] gets teacher hidden injection. The teacher is unmodified.
    Student probabilities remain attached in KL(student || teacher).
    """
    import torch
    from full_vocab_kl import full_vocab_kl
    if len(prompt) < 2 or not response:
        raise ValueError('A prompt and a nonempty sampled response are required')
    ids = torch.tensor([prompt + response[:-1]], device=bridge.student.device)
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    with torch.no_grad():
        output = bridge.teacher.model(input_ids=ids, attention_mask=mask,
            position_ids=positions, use_cache=False, output_hidden_states=True)
        teacher_prediction = output.last_hidden_state[:, len(prompt)-1:].contiguous()
        features = tuple(h[:, :len(prompt)].contiguous().clone()
                         for h in bridge.teacher_features(output.hidden_states))
        del output
    injection = torch.ones((1, len(prompt)), device=ids.device, dtype=torch.bool)
    injection[:, -1] = False
    with bridge.inject(features, injection):
        student = bridge.student.model(input_ids=ids, attention_mask=mask,
            position_ids=positions, use_cache=False).last_hidden_state[:, len(prompt)-1:]
    active = torch.ones(student.shape[:2], device=ids.device, dtype=torch.bool)
    assert student.shape[1] == teacher_prediction.shape[1] == len(response)
    return full_vocab_kl(student, teacher_prediction, bridge.student.lm_head,
        bridge.teacher.lm_head, active, direction='reverse', temperature=1.,
        chunk_tokens=chunk_tokens)


def first_tokens(logits, rows, cap, sampling):
    import torch
    # The first sampled token is retained from the original HF bridge prefill.
    uniforms = torch.stack([torch.rand(cap, device=logits.device,
        generator=torch.Generator(device=logits.device).manual_seed(r['sampling_seed']))[0]
        for r in rows])
    if sampling == LEGACY_EVAL_SAMPLING:
        from eval_qwen35_35b_4b_math_bridge_8gpu import sample_tokens
        return sample_tokens(logits, torch.zeros_like(logits, dtype=torch.bool), uniforms).tolist()
    if sampling == GENERAL_SAMPLING:
        # Match vLLM's top-k threshold (including ties), followed by ascending
        # nucleus filtering. No response tokens exist yet, so penalties are zero.
        values, indices = logits.float().sort(dim=-1, descending=False)
        threshold = values[:, values.shape[-1] - min(20, values.shape[-1]), None]
        values = values.masked_fill(values < threshold, -torch.inf)
        probabilities = values.softmax(-1)
        excluded = probabilities.cumsum(-1) <= 1 - sampling['top_p']
        excluded[:, -1] = False
        probabilities = values.masked_fill(excluded, -torch.inf).softmax(-1)
        chosen = (probabilities.cumsum(-1) < uniforms[:, None]).sum(-1, keepdim=True)
        return indices.gather(-1, chosen.clamp_max(indices.shape[-1] - 1)).squeeze(-1).tolist()
    assert sampling == TRAIN_SAMPLING
    probabilities = logits.float().softmax(-1)
    chosen = (probabilities.cumsum(-1) < uniforms[:, None]).sum(-1)
    return chosen.clamp_max(logits.shape[-1] - 1).tolist()


def quarter_epoch_schedule(examples, batch_size=64, epochs=3, evaluations_per_epoch=4):
    """Include the last short batch and round each evaluation boundary upward."""
    assert examples > 0 and batch_size > 0 and epochs > 0 and evaluations_per_epoch > 0
    steps = math.ceil(examples / batch_size)
    return steps, sorted({epoch * steps + math.ceil(part * steps / evaluations_per_epoch)
                         for epoch in range(epochs) for part in range(1, evaluations_per_epoch+1)})


def early_stop_config(interval, patience=2, max_steps=None, clip_abs=.05, clip_ratio=2., skip_initial_eval=False):
    """Validation-based early stop (all optional; absent from protocol => original behaviour).

    An evaluation counts as *not better* when avg@4 is not strictly above the best so far, or when
    its clipping rate is above clip_abs, or at least clip_ratio times the clipping rate of the
    current best evaluation (but only once that rate is above 1%). Stop after `patience`
    consecutive not-better evaluations; the checkpoint of the best evaluation is the result.
    """
    assert interval > 0 and patience > 0
    assert max_steps is None or (max_steps >= interval and max_steps % interval == 0), \
        'max_steps must be a multiple of the validation interval so the cap coincides with an evaluation'
    return dict(interval=interval, patience=patience, max_steps=max_steps, clip_abs=clip_abs,
        clip_ratio=clip_ratio, skip_initial_eval=bool(skip_initial_eval),
        rule='Stop after `patience` consecutive evaluations that are not strictly better than the best avg@4 '
             'or whose clipping rate breaches the guard; best-step checkpoint is retained')


def early_stop_update(state, step, accuracy, clipping, config):
    """Pure update of the early-stop state after one validation; returns (new_state, stop_reason|None)."""
    state = dict(state or dict(best_step=None, best_accuracy=None, best_clipping=None, fails=0, history=[]))
    history = list(state['history'])
    best_acc, best_clip = state['best_accuracy'], state['best_clipping']
    state['samples_per_question'] = config.get('samples_per_question', 4)
    guard = ((config.get('clip_abs') is not None and clipping > config['clip_abs']) or
             (config.get('clip_ratio') is not None and best_clip is not None and best_clip > .01 and
              clipping >= config['clip_ratio']*best_clip))
    improved = best_acc is None or (accuracy > best_acc and not guard)
    if improved:
        state.update(best_step=step, best_accuracy=accuracy, best_clipping=clipping, fails=0)
    else:
        state['fails'] += 1
    history.append(dict(step=step, **{f'avg_at_{state["samples_per_question"]}':accuracy},
        clipping_rate=clipping, improved=improved, clip_guard=guard, fails=state['fails']))
    state['history'] = history
    if state['fails'] >= config['patience']:
        return state, f"patience: {config['patience']} consecutive evaluations not better than step {state['best_step']}"
    if config['max_steps'] is not None and step >= config['max_steps']:
        return state, f"max_steps {config['max_steps']} reached"
    return state, None


def prepare(base=BASE, objective='opd',teacher='Qwen3.5-35B-A3B',layer_mapping='last_aligned',gpu_count=8,teacher_precision='bf16',dataset='math',max_response_length=4096,epochs=None,evaluations_per_epoch=None,tuning=None):
    from transformers import AutoTokenizer
    tuning = dict(tuning or {})
    assert gpu_count in (6,8)
    assert dataset in ('math','dapo') and max_response_length > 0
    if epochs is not None or evaluations_per_epoch is not None:
        assert dataset=='dapo', 'Schedule overrides currently apply to DAPO only'
    assert epochs is None or epochs > 0
    assert evaluations_per_epoch is None or evaluations_per_epoch > 0
    if dataset=='dapo':assert objective=='grpo'
    assert teacher_precision in ('bf16','fp8')
    if teacher_precision=='fp8':
        from qwen35_122b_4b_math_bridge import FP8_KERNEL_REVISION
        assert teacher in ('Qwen3.5-122B-A10B','Qwen3.5-397B-A17B') and objective=='grpo'
        assert layer_mapping=='relative_depth_groups4' and gpu_count==8
    if teacher=='Qwen3.5-122B-A10B':assert gpu_count==8
    if teacher=='Qwen3.5-397B-A17B':assert gpu_count==8 and teacher_precision=='fp8'
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    if (base / 'protocol.json').exists():
        existing=json.loads((base / 'protocol.json').read_text())
        assert existing.get('method','opd')==objective
        assert existing['teacher']==teacher
        assert existing.get('layer_mapping','last_aligned')==layer_mapping
        assert existing['gpu_ids']==list(range(gpu_count)), 'Use a separate output directory when changing GPU count'
        assert existing.get('teacher_weight_precision','bf16')==teacher_precision, 'FP8 needs a separate output directory'
        assert existing.get('dataset','math')==dataset, 'Use a separate output directory when changing datasets'
        assert existing['max_response_length']==max_response_length, 'Use a separate output directory when changing response length'
        assert existing['eval_sampling']==EVAL_SAMPLING, 'Use a separate output directory when changing evaluation sampling'
        if epochs is not None:
            assert existing['epochs']==epochs, 'Use a separate output directory when changing epochs'
        if evaluations_per_epoch is not None:
            assert existing.get('evaluations_per_epoch',4)==evaluations_per_epoch, 'Use a separate output directory when changing evaluation frequency'
        if layer_mapping==MSE_ARGMAX_MAPPING:
            mapping,selection=mse_argmax_selection()
            assert existing['layer_map']==mapping and existing['mse_selection']==selection
        if tuning:
            # An existing protocol is immutable: a differing request must go to a new output root.
            wanted=tuning_updates(tuning,existing)
            wanted.pop('wandb_run_name',None)
            changed={k:(existing.get(k),v) for k,v in wanted.items() if existing.get(k)!=v}
            assert not changed, f'Use a separate output directory when changing tuning options: {changed}'
        return existing
    assert layer_mapping in ('last_aligned','relative_depth_groups4',MSE_ARGMAX_MAPPING)
    if layer_mapping=='relative_depth_groups4':assert objective=='grpo'
    if layer_mapping==MSE_ARGMAX_MAPPING:assert teacher=='Qwen3.5-35B-A3B' and objective=='grpo'
    source = ROOT / 'outputs/llm_communicate/qwen35_train_generation/inputs/math.jsonl'
    rows = [json.loads(line) for line in source.open()] if dataset=='math' else []
    if dataset=='math':assert [r['index'] for r in rows] == list(range(7500))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'data/models/Qwen3.5-4B', local_files_only=True)
    teacher_checkpoint=ROOT/'data/models'/(teacher+'-FP8' if teacher_precision=='fp8' else teacher)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint, local_files_only=True)
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    prompts = []
    for row in rows:
        message = [dict(role='user', content=row['input_text'])]
        ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        other = teacher_tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        if isinstance(ids, dict) or hasattr(ids, 'keys'):
            ids = ids['input_ids']
        if isinstance(other, dict) or hasattr(other, 'keys'):
            other = other['input_ids']
        assert isinstance(ids, list) and all(isinstance(token, int) for token in ids)
        assert ids == other and 2 <= len(ids) <= 2048
        prompts.append(dict(index=row['index'], question=row['question'], gold=row['gold'],
                            input_text=row['input_text'], prompt_token_ids=ids))
    evaluation = ROOT / 'outputs/llm_communicate/qwen35_math500_official_sample4_cap4096/Qwen3.5-4B/prompts.json'
    if dataset=='dapo':
        from prepare_qwen35_dapo_grpo import tokenized_splits
        prompts,eval_rows=tokenized_splits(base,tokenizer,teacher_tokenizer)
        source=ROOT/'data/dapo_math_17k/train.parquet'
        evaluation=ROOT/'data/dapo_math_17k/validation.parquet'
    else:
        eval_rows = json.loads(evaluation.read_text())
    assert len(eval_rows) == 500 and len({r['index'] for r in eval_rows}) == 500
    assert not {tuple(r['prompt_token_ids']) for r in prompts} & {tuple(r['prompt_token_ids']) for r in eval_rows}
    save(base / 'train_prompts.json', prompts)
    save(base / 'eval_prompts.json', eval_rows)
    protocol = dict(teacher='Qwen3.5-35B-A3B', student='Qwen3.5-4B',
        formula='h_s[i] + P_i(h_t[i+8])', layer_map=list(range(8,40)),layer_mapping=layer_mapping,
        hidden_space='decoder block input before RMSNorm', injection='prompt[:-1] only',
        constant_gate=1., initialization='uniform +/-1e-4/sqrt(2048), seed42; no SFT or pretrain weights',
        frozen_backbones=True, projection_dtype='float32', trainable_parameters=167772160,
        objective='full vocabulary KL(student || unmodified teacher) on student rollout positions; global token mean',
        kl_temperature=1., kl_chunk_tokens=32, optimizer='AdamW', learning_rate=1e-5,
        weight_decay=0., betas=[.9,.999], grad_clip=1., warmup_steps=0, scheduler='constant',
        epochs=3, examples_per_epoch=7500, batch_size=128, group_size=1,
        steps_per_epoch=59, total_steps=177, micro_batch=1, world_size=gpu_count, gpu_ids=list(range(gpu_count)),
        enable_thinking=False, max_prompt_length=2048, max_response_length=max_response_length,
        train_sampling=TRAIN_SAMPLING, eval_sampling=EVAL_SAMPLING,
        sampling_source='https://huggingface.co/Qwen/Qwen3.5-4B#using-qwen35-via-the-chat-completions-api',
        eval_sampling_default_since='evaluation default: top_p .95, top_k 20, presence penalty 1.5',
        eval_steps=[15,30,45,59,74,89,104,118,133,148,163,177],
        eval_questions=500, eval_samples=4, eval_seed='1234 + 4*index + sample_index',
        eval_selection='No early stopping; store all epoch checkpoints',
        training_rollout_backend='HF bridge prefill + native vLLM hybrid-cache decode',
        evaluation_backend='HF bridge prefill + native vLLM hybrid-cache decode',
        vllm_tensor_parallel_size=1, vllm_replicas=gpu_count, vllm_memory_fraction=.12,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        eval_source_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        filter='None; retain clipped, incorrect and empty-gold training samples',
        resume=False, validation_before_training=True, initial_eval_step=0,
        total_evaluations=13,
        train_seed_formula='42 + 100000*zero_based_epoch + 1000*global_step + question_index',
        data_order='Fresh torch.randperm(7500), seed 42 + zero_based_epoch',
        loss_distribution='Unfiltered, unpenalized complete LM distributions',
        eval_coverage='MATH500 x4 before training, then MATH500 x4 every quarter epoch')
    protocol['method']=objective
    if objective=='grpo':
        protocol.update(objective='GRPO binary boxed MATH reward; group-normalized advantage',
            reward='VERL math-verify acc, 0/1; missing-gold groups have zero advantage',
            batch_size=64,group_size=8,responses_per_full_batch=512,
            steps_per_epoch=118,total_steps=354,ppo_epochs=1,
            ppo_mini_batch_size_prompts=64,ppo_mini_batch_size_responses=512,
            loss_agg_mode='seq-mean-token-mean',clip_ratio=.2,clip_ratio_c=3.,
            norm_adv_by_std=True,advantage_std_correction=1,advantage_epsilon=1e-6,
            use_kl_loss=False,use_kl_in_reward=False,entropy_coefficient=0.,
            rollout_correction='detached token-level IS, cap 2; native vLLM sampled log probabilities',
            old_policy='actor log_probs.detach(), one on-policy minibatch and one PPO epoch (VERL default)',
            teacher_response_forward=False,
            eval_steps=[30,59,89,118,148,177,207,236,266,295,325,354],
            train_seed_formula='42 + 100000*zero_based_epoch + 1000*global_step + 8*question_index + sample_index',
            loss_distribution='Unfiltered actor log probabilities at sampled response tokens')
        protocol.pop('kl_temperature',None)
        protocol.pop('kl_chunk_tokens',None)
    else:
        assert objective=='opd'
    if teacher=='Qwen3.5-122B-A10B':
        assert objective=='grpo'
        protocol.update(teacher=teacher,formula='h_s[i] + P_i(h_t[i+16])',layer_map=list(range(16,48)),
            initialization='uniform +/-1e-4/sqrt(3072), seed42; no SFT or pretrain weights',
            trainable_parameters=251658240,world_size=4,gpu_ids=list(range(8)),
            decoder_gpu_ids=[0,2,4,6],teacher_gpu_pairs=[[0,1],[2,3],[4,5],[6,7]],
            teacher_model_parallel_size=2,teacher_layer_split=[16,32],vllm_replicas=4,
            teacher_precision='Original BF16, no quantization or CPU offload',
            save_evaluation_bridges=True,
            wandb_run_name='qwen35_122b_to_4b_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_8gpu')
    elif teacher=='Qwen3.5-397B-A17B':
        assert objective=='grpo' and teacher_precision=='fp8'
        protocol.update(teacher=teacher,initialization='uniform +/-1e-4/sqrt(4096), seed42; no SFT or pretrain weights',
            trainable_parameters=335544320,world_size=2,gpu_ids=list(range(8)),
            decoder_gpu_ids=[0,4],teacher_gpu_groups=[[0,1,2,3],[4,5,6,7]],
            teacher_model_parallel_size=4,teacher_layer_split=[12,16,16,16],vllm_replicas=2,
            save_evaluation_bridges=True)
    else:
        assert teacher=='Qwen3.5-35B-A3B'
        if layer_mapping==MSE_ARGMAX_MAPPING:
            mapping,selection=mse_argmax_selection()
            save(base/'mapping.json',dict(teacher_layers_0based=mapping,
                teacher_layers_1based=[j+1 for j in mapping],selection=selection))
            protocol.update(formula='h_s[i] + P_i(h_t[layer_map[i]])',layer_map=mapping,
                mse_selection=selection,group_constraint=False,monotone_constraint=False,
                attention_types_preserved=False,use_mse_projector_weights=False,
                save_evaluation_bridges=True,
                wandb_run_name=f'qwen35_35b_to_4b_mse20m_shared_argmax_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_{gpu_count}gpu')
    if layer_mapping=='relative_depth_groups4':
        from qwen35_122b_4b_math_bridge import relative_depth_layer_map
        teacher_layers={'Qwen3.5-122B-A10B':48,'Qwen3.5-397B-A17B':60,'Qwen3.5-35B-A3B':40}[teacher]
        layers=relative_depth_layer_map(32,teacher_layers)
        group_map=[j//4 for j in layers[::4]]
        label={48:'122b',60:'397b',40:'35b'}[teacher_layers]
        gpu_count=len(protocol['gpu_ids'])
        rule=f'round_half_up(student_group*{teacher_layers//4-1}/7); preserve intra-group offset; zero-based indices'
        protocol.update(formula='h_s[i] + P_i(h_t[layer_map[i]])',layer_map=layers,
            mapping_group_size=4,teacher_group_map=group_map,mapping_rule=rule,
            group_constraint=True,monotone_constraint=True,attention_types_preserved=True,
            use_mse_projector_weights=False,save_evaluation_bridges=True,
            wandb_run_name=f'qwen35_{label}_to_4b_relative_depth_groups4_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_{gpu_count}gpu')
        save(base/'mapping.json',dict(teacher_layers_0based=layers,
            teacher_layers_1based=[j+1 for j in layers],teacher_groups_0based=group_map,
            teacher_groups_1based=[j+1 for j in group_map],group_size=4,rule=rule))
    if teacher_precision=='fp8':
        label='397b' if teacher=='Qwen3.5-397B-A17B' else '122b'
        download=ROOT/f'outputs/model_downloads/qwen35_{label}_fp8'
        manifest=json.loads((download/'manifest.json').read_text())
        status=json.loads((download/'state.json').read_text())
        assert status['status']=='complete' and status['completed_shards']==(94 if label=='397b' else 39)
        quant=json.loads((teacher_checkpoint/'config.json').read_text())['quantization_config']
        assert quant['quant_method']=='fp8' and quant['weight_block_size']==[128,128]
        protocol.update(teacher_weight_precision='fp8',teacher_checkpoint=str(teacher_checkpoint),
            teacher_checkpoint_revision=manifest['revision'],
            teacher_hidden_transfer_dtype='bfloat16',
            teacher_precision='Official FP8 block128 weights/dynamic activations; BF16 hidden transfer; no CPU offload',
            teacher_quantization={k:v for k,v in quant.items() if k!='modules_to_not_convert'},
            fp8_kernel_cache=str(ROOT/'.runtime/fp8-kernels-cache'),
            fp8_kernel_revision=FP8_KERNEL_REVISION,
            wandb_run_name=f'qwen35_{label}_fp8_to_4b_relative_depth_groups4_constant_gate1_grpo_g8_bs64_4k_lr1e5_3epoch_eval13_vllm_8gpu')
    if dataset=='dapo':
        protocol['epochs']=epochs if epochs is not None else protocol['epochs']
        frequency=evaluations_per_epoch if evaluations_per_epoch is not None else 4
        steps,eval_steps=quarter_epoch_schedule(len(prompts),protocol['batch_size'],protocol['epochs'],frequency)
        protocol.update(dataset='dapo',examples_per_epoch=len(prompts),steps_per_epoch=steps,
            evaluations_per_epoch=frequency,
            total_steps=steps*protocol['epochs'],eval_steps=eval_steps,checkpoint_steps=eval_steps,
            total_evaluations=1+len(eval_steps),eval_questions=len(eval_rows),
            reward='VERL math-verify binary acc, full-response Answer: and boxed extraction',
            objective='GRPO binary DAPO reward; group-normalized advantage',
            math_verify_boxed_only=False,
            val_metric='answer',
            eval_metric=('avg@4 = last "Answer:" match verified with math-verify, length-capped responses 0 '
                         '(val/avg@4); pass@4 separately; legacy full-response parse logged as val/avg@4_full, '
                         'training format rule as val/avg@4_strict'),
            eval_coverage=f'DAPO validation500 x4 before training and every 1/{frequency} epoch',
            eval_selection=f'No early stopping; full bridge/optimizer/rank-RNG checkpoint after every 1/{frequency}-epoch eval',
            save_evaluation_bridges=False,wandb_project='hidden-bridge-dapo-grpo',
            student_activation_checkpointing=True,
            activation_checkpointing='non-reentrant per-block forward; bridge injection outside replay; eval mode retained',
            data_order=f'Fresh torch.randperm({len(prompts)}), seed 42 + zero_based_epoch')
        label=teacher.split('-')[1].lower()+('_fp8' if teacher_precision=='fp8' else '')
        protocol['wandb_run_name']=f'qwen35_{label}_to_4b_dapo_relative_depth_groups4_constant_gate1_grpo_g8_bs64_16k_lr1e5_{protocol["epochs"]}epoch_eval{1+len(eval_steps)}_vllm_8gpu_evalgeneral_gc'
    protocol['vllm_max_model_len']=protocol['max_prompt_length']+max_response_length
    if teacher=='Qwen3.5-35B-A3B' and dataset=='dapo' and max_response_length==16384:
        # Resident BF16 teacher + HF actor + vLLM, measured on the original eight-GPU hosts.
        # Existing protocol files return above and retain their original settings.
        protocol.update(vllm_max_num_seqs=64,vllm_decode_batch_size=0,vllm_memory_fraction=.30,
            vllm_rollout_tuning='rollout_queue_benchmark; 64 active, all local requests queued; 16K actor backward passed')
    protocol.update(tuning_updates(tuning,protocol))
    save(base / 'protocol.json', protocol)
    return protocol


def tuning_updates(tuning, protocol):
    """Protocol fields requested on the command line (empty for the reference behaviour).

    early_stop: dict from early_stop_config(); vllm_memory_fraction / vllm_max_num_seqs /
    vllm_decode_batch_size override the per-teacher defaults; wandb_run_suffix is appended to the
    W&B run name; skip_initial_eval removes the step-0 validation.
    """
    updates={}
    tuning=dict(tuning or {})
    unknown=set(tuning)-{'early_stop','vllm_memory_fraction','vllm_max_num_seqs','vllm_decode_batch_size','wandb_run_suffix','skip_initial_eval','format_penalty','learning_rate'}
    assert not unknown, f'Unknown tuning options: {sorted(unknown)}'
    for key in ('vllm_memory_fraction','vllm_max_num_seqs','vllm_decode_batch_size'):
        if tuning.get(key) is not None:
            updates[key]=tuning[key]
    if updates:
        updates['vllm_rollout_tuning']='command-line override: '+', '.join(f'{k}={updates[k]}' for k in sorted(updates))
    if tuning.get('format_penalty'):
        assert protocol.get('dataset')=='dapo', 'The format penalty is defined for the DAPO "Answer:" prompt'
        updates.update(format_penalty=True,
            reward='VERL math-verify binary acc with format penalty: 0 if finish_reason==length or the last non-empty '
                   'line is not "Answer: X"; otherwise only that line is verified. Validation is unpenalised.')
    name=protocol['wandb_run_name']
    if tuning.get('learning_rate') is not None and tuning['learning_rate']!=protocol['learning_rate']:
        lr=float(tuning['learning_rate'])
        assert 0<lr<1e-2 and protocol.get('scheduler')=='constant' and protocol.get('warmup_steps')==0
        updates['learning_rate']=lr
        mantissa,exponent=f'{lr:.0e}'.split('e')
        tag=f"_lr{mantissa}e{abs(int(exponent))}_"
        assert '_lr1e5_' in name, f'Cannot relabel the learning rate in {name}'
        name=name.replace('_lr1e5_',tag)
    if tuning.get('wandb_run_suffix'):
        name+=tuning['wandb_run_suffix']
    if name!=protocol['wandb_run_name']:
        updates['wandb_run_name']=name
    early=tuning.get('early_stop')
    skip=bool(tuning.get('skip_initial_eval'))
    if early:
        assert protocol.get('dataset')=='dapo', 'Validation-based early stop is defined for the DAPO runs'
        steps=protocol['eval_steps']
        interval=steps[0]
        assert steps==[interval*k for k in range(1,len(steps)+1)], 'Early stop needs evenly spaced validation steps'
        early=early_stop_config(interval,early.get('patience',2),early.get('max_steps'),early.get('clip_abs',.05),
            early.get('clip_ratio',2.),skip)
        eval_steps=[s for s in steps if early['max_steps'] is None or s<=early['max_steps']]
        updates.update(early_stop=early,eval_steps=eval_steps,checkpoint_steps=eval_steps,
            total_evaluations=len(eval_steps)+(0 if skip else 1),
            eval_selection=f"Early stop (patience {early['patience']}, interval {early['interval']}, max_steps {early['max_steps']}, "
                           f"clip guard abs {early['clip_abs']} / ratio {early['clip_ratio']}); full checkpoint after every evaluation")
    if skip:
        steps_text=updates.get('eval_steps',protocol['eval_steps'])
        updates.update(validation_before_training=False,initial_eval_step=None,
            total_evaluations=len(steps_text),
            eval_coverage=f"DAPO validation500 x4 at steps {steps_text} (step-0 evaluation skipped)")
    return updates
