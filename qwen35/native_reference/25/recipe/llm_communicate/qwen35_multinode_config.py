"""Pure topology and immutable protocol preparation for two-node FP8 bridge GRPO."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from qwen35_bridge_opd_core import ROOT, save, early_stop_config
from formal_config import MAPPING, RULE, layer_map_b, validation_schedule

ORIGINAL = ROOT.parents[2]
BUNDLE = ROOT.parent
TEACHERS = {'397b': 'Qwen3.5-397B-A17B'}
PARALLEL = {'397b': 4}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def topology(teacher, nnodes=2, gpus_per_node=8):
    if teacher not in PARALLEL or nnodes != 2:
        raise ValueError('This launcher supports exactly two nodes and teacher 122b/397b')
    parallel = PARALLEL[teacher]
    if gpus_per_node < parallel or gpus_per_node % parallel:
        raise ValueError('Local GPU count must be divisible by teacher model-parallel size')
    local_world = gpus_per_node // parallel
    return dict(nnodes=nnodes, gpus_per_node=gpus_per_node, local_world_size=local_world,
                world_size=nnodes*local_world, total_gpus=nnodes*gpus_per_node,
                teacher_model_parallel_size=parallel)


def rank_layout(protocol, environment=None):
    env = os.environ if environment is None else environment
    rank, world, local = (int(env[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    local_world = int(env['LOCAL_WORLD_SIZE'])
    node = int(env.get('GROUP_RANK', rank//local_world))
    expected = topology(protocol['teacher_key'], protocol['nnodes'], protocol['gpus_per_node'])
    if world != expected['world_size'] or local_world != expected['local_world_size']:
        raise ValueError('torchrun world/local-world sizes do not match the saved topology')
    if not (0 <= node < expected['nnodes'] and 0 <= local < local_world and rank == node*local_world+local):
        raise ValueError('Global rank, node rank and local rank disagree')
    primary = local*expected['teacher_model_parallel_size']
    return dict(rank=rank, world=world, local_rank=local, local_world=local_world, node_rank=node,
                primary=primary, devices=list(range(primary, primary+expected['teacher_model_parallel_size'])))


def initialize_zero(bridge):
    import torch
    with torch.no_grad():
        for parameter in bridge.bridge_parameters():
            parameter.zero_()
    bridge.initialization = 'zero'
    assert all(torch.count_nonzero(p).item() == 0 for p in bridge.bridge_parameters())
    return dict(kind='zero', seed=None, bound=0., scale=0., constant_gate=1., sft_checkpoint_loaded=False)


def source_manifest():
    return json.loads((BUNDLE/'source_manifest.json').read_text())


def verify_sources():
    manifest = source_manifest()
    for relative, value in manifest['files'].items():
        if digest(ROOT/relative) != value:
            raise RuntimeError(f'Multinode source changed: {relative}')
    if digest(ORIGINAL/manifest['launcher']) != manifest['launcher_sha256']:
        raise RuntimeError('Multinode shell launcher changed')
    return digest(BUNDLE/'source_manifest.json')


def configure_protocol(template, student, teacher, options):
    layout = topology(options['teacher'], options['nnodes'], options['gpus_per_node'])
    p = dict(template)
    mapping = layer_map_b(student['layer_types'], teacher['layer_types'])
    dataset = options['dataset']
    epochs = options['epochs']
    samples = 4 if dataset == 'dapo' else 16
    cap = options['max_steps']
    steps, evaluation = validation_schedule(p['examples_per_epoch'], p['batch_size'], epochs,
                                            options['eval_interval'], cap)
    early = early_stop_config(min(options['eval_interval'], steps), patience=options['patience'],
                             max_steps=None, clip_abs=.05 if dataset == 'dapo' else None,
                             clip_ratio=2. if dataset == 'dapo' else None)
    early.update(max_steps=cap, samples_per_question=samples, metric=f'avg@{samples}')
    if dataset=='deepcoder':
        evaluation=[step for step in evaluation if step!=294]
        early.update(warmup_steps=0,minimum_steps=0,patience=options['patience'],
                     clip_abs=.35,clip_ratio=2.,clip_ratio_floor=.15,
                     rule='Stop after two consecutive complete held-out DeepCoder avg@16 validations without guarded strict improvement; skip redundant step294, keep terminal295')
    parallel = layout['teacher_model_parallel_size']
    groups = [list(range(i*parallel, (i+1)*parallel)) for i in range(layout['local_world_size'])]
    p.pop('teacher_group_map', None)
    p.pop('teacher_gpu_pairs', None)
    p.update(layout, teacher_key=options['teacher'], dataset=dataset,
        gpu_ids=list(range(layout['gpus_per_node'])), decoder_gpu_ids=[g[0] for g in groups],
        teacher_gpu_groups=groups, gpu_index_scope='node-local; trainer output/RNG ranks are global',
        epochs=epochs, steps_per_epoch=steps, total_steps=min(steps*epochs,cap), eval_steps=evaluation,
        checkpoint_steps=evaluation, total_evaluations=1+len(evaluation), eval_samples=samples,
        eval_seed_stride=samples, eval_seed=f'1234 + {samples}*index + sample_index',
        initialization='zero', layer_mapping=MAPPING, layer_map=mapping, mapping_rule=RULE,
        mapping_group_size=1, group_constraint=False, monotone_constraint=True, full_layer_constraint=True,
        nearest_full_tie_break='shallower', attention_types_preserved=False, cross_attention_mapping_allowed=True,
        attention_layer_pairs=[dict(student_layer=i, teacher_layer=j,
            student_attention=student['layer_types'][i], teacher_attention=teacher['layer_types'][j])
            for i, j in enumerate(mapping)], formula='h_s[i] + P_i(h_t[layer_map[i]])',
        validation_before_training=True, initial_eval_step=0, early_stop=early,
        eval_selection=f'Validation best avg@{samples}; no test-based selection',
        eval_coverage=f'{p["eval_questions"]} questions x{samples} at 0 and {evaluation}',
        learning_rate=options['learning_rate'], format_penalty=True, vllm_memory_fraction=options['memory_fraction'],
        vllm_max_num_seqs=16, vllm_decode_batch_size=0, vllm_replicas=layout['world_size'],
        save_evaluation_bridges=False,stress_all_ranks=True,smoke_full_batch=True,
        wandb_project=options['project'],
        wandb_run_name=f'platour_{options["teacher"]}_fp8_to_4b_{dataset}_monotone_b_zero_grpo_lr{options["learning_rate"]:g}_fmtpen_g8_bs64_16k_2node_valavg{samples}',
        source_execution_root=str(ROOT), test_samples=8 if dataset == 'lcb' else None)
    return p


def prepare_protocol(base, options):
    from transformers import AutoTokenizer
    from qwen35_bridge_opd_core import prepare as core_prepare
    import code_prompts
    import code_judge
    base = Path(base)
    if (base/'protocol.json').exists():
        existing = json.loads((base/'protocol.json').read_text())
        if existing.get('multinode_options') != options or existing.get('source_manifest_sha256')!=verify_sources():
            raise ValueError('Saved protocol or source version differs; use a fresh --root')
        return existing
    if options['dataset']=='deepcoder':
        from training_policy import common_policy
        approved=common_policy()
        if options['teacher']!='397b' or options['epochs']!=1 or options['max_steps']!=approved['total_steps'] or options['eval_interval']!=42 or options['patience']!=2:
            raise RuntimeError('397B DeepCoder must match final one-epoch shared 295/42/2 code policy')
        template=ORIGINAL/'outputs/multinode_runs/397b_lcb_zero_b_v5/protocol.json'
        p=json.loads(template.read_text())
        train=json.loads(Path(approved['train_prompts_path']).read_text())
        validation=json.loads(Path(approved['eval_prompts_path']).read_text())
        data=dict(train=len(train),validation=len(validation),source='filtered_DeepCoder_v5',
                  source_revision=approved['source_revision'],checker_policy_sha256=approved['checker_policy_sha256'])
        p.update(examples_per_epoch=len(train),eval_questions=len(validation),
                 source_sha256=approved['train_prompts_sha256'],eval_source_sha256=approved['eval_prompts_sha256'],
                 teacher='Qwen3.5-397B-A17B',student='Qwen3.5-4B',dataset='deepcoder',
                 teacher_precision='fp8',teacher_checkpoint=str(ORIGINAL/'data/models/Qwen3.5-397B-A17B-FP8'),
                 reward='Official complete pinned checker over all cases of approved filtered DeepCoder question; train truncation 0, original format penalty',
                 prompt_format='Shared filtered DeepCoder generic Python nonthinking prompt')
    else:
        with tempfile.TemporaryDirectory(dir=str(base)) as scratch:
            temporary = Path(scratch)/'template'
            p = core_prepare(temporary, 'grpo', TEACHERS[options['teacher']], 'relative_depth_groups4',
                             gpu_count=8, teacher_precision='fp8', dataset='dapo', max_response_length=16384,
                             epochs=1, evaluations_per_epoch=8,
                             tuning=dict(format_penalty=True, learning_rate=options['learning_rate']))
            train = json.loads((temporary/'train_prompts.json').read_text())
            validation = json.loads((temporary/'eval_prompts.json').read_text())
            data = json.loads((temporary/'dataset.json').read_text())
    student = json.loads((ROOT/'data/models/Qwen3.5-4B/config.json').read_text())['text_config']
    model = ROOT/'data/models'/(TEACHERS[options['teacher']]+'-FP8')
    teacher = json.loads((model/'config.json').read_text())['text_config']
    if options['dataset']=='deepcoder':
        from shared_judge import service_info, SERVICE
        service=service_info()
        from transformers import AutoTokenizer as _Tokenizer
        receiver_tokenizer=_Tokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-4B',local_files_only=True)
        teacher_tokenizer=_Tokenizer.from_pretrained(model,local_files_only=True)
        if teacher_tokenizer.get_vocab()!=receiver_tokenizer.get_vocab():
            raise RuntimeError('397B DeepCoder sender/receiver token vocab differs')
        for row in validation:
            encoded=teacher_tokenizer.apply_chat_template(row['messages'],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if hasattr(encoded,'keys'):encoded=encoded['input_ids']
            if list(encoded)!=row['prompt_token_ids']:
                raise RuntimeError('397B and 4B full201val token sequence differs')
        if not all(len(row['prompt_token_ids'])<=2048 and row['judge']['kind']=='deepcoder' for row in train+validation):
            raise RuntimeError('DeepCoder prompts or checker identity differ')
        p.update(judge_backend='deepcoder_shared_docker',judge_service_root=str(SERVICE),
            judge_service_spec_id=service['spec_id'],score_worker='deepcoder_shared_scoring',
            reward='All official cases of filtered DeepCoder question pass; format penalty on truncated train only',
            prompt_format='Shared DeepCoder generic nonthinking Python prompt',
            eval_metric='avg@16 fully scored held-out filtered DeepCoder responses / 3216',
            val_metric='full',full_test_benchmarks=['lcb','humaneval','mbpp'],
            test_samples=8,test_max_new_tokens=16384,test_max_model_len=20480)
        for stale in ('answer_scoring_version','math_verify_boxed_only','teacher_group_map'):
            p.pop(stale,None)
    if options['dataset'] == 'lcb':
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen3.5-4B', local_files_only=True)
        teacher_tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        split = json.loads((ROOT/'data/livecodebench/split.json').read_text())
        problems = code_prompts.lcb_problems()
        order = {'easy': 0, 'medium': 1, 'hard': 2}
        keys = sorted(split['validation'], key=lambda k: (order[problems[k]['difficulty']], problems[k]['contest_date'], k))
        train = code_prompts.lcb_rows(split['train'], tokenizer, problems)
        validation = code_prompts.lcb_rows(keys, tokenizer, problems)
        assert len(train) == 780 and len(validation) == 100
        assert not {r['key'] for r in train+validation} & set(split['test'])
        for row in train+validation:
            assert code_prompts.encode(teacher_tokenizer, row['messages']) == row['prompt_token_ids']
            assert len(row['prompt_token_ids']) <= 2048
        from shared_code_judge_client import service_info, SERVICE
        service=service_info()
        image=service['image_id']
        p.update(judge_backend='shared_files_docker',judge_service_root=str(SERVICE),
                 judge_service_spec_id=service['spec_id'])
        p.pop('val_metric', None)
        p.update(examples_per_epoch=len(train), eval_questions=len(validation), score_worker='lcb_score_worker.py',
            reward='Official LCB all public/private tests pass; truncated training responses 0; val unpenalised',
            eval_metric='avg@16 pass-all-tests responses / 1600; pass@16 separately',
            prompt_format='Official LiveCodeBench generic prompt',
            source_sha256=digest(ROOT/'data/livecodebench/split.json'),
            eval_source_sha256=digest(ROOT/'data/livecodebench/problems.jsonl'),
            code_judge=dict(code_judge.DEFAULTS, image=code_judge.IMAGE, image_id=image))
        data = dict(train=780, validation=100, split_rule=split['rule'], v6_test_questions=175)
    p = configure_protocol(p, student, teacher, options)
    if options['teacher']=='397b' and options['dataset'] in ('lcb','deepcoder'):
        from formal_qualification import verify_evidence
        p.update(formal_startup_checks=True,qualification_evidence=verify_evidence(ORIGINAL),
                 diagnostic_full_validation=True)
    if options['dataset']=='deepcoder':
        approved=common_policy()
        p['early_stop'].update(patience=2,clip_abs=.35,clip_ratio=2.,clip_ratio_floor=.15)
        p.update(code_clip_abs=.35,code_clip_ratio=2.,code_clip_floor=.15,
            hard_cap_candidates=[168,210,252,295],hard_cap_selection_pending='Current35B manual validated cap',
            train_prompts_sha256=approved['train_prompts_sha256'],
            eval_prompts_sha256=approved['eval_prompts_sha256'],checker_policy_sha256=approved['checker_policy_sha256'],
            minimum_steps=0,model='Qwen3.5-4B',method='grpo',format_penalty=True,
            max_num_seqs=16,vllm_max_num_seqs=16,vllm_max_model_len=18432,
            checkpoint_steps=[0,*p['eval_steps']],per_step_checkpoints=True,
            non_validation_checkpoint_retention=1,
            user_approved_omitted_redundant_validation_step=294,terminal_validation_step=295,
            val_metric='full',
            max_prompt_length=2048,max_response_length=16384,eval_questions=201,eval_samples=16,
            teacher_model_loaded=True,previous_coding_checkpoint_loaded=False,
            train_seed_formula='42 + 100000*epoch + 1000*global_step + 8*question_index + sample_index',
            test_specs=approved['test_specs'],test_total_responses=6712,
            wandb_run_name='platour_397b_to_4b_deepcoder_terminal_only295_v7_18873train_201val_lr1e5_16k')
    if options['dataset']=='dapo':
        from math_answer_parser import SCORING_VERSION
        p.update(val_metric='answer',math_verify_boxed_only=False,answer_scoring_version=SCORING_VERSION,
            test_samples=dict(aime=16,hmmt=16,imo=4),test_max_new_tokens=32768,
            training_task_independent=True,maximum_steps_policy=132,
            eval_metric='Last Answer match, whole expression verified; clipped responses zero; val avg@4',
            previous_coding_checkpoint_loaded=False)
    if options['dataset']!='deepcoder':
        p['wandb_run_name']+='_v6_answer_math_matched_budget_active16'
    p.update(multinode_options=options, source_manifest_sha256=verify_sources())
    if options['dataset']=='deepcoder':
        (base/'train_prompts.json').write_bytes(Path(approved['train_prompts_path']).read_bytes())
        (base/'eval_prompts.json').write_bytes(Path(approved['eval_prompts_path']).read_bytes())
        assert digest(base/'train_prompts.json')==approved['train_prompts_sha256']
        assert digest(base/'eval_prompts.json')==approved['eval_prompts_sha256']
    else:
        save(base/'train_prompts.json', train)
        save(base/'eval_prompts.json', validation)
    data.update(train_prompts_sha256=digest(base/'train_prompts.json'),
                eval_prompts_sha256=digest(base/'eval_prompts.json'))
    p.update(train_prompts_sha256=data['train_prompts_sha256'],eval_prompts_sha256=data['eval_prompts_sha256'])
    save(base/'dataset.json', data)
    save(base/'mapping.json', dict(layer_mapping=MAPPING, teacher_layers_0based=p['layer_map'],
                                   teacher_layers_1based=[j+1 for j in p['layer_map']], rule=RULE))
    save(base/'protocol.json', p)
    return p
