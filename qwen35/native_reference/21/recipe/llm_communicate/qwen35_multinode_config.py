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
TEACHERS = {'4b': 'Qwen3.5-4B'}
PARALLEL = {'4b': 2}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def topology(teacher, nnodes=1, gpus_per_node=8):
    if teacher != '4b' or nnodes != 1 or gpus_per_node != 8:
        raise ValueError('Native4B bridge requires exactly one eight-GPU host and four independent two-GPU replicas')
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
        nearest_full_tie_break='shallower', attention_types_preserved=True, cross_attention_mapping_allowed=False,
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
        wandb_run_name=f'platour_native4b_to_native4b_{dataset}_identity32_zero_grpo_lr{options["learning_rate"]:g}_fmtpen_g8_bs64_16k_1node_valavg{samples}_v1',
        source_execution_root=str(ROOT), test_samples=8 if dataset == 'lcb' else None,
        source_teacher='independent_frozen_native_Qwen3.5-4B', source_receiver='independent_frozen_native_Qwen3.5-4B',
        source_model_revision=str(ROOT/'data/models/Qwen3.5-4B'),
        model_parallelism='single host; four independent paired GPUs, student on even and sender on odd indices',
        teacher_model_loaded=True,trainable_modules='32 independent FP32 zero-initialized dense projectors only')
    return p


def prepare_protocol(base, options):
    from transformers import AutoTokenizer
    import code_prompts
    import code_judge
    base = Path(base)
    if (base/'protocol.json').exists():
        existing = json.loads((base/'protocol.json').read_text())
        if existing.get('multinode_options') != options or existing.get('source_manifest_sha256')!=verify_sources():
            raise ValueError('Saved protocol or source version differs; use a fresh --root')
        return existing
    assert options['teacher']=='4b' and options['dataset'] in ('dapo','lcb')
    reference = ORIGINAL/'outputs/full4b_grpo_v5'/options['dataset']
    for filename in ('train_prompts.json','eval_prompts.json'):
        (base/filename).write_bytes((reference/filename).read_bytes())
    train = json.loads((base/'train_prompts.json').read_text())
    validation = json.loads((base/'eval_prompts.json').read_text())
    template = ORIGINAL/'outputs/multinode_runs'/('397b_dapo_zero_b_v6' if options['dataset']=='dapo' else '397b_lcb_zero_b_v5')
    p = json.loads((template/'protocol.json').read_text())
    model_path = ROOT/'data/models/Qwen3.5-4B'
    for key in ('fp8_kernel_cache','fp8_kernel_revision','teacher_checkpoint_revision','teacher_quantization'):
        p.pop(key,None)
    p.update(teacher='Qwen3.5-4B',student='Qwen3.5-4B',teacher_weight_precision='bf16',
        teacher_checkpoint=str(model_path),teacher_precision='Original BF16, no FP8, no CPU offload',
        trainable_parameters=32*2560*2560, examples_per_epoch=len(train),eval_questions=len(validation),
        source_template_protocol_sha256=digest(template/'protocol.json'),
        source_data_protocol_sha256=digest(reference/'protocol.json'))
    data = dict(train=len(train),validation=len(validation),source_dataset='identical Qwen3.5-4B frozen train/val files',
                source_data_root=str(reference),source_data_protocol_sha256=digest(reference/'protocol.json'))
    student = json.loads((ROOT/'data/models/Qwen3.5-4B/config.json').read_text())['text_config']
    model = ROOT/'data/models'/TEACHERS[options['teacher']]
    teacher = json.loads((model/'config.json').read_text())['text_config']
    assert student['hidden_size']==teacher['hidden_size']==2560 and len(student['layer_types'])==len(teacher['layer_types'])==32
    assert student['layer_types']==teacher['layer_types']
    if options['dataset'] == 'lcb':
        teacher_tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        split = json.loads((ROOT/'data/livecodebench/split.json').read_text())
        assert len(train) == 780 and len(validation) == 100
        assert {r['key'] for r in train} == set(split['train'])
        assert {r['key'] for r in validation} == set(split['validation'])
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
    if options['teacher']=='397b' and options['dataset']=='lcb':
        from formal_qualification import verify_evidence
        p.update(formal_startup_checks=True,qualification_evidence=verify_evidence(ORIGINAL),
                 diagnostic_full_validation=True)
    if options['dataset']=='dapo':
        from math_answer_parser import SCORING_VERSION
        p.update(val_metric='answer',math_verify_boxed_only=False,answer_scoring_version=SCORING_VERSION,
            test_samples=dict(aime=16,hmmt=16,imo=4),test_max_new_tokens=32768,
            training_task_independent=True,maximum_steps_policy=132,
            eval_metric='Last Answer match, whole expression verified; clipped responses zero; val avg@4',
            previous_coding_checkpoint_loaded=False)
    p['wandb_run_name'] += '_native4b_identity32_1node_v4_local_tmpfs_compile'
    p.update(multinode_options=options, source_manifest_sha256=verify_sources(),
             previous_startup_failure_root=str(ORIGINAL/'outputs/multinode_runs/4b_native_dapo_zero_v3'),
             runtime_only_change='Isolated per-rank Triton/Inductor/vLLM JIT cache on local tmpfs; scientific data, sampler, optimizer, scorer and checkpoint semantics unchanged',
             runtime_cache_kind='local_tmpfs_per_rank',runtime_cache_root='/tmp/platour_bridge4b_native_v4',
             native4b_formal=True, small_model_diagnostic=False, teacher_weight_precision='bf16',
             teacher_layer_split=[32], world_size=4,nnodes=1,teacher_model_parallel_size=2,
             diagnostic_full_validation=True,formal_startup_checks=False,
             require_new_single_host_smoke=True, cpu_source_reference='isolated_bridge4b_native_v4')
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
