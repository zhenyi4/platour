"""Independent frozen Qwen3.5-4B sender/receiver, baseline DAPO bridge GRPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from qwen35_bridge_opd_core import ROOT,save

REFERENCE=ROOT/'outputs/llm_communicate/qwen35_35b_to_4b_dapo_grpo_relative_depth_groups4_constant1_lr1e5_bs64_g8_cap16384_8gpu_vllm_evalgeneral_gc_ep1_eval8'
BASE=ROOT/'outputs/llm_communicate/qwen35_4b_to_4b_dapo_grpo_same_index_constant1_lr1e5_bs64_g8_cap16384_ep1_eval8_8gpu'
CHANGED_KEYS={'teacher','teacher_checkpoint','student_checkpoint','layer_map','layer_mapping',
              'mapping_rule','teacher_group_map','initialization','trainable_parameters',
              'wandb_run_name','smoke_full_batch','verify_source_hashes','independent_backbone_instances'}


def same_model_protocol(baseline):
    result=dict(baseline)
    path=ROOT/'data/models/Qwen3.5-4B'
    result.update(teacher='Qwen3.5-4B',teacher_checkpoint=str(path),student_checkpoint=str(path),
        layer_mapping='same_index',layer_map=list(range(32)),teacher_group_map=list(range(8)),
        mapping_rule='Identity: teacher_layer=student_layer; equal-depth relative group mapping',
        initialization='uniform +/-1e-4/sqrt(2560), seed42; no SFT or pretrain weights',
        trainable_parameters=32*2560*2560,independent_backbone_instances=True,
        wandb_run_name='ablation_qwen35_4b_to_4b_dapo_same_index_constant_gate1_grpo_g8_bs64_16k_lr1e5_1epoch_eval9_vllm_8gpu',
        smoke_full_batch=True,verify_source_hashes=True)
    assert baseline['epochs']==1 and baseline['total_steps']==264 and baseline['examples_per_epoch']==16891
    assert all(result.get(k)==v for k,v in baseline.items() if k not in CHANGED_KEYS)
    return result


def load_bridge():
    import torch
    from qwen35_direct_math_bridge import accelerated_models
    from qwen35_student import load_student
    from qwen35_last_aligned_math_bridge import LastAlignedResidualBridge
    runtime=accelerated_models()
    path=ROOT/'data/models/Qwen3.5-4B'
    student=load_student(path)
    teacher=load_student(path)
    assert student is not teacher
    assert student.config.num_hidden_layers==teacher.config.num_hidden_layers==32
    assert student.config.hidden_size==teacher.config.hidden_size==2560
    assert student.config.layer_types==teacher.config.layer_types
    assert all(a.data_ptr()!=b.data_ptr() and torch.equal(a,b)
               for a,b in zip(student.parameters(),teacher.parameters(),strict=True))
    bridge=LastAlignedResidualBridge(student,teacher).eval()
    assert bridge.layer_map==list(range(32))
    assert sum(p.numel() for p in bridge.bridge_parameters())==209715200
    original_metadata=bridge.metadata
    bridge.metadata=lambda:dict(original_metadata(),layer_mapping='same_index',
        sender='Qwen3.5-4B',receiver='Qwen3.5-4B',independent_backbone_instances=True)
    runtime.update(teacher='Qwen3.5-4B',student='Qwen3.5-4B',teacher_layers=32,student_layers=32,
        teacher_dim=2560,student_dim=2560,layer_mapping='same_index',layer_map=bridge.layer_map,
        teacher_precision='original BF16',teacher_class=type(teacher).__name__,
        student_class=type(student).__name__,independent_backbone_instances=True,
        identical_initial_backbone_weights_verified=True,cuda_device=torch.cuda.current_device())
    return bridge,runtime


def prepare(base):
    base=Path(base).absolute()
    original=json.loads((REFERENCE/'protocol.json').read_text())
    expected=same_model_protocol(original)
    if (base/'protocol.json').exists():
        assert json.loads((base/'protocol.json').read_text())==expected
        for name in ('train_prompts.json','eval_prompts.json'):
            assert (base/name).read_bytes()==(REFERENCE/name).read_bytes()
        return expected
    base.mkdir(parents=True,exist_ok=True)
    for name in ('train_prompts.json','eval_prompts.json','dataset.json'):
        shutil.copyfile(REFERENCE/name,base/name)
    assert len(json.loads((base/'train_prompts.json').read_text()))==16891
    assert len(json.loads((base/'eval_prompts.json').read_text()))==500
    # Sender and receiver use the same checkpoint and the exact saved baseline prompts.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(expected['student_checkpoint'],local_files_only=True)
    assert tokenizer.get_vocab()
    save(base/'protocol.json',expected)
    save(base/'mapping.json',dict(layer_mapping='same_index',teacher_layers_0based=list(range(32)),
                                teacher_layers_1based=list(range(1,33))))
    differences={k:dict(baseline=original.get(k),same_model=expected.get(k))
                 for k in sorted(original.keys()|expected.keys()) if original.get(k)!=expected.get(k)}
    assert set(differences)<=CHANGED_KEYS
    save(base/'baseline_comparison.json',dict(reference_root=str(REFERENCE),changes=differences,
        same_data_and_sampling=True,same_one_epoch_schedule=True,
        reference_protocol_sha256=hashlib.sha256((REFERENCE/'protocol.json').read_bytes()).hexdigest(),
        train_prompts_sha256=hashlib.sha256((base/'train_prompts.json').read_bytes()).hexdigest(),
        eval_prompts_sha256=hashlib.sha256((base/'eval_prompts.json').read_bytes()).hexdigest()))
    dependencies=[Path(__file__),*[Path(__file__).with_name(name) for name in (
        'run_qwen35_4b_4b_dapo_grpo_16k.sh','run_qwen35_bridge_opd_vllm.py',
        'train_qwen35_bridge_opd_vllm.py','qwen35_bridge_opd_core.py','qwen35_bridge_grpo_core.py',
        'qwen35_last_aligned_math_bridge.py','qwen35_direct_math_bridge.py','qwen35_student.py',
        'qwen35_bridge_opd_vllm_worker.py','qwen35_bridge_opd_score_worker.py',
        'qwen35_bridge_logits_processors.py','qwen35_hybrid_memory_connector.py',
        'prepare_qwen35_bridge_vllm.py','eval_qwen35_35b_4b_math_bridge_8gpu.py',
        'generate_qwen35_train_suite.py')],*[ROOT/'scripts'/name for name in (
        'direct_residual_chat_bridge.py','shared_subspace_residual_bridge.py',
        'shared_subspace_chat_bridge.py','hidden_bridge.py')]]
    for p in dependencies:
        target=base/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    save(base/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in dependencies})
    return expected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--queue',action='store_true')
    args=parser.parse_args()
    base=(args.root or (BASE if args.train else BASE.with_name(BASE.name+'_smoke'))).absolute()
    prepare(base)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,root=str(base))),flush=True)
        return
    if args.detach:
        if (base/'controller.json').exists():
            pid=json.loads((base/'controller.json').read_text())['pid']
            p=Path(f'/proc/{pid}/cmdline')
            if p.exists() and str(base).encode() in p.read_bytes():raise RuntimeError('Job already active')
        python=Path(os.environ.get('PLATOUR_BRIDGE_PYTHON',ROOT/'.runtime/readme-check/bridge/bin/python')).absolute()
        command=[str(python),'-u',str(Path(__file__).absolute()),'--queue','--root',str(base),
                 '--train' if args.train else '--smoke-only']
        with (base/'controller.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
        save(base/'controller.json',dict(pid=child.pid,started=time.time(),command=command,smoke_only=not args.train))
        print(json.dumps(dict(pid=child.pid,root=str(base),smoke_only=not args.train)),flush=True)
    else:
        from run_qwen35_bridge_opd_vllm import queue
        queue(base,'grpo',smoke_only=not args.train)


if __name__=='__main__':main()
