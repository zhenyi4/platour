"""Queue a matched batched-prefill C2C run after the serial baseline completes."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from arc_bridge_sft_common import ROOT,HERE,save,sha
from generate_qwen35_train_suite import stop_job
from run_qwen3_4b_17b_gsm8k_sft import environment

BASE=ROOT/'outputs/llm_communicate/qwen3_4b_to_17b_gsm8k_c2c_batched_sft_ep3_bs64_8gpu'
REFERENCE=ROOT/'outputs/llm_communicate/qwen3_4b_to_17b_gsm8k_c2c_sft_ep3_bs64_8gpu'
SCRIPT=Path(__file__).resolve()


def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'protocol.json').exists():return
    p=json.loads((REFERENCE/'protocol.json').read_text())
    p.update(cache_prefill='batched right-padded prompt KV; per-sample RoPE positions; compact valid KV for vLLM',
        batch_prefill=True,reference_serial_run=str(REFERENCE),fresh_initialization=True)
    save(BASE/'protocol.json',p);shutil.copyfile(REFERENCE/'eval_prompts.json',BASE/'eval_prompts.json')
    files=json.loads((REFERENCE/'source_sha256.json').read_text())
    for f in [SCRIPT,HERE/'qwen3_c2c_batched_sft.py']:files[str(f.relative_to(ROOT))]=sha(f)
    save(BASE/'source_sha256.json',files)
    for name in files:
        target=BASE/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)


def install_adapter():
    configure_attention()
    from qwen3_c2c_batched_sft import load_bridge
    import qwen235_c2c_sft
    import arc_bridge_sft_common
    qwen235_c2c_sft.load_bridge=load_bridge
    arc_bridge_sft_common.load_hidden=load_bridge
    return load_bridge


def configure_attention():
    # Keep padded and unpadded calls on the same SDPA implementation. Fused
    # backend dispatch otherwise gives materially different BF16 gradients.
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False


def audit():
    import os
    precision=os.environ.get('C2C_AUDIT_PRECISION','bf16')
    if precision=='bf16':
        # Verify semantics in high precision before judging production BF16 drift.
        subprocess.run([sys.executable,'-u',str(SCRIPT),'audit'],cwd=ROOT,
            env={**os.environ,'C2C_AUDIT_PRECISION':'fp32'},check=True)
    import torch
    from nemotron_chat_data import ChatTokens,collate
    from qwen3_17b_hidden_sft import digest
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    bridge=install_adapter()(BASE,devices=[0,1],student_device=0).eval()
    initial=digest(bridge)
    reference=json.loads((REFERENCE/'train/complete.json').read_text())
    assert initial==reference['initial_sha256'],'Batched run must use identical initial projector weights'
    if precision=='fp32':bridge.float()
    assert precision in ['bf16','fp32']
    p=json.loads((BASE/'protocol.json').read_text());data=ChatTokens(Path(p['data_root']))
    ids=json.loads((Path(p['data_root'])/'split.json').read_text())['train']
    ids=sorted(ids,key=lambda i:int(data.index[i]['prefix']))
    selected=[ids[0],ids[len(ids)//2],ids[-1]]
    # Open every gate so a comparison cannot pass by falling back to native KV.
    for projector in bridge.maps:
        projector.key_gate_logit.data.fill_(.1);projector.value_gate_logit.data.fill_(.1)
    batch=collate(data,selected,bridge.pad_token_id,'cuda')
    batched=bridge.chat_loss(batch)[1];assert torch.isfinite(batched)
    batched.backward()
    parameters=bridge.bridge_parameters();gradients=[None if x.grad is None else x.grad.detach().clone() for x in parameters]
    bridge.zero_grad(set_to_none=True)
    individual=bridge.serial_chat_loss(batch)[1];assert torch.isfinite(individual)
    individual.backward()
    diff=torch.zeros((),device='cuda',dtype=torch.float64);norm=torch.zeros_like(diff)
    actual_norm=torch.zeros_like(diff);dot=torch.zeros_like(diff)
    for parameter,expected in zip(parameters,gradients,strict=True):
        actual=parameter.grad
        if expected is None:assert actual is None;continue
        assert actual is not None and torch.isfinite(actual).all() and torch.isfinite(expected).all()
        diff+=(actual.float()-expected.float()).double().square().sum();norm+=expected.double().square().sum()
        actual_norm+=actual.double().square().sum();dot+=(actual.double()*expected.double()).sum()
    relative=float((diff/norm).sqrt())
    cosine=float(dot/(norm*actual_norm).sqrt())
    save(BASE/f'audit_gradient_{precision}.json',dict(precision=precision,
        batched_loss=float(batched.detach()),individual_loss=float(individual.detach()),
        gradient_relative_l2=relative,gradient_cosine=cosine))
    if precision=='fp32':assert relative<.001,relative
    else:assert relative<.1 and cosine>.95,(relative,cosine)
    torch.testing.assert_close(batched.detach(),individual.detach(),atol=.01,rtol=.02)
    rows=[dict(prompt_token_ids=data[i][0][:data[i][1]].tolist()) for i in selected]
    with torch.inference_mode():
        together=bridge.prefill_rows(rows,None)
        singles=bridge.serial_prefill_rows(rows,None)
        error=(together.logits.float()-singles.logits.float()).abs()
        mean_error=float(error.mean());max_error=float(error.max())
        a=together.logits.float().log_softmax(-1);b=singles.logits.float().log_softmax(-1)
        logit_kl=float((b.exp()*(b-a)).sum(-1).max())
        if precision=='fp32':assert mean_error<.001 and max_error<.01,(mean_error,max_error)
        cache_max=0.;cache_relative=0.
        for a,b in zip(together.past_key_values.layers,singles.past_key_values.layers,strict=True):
            for i,row in enumerate(rows):
                n=len(row['prompt_token_ids'])
                for x,y in [(a.keys,b.keys),(a.values,b.values)]:
                    v=float((x[i,:,:n].float()-y[i,:,:n].float()).abs().max());cache_max=max(cache_max,v)
                    error_norm=(x[i,:,:n].float()-y[i,:,:n].float()).norm()
                    reference_norm=y[i,:,:n].float().norm().clamp_min(1e-12)
                    cache_relative=max(cache_relative,float(error_norm/reference_norm))
    assert all(not x.requires_grad and x.grad is None for m in [bridge.student,bridge.teacher] for x in m.parameters())
    report=dict(passed=True,precision=precision,initial_sha256=initial,identical_initial_projectors=True,
        selected_training_indices=selected,prompt_lengths=[len(r['prompt_token_ids']) for r in rows],
        all_gates_open=True,batched_loss=float(batched.detach()),individual_loss=float(individual.detach()),
        gradient_relative_l2=relative,gradient_cosine=cosine,logit_mean_abs_error=mean_error,logit_max_abs_error=max_error,
        next_token_kl_max=logit_kl,compact_cache_relative_l2_max=cache_relative,
        compact_cache_max_abs_error=cache_max,backbones_frozen=True,discarded_audit_weights=True)
    # BF16 equality is not bitwise: use scale-aware KV error and probability
    # divergence, while the independent FP32 audit enforces semantic equality.
    report['acceptance']=dict(gradient_relative_l2_max=.001 if precision=='fp32' else .1,
        cache_relative_l2_max=.0001 if precision=='fp32' else .05,
        next_token_kl_max=.02 if precision=='bf16' else .00001)
    save(BASE/f'batch_prefill_diagnostics_{precision}.json',report)
    assert cache_relative<report['acceptance']['cache_relative_l2_max'],cache_relative
    assert logit_kl<report['acceptance']['next_token_kl_max'],logit_kl
    save(BASE/f'batch_prefill_audit_{precision}.json',report)
    if precision=='bf16':
        report['fp32_semantic_audit']=json.loads((BASE/'batch_prefill_audit_fp32.json').read_text())
        save(BASE/'batch_prefill_audit.json',report)


def eval_worker(checkpoint,output,rank,smoke):
    install_adapter()
    import eval_qwen3_17b_hidden_sft as evaluation
    evaluation.worker(BASE,rank,checkpoint,output,smoke)


def evaluate(checkpoint,output,smoke):
    from qwen3_17b_hidden_sft import env
    import eval_qwen3_17b_hidden_sft as evaluation
    children=[];output.mkdir(parents=True,exist_ok=True)
    try:
        for rank in range(1 if smoke else 8):
            with (output/f'rank-{rank:02d}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'eval_worker','--checkpoint',str(checkpoint),
                    '--output',str(output),'--rank',str(rank),*(['--smoke'] if smoke else [])],cwd=ROOT,
                    env=env([rank]),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
        for child in children:
            code=child.wait()
            if code:raise RuntimeError(f'Batched evaluation worker failed ({code})')
        evaluation.merge(BASE,output,smoke)
    finally:
        for child in children:
            if child.poll() is None:stop_job(child)


def queue():
    import psutil
    prepare();active=None
    def stopped(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        pid=int((REFERENCE/'controller.pid').read_text())
        save(BASE/'phase.json',dict(phase='waiting_for_serial_c2c',reference=str(REFERENCE),pid=pid,time=time.time()))
        while psutil.pid_exists(pid):
            proc=psutil.Process(pid)
            if proc.status()==psutil.STATUS_ZOMBIE:break
            if 'run_qwen3_4b_17b_gsm8k_c2c_sft.py' not in ' '.join(proc.cmdline()):break
            time.sleep(10)
        assert json.loads((REFERENCE/'phase.json').read_text())['phase']=='complete','Serial reference did not finish successfully'
        assert json.loads((REFERENCE/'eval/summary.json').read_text())['complete']
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert all(int(x)<512 for x in usage.splitlines()),'Eight GPUs must be idle'
        hashes=json.loads((BASE/'source_sha256.json').read_text())
        def execute(stage,command):
            nonlocal active
            assert all(sha(ROOT/name)==digest for name,digest in hashes.items()),'Source changed during run'
            save(BASE/'phase.json',dict(phase=stage,time=time.time()))
            with (BASE/f'{stage}.log').open('a') as log:
                active=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                code=active.wait()
                if code:raise RuntimeError(f'{stage} failed ({code}); inspect {BASE}/{stage}.log')
            active=None
        audit_path=BASE/'batch_prefill_audit.json'
        audited=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if not (audited.get('passed') and audited.get('precision')=='bf16' and audited.get('fp32_semantic_audit',{}).get('passed')):
            execute('batch_audit',[sys.executable,'-u',str(SCRIPT),'audit'])
        trainer=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(SCRIPT),'train_worker']
        for smoke in [True,False]:
            stage='smoke' if smoke else 'train'
            if not (BASE/stage/'complete.json').exists():execute(stage,[*trainer,*(['--smoke'] if smoke else [])])
            done=json.loads((BASE/stage/'complete.json').read_text());output=BASE/('smoke_eval' if smoke else 'eval')
            if not (output/'summary.json').exists():execute(stage+'_eval',[sys.executable,'-u',str(SCRIPT),'eval_queue',
                '--checkpoint',done['checkpoint'],'--output',str(output),*(['--smoke'] if smoke else [])])
        save(BASE/'phase.json',dict(phase='complete',time=time.time()))
    except BaseException as error:
        save(BASE/'phase.json',dict(phase='failed',error=repr(error),time=time.time()));raise
    finally:
        if active is not None and active.poll() is None:stop_job(active)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['queue','audit','train_worker','eval_queue','eval_worker'])
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage=='queue':queue()
    elif args.stage=='audit':audit()
    elif args.stage=='train_worker':
        install_adapter()
        import train_qwen235_trivia_distributed as trainer
        trainer.train(BASE,args.smoke)
    elif args.stage=='eval_queue':evaluate(args.checkpoint,args.output,args.smoke)
    else:eval_worker(args.checkpoint,args.output,args.rank,args.smoke)
