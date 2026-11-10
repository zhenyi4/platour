"""Train-only holdout selection and one final test for frozen native GRPO.

Copied verbatim into each run so its policy is frozen independently of later
checkout edits. Historical model/loss/scorer logic is unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import random
from fractions import Fraction

POLICY='train_holdout_best_validation_v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def split_training(pool,test,seed=42):
    if not 0<len(test)<len(pool):raise ValueError('Holdout must match nonempty test size and leave training examples')
    indices=[r['index'] for r in pool]
    if len(set(indices))!=len(pool):raise ValueError('Original training indices are not unique')
    chosen=set(random.Random(seed).sample(range(len(pool)),len(test)))
    train_positions=[i for i in range(len(pool)) if i not in chosen]
    validation_positions=sorted(chosen)
    def subset(positions):
        return [dict(pool[pos],index=i,original_train_index=pool[pos]['index']) for i,pos in enumerate(positions)]
    return subset(train_positions),subset(validation_positions),dict(seed=seed,algorithm='python.random.Random.sample; sorted source positions; remaining source order retained',
        original_train_count=len(pool),train_count=len(train_positions),validation_count=len(chosen),test_count=len(test),
        train_source_indices=[indices[i] for i in train_positions],validation_source_indices=[indices[i] for i in validation_positions],
        reindexing='contiguous index within each split; original_train_index preserved; id/prompt/gold unchanged')


def published_splits(train, validation, test):
    """Consume materialized dataset splits without drawing another holdout."""
    if not train or not validation or not test:
        raise ValueError('Published TRAIN/VAL/TEST must be nonempty')
    groups = [{tuple(r['prompt_token_ids']) for r in rows} for rows in (train, validation, test)]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Published TRAIN/VAL/TEST prompts overlap')
    for rows in (train, validation, test):
        if len({r['index'] for r in rows}) != len(rows):
            raise ValueError('Published split indices must be unique')
    left = [r['original_train_index'] for r in train]
    right = [r['original_train_index'] for r in validation]
    if len(set(left + right)) != len(left) + len(right):
        raise ValueError('Published TRAIN/VAL source indices overlap')
    return list(train), list(validation), dict(seed=42, source='published_train_holdout_v2',
        original_train_count=len(train)+len(validation), train_count=len(train),
        validation_count=len(validation), test_count=len(test),
        train_source_indices=left, validation_source_indices=right,
        algorithm='read published TRAIN/VAL directly; no runtime resplit')


def configure(root,protocol,seed=42,validation_rows=None):
    root=Path(root)
    pool=json.loads((root/'train_prompts.json').read_text())
    test=json.loads((root/'eval_prompts.json').read_text())
    train,validation,split=(split_training(pool,test,seed) if validation_rows is None
                            else published_splits(pool,validation_rows,test))
    if any(not str(row.get('gold','')).strip() for row in validation):raise ValueError('Random validation contains missing gold; do not silently change the split')
    if {tuple(row['prompt_token_ids']) for row in validation}&{tuple(row['prompt_token_ids']) for row in test}:raise ValueError('Validation contains a test prompt')
    if 'eval_samples' in protocol:
        protocol['test_eval_samples'] = protocol['eval_samples']
        protocol['test_eval_seed_stride'] = protocol.get('eval_seed_stride', protocol['eval_samples'])
        protocol['eval_samples'] = protocol.get('validation_samples', protocol['eval_samples'])
        protocol['eval_seed_stride'] = protocol['eval_samples']
    before={name:protocol.get(name) for name in ['examples_per_epoch','steps_per_epoch','total_steps','eval_steps','checkpoint_steps','total_evaluations']}
    save(root/'test_prompts.json',test)
    save(root/'train_prompts.json',train)
    save(root/'eval_prompts.json',validation)
    save(root/'validation_prompts.json',validation)
    split['sha256']={name:sha(root/name) for name in ['train_prompts.json','eval_prompts.json','validation_prompts.json','test_prompts.json']}
    save(root/'validation_split.json',split)
    epoch_steps=math.ceil(len(train)/protocol['batch_size'])
    epochs=protocol['epochs']
    eval_steps=sorted({math.ceil(k*epoch_steps/2) for k in range(1,2*epochs+1)})
    protocol.update(evaluation_policy=POLICY,validation_split_seed=seed,validation_split_sha256=sha(root/'validation_split.json'),
        train_prompts_sha256=split['sha256']['train_prompts.json'],eval_prompts_sha256=split['sha256']['eval_prompts.json'],
        validation_prompts_sha256=split['sha256']['validation_prompts.json'],test_prompts_sha256=split['sha256']['test_prompts.json'],examples_per_epoch=len(train),steps_per_epoch=epoch_steps,
        total_steps=epoch_steps*epochs,eval_questions=len(validation),test_questions=len(test),eval_steps=eval_steps,checkpoint_steps=eval_steps,
        total_evaluations=len(eval_steps)+1,validation_evaluations=len(eval_steps)+1,final_test_evaluations=1,validation_before_training=True,save_evaluation_bridges=True,
        validation_selection_metric='accuracy (avg@eval_samples), include step0, tie -> earliest step',
        final_test_policy='one full test evaluation after all validation scores; load selected bridge; restore terminal training weights',
        historical_budget_before_holdout=before)
    return split


def verify_inputs(root,protocol):
    root=Path(root)
    if protocol.get('evaluation_policy')!=POLICY:raise ValueError('Expected train-holdout validation policy; prepare a fresh run')
    for filename,key in [('train_prompts.json','train_prompts_sha256'),('eval_prompts.json','eval_prompts_sha256'),('test_prompts.json','test_prompts_sha256'),('validation_prompts.json','validation_prompts_sha256'),('validation_split.json','validation_split_sha256')]:
        if sha(root/filename)!=protocol[key]:raise ValueError(f'Prepared validation input changed: {filename}')
    split=json.loads((root/'validation_split.json').read_text())
    if set(split['train_source_indices'])&set(split['validation_source_indices']):raise ValueError('Training/validation overlap')
    if len(split['validation_source_indices'])!=split['validation_count']:raise ValueError('Validation count differs')


def select_best(run_dir,protocol):
    expected=[0,*protocol['eval_steps']]
    candidates=[]
    for step in expected:
        directory=Path(run_dir)/'evaluation'/f'step-{step:04d}'
        summary=json.loads((directory/'summary.json').read_text())
        denominator=protocol['eval_questions']*protocol['eval_samples']
        if not summary['complete'] or summary['responses']!=denominator or summary['questions']!=protocol['eval_questions']:
            raise ValueError(f'Incomplete validation coverage at step {step}')
        if summary.get('evaluation_role')!='validation':raise ValueError('Selection only accepts validation-tagged results')
        if summary.get('dataset_sha256')!=protocol['eval_prompts_sha256']:raise ValueError('Validation dataset identity differs')
        correct=summary['correct']
        if not 0<=correct<=denominator or abs(summary['accuracy']-correct/denominator)>1e-12:raise ValueError('Invalid validation accuracy')
        candidates.append(dict(step=step,correct=correct,responses=denominator,accuracy=correct/denominator,
            bridge_state_sha256=summary['bridge_state_sha256'],checkpoint=str(directory/'bridge.pt')))
    best=max(candidates,key=lambda row:(Fraction(row['correct'],row['responses']),-row['step']))
    if not Path(best['checkpoint']).is_file():raise FileNotFoundError(best['checkpoint'])
    return dict(policy=POLICY,selection_metric='validation avg@'+str(protocol['eval_samples']),tie_break='earliest_step',
        selected_step=best['step'],validation_accuracy=best['accuracy'],selected_checkpoint=best['checkpoint'],
        selected_bridge_state_sha256=best['bridge_state_sha256'],validation_dataset_sha256=protocol['eval_prompts_sha256'],
        test_used_for_selection=False,candidates=candidates)


def test_evaluation_directory(root, protocol):
    """Give the frozen evaluator a TEST-only protocol without rewriting validation's."""
    root = Path(root)
    samples = protocol.get('test_eval_samples', protocol['eval_samples'])
    stride = protocol.get('test_eval_seed_stride', protocol.get('eval_seed_stride', samples))
    if samples == protocol['eval_samples'] and stride == protocol.get('eval_seed_stride', samples):
        return root/'test'
    test_protocol = dict(protocol, eval_samples=samples, eval_seed_stride=stride,
                         eval_questions=protocol['test_questions'], evaluation_role='test')
    save(root/'test/protocol.json', test_protocol)
    return root/'test/generation'


def evaluate_selected(trainer,root,bridge,engine,scorer,run_dir,rank,world,tokenizer,eos,protocol):
    import torch
    import torch.distributed as dist
    root=Path(root);run_dir=Path(run_dir)
    payload=[select_best(run_dir,protocol) if rank==0 else None]
    if world>1:dist.broadcast_object_list(payload,src=0)
    selection=payload[0]
    if rank==0:save(root/'selection.json',selection)
    # Never overwrite final-step training/optimizer checkpoints with best-val
    # parameters. Restore terminal tensors and RNG before the trainer continues.
    terminal={k:v.detach().cpu().clone() for k,v in bridge.maps.state_dict().items()}
    terminal_digest=trainer.weight_digest(bridge)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        checkpoint=Path(selection['selected_checkpoint'])
        identity=json.loads((checkpoint.parent/'checkpoint.json').read_text())
        if sha(checkpoint)!=identity['bridge_sha256']:raise ValueError('Selected checkpoint file identity differs')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        bridge.maps.load_state_dict(state['P']);del state
        if trainer.weight_digest(bridge)!=selection['selected_bridge_state_sha256']:raise ValueError('Loaded bridge differs from selected validation checkpoint')
        test=json.loads((root/'test_prompts.json').read_text())
        if len(test)!=protocol.get('test_questions',protocol['eval_questions']):raise ValueError('Final test count differs from signed test size')
        destination = [str(test_evaluation_directory(root, protocol)) if rank == 0 else None]
        if world > 1: dist.broadcast_object_list(destination, src=0)
        result=trainer._evaluate_once(bridge,engine,scorer,test,Path(destination[0]),selection['selected_step'],rank,world,tokenizer,eos,None,smoke=False)
        if rank==0:
            result.update(evaluation_role='test',dataset_sha256=protocol['test_prompts_sha256'],
                selected_step=selection['selected_step'],selection_metric=selection['selection_metric'],
                validation_accuracy=selection['validation_accuracy'],validation_before_training=False,test_used_for_selection=False)
            save(root/'test/summary.json',result)
            save(root/'test/evaluation'/f"step-{selection['selected_step']:04d}"/'summary.json',result)
    finally:
        bridge.maps.load_state_dict(terminal);del terminal
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
        if trainer.weight_digest(bridge)!=terminal_digest:raise RuntimeError('Terminal training weights were not restored after test')
    if world>1:dist.barrier()


def install(trainer,root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());verify_inputs(root,protocol)
    original=trainer.evaluate
    def evaluate(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,tokenizer,eos,wandb_run,smoke=False):
        result=original(bridge,engine,scorer,eval_rows,run_dir,step,rank,world,tokenizer,eos,wandb_run,smoke=smoke)
        if rank==0:
            summary_path=Path(run_dir)/'evaluation'/f'step-{step:04d}'/'summary.json'
            summary=json.loads(summary_path.read_text());summary.update(evaluation_role='validation',dataset_sha256=protocol['eval_prompts_sha256'])
            save(summary_path,summary)
        if world>1:
            import torch.distributed as dist
            dist.barrier()
        if not smoke and step==protocol['total_steps']:
            evaluate_selected(trainer,root,bridge,engine,scorer,run_dir,rank,world,tokenizer,eos,protocol)
        return result
    trainer.evaluate=evaluate
