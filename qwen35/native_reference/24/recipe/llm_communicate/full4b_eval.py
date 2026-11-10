"""Native evaluation of full-parameter 4B checkpoint; test immediately after its training."""
import argparse
import json
from pathlib import Path
from full4b_train import Decoder,read,sample,sha
from qwen35_bridge_opd_core import ROOT,EVAL_SAMPLING,save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-step',type=int,required=True)
    parser.add_argument('--rank',type=int,required=True)
    args=parser.parse_args();rank=args.rank
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=str(rank)
    metadata=read(args.root/'train/checkpoints'/f'step-{args.checkpoint_step:04d}'/'checkpoint.json')
    assert metadata['complete'] and sha(metadata['weights'])==metadata['weights_sha256']
    plan=read(args.output/'plan.json')
    context=34816 if plan['dataset']=='dapo' else 18432
    decoder=Decoder(args.output/'generation'/f'rank-{rank:02d}',rank,ROOT/'data/models/Qwen3.5-4B',max_model_len=context)
    try:
        result=decoder.call(dict(command='sync',path=metadata['weights'],file_sha256=metadata['weights_sha256'],
                                state_sha256=metadata['state_sha256']))
        assert result['complete']
        plan=read(args.output/'plan.json')
        for benchmark in plan['benchmarks']:
            directory=args.output/benchmark
            protocol=read(directory/'protocol.json')
            prompts=read(directory/'prompts.json')
            requests=[dict(row,sample_index=i,sampling_seed=1234+protocol['samples_per_question']*row['index']+i)
                      for row in prompts[rank::8] for i in range(protocol['samples_per_question'])]
            target=directory/'generation'/f'rank-{rank:02d}'/'responses.json'
            sample(decoder,requests,metadata['state_sha256'],EVAL_SAMPLING,protocol['max_new_tokens'],target)
            save(directory/'generation'/f'rank-{rank:02d}'/'complete.json',dict(complete=True,responses=len(requests),
                  checkpoint_sha256=metadata['weights_sha256']))
    finally:decoder.close()


if __name__=='__main__':main()
