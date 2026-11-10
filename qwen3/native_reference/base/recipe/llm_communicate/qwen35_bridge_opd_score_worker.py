"""CPU-only persistent boxed VERL scorer, separate from model processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from functools import partial
import json
import multiprocessing
from pathlib import Path
import sys
import traceback
from qwen35_bridge_opd_core import ROOT, save
sys.path.insert(0,str(ROOT/'third_party/verl'))
from generate_qwen35_train_suite import score


class BatchScoringPool(AbstractContextManager):
    """Recycle whole pools between requests, never workers during a request."""
    def map(self, scorer, rows, chunksize):
        with ProcessPoolExecutor(max_workers=6,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            return list(pool.map(scorer, rows, chunksize=chunksize))

    def __exit__(self, *args):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    # Fresh pools bound parser caches without retiring workers while a map has
    # pending tasks. The persistent RPC process remains available across batches.
    with BatchScoringPool() as pool:
        save(args.directory/'ready.json',dict(cpu_only=True,workers=6))
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('command')=='stop':
                break
            try:
                rows=[]
                for filename in request['inputs']:
                    payload=json.loads(Path(filename).read_text())
                    assert payload['complete']
                    rows.extend(payload['rows'])
                rows.sort(key=lambda r:(r['index'],r.get('sample_index',0)))
                assert len(rows)==request['expected_responses']
                assert len({(r['index'],r.get('sample_index',0)) for r in rows})==len(rows)
                penalty=bool(request.get('format_penalty',False))
                scorer=score
                if penalty:
                    from generation_format_reward import score_with_format
                    scorer=partial(score_with_format,base_scorer=score,
                        answer_format=request.get('reward_answer_format','boxed'))
                result=list(pool.map(scorer,rows,chunksize=4))
                timeouts=sum(r['scoring_timed_out'] for r in result)
                if timeouts:
                    # Retry serially on the main CPU process; never silently turn
                    # verifier timeouts into incorrect labels.
                    for i,row in enumerate(result):
                        if row['scoring_timed_out']:
                            result[i]=scorer(rows[i])
                unresolved=[r for r in result if r['scoring_timed_out']]
                if unresolved:
                    diagnostic=Path(request['result']).with_name('unresolved_scoring.json')
                    save(diagnostic,dict(complete=False,responses=unresolved))
                    raise RuntimeError('Unresolved math scoring timeouts: '+
                        ', '.join(f"{r['index']}/{r.get('sample_index',0)}" for r in unresolved)+
                        '; original responses saved to '+str(diagnostic))
                eligible=[r for r in result if not r['gold_missing']]
                assert all(r['acc'] in (0,1) for r in eligible)
                groups={}
                for row in eligible:
                    groups.setdefault(row['index'],[]).append(row['acc'])
                summary=dict(complete=True, responses=len(result), scored_responses=len(eligible),
                    correct=sum(r['acc'] for r in eligible), accuracy=sum(r['acc'] for r in eligible)/len(eligible),
                    questions=len(groups), passed_questions=sum(any(v) for v in groups.values()),
                    pass_at_k=sum(any(v) for v in groups.values())/len(groups),
                    clipping_rate=sum(r['finish_reason']=='length' for r in result)/len(result),
                    clipped=sum(r['finish_reason']=='length' for r in result),
                    mean_tokens=sum(r['tokens'] for r in result)/len(result),
                    scoring_timeouts=0,gold_missing=sum(r['gold_missing'] for r in result))
                if penalty:
                    summary.update(format_penalty=True,reward_answer_format=request.get('reward_answer_format','boxed'),
                        accuracy_unpenalized=sum(r['acc_unpenalized'] for r in eligible)/len(eligible),
                        format_violation_rate=sum(not r['format_ok'] for r in eligible)/len(eligible),
                        format_zeroed_rate=sum(r['acc_unpenalized']==1 and r['acc']==0 for r in eligible)/len(eligible))
                scored=Path(request['result']).with_name('scored.jsonl')
                scored.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
                save(request['result'],summary)
            except BaseException as error:
                traceback.print_exc()
                save(request['result'],dict(complete=False,error=f'{type(error).__name__}: {error}'))
                raise


if __name__=='__main__':
    main()
