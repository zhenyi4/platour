"""CPU-only persistent boxed VERL scorer, separate from model processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import traceback
from qwen35_bridge_opd_core import ROOT, save
sys.path.insert(0,str(ROOT/'third_party/verl'))
from generate_qwen35_train_suite import score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=6) as pool:
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
                result=list(pool.map(score,rows,chunksize=4))
                timeouts=sum(r['scoring_timed_out'] for r in result)
                if timeouts:
                    # Retry serially on the main CPU process; never silently turn
                    # verifier timeouts into incorrect labels.
                    for i,row in enumerate(result):
                        if row['scoring_timed_out']:
                            result[i]=score(rows[i])
                assert not any(r['scoring_timed_out'] for r in result)
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
                scored=Path(request['result']).with_name('scored.jsonl')
                scored.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
                save(request['result'],summary)
            except BaseException as error:
                traceback.print_exc()
                save(request['result'],dict(complete=False,error=str(error)))
                raise


if __name__=='__main__':
    main()
