"""CPU-only persistent boxed VERL scorer, separate from model processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import traceback
from qwen35_bridge_opd_core import ROOT, save
sys.path.insert(0,str(ROOT/'third_party/verl'))
from generate_qwen35_train_suite import score

ANSWER_LINE=re.compile(r'^Answer:\s*\S')
# Answer extraction of the original DAPO reward (verl math_dapo.is_correct_minerva): last match wins.
DAPO_ANSWER=re.compile(r'(?i)Answer\s*:\s*([^\n]+)')


def final_line(text):
    lines=[x.strip() for x in text.rstrip().split('\n') if x.strip()]
    return lines[-1] if lines else ''


def score_with_format(row):
    """Format-penalised reward (opt-in, training rollouts only).

    The reward is 0 unless the response stopped on its own (not finish_reason=='length') and its last
    non-empty line is `Answer: X`; only that line is then verified. The unpenalised full-response
    result stays in acc_unpenalized / score_unpenalized.
    """
    base=score(row)
    if base['gold_missing'] or base['scoring_timed_out']:
        return base
    ok=row['finish_reason']!='length' and bool(ANSWER_LINE.match(final_line(row['response'])))
    strict=score(dict(row,response=final_line(row['response']))) if ok else None
    if ok and strict['scoring_timed_out']:
        return dict(base,scoring_timed_out=True)
    return dict(base,acc_unpenalized=base['acc'],score_unpenalized=base['score'],format_ok=ok,
        acc=strict['acc'] if ok else 0,score=strict['score'] if ok else 0.,pred=strict['pred'] if ok else base['pred'])


def _verify_line(row, line):
    """math-verify on a single answer line; one retry on timeout; None if it still times out."""
    for _ in range(2):
        result=score(dict(gold=row['gold'],response=line))  # row may already carry score fields
        if not result['scoring_timed_out']:
            return result['acc']
    return None


def add_answer_metrics(row):
    """Corrected DAPO-val metrics for the `Answer:` prompt (added fields only; `acc` is untouched).

    acc_answer: verify only the last `Answer: ...` match in the response (original DAPO extraction);
                responses cut at the length cap count as 0; no match counts as 0.
    acc_strict: the training format rule: not cut and the last non-empty line is a correct `Answer: X`.
    Full-response parsing (the legacy val metric) can pick an unrelated expression from the reasoning
    (e.g. a "\\frac{m}{n}" template) instead of the final answer; these fields avoid that.
    """
    clipped=row['finish_reason']=='length'
    found=DAPO_ANSWER.findall(row['response'])
    last=final_line(row['response'])
    extra=dict(pred_answer=found[-1].strip() if found else None,
        answer_format_ok=not clipped and bool(ANSWER_LINE.match(last)))
    if row['gold_missing']:
        return dict(row,acc_answer=None,acc_strict=None,answer_metric_timeout=False,**extra)
    answer=None if clipped or not found else _verify_line(row,'Answer: '+found[-1].strip())
    strict=None if not extra['answer_format_ok'] else _verify_line(row,last)
    timeout=(not clipped and bool(found) and answer is None) or (extra['answer_format_ok'] and strict is None)
    return dict(row,acc_answer=int(answer or 0),acc_strict=int(strict or 0),answer_metric_timeout=timeout,**extra)


def use_answer_metric(row):
    """metric='answer': the corrected answer-line result becomes the primary `acc`; legacy kept as *_full."""
    if row['gold_missing']:
        return dict(row,acc_full=None,score_full=None,pred_full=row.get('pred'))
    return dict(row,acc_full=row['acc'],score_full=row['score'],pred_full=row.get('pred'),
        acc=row['acc_answer'],score=float(row['acc_answer']),pred=row['pred_answer'])


def answer_summary(eligible):
    def stats(key):
        groups={}
        for r in eligible:
            groups.setdefault(r['index'],[]).append(r[key])
        return sum(r[key] for r in eligible)/len(eligible), sum(any(v) for v in groups.values())/len(groups)
    accuracy_answer,pass_answer=stats('acc_answer')
    accuracy_strict,pass_strict=stats('acc_strict')
    return dict(answer_metrics=True,accuracy_answer=accuracy_answer,pass_at_k_answer=pass_answer,
        accuracy_strict=accuracy_strict,pass_at_k_strict=pass_strict,
        answer_format_violation_rate=sum(not r['answer_format_ok'] for r in eligible)/len(eligible),
        answer_metric_timeouts=sum(r['answer_metric_timeout'] for r in eligible))


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
                penalty=bool(request.get('format_penalty',False))
                scorer=score_with_format if penalty else score
                result=list(pool.map(scorer,rows,chunksize=4))
                timeouts=sum(r['scoring_timed_out'] for r in result)
                if timeouts:
                    # Retry serially on the main CPU process; never silently turn
                    # verifier timeouts into incorrect labels.
                    for i,row in enumerate(result):
                        if row['scoring_timed_out']:
                            result[i]=scorer(rows[i])
                assert not any(r['scoring_timed_out'] for r in result)
                # metric='answer' (corrected DAPO val/eval metric) makes the last `Answer:` match the primary acc.
                metric=request.get('metric','full')
                assert metric in ('full','answer') and not (metric=='answer' and penalty)
                extra=bool(request.get('answer_metrics',False)) or metric=='answer'
                if extra:
                    # The answer line must be parsed as plain text, not reduced to a (missing) \boxed{}.
                    assert os.environ.get('MATH_VERIFY_BOXED_ONLY','1') in ('0','false','False'), \
                        'answer_metrics requires MATH_VERIFY_BOXED_ONLY=0'
                    result=list(pool.map(add_answer_metrics,result,chunksize=4))
                    for i,row in enumerate(result):
                        if row['answer_metric_timeout']:  # serial retry, as for the primary scorer
                            result[i]=add_answer_metrics(row)
                if metric=='answer':
                    result=[use_answer_metric(r) for r in result]
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
                    summary.update(format_penalty=True,
                        accuracy_unpenalized=sum(r['acc_unpenalized'] for r in eligible)/len(eligible),
                        format_violation_rate=sum(not r['format_ok'] for r in eligible)/len(eligible),
                        format_zeroed_rate=sum(r['acc_unpenalized']==1 and r['acc']==0 for r in eligible)/len(eligible))
                if extra:
                    summary.update(answer_summary(eligible))
                if metric=='answer':
                    full={}
                    for row in eligible:
                        full.setdefault(row['index'],[]).append(row['acc_full'])
                    summary.update(metric='answer',
                        metric_definition='last "Answer:" match verified with math-verify; length-capped responses = 0',
                        accuracy_full=sum(r['acc_full'] for r in eligible)/len(eligible),
                        pass_at_k_full=sum(any(v) for v in full.values())/len(full),
                        format_violation_rate=summary['answer_format_violation_rate'])
                scored=Path(request['result']).with_name('scored.jsonl')
                scored.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
                save(request['result'],summary)
            except BaseException as error:
                traceback.print_exc()
                save(request['result'],dict(complete=False,error=str(error)))
                raise


if __name__=='__main__':
    main()
