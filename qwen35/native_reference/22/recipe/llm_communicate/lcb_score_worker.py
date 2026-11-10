"""CPU-only persistent code scorer with the RPC contract of qwen35_bridge_opd_score_worker.py.

Request: {"inputs": [responses.json ...], "expected_responses": N, "result": path, "format_penalty": bool}
Every row carries a `judge` spec (code_prompts.py); each response is judged in its own disposable container
(code_judge.py). acc = 1 iff every test passes. With format_penalty (training rollouts only), a response cut at the
length cap gets reward 0; the unpenalised result is kept in acc_unpenalized. Infrastructure failures are retried and
then fail the request; they are never turned into wrong answers.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_judge  # noqa: E402
from qwen35_bridge_opd_core import save  # noqa: E402

WORKERS = int(os.environ.get('PLATOUR_CODE_JUDGE_WORKERS', '64'))
SETTINGS = json.loads(os.environ.get('PLATOUR_CODE_JUDGE_SETTINGS', '{}'))


def judge_row(row):
    for attempt in range(3):
        try:
            verdict = code_judge.judge(row['judge'], row['response'], SETTINGS)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    results = verdict['results']
    return dict(row, acc=int(verdict['correct']), score=float(verdict['correct']), pred=None, gold_missing=False,
        scoring_timed_out=False, no_code=verdict['no_code'], judge_error_code=verdict['error_code'],
        judge_tests_passed=sum(x > 0 for x in results), judge_results=results[:64],
        judge_seconds=verdict['seconds'], judge_metadata=verdict['metadata'] if not verdict['correct'] else None)


def apply_penalty(row):
    clipped = row['finish_reason'] == 'length'
    ok = not clipped and not row['no_code']
    return dict(row, acc_unpenalized=row['acc'], score_unpenalized=row['score'], format_ok=ok,
                acc=0 if clipped else row['acc'], score=0. if clipped else row['score'])


def summarize(result, penalty, seconds):
    groups = {}
    for row in result:
        groups.setdefault(row['index'], []).append(row['acc'])
    n = len(result)
    summary = dict(complete=True, responses=n, scored_responses=n, correct=sum(r['acc'] for r in result),
        accuracy=sum(r['acc'] for r in result)/n, questions=len(groups),
        passed_questions=sum(any(v) for v in groups.values()), pass_at_k=sum(any(v) for v in groups.values())/len(groups),
        clipping_rate=sum(r['finish_reason'] == 'length' for r in result)/n,
        clipped=sum(r['finish_reason'] == 'length' for r in result), mean_tokens=sum(r['tokens'] for r in result)/n,
        scoring_timeouts=0, gold_missing=0, no_code_rate=sum(r['no_code'] for r in result)/n,
        judge_error_codes={str(k): v for k, v in sorted(Counter(r['judge_error_code'] for r in result).items())},
        judge_wall_seconds=seconds, judge_workers=WORKERS, judge_settings=dict(code_judge.DEFAULTS, **SETTINGS))
    if penalty:
        summary.update(format_penalty=True, accuracy_unpenalized=sum(r['acc_unpenalized'] for r in result)/n,
            format_violation_rate=sum(not r['format_ok'] for r in result)/n,
            format_zeroed_rate=sum(r['acc_unpenalized'] == 1 and r['acc'] == 0 for r in result)/n)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    code_judge.cleanup_exited()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        save(args.directory/'ready.json', dict(cpu_only=True, workers=WORKERS, judge='container per response',
                                               settings=dict(code_judge.DEFAULTS, **SETTINGS)))
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('command') == 'stop':
                break
            try:
                rows = []
                for filename in request['inputs']:
                    payload = json.loads(Path(filename).read_text())
                    assert payload['complete']
                    rows.extend(payload['rows'])
                rows.sort(key=lambda r: (r['index'], r.get('sample_index', 0)))
                assert len(rows) == request['expected_responses']
                assert len({(r['index'], r.get('sample_index', 0)) for r in rows}) == len(rows)
                assert all('judge' in r for r in rows), 'code rows must carry a judge spec'
                started = time.monotonic()
                result = list(pool.map(judge_row, rows))
                penalty = bool(request.get('format_penalty', False))
                if penalty:
                    result = [apply_penalty(r) for r in result]
                summary = summarize(result, penalty, time.monotonic()-started)
                scored = Path(request['result']).with_name('scored.jsonl')
                scored.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result))
                save(request['result'], summary)
            except BaseException as error:
                traceback.print_exc()
                save(request['result'], dict(complete=False, error=str(error)))
                raise
            finally:
                code_judge.cleanup(owner=os.getpid())


if __name__ == '__main__':
    main()
