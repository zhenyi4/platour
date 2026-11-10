"""Original SDPO Chemistry XML accuracy through the existing GRPO scoring RPC."""
import qwen35_bridge_opd_score_worker as worker
from eval_qwen3_17b_chemistry import score as chemistry_score


def score(row):
    result=chemistry_score(dict(row,mode='nonthinking_cot'))
    return dict(result,score=result['acc'],pred=result['extracted_answer'],
                gold_missing=False,scoring_timed_out=False)


worker.score=score
if __name__=='__main__':worker.main()
