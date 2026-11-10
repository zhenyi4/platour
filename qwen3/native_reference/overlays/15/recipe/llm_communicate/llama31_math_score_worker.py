"""Persistent math-verify scoring with the native Llama MATH500 scorer."""
import qwen35_bridge_opd_score_worker as worker
from llama31_math_grpo import score
worker.score=score
if __name__=='__main__':worker.main()
