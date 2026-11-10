# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import logging
import multiprocessing
import os
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool

_MATH_VERIFY_AVAILABLE = False

try:
    # Suppress math_verify's noisy "Timeout is disabled" warnings
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Timeout is disabled.*")

    import math_verify as _math_verify_module

    print(f"[math_verify] Loaded version: {getattr(_math_verify_module, '__version__', 'unknown')}")
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    _MATH_VERIFY_AVAILABLE = True
except Exception as e:
    print(f"[math_verify] Failed to import: {type(e).__name__}: {e}")
    print("[math_verify] Install with: pip install math-verify")


# signal.alarm() (used internally by math_verify for timeouts) only works
# on the main thread of a process.  Ray's async reward pipeline runs
# compute_score inside asyncio's default ThreadPoolExecutor, so every
# call lands on a non-main thread — regardless of CPU architecture.
#
# Two-tier strategy:
#   - Main thread: call parse()/verify() directly with 5s signal-based
#     timeouts (cheapest path, used by tests and one-off callers).
#   - Non-main thread: dispatch to a *persistent* ProcessPoolExecutor and
#     enforce a wall-clock timeout on the future.  The pool is created
#     lazily on first use and reused for the lifetime of the process, so
#     the IPC round-trip cost is sub-millisecond per call (vs the ~10-30
#     ms-per-call cost of spawning a fresh pool, which f1912d1 removed).
#     Without this hard timeout, a pathological SymPy parse can wedge a
#     reward worker indefinitely — manifests as a training hang at a
#     non-deterministic step (the sample with the bad input has to
#     actually appear in a batch).


_PERSISTENT_POOL = None
_PERSISTENT_POOL_LOCK = threading.Lock()
_PERSISTENT_POOL_EXEC_LOCK = threading.Lock()
_PERSISTENT_POOL_READY = False
_PERSISTENT_POOL_TASKS = 0
_PERSISTENT_POOL_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_TIMEOUT_S", "3"))
_MATH_VERIFY_INNER_TIMEOUT_S = max(
    1,
    int(float(os.environ.get(
        "MATH_VERIFY_INNER_TIMEOUT_S",
        str(_PERSISTENT_POOL_TIMEOUT_S),
    ))),
)
_PERSISTENT_POOL_STARTUP_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_POOL_STARTUP_TIMEOUT_S", "30"))
# Recycle the helper subprocess after this many calls so SymPy's expression
# cache can't grow without bound over a long run.
_PERSISTENT_POOL_MAX_TASKS = int(os.environ.get("MATH_VERIFY_POOL_MAX_TASKS", "200"))
_PERSISTENT_POOL_SHUTDOWN_GRACE_S = float(os.environ.get("MATH_VERIFY_POOL_SHUTDOWN_GRACE_S", "1"))
# Parse ONLY the final \boxed{} answer instead of the full response.  Parsing a
# long, LaTeX-dense CoT (e.g. Qwen3-4B's verbose outputs) in SymPy can exceed
# MATH_VERIFY_TIMEOUT_S and silently zero a CORRECT answer; the boxed substring
# parses in microseconds.  If no \boxed{} is present, the answer is format-
# invalid and scores 0 without parsing the full response.  Set
# MATH_VERIFY_BOXED_ONLY=0 to restore the legacy full-parse fallback.
_BOXED_ONLY = os.environ.get("MATH_VERIFY_BOXED_ONLY", "1") not in ("0", "false", "False")


def _get_persistent_pool():
    global _PERSISTENT_POOL
    if _PERSISTENT_POOL is None:
        with _PERSISTENT_POOL_LOCK:
            if _PERSISTENT_POOL is None:
                # Use 'spawn', NOT 'fork'.  This helper is created lazily from a
                # *multithreaded* Ray asyncio reward worker, and forking such a
                # process is both unsafe (can deadlock on inherited locks) and a
                # memory-accounting trap: the fork child copy-on-write shares the
                # worker's large heap (tokenizer, etc.).  Ray's OOM monitor sums
                # RSS across processes and double-counts those COW-shared pages,
                # so N forked reward helpers each report ~N x the worker heap and
                # trip a *false* node-OOM kill.  A spawned child starts from a
                # clean, small heap, so it is accounted for what it actually uses.
                #
                # The caller recycles the helper every N scores to bound SymPy
                # cache growth, including on Python 3.10.
                # _compute_score_inner is a module-level function and its args are
                # plain str/bool, so spawn's pickle-by-name requirement is met.
                ctx = multiprocessing.get_context("spawn")
                _PERSISTENT_POOL = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                )
                atexit.register(_shutdown_persistent_pool)
    return _PERSISTENT_POOL


def _shutdown_persistent_pool():
    global _PERSISTENT_POOL, _PERSISTENT_POOL_READY
    with _PERSISTENT_POOL_LOCK:
        pool = _PERSISTENT_POOL
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_READY = False
    if pool is None:
        return

    # ProcessPoolExecutor.shutdown(wait=False) cancels queued work but can leave
    # the currently running child alive.  For math_verify timeouts that child is
    # exactly the risky process: it may be wedged in SymPy with hundreds of GB of
    # temporary state, and Ray's OOM monitor will keep counting its RSS.  Kill the
    # private worker processes explicitly, then shut the executor down.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    for proc in processes:
        try:
            proc.join(timeout=_PERSISTENT_POOL_SHUTDOWN_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=_PERSISTENT_POOL_SHUTDOWN_GRACE_S)
        except Exception:
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _math_verify_pool_warmup():
    return True


def _boxed_answer_for_parse(model_output):
    if not _BOXED_ONLY:
        return None
    try:
        from .math_reward import last_boxed_only_string

        return last_boxed_only_string(model_output)
    except Exception:
        return None


def _answer_text_for_parse(model_output):
    """Restrict SymPy parsing to the final \\boxed{} answer when present.

    Parsing the FULL verbose response is what blows past MATH_VERIFY_TIMEOUT_S
    and zeroes correct answers on long, LaTeX-dense CoT.  The boxed substring
    parses in microseconds and matches the verifier's intended "final boxed
    answer" semantics (compute_score_dict already extracts pred this way).
    Full-output parsing is available only when MATH_VERIFY_BOXED_ONLY=0.
    """
    if not _BOXED_ONLY:
        return model_output
    boxed = _boxed_answer_for_parse(model_output)
    if boxed:
        return boxed
    return model_output


def _compute_score_inner(model_output, ground_truth, ground_truth_boxed, use_timeout: bool):
    parse_timeout = _MATH_VERIFY_INNER_TIMEOUT_S if use_timeout else None
    verify_timeout = _MATH_VERIFY_INNER_TIMEOUT_S if use_timeout else None
    gold_parsed = parse(ground_truth_boxed,
                        extraction_config=[LatexExtractionConfig()],
                        parsing_timeout=parse_timeout)
    if not gold_parsed:
        gold_parsed = parse(str(ground_truth),
                            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                            parsing_timeout=parse_timeout)
    if not gold_parsed:
        return None  # sentinel: gold parse failed
    answer_parsed = parse(_answer_text_for_parse(model_output),
                          extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                          parsing_timeout=parse_timeout)
    return float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))


def _short_boxed_answer(model_output: str) -> str | None:
    if not _BOXED_ONLY:
        return None
    answer_text = _boxed_answer_for_parse(model_output)
    if answer_text is None:
        return None
    # Keep the no-timeout, in-thread path only for genuinely small final-answer
    # snippets.  This avoids ProcessPool spawn/import overhead for normal boxed
    # answers while preserving the subprocess guard for long/unboxed outputs.
    if len(answer_text) > 512:
        return None
    return answer_text


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    global _PERSISTENT_POOL_READY, _PERSISTENT_POOL_TASKS

    if not _MATH_VERIFY_AVAILABLE:
        raise RuntimeError("math_verify is not available — check import errors above")

    if ground_truth is None:
        print(f"[math_verify] Invalid ground_truth: type={type(ground_truth)}, value={ground_truth!r}")
        return 0.0
    if not isinstance(ground_truth, str):
        # Some parquet datasets store numeric answers as int/float scalars.
        # Treat them as the same literal answer the JSON trajectory dump shows,
        # instead of silently marking every such row wrong.
        ground_truth = str(ground_truth)

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    on_main_thread = threading.current_thread() is threading.main_thread()
    boxed_answer = _short_boxed_answer(model_output)
    boxed_for_parse = _boxed_answer_for_parse(model_output) if _BOXED_ONLY else None

    try:
        if _BOXED_ONLY and boxed_for_parse is None:
            return 0.0
        if boxed_answer is not None and on_main_thread:
            # math_verify's timeout mechanism relies on signal.alarm, which is
            # available here.  Keep its parser/verifier timeout enabled even for
            # short boxed strings: malformed symbolic expressions can otherwise
            # wedge a standalone evaluation worker indefinitely.
            result = _compute_score_inner(boxed_answer, ground_truth, ground_truth_boxed, use_timeout=True)
        elif on_main_thread:
            result = _compute_score_inner(boxed_for_parse or model_output, ground_truth, ground_truth_boxed, use_timeout=True)
        else:
            # Subprocess path: the worker function runs on the subprocess main
            # thread, so math_verify's signal-based parser/verifier timeouts are
            # valid here.  Keep the outer future timeout as a hard wall-clock
            # guard and pool-recycle trigger for pathological cases.  The
            # executor has one worker; serialize submissions so the timeout
            # measures parse/verify time, not time waiting behind queued samples.
            for pool_attempt in range(2):
                with _PERSISTENT_POOL_EXEC_LOCK:
                    if _PERSISTENT_POOL_TASKS >= _PERSISTENT_POOL_MAX_TASKS:
                        _shutdown_persistent_pool()
                        _PERSISTENT_POOL_TASKS = 0
                    pool = _get_persistent_pool()
                    try:
                        if not _PERSISTENT_POOL_READY:
                            warmup = pool.submit(_math_verify_pool_warmup)
                            warmup.result(timeout=_PERSISTENT_POOL_STARTUP_TIMEOUT_S)
                            _PERSISTENT_POOL_READY = True

                        future = pool.submit(
                            _compute_score_inner,
                            boxed_for_parse or model_output,
                            ground_truth,
                            ground_truth_boxed,
                            True,
                        )
                        result = future.result(timeout=_PERSISTENT_POOL_TIMEOUT_S)
                        _PERSISTENT_POOL_TASKS += 1
                        break
                    except FutureTimeoutError:
                        # Subprocess is wedged in SymPy; recycle the pool so the
                        # next call doesn't queue behind the dead worker.
                        timeout_s = (
                            _PERSISTENT_POOL_STARTUP_TIMEOUT_S
                            if not _PERSISTENT_POOL_READY else _PERSISTENT_POOL_TIMEOUT_S
                        )
                        phase = "startup" if not _PERSISTENT_POOL_READY else "scoring"
                        _shutdown_persistent_pool()
                        print(
                            f"[math_verify] Pool {phase} timeout ({timeout_s}s) "
                            f"for gold={ground_truth!r} model={model_output[:200]!r}"
                        )
                        return timeout_score
                    except BrokenProcessPool:
                        _shutdown_persistent_pool()
                        print("[math_verify] Pool broke; recycling")
                        if pool_attempt == 0:
                            continue
                        return timeout_score
            else:
                return timeout_score
        if result is None:
            print(f"[math_verify] Failed to parse gold answer: {ground_truth!r}")
            return 0.0
        return result
    except TimeoutError:
        print(f"[math_verify] Timeout for gold={ground_truth!r} model={model_output[:200]!r}")
        return timeout_score
    except Exception as e:
        print(f"[math_verify] Error for gold={ground_truth!r} model={model_output[:200]!r}: {type(e).__name__}: {e}")
        return 0.0


def compute_score_dict(solution_str: str, ground_truth: str, timeout_score: float = 0) -> dict:
    """Dict-returning wrapper around ``compute_score`` for the verl metric pipeline.

    The validation metrics aggregator (``process_validation_metrics``)
    iterates over keys of the per-sample reward output, so dict callers
    expose ``acc`` and ``pred`` separately from ``score``.  This wrapper
    keeps math_verify's SymPy-based correctness check (no last-300-chars
    truncation, full \\boxed{} extraction) while emitting the same dict
    shape ``math_dapo.compute_score`` does so existing dashboards keep
    working (e.g. ``val-core/aime24/acc/mean@N``).
    """
    score = compute_score(solution_str, ground_truth, timeout_score=timeout_score)
    pred = None
    try:
        from .math_reward import last_boxed_only_string, remove_boxed, strip_string

        boxed = last_boxed_only_string(solution_str)
        if boxed is not None:
            pred = strip_string(remove_boxed(boxed))
    except Exception:
        pred = None
    return {
        "score": float(score),
        "acc": int(score >= 0.5),
        "pred": pred,
    }
