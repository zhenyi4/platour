"""Portable reference runner for the 18 released Qwen3 bridge recipes."""

from contextlib import contextmanager, nullcontext
import json
import hashlib
import math
from fractions import Fraction
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bridge import Bridge
from .checkpoints import (atomic_json, barrier, capture_rng, load_step, rank,
                          restore_rng, save_step, world_size)
from .scoring import evaluation_score, score


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array or JSONL dataset")
    return rows


def load_bridge(sender: str, receiver: str, mapping: str,
                initialization: str = "zero", dtype: str = "bfloat16", *,
                device_map: str = "auto") -> tuple[Bridge, object]:
    if initialization != "zero":
        raise ValueError("New bridge projections require zero initialization")
    precision = getattr(torch, dtype)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(sender, local_files_only=True)
    fp8 = getattr(config, 'quantization_config', {}).get('quant_method') == 'fp8'
    if fp8:
        from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
        from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING, allow_all_hub_kernels
        _HUB_KERNEL_MAPPING['finegrained-fp8'] = {
            'repo_id': 'kernels-community/finegrained-fp8',
            'revision': '8c178950e8710e26a2210e4e909cd02dc16c8715'}
        os.environ.setdefault('TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR', '1')
        # The exact commit is pinned, so skip the network publisher lookup that
        # `kernels` performs otherwise (it fails intermittently without a token).
        with allow_all_hub_kernels():
            load_finegrained_fp8_kernel()
    if device_map != 'auto' and not (device_map == 'cpu' or
                                    device_map.startswith('cuda:') and
                                    device_map[5:].isdigit()):
        raise ValueError('Model device_map must be auto, cpu, or one explicit CUDA device')
    if device_map == 'auto' and not fp8 and torch.cuda.device_count() == 1:
        device_map = 'cuda:0'
    if device_map == 'auto':
        from .eval_parallel import SHARD_ENV, sender_device_map
        replica = SHARD_ENV in os.environ or 'PLATOUR_GPU_GROUP' in os.environ
        placement = (sender_device_map('qwen3', config.num_hidden_layers, torch.cuda.device_count())
                     if replica else 'auto')
        source = AutoModelForCausalLM.from_pretrained(sender, torch_dtype=precision, device_map=placement)
        target = AutoModelForCausalLM.from_pretrained(receiver, torch_dtype=precision,
                                                     device_map='cuda:0' if replica else 'auto')
    else:
        if fp8:
            raise ValueError('Explicit single-device mode is for non-FP8 senders')
        source = AutoModelForCausalLM.from_pretrained(sender, torch_dtype=precision).to(device_map)
        target = AutoModelForCausalLM.from_pretrained(receiver, torch_dtype=precision).to(device_map)
    tokenizer = AutoTokenizer.from_pretrained(receiver)
    bridge = Bridge(source, target, mapping, initialization,
                    prefill_layout='fp8_prefix' if fp8 else 'dense_prompt')
    bridge.pad_token_id = tokenizer.pad_token_id
    bridge.training_sources = {'sender': str(Path(sender).resolve()),
                               'receiver': str(Path(receiver).resolve())}
    return bridge, tokenizer


def sample_indices(count: int, step: int, per_epoch: int, batch_size: int) -> list[int]:
    epoch, offset = divmod(step - 1, per_epoch)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
    return order[offset * batch_size:(offset + 1) * batch_size]


def prepare_grpo_holdout(data: list[dict], test: list[dict], config: dict,
                         *, seed: int = 42, validation_rows=None) -> tuple[list[dict], list[dict], dict, dict]:
    """Split training questions before fitting; preserve the independent test.

    Published v2 data supplies validation_rows and is never split again. Legacy pinned
    data retains its historical seed-42 test-sized holdout for reproducibility.
    """
    if config["method"] != "grpo" or not data or not test:
        raise ValueError("GRPO needs nonempty original training and test splits")
    if validation_rows is not None:
        from .native_grpo_validation import published_splits
        training, validation, published = published_splits(data, validation_rows, test)
    else:
        if any('original_train_index' in row for row in data):
            raise ValueError('Published TRAIN needs its explicit validation split; refusing to resplit')
        indices = [row["index"] for row in data]
        if len(set(indices)) != len(data):
            raise ValueError("Original training indices must be unique")
        size = len(test)
        if not 0 < size < len(data):
            raise ValueError("Training holdout must leave a nonempty training split")
        positions = set(random.Random(seed).sample(range(len(data)), size))
        training_positions = [i for i in range(len(data)) if i not in positions]
        validation_positions = sorted(positions)

        def select(positions):
            return [dict(data[position], index=i, original_train_index=indices[position])
                    for i, position in enumerate(positions)]

        training, validation = select(training_positions), select(validation_positions)
    if any(not str(row.get("gold", "")).strip() for row in validation):
        raise ValueError("Holdout contains a question without a gold answer")
    if ({tuple(row["prompt_token_ids"]) for row in validation} &
            {tuple(row["prompt_token_ids"]) for row in test}):
        raise ValueError("Training holdout overlaps the independent test prompts")
    protocol = dict(config, train_examples=len(training),
                    steps_per_epoch=math.ceil(len(training) / config["batch_size"]),
                    validation_policy="train_holdout_best_validation_v1",
                    test_eval_samples=config["eval_samples"],
                    eval_samples=config.get("validation_samples", config["eval_samples"]))
    protocol["total_steps"] = protocol["steps_per_epoch"] * 3
    protocol["eval_steps"] = sorted({math.ceil(i * protocol["steps_per_epoch"] / 2)
                                     for i in range(1, 7)})
    split = {"seed": seed, "train_source_indices": (published["train_source_indices"] if validation_rows is not None else [indices[i] for i in training_positions]),
             "validation_source_indices": (published["validation_source_indices"] if validation_rows is not None else [indices[i] for i in validation_positions]),
             "validation_questions": len(validation), "test_questions": len(test),
             "validation_samples": protocol["eval_samples"],
             "test_samples": protocol["test_eval_samples"],
             "eval_steps": [0, *protocol["eval_steps"]]}
    if validation_rows is not None:
        split['source'] = published['source']
        protocol['validation_split_source'] = published['source']
    return training, validation, protocol, split


V1101_SFT_SHARDS = 4


@contextmanager
def reference_sft_numerics():
    """Reference SFT arithmetic: no TF32 or reduced-precision bf16 matmul, math-only SDPA."""
    cuda = torch.backends.cuda
    saved = (cuda.matmul.allow_tf32, cuda.matmul.allow_bf16_reduced_precision_reduction,
             cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.cudnn_sdp_enabled(),
             cuda.math_sdp_enabled())
    cuda.matmul.allow_tf32 = False
    cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cuda.enable_flash_sdp(False)
    cuda.enable_mem_efficient_sdp(False)
    cuda.enable_cudnn_sdp(False)
    cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        cuda.matmul.allow_tf32, cuda.matmul.allow_bf16_reduced_precision_reduction = saved[:2]
        cuda.enable_flash_sdp(saved[2])
        cuda.enable_mem_efficient_sdp(saved[3])
        cuda.enable_cudnn_sdp(saved[4])
        cuda.enable_math_sdp(saved[5])


def reference_sft_chunks(indices: list[int], rows: list[dict], micro_batch: int) -> list[list[int]]:
    """This rank's padded SFT micro-batches in the reference layout.

    The reference run split each 64-question batch over four ranks by stride, sorted each share by
    prompt+answer length (longest first) and cut it into ``micro_batch`` rows. The same
    micro-batches are used for any number of processes: share ``s``'s chunk ``c`` runs on
    rank ``(s + 4c) % world``, so the losses do not depend on the GPU count.
    """
    if micro_batch < 1:
        raise ValueError("micro_batch must be positive")
    chunks = []
    for share in range(V1101_SFT_SHARDS):
        local = sorted(indices[share::V1101_SFT_SHARDS], reverse=True,
                       key=lambda i: len(rows[i]['prompt_token_ids']) + len(rows[i]['response_token_ids']))
        for number, start in enumerate(range(0, len(local), micro_batch)):
            if (share + V1101_SFT_SHARDS * number) % world_size() == rank():
                chunks.append(local[start:start + micro_batch])
    return chunks


def sft_update(bridge: Bridge, optimizer: torch.optim.Optimizer,
               rows: list[dict], indices: list[int], micro_batch: int,
               global_tokens: int) -> tuple[float, int]:
    """Backward this rank's share of one global SFT batch (``indices``, all ranks).

    Returns this rank's token-weighted NLL sum and its response-token count; the global
    NLL is the all-rank sum divided by ``global_tokens``.
    """
    optimizer.zero_grad(set_to_none=True)
    if global_tokens < 1:
        raise ValueError("SFT requires supervised response tokens")
    if bridge.pad_token_id is None:
        raise ValueError('Historical padded SFT needs the signed receiver pad token')
    weighted_nll, answer_tokens = 0.0, 0
    for chunk in reference_sft_chunks(indices, rows, micro_batch):
        micro = [rows[index] for index in chunk]
        tokens = sum(len(row['response_token_ids']) for row in micro)
        loss = bridge.batch_response_loss(micro, bridge.pad_token_id)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite SFT loss')
        (loss * (tokens / global_tokens)).backward()
        weighted_nll += float(loss.detach().double()) * tokens
        answer_tokens += tokens
    for parameter in bridge.P.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    return weighted_nll, answer_tokens


def clip_flat_gradients(parameters) -> torch.Tensor:
    """One flattened SUM all-reduce, then scale to global L2 norm at most 1."""
    parameters = tuple(parameters)
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
    if world_size() > 1:
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if not torch.isfinite(flat).all():
        raise FloatingPointError('Nonfinite global bridge gradient')
    norm = torch.linalg.vector_norm(flat)
    flat.mul_((1. / (norm + 1e-6)).clamp(max=1.))
    offset = 0
    for parameter in parameters:
        parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
        offset += parameter.numel()
    return norm


@torch.no_grad()
def heldout_nll(bridge: Bridge, rows: list[dict], micro_batch: int) -> tuple[float, int]:
    """Held-out NLL: token-weighted over padded micro-batches of the same layout."""
    if bridge.pad_token_id is None:
        raise ValueError('Held-out SFT validation needs the signed receiver pad token')
    stats = torch.zeros(2, dtype=torch.float64, device=next(bridge.P.parameters()).device)
    for chunk in reference_sft_chunks(list(range(len(rows))), rows, micro_batch):
        micro = [rows[index] for index in chunk]
        tokens = sum(len(row['response_token_ids']) for row in micro)
        loss = bridge.batch_response_loss(micro, bridge.pad_token_id)
        stats[0] += loss.double().to(stats.device) * tokens
        stats[1] += tokens
    if world_size() > 1:
        dist.all_reduce(stats)
    expected = sum(len(row['response_token_ids']) for row in rows)
    if not stats[1] or int(stats[1]) != expected:
        raise RuntimeError("Held-out validation token denominator incomplete")
    return float(stats[0] / stats[1]), expected


def policy_loss(new_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor,
                advantage: float, clip_ratio: float = .2, dual_clip: float = 3.,
                correction_cap: float = 2.) -> torch.Tensor:
    """On-policy clipped GRPO, with detached token rollout importance weights."""
    old = new_log_probs.detach()
    ratio = (new_log_probs - old).clamp(-20., 20.).exp()
    a = new_log_probs.new_tensor(advantage)
    raw = -a * ratio
    clipped = -a * ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    bounded = torch.maximum(raw, clipped)
    if advantage < 0:
        bounded = torch.minimum(bounded, -a * dual_clip)
    correction = (old - rollout_log_probs.detach().to(old)).clamp(-20., 20.).exp()
    correction = correction.clamp_max(correction_cap)
    return (bounded * correction).mean()


def grpo_update(bridge: Bridge, tokenizer: object, rows: list[dict],
                indices: list[int], config: dict, step: int,
                global_groups: int, *, reward_fn=None) -> tuple[float, int]:
    group_size = config["group_size"]
    max_tokens = config["response_cap"]
    sampling = config["train_sampling"]
    denominator = global_groups * group_size
    objective = 0.0
    correct_count = 0
    from .vllm_decode import VLLMDecoder
    from training_protocol import rollout_log_probs
    if 'rollout_output' not in config:
        raise ValueError('GRPO requires a private vLLM rollout output directory')
    selected = [dict(rows[index], index=index) for index in indices]
    if not selected:
        return 0., 0.
    with torch.no_grad(), VLLMDecoder(bridge, Path(config['rollout_output']) /
            f'step-{step:04d}-rank-{rank()}',
            max_model_len=max(len(row['prompt_token_ids']) for row in selected) + max_tokens,
            worker_python=config.get('vllm_python')) as decoder:
        generated = decoder.generate(selected, count=group_size, cap=max_tokens,
            sampling=sampling, parity=True, seed_base=42 + 1000 * step)
    expected = [(index, sample) for index in indices for sample in range(group_size)]
    if [(row['index'], row['sample_index']) for row in generated] != expected:
        raise RuntimeError('GRPO vLLM rollout coverage differs')
    generated = iter(generated)
    for index in indices:
        record = rows[index]
        prompt = record["prompt_token_ids"]
        answers = []
        rewards = []
        old_log_probs = []
        for sample in range(group_size):
            output = next(generated)
            answer = output['response_token_ids']
            log_probs = rollout_log_probs(output, max_tokens)
            result = (score(config["task"], tokenizer.decode(answer), record["gold"],
                            truncated=len(answer) >= max_tokens) if reward_fn is None else
                      reward_fn(record, tokenizer.decode(answer), len(answer) >= max_tokens))
            if not isinstance(result, (int, float)) or not math.isfinite(result) or not 0 <= result <= 1:
                raise ValueError('Custom GRPO reward must be a finite value in [0, 1]')
            answers.append(answer)
            rewards.append(result)
            old_log_probs.append(log_probs)
            correct_count += result
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (
            rewards_tensor.std(correction=1) + 1e-6)
        for sample, answer in enumerate(answers):
            if not float(advantages[sample]):
                continue
            # Backpropagate each answer at once (the objective is a sum over answers), so
            # only one answer's graph is alive: a full 64 x 8 batch fits on one GPU.
            current = bridge.response_log_probs(prompt, answer)
            loss = policy_loss(current, old_log_probs[sample], float(advantages[sample])) / denominator
            loss.backward()
            objective += float(loss.detach())
            del current, loss
    return objective, correct_count


def synchronize_gradients(bridge: Bridge) -> None:
    if world_size() == 1:
        return
    for param in bridge.P.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad)


@torch.no_grad()
def evaluate(bridge: Bridge, tokenizer: object, rows: list[dict], config: dict,
             *, count: int, cap: int) -> dict:
    task = config["task"]
    correct_count = 0
    total = 0
    for record in rows[rank()::world_size()]:
        for sample in range(count):
            generated = bridge.generate(record["prompt_token_ids"],
                seed=1234 + count * record["index"] + sample,
                max_new_tokens=cap,
                temperature=config["eval_sampling"]["temperature"],
                top_p=config["eval_sampling"]["top_p"],
                top_k=config["eval_sampling"].get("top_k", -1))
            correct_count += evaluation_score(task, tokenizer.decode(generated), record["gold"],
                                              valid_labels=record.get("valid_labels", record.get("labels")))
            total += 1
    result = torch.tensor([correct_count, total], dtype=torch.long,
                          device=next(bridge.P.parameters()).device)
    if world_size() > 1:
        dist.all_reduce(result)
    actual, denominator = map(int, result.tolist())
    if denominator != len(rows) * count:
        raise RuntimeError("Evaluation response denominator incomplete")
    return {"correct": actual, "responses": denominator,
            "accuracy": actual / denominator}


@torch.no_grad()
def evaluate_vllm(bridge: Bridge, tokenizer: object, rows: list[dict], config: dict,
                  *, count: int, cap: int, output: Path, batch_size: int = 32,
                  memory_fraction: float = .25, worker_python: str | None = None,
                  _local: bool = False) -> dict:
    """Use the fixed-branch Qwen3 HF-prefill / native-vLLM-0.17.1 decoder.

    Under torchrun every rank decodes a disjoint question shard.
    """
    from .vllm_decode import VLLMDecoder
    if not _local and world_size() > 1:
        from .parallel import collective_evaluation
        return collective_evaluation(rows, count, output, lambda local, path: evaluate_vllm(
            bridge, tokenizer, local, config, count=count, cap=cap, output=path,
            batch_size=batch_size, memory_fraction=memory_fraction, worker_python=worker_python, _local=True))
    if not rows or count < 1 or cap < 1 or batch_size < 1:
        raise ValueError('vLLM evaluation needs rows and positive budgets')
    if len({row['index'] for row in rows}) != len(rows):
        raise ValueError('Evaluation question indices must be unique')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError('Use a fresh vLLM evaluation directory')
    output.mkdir(parents=True)
    choice = config['task'] in {'arc_c', 'mmlu_redux', 'chemistry_no_cot'}
    shards = 4 if choice else 1
    required = max(len(row['prompt_token_ids']) for row in rows) + cap
    budget = max(3072, required) if choice else required
    correct = total = clipped = 0
    parity = None
    with VLLMDecoder(bridge, output, max_model_len=budget,
            memory_fraction=memory_fraction, max_num_seqs=32 if choice else 16,
            max_num_batched_tokens=4096 if choice else None,
            worker_python=worker_python) as decoder:
        if config.get('parity_report', False):
            from .vllm_decode import check_parity
            parity = check_parity(bridge, decoder, rows)
            atomic_json(parity, output / 'parity.json')
        with (output / 'scored.jsonl').open('w') as stream:
            for shard in range(shards):
                part = rows[shard::shards]
                for start in range(0, len(part), batch_size):
                    batch = part[start:start + batch_size]
                    responses = decoder.generate(batch, count=count, cap=cap,
                                                 sampling=config['eval_sampling'])
                    expected = [(row['index'], sample) for row in batch for sample in range(count)]
                    if [(row['index'], row['sample_index']) for row in responses] != expected:
                        raise RuntimeError('vLLM response order or coverage differs')
                    signed_rows = {row['index']: row for row in batch}
                    for response in responses:
                        row = signed_rows[response['index']]
                        details = {}
                        if config['task'] == 'gsm8k':
                            from .scoring import gsm8k_evaluation_details
                            details = gsm8k_evaluation_details(response['response'], row['gold'])
                        elif config['task'] == 'math500':
                            from .scoring import math_evaluation_details
                            details = math_evaluation_details(response['response'], row['gold'])
                        value = (details['correct'] if details else evaluation_score(
                            config['task'], response['response'], row['gold'],
                            valid_labels=row.get('valid_labels', row.get('labels'))))
                        stream.write(json.dumps({**response, **details,
                            'gold': row['gold'], 'correct': value,
                            'at_token_cap': response['tokens'] >= cap}, ensure_ascii=False) + '\n')
                        correct += value
                        total += 1
                        clipped += response['tokens'] >= cap
                    stream.flush()
                    atomic_json({'correct': correct, 'responses': total,
                                 'expected_responses': len(rows) * count}, output / 'live.json')
        runtime = decoder.runtime
    if total != len(rows) * count:
        raise RuntimeError('Qwen3 vLLM full test denominator differs')
    return {'correct': correct, 'responses': total, 'accuracy': correct / total,
            'at_token_cap': clipped, 'backend': 'vllm', 'runtime': runtime,
            'parity': parity}


@torch.no_grad()
def validate(bridge: Bridge, tokenizer: object, rows: list[dict], config: dict) -> dict:
    if config["method"] != "sft":
        return evaluate(bridge, tokenizer, rows, config,
                        count=config["eval_samples"], cap=config["response_cap"])
    with reference_sft_numerics():
        value, tokens = heldout_nll(bridge, rows, config["micro_batch"])
    return {"heldout_nll": value, "response_tokens": tokens}


def reconcile_metrics(output: Path, start_step: int) -> None:
    """Resume after a crash between checkpoints: keep steps 1..start_step, archive the rest.

    Metrics after the signed checkpoint were produced by updates that are recomputed
    on resume; they are moved to ``recovery/`` rather than deleted. Gaps still fail.
    """
    metric_file = output / "metrics.jsonl"
    if not metric_file.is_file():
        return
    raw = metric_file.read_bytes()
    lines = raw.split(b"\n")
    kept, steps, torn = [], [], False
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if number != len(lines) - 1 or raw.endswith(b"\n"):
                raise RuntimeError("A committed metric line is corrupt") from None
            torn = True
            continue
        steps.append(record["step"])
        if record["step"] <= start_step:
            kept.append(line)
    if steps != list(range(1, len(steps) + 1)) or len(steps) < start_step:
        raise RuntimeError("Only a contiguous checkpoint-matched training history is resumable")
    if len(steps) == start_step and not torn:
        return
    recovery = output / "recovery"
    recovery.mkdir(exist_ok=True)
    (recovery / f"metrics-before-resume-step-{start_step:04d}-{time.time_ns()}.jsonl").write_bytes(raw)
    temporary = metric_file.with_suffix(".jsonl.tmp")
    temporary.write_bytes(b"".join(line + b"\n" for line in kept))
    os.replace(temporary, metric_file)


def on_rank_zero(function):
    """Run a shared operation on rank zero and propagate its value or failure."""
    if world_size() == 1:
        return function()
    error = value = failure = None
    if rank() == 0:
        try:
            value = function()
        except BaseException as exc:
            failure, error = exc, f'{type(exc).__name__}: {exc}'
    holder = [(error, value)]
    dist.broadcast_object_list(holder, src=0)
    if holder[0][0] is not None:
        if failure is not None:
            raise failure
        raise RuntimeError('Rank-zero operation failed: ' + holder[0][0])
    return holder[0][1]


def validation_without_rng_side_effect(bridge: Bridge, tokenizer: object,
                                       rows: list[dict], config: dict,
                                       output: Path | None = None) -> dict:
    saved = capture_rng()
    try:
        if config['method'] == 'grpo' and config.get('eval_backend', 'vllm') == 'vllm':
            if output is None:
                raise ValueError('GRPO vLLM validation needs an isolated output directory')
            return evaluate_vllm(
                bridge, tokenizer, rows, config, count=config['eval_samples'],
                cap=config['response_cap'], output=output,
                worker_python=config.get('vllm_python'))
        return validate(bridge, tokenizer, rows, config)
    finally:
        restore_rng(saved)


def best_sft_checkpoint(output: Path, config: dict, steps: list[int]) -> dict:
    """SFT: use the final training checkpoint; held-out NLL is monitoring only."""
    candidates = []
    for step in sorted(steps):
        marker = json.loads((output / "checkpoints" / f"step-{step:04d}" / "complete.json").read_text())
        result = json.loads((output / "evaluation" / f"step-{step:04d}.json").read_text())
        nll = result.get("heldout_nll")
        if (not marker.get("complete") or marker.get("step") != step or result.get("step") != step or
                not isinstance(nll, float) or not math.isfinite(nll)):
            raise ValueError("SFT held-out validation checkpoint coverage differs")
        candidates.append({"step": step, "heldout_nll": nll})
    if config.get("test_checkpoint", "last") != "last":
        raise ValueError('SFT TEST requires the final training checkpoint')
    selected = next((row for row in candidates if row["step"] == config["total_steps"]), None)
    if selected is None:
        raise ValueError("SFT final checkpoint is missing from validation history")
    policy = "final_sft_checkpoint_v1"
    return {"policy": policy, "selected_step": selected["step"],
            "validation_heldout_nll": selected["heldout_nll"],
            "test_used_for_selection": False, "candidates": candidates}


def best_holdout_checkpoint(output: Path, config: dict, validation: list[dict]) -> dict:
    """Choose only from complete train-holdout validation, including step zero."""
    denominator = len(validation) * config["eval_samples"]
    candidates = []
    for step in (0, *config["eval_steps"]):
        checkpoint = output / "checkpoints" / f"step-{step:04d}"
        marker = json.loads((checkpoint / "complete.json").read_text())
        result = json.loads((output / "evaluation" / f"step-{step:04d}.json").read_text())
        if (not marker.get("complete") or marker.get("step") != step or
                result.get("step") != step or result.get("evaluation_role") != "validation" or
                result.get("questions") != len(validation) or
                result.get("responses") != denominator or
                type(result.get("correct")) is not int or
                not 0 <= result["correct"] <= denominator or
                abs(result["accuracy"] - result["correct"] / denominator) > 1e-12):
            raise ValueError("Holdout checkpoint or validation coverage differs")
        candidates.append({"step": step, "correct": result["correct"],
                           "responses": denominator, "accuracy": result["accuracy"]})
    selected = max(candidates, key=lambda row: (Fraction(row["correct"], row["responses"]),
                                                -row["step"]))
    return {"policy": "train_holdout_best_validation_v1",
            "selected_step": selected["step"], "validation_accuracy": selected["accuracy"],
            "test_used_for_selection": False, "candidates": candidates}


def prepare_sft_test(bridge, optimizer):
    """Keep live training state on CPU while the final decoder uses the GPU.

    The caller restores the signed terminal optimizer/RNG after TEST, including
    when decoding fails. This must not be used during gradient accumulation.
    """
    import gc
    bridge.zero_grad(set_to_none=True)
    bridge.eval()
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    # Drop the loop's reference to the last original GPU tensor before collecting.
    value = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_selected_once(config: dict, bridge: Bridge, tokenizer: object,
                           validation: list[dict], test: list[dict], output: Path,
                           optimizer: torch.optim.Optimizer, *, selection: dict | None = None) -> None:
    if selection is None:
        selection = best_holdout_checkpoint(output, config, validation)
    samples = config.get("test_eval_samples", config["eval_samples"])
    if rank() == 0:
        atomic_json(selection, output / "selection.json")
    barrier()
    destination = output / "test" / "summary.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        if (result.get("complete") is not True or
                result.get("selected_step") != selection["selected_step"] or
                result.get("questions") != len(test) or
                result.get("responses") != len(test) * samples):
            raise ValueError("Previous final test differs from the signed selection")
        return
    terminal = output / "checkpoints" / f"step-{config['total_steps']:04d}"
    selected = output / "checkpoints" / f"step-{selection['selected_step']:04d}"
    training_modes = [(module, module.training) for module in bridge.modules()]
    try:
        load_step(selected, bridge, optimizer)
        if config['method'] == 'sft':
            prepare_sft_test(bridge, optimizer)
        if config.get('eval_backend', 'vllm') == 'vllm':
            folder = output / 'test' / 'vllm'
            if rank() == 0 and folder.exists():
                recovery = output / 'recovery'
                recovery.mkdir(exist_ok=True)
                folder.rename(recovery / f'incomplete-test-vllm-{os.getpid()}-{random.getrandbits(32)}')
            barrier()
            result = evaluate_vllm(
                bridge, tokenizer, test, config, count=samples, cap=config["response_cap"],
                output=folder, worker_python=config.get('vllm_python'))
        else:
            result = evaluate(bridge, tokenizer, test, config,
                              count=samples, cap=config["response_cap"])
        if result["responses"] != len(test) * samples:
            raise RuntimeError("Final independent test coverage differs")
        if rank() == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json({"complete": True, "questions": len(test), "selected_step":
                         selection["selected_step"], "selection_policy": selection["policy"],
                         "evaluation_role": "test", **result}, destination)
        barrier()
    finally:
        load_step(terminal, bridge, optimizer)
        for module, training in training_modes:
            module.training = training


def train(config: dict, bridge: Bridge, tokenizer: object, data: list[dict],
          validation: list[dict], output: Path, *, resume: Path | None = None,
          final_test: list[dict] | None = None, split: dict | None = None) -> None:
    """Train one recipe; SFT runs under the reference arithmetic settings."""
    with reference_sft_numerics() if config["method"] == "sft" else nullcontext():
        _train(config, bridge, tokenizer, data, validation, output, resume=resume,
               final_test=final_test, split=split)


def training_identity(config, bridge, data, validation, final_test, output):
    """Bind restart state to the recipe, inputs and frozen backbone contents."""
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    models = {}
    for role in ('sender', 'receiver'):
        source = getattr(bridge, 'training_sources', {}).get(role)
        model = getattr(bridge, role, None)
        if source is not None:
            directory = Path(source)
            files = sorted({p for pattern in ('*.json', '*.safetensors', '*.bin', '*.model', '*.tiktoken')
                            for p in directory.glob(pattern) if p.is_file()})
            if not files:
                raise ValueError('Resume identity requires local backbone files')
            hashes = {}
            for path in files:
                with path.open('rb') as stream:
                    hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            models[role] = {'path': source, 'files': hashes}
        elif model is not None:
            checksum = hashlib.sha256()
            for name, tensor in sorted(model.state_dict().items()):
                tensor = tensor.detach().cpu().contiguous()
                checksum.update(str((name, str(tensor.dtype), tuple(tensor.shape))).encode())
                checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            models[role] = {'class': type(model).__qualname__, 'sha256': checksum.hexdigest()}
    return {'run_root': str(Path(output).resolve()), 'recipe': config, 'models': models,
            'train_sha256': digest(data), 'validation_sha256': digest(validation),
            'test_sha256': digest(final_test)}


def audit_portable_resume(output, resume):
    root, saved = Path(output).resolve(), Path(resume)
    if saved.is_symlink() or saved.resolve().parent != root / 'checkpoints':
        raise ValueError('Portable resume requires a checkpoint in this output root')
    candidates = []
    for marker in (root / 'checkpoints').glob('step-*/complete.json'):
        record = json.loads(marker.read_text())
        if record.get('complete') and marker.parent.name == f"step-{record.get('step', -1):04d}":
            candidates.append((record['step'], marker.parent))
    if not candidates or saved.resolve() != max(candidates)[1]:
        raise ValueError('Portable resume requires the newest complete checkpoint')


def _train(config: dict, bridge: Bridge, tokenizer: object, data: list[dict],
           validation: list[dict], output: Path, *, resume: Path | None = None,
           final_test: list[dict] | None = None, split: dict | None = None) -> None:
    """At every validation boundary: finish checkpoint, commit marker, THEN evaluate."""
    if config.get('eval_backend', 'vllm') != 'vllm':
        raise ValueError('Generated validation and TEST require vLLM')
    config = dict(config, eval_backend='vllm')
    per_epoch = math.ceil(len(data) / config['batch_size'])
    if not config.get('smoke', False) and config['total_steps'] != 3 * per_epoch:
        raise ValueError('Formal Qwen3 training requires three complete epochs')
    if resume is not None:
        audit_portable_resume(output, resume)
    protocol = dict(version=3, epochs=None if config.get('smoke') else 3,
        smoke_only=bool(config.get('smoke')), total_steps=config['total_steps'],
        steps_per_epoch=config['steps_per_epoch'], batch_size=config['batch_size'],
        rollout_backend='vllm' if config['method'] == 'grpo' else None,
        evaluation_backend='vllm', sampling=config.get('train_sampling'), group_size=config.get('group_size'),
        validation_samples=config.get('eval_samples'),
        selection='final_sft_checkpoint_v1' if config['method'] == 'sft' else 'best-validation')
    # Hash backbones once per invocation, on rank zero, before any updates.
    protocol['resume_identity'] = on_rank_zero(
        lambda: training_identity(config, bridge, data, validation, final_test, output))
    path = Path(output) / 'training_protocol.json'
    if config['method'] == 'sft':
        protocol['micro_batch'] = config['micro_batch']
    from .parallel import sign_topology
    sign_topology(protocol)
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('Training backend/budget protocol changed; use a new run')
    elif resume is not None:
        raise RuntimeError('Resume lacks the current vLLM training protocol')
    elif rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(protocol, path)
    barrier()
    grpo_holdout = final_test is not None and config["method"] == "grpo"
    if grpo_holdout and (config.get("validation_policy") !=
                         "train_holdout_best_validation_v1" or split is None):
        raise ValueError("A final test requires an independent signed GRPO holdout")
    if final_test is not None and config["method"] == "sft" and split is not None:
        raise ValueError("SFT uses its signed held-out split; no GRPO holdout split")
    output.mkdir(parents=True, exist_ok=True)
    if grpo_holdout:
        record = output / "validation_split.json"
        if record.exists():
            if json.loads(record.read_text()) != split:
                raise ValueError("Cannot resume with a different GRPO train/validation split")
        elif resume:
            raise FileNotFoundError("Missing original signed holdout split on resume")
        elif rank() == 0:
            atomic_json(split, record)
    barrier()
    torch.manual_seed(42)
    random.seed(42)
    expected_steps = config["total_steps"]
    per_epoch = math.ceil(len(data) / config["batch_size"])
    if per_epoch != config["steps_per_epoch"]:
        raise ValueError("Train split or fixed batch size differs from the experiment config")
    optimizer = torch.optim.AdamW(bridge.P.parameters(), lr=config["learning_rate"],
                                  betas=(.9, .999), weight_decay=0.)
    if config["method"] == "sft":
        warmup = config["warmup_steps"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda current: min(
            current / max(1, warmup), max(0, (expected_steps - current) /
                                           max(1, expected_steps - warmup))))
    else:
        scheduler = None
    start_step = load_step(resume, bridge, optimizer, scheduler,
                           expected_identity=identity) if resume else 0
    if start_step > expected_steps:
        raise RuntimeError("Checkpoint exceeds the experiment budget")
    metric_file = output / "metrics.jsonl"
    if start_step == 0 and not resume and metric_file.exists():
        raise FileExistsError("Use a fresh output directory or an explicit checkpoint")
    if resume and rank() == 0:
        reconcile_metrics(output, start_step)
    barrier()
    validation_steps = set(range(per_epoch, expected_steps + 1, per_epoch)) if (
        config["method"] == "sft") else set(config["eval_steps"])
    if config["method"] == "sft" and not resume:
        # The reference run records the zero-bridge held-out NLL before the first update (not a
        # selection candidate; selection uses the epoch checkpoints).
        initial = validate(bridge, tokenizer, validation, config)
        if rank() == 0:
            atomic_json({"step": 0, "evaluation_role": "initial", **initial},
                        output / "initial_validation.json")
        barrier()
    if grpo_holdout:
        validation_steps.add(0)
        if not resume:
            save_step(output, 0, bridge, optimizer, scheduler, identity=identity)
            baseline = validation_without_rng_side_effect(bridge, tokenizer, validation, config,
                output / 'evaluation' / 'step-0000-vllm')
            if rank() == 0:
                destination = output / "evaluation" / "step-0000.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_json({"step": 0, "questions": len(validation),
                             "evaluation_role": "validation", **baseline}, destination)
            barrier()
    if resume and start_step in validation_steps:
        previous = output / "evaluation" / f"step-{start_step:04d}.json"
        if not previous.is_file():
            # A previous evaluation may have timed out. Its completed checkpoint
            # is sufficient to repeat validation without losing optimizer work.
            import uuid
            vllm_root = output / 'evaluation' / f'step-{start_step:04d}-vllm-{uuid.uuid4().hex[:12]}'
            result = validation_without_rng_side_effect(bridge, tokenizer, validation, config,
                                                        vllm_root)
            if rank() == 0:
                previous.parent.mkdir(parents=True, exist_ok=True)
                details = ({"questions": len(validation), "evaluation_role": "validation"}
                           if final_test is not None else {})
                atomic_json({"step": start_step, **details, **result}, previous)
        barrier()
    for step in range(start_step + 1, expected_steps + 1):
        indices = sample_indices(len(data), step, per_epoch, config["batch_size"])
        if not indices:
            raise RuntimeError("Empty training minibatch")
        local = indices[rank()::world_size()]
        if config["method"] == "sft":
            global_tokens = sum(len(data[index]["response_token_ids"]) for index in indices)
            loss, count = sft_update(bridge, optimizer, data, indices,
                                     config["micro_batch"], global_tokens)
            reward = None
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, count = grpo_update(bridge, tokenizer, data, local,
                                      dict(config, rollout_output=str(output / 'rollouts')), step, len(indices))
            reward = count / (len(indices) * config["group_size"])
        metrics = torch.tensor([loss, float(count)], dtype=torch.float64,
                               device=next(bridge.P.parameters()).device)
        if world_size() > 1:
            dist.all_reduce(metrics)
        if config["method"] == "grpo":
            reward = float(metrics[1]) / (len(indices) * config["group_size"])
            loss = float(metrics[0])
            synchronize_gradients(bridge)
            gradient = torch.nn.utils.clip_grad_norm_(bridge.P.parameters(), 1.,
                                                     error_if_nonfinite=True)
        else:
            if int(metrics[1]) != global_tokens:
                raise RuntimeError("SFT micro-batches do not cover the global batch")
            loss = float(metrics[0]) / global_tokens
            gradient = clip_flat_gradients(bridge.P.parameters())
        applied_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler:
            scheduler.step()
        if rank() == 0:
            line = {"step": step, "train/loss": loss, "train/grad_norm": float(gradient),
                    "train/learning_rate": applied_lr}
            if reward is not None:
                line["train/reward"] = reward
            with metric_file.open("a") as file:
                file.write(json.dumps(line) + "\n")
        barrier()
        if step in validation_steps or step == expected_steps:
            save_step(output, step, bridge, optimizer, scheduler, identity=identity)
            # The checkpoint marker is now durable across all ranks. A scoring
            # timeout leaves a valid step from which validation can be retried.
            result = validation_without_rng_side_effect(bridge, tokenizer, validation, config,
                output / 'evaluation' / f'step-{step:04d}-vllm')
            if rank() == 0:
                destination = output / "evaluation" / f"step-{step:04d}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                details = ({"questions": len(validation), "evaluation_role": "validation"}
                           if final_test is not None else {})
                atomic_json({"step": step, **details, **result}, destination)
            barrier()
    if final_test is not None:
        selection = (best_sft_checkpoint(output, config, sorted(validation_steps | {expected_steps}))
                     if config["method"] == "sft" else None)
        evaluate_selected_once(config, bridge, tokenizer, validation, final_test,
                               output, optimizer, selection=selection)
