# PLatour

**[On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer](https://arxiv.org/abs/2611.06976)**

PLatour (**P**refill-only **LAT**ent c**O**mm**U**nication for **R**easoning) is a framework for Prefill-to-Autoregressive Cross-model Transfer (PACT): a larger sender contributes prompt representations during prefill, and a smaller receiver generates the response independently. The GRPO method is called PLatour (also known as PLatour-GRPO); its supervised variant is PLatour-SFT.

## What’s included

- **Training and evaluation code** for PLatour (GRPO) and PLatour-SFT, with recipes and test protocols for all 30 experiments. [`platour-exps` CLI](platour_exps.py) · [Python API](platour_api.py) · [Qwen3 recipes](qwen3/README.md) · [Qwen3.5 recipes](qwen35/README.md)
- **PACT-Check code** for Dummy Transfer, Irrelevant Transfer, Self-Communication, and Trained Receiver. [Source](pact_check.py) · [Usage guide](#pact-check)
- **All 30 model checkpoints** for the Qwen3 and Qwen3.5 experiments, including released baselines. [Hugging Face collection](https://huggingface.co/collections/zen-E/platour-6aee0586243e1c15a5dd2cc4) · [Checkpoint index and loading guide](models/README.md#downloadable-checkpoints)
- **Three datasets** with the published experiment inputs, documented splits, and checksums. [Qwen3](https://huggingface.co/datasets/zen-E/PLatour-Qwen3-Data) · [DAPO](https://huggingface.co/datasets/zen-E/PLatour-DAPO-Data) · [DeepCoder](https://huggingface.co/datasets/zen-E/PLatour-DeepCoder-Data) · [Data guide](data/README.md)

![PLatour benchmark results and PACT-Check comparisons](https://raw.githubusercontent.com/zhenyi4/platour/main/assets/hero.png)

## Overview

- **Lightweight communication.** Zero-initialized linear bridges connect frozen sender and receiver models through hidden-state transfer.
- **Direct reward optimization.** PLatour (GRPO) trains the bridges from task rewards, without an alignment or distillation warm-up.
- **Broad applicability.** The framework supports hybrid-attention architectures, generation budgets up to 16K tokens, and sender-to-receiver parameter ratios exceeding 100:1.
- **Controlled evaluation.** PACT-Check examines whether performance gains depend on useful sender information or can be reproduced through receiver-side adaptation.

The paper reports improvements over open-source baselines on generation tasks and over Mostik's reported results on shared mathematics and coding benchmarks with a smaller sender. It also shows that strong benchmark performance alone does not establish useful cross-model information transfer.

## Architecture

![PLatour architecture: frozen sender and receiver connected by trainable prefill-only linear bridges](https://raw.githubusercontent.com/zhenyi4/platour/main/assets/architecture.png)

During prefill, linear projectors map hidden states from sender layers at corresponding relative depths for GRPO (last-n alignment for Qwen3 SFT) into the receiver's residual stream. The receiver then performs all autoregressive decoding without further sender involvement. Both model backbones stay frozen. Only the bridge parameters are trained, using GRPO rewards for PLatour (GRPO) or supervised response loss for PLatour-SFT.

## Getting started

Run from a local checkout with Python 3.11 and install the training environment:

```sh
pip install -e . "torch==2.10.0" "transformers==5.17.0" "fla-core==0.5.2" "kernels==0.16.0"
```

For Qwen3.5 GPU training, complete the environment after the base install:

```sh
python scripts/install_qwen35_extras.py
```

This adds the FLA package, PEFT, data/logging and build tools, and an ABI-matched
`causal-conv1d` wheel while preserving the installed core stack. The wheel path
supports Linux x86_64, Python 3.11 and Torch 2.10 with CUDA 12 and CXX11 ABI;
the installer preserves the system CUDA Toolkit. Use `--dry-run`
to inspect the plan. Finish active jobs in that environment before installing.
See [Qwen3.5 environment completion](qwen35/README.md#complete-the-gpu-training-environment).

Use a separate Python environment with **vLLM 0.17.1** for GRPO rollouts and
validation/test answer generation. Pass its Python executable through
`--vllm-python`, shown as `/path/to/vllm/bin/python` in the examples below.
SFT validation measures loss with HF teacher forcing and does not generate answers.
See the [environment guide](AGENT.md#setup) for dependencies and native-backend requirements.

Before GPU training, check the generation environment with a real FlashAttention
kernel (a version check alone does not detect driver/PTX incompatibility):

```sh
/path/to/vllm/bin/python scripts/check_vllm_runtime.py
```

Run it with the same GPU visibility and library paths as the decoder. This small
BF16 decode check loads no model and does not qualify every model-specific kernel.
The vLLM 0.17.1 wheel can contain CUDA 12.9 PTX even when Torch reports CUDA 12.8.
On supported data-center GPUs with an older driver, follow NVIDIA's
[forward-compatibility instructions](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
for a matching compatibility package. Put its `compat` directory first in
`LD_LIBRARY_PATH` **before** starting the check and training; installing the
package alone does not activate it. System driver upgrades are not performed by
this repository.

- **Training backbones:** download the sender and receiver beforehand and pass their local directories.
- **Evaluation backbones:** `--sender` and `--receiver` accept local directories or HF model IDs; missing HF models are downloaded automatically.
- **Experiment data and released checkpoints:** the relevant entry downloads missing inputs from HF and verifies their hashes. `evaluate` retrieves the released checkpoint when `--weights` is omitted. New PLatour and PLatour-SFT training starts from a zero-initialized bridge.

To prepare all published task data in advance, use `platour-exps download-data --all`;
add `--include-it` to also retrieve Qwen3 IT inputs. See the
[data guide](data/README.md) for offline use.

Detailed usage is covered by the [Qwen3 guide](qwen3/README.md) and
[Qwen3.5 guide](qwen35/README.md). [AGENT.md](AGENT.md) provides instructions
for coding agents, including distributed training, resume, the Python API and the coding judge.

## Train PLatour and PLatour-SFT

After [environment setup](#getting-started), choose an experiment ID to select
the task, model pair and training recipe. Both methods freeze the sender and
receiver and train only the communication bridges: PLatour (GRPO) uses task rewards,
while PLatour-SFT uses supervised responses.

These examples train **Qwen3-8B → Qwen3-1.7B on GSM8K** for three epochs using
the default portable backend. Replace the backbone paths with downloaded local
model directories and the vLLM path with your generation environment's Python.
Missing published datasets are downloaded and verified automatically. Use a
fresh output directory for each run.

```sh
# PLatour-SFT (experiment 11): monitor VAL loss each epoch, then test the final checkpoint.
platour-exps train --experiment 11 \
  --sender /path/to/Qwen3-8B --receiver /path/to/Qwen3-1.7B \
  --test-data data/qwen3/11_qwen3_8b_to_1p7b_sft_gsm8k/test.jsonl \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/11-platour-sft

# PLatour (GRPO, experiment 17): select by VAL accuracy, then test that checkpoint.
platour-exps train --experiment 17 \
  --sender /path/to/Qwen3-8B --receiver /path/to/Qwen3-1.7B \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/17-platour-grpo
```

Qwen3 formal training runs for three epochs; Qwen3.5 runs for one epoch.
The default training backend is `portable`; model-specific GPU and native-backend
requirements are described in the [Qwen3 guide](qwen3/README.md) and
[Qwen3.5 guide](qwen35/README.md).

| Training objective | Validation | Checkpoint used for TEST |
| --- | --- | --- |
| SFT | Record held-out loss each epoch for monitoring | Final training checkpoint |
| GRPO | Generate and score validation answers with vLLM | Validation-best checkpoint |

Formal optimizer defaults are shared by training and PACT controls:

| Objective | Bridge / SC / LoRA / Virtual Sender LR | Full-parameter LR | Schedule |
| --- | --- | --- | --- |
| SFT | `1e-4` | `1e-5` | Linear warmup (10%) then linear decay |
| GRPO | `1e-5` | `1e-6` | Constant, no warmup |

SFT validation always measures response-token NLL with HF teacher forcing, once per epoch; TEST still generates and scores answers. Qwen3.5 published experiments are all GRPO. Their LoRA standard is rank/alpha `128/128`, dropout `0`, and weight decay `0`.

GRPO selects the highest VAL accuracy, with earliest-step ties; response clipping is diagnostic only. Qwen3 validates every half epoch. Qwen3.5 DAPO validates every 33 updates, DeepCoder every 42, and both include the final step (DeepCoder IDs 26/28 also include step 294). Initial step-zero validation is retained.

Qwen3 GRPO uses the corrected disjoint TRAIN/VAL/TEST partitions: Chemistry `1680/210/210`, GSM8K `6973/500/1319`, and MATH `7000/500/500`. GSM8K PLatour, SC and TR all use **500 VAL questions × 4 samples**, and **1,319 TEST questions × 2 samples**. Chemistry CoT requires a complete `<answer>...</answer>` field; missing fields are incorrect.

Qwen3.5 API and PACT defaults match the formal recipe: batch 64, TRAIN/VAL response cap 16,384, DAPO validation avg@4 and DeepCoder avg@16. TEST sampling and caps are separate; see [Qwen3.5 formal defaults](qwen35/README.md#shared-formal-defaults).

These checkpoint rules also apply to PACT-Check SC/TR. In the Qwen3
PLatour-SFT training command, final TEST runs only when `--test-data` is supplied,
as above. Qwen3 PLatour (GRPO) uses the published TRAIN/VAL/TEST splits by default
and evaluates TEST after checkpoint selection. All TEST answer generation uses vLLM.

## Single-GPU and eight-GPU training / PACT-Check

`train`, `evaluate`, and `pact-check` accept the **optional** `--n-gpu N`
(alias `--n_gpu N`). It is unset by default. N specifies the **total GPUs on
this host**, not the number of replicas. For standalone evaluation and portable
training/PACT, the behavior is:

| Setting | Behavior |
|---|---|
| Omit `--n-gpu` | Keep the existing device placement and launch behavior. A direct invocation uses one process and may place model components on multiple visible GPUs. Supported training/PACT paths can instead be launched explicitly with `torchrun`. No replica launcher is started automatically. |
| `--n-gpu 1` | Restrict the run to the first visible GPU and use one replica, provided its configured group size is 1 and the workload fits. |
| `--n-gpu N` | Use the first N visible GPUs, partitioned into groups of G GPUs each; run N / G model replicas. G is the default or explicit `--gpus-per-replica` value. |

N and G must be positive integers, N must be divisible by G, and N cannot exceed
the visible GPU count. The examples use 1 and 8, but other counts satisfying
these constraints are supported. For **8B→1.7B**, G defaults to 1, so
`--n-gpu 8` starts eight replicas. Exposing eight GPUs alone does not enable
data parallelism, and omitting the flag does not restrict the run to one GPU.

Use `CUDA_VISIBLE_DEVICES` to select and order the physical GPUs; the launcher
uses its first N devices. Do not combine `--n-gpu` with an external `torchrun`.
Native training backends retain their documented fixed per-host and multi-host
topologies; this flag does not make them support arbitrary GPU counts.

```sh
# PLatour-SFT, eight replicas; change only --n-gpu to 1 for single-GPU training.
platour-exps train --experiment 11 \
  --sender /path/to/Qwen3-8B --receiver /path/to/Qwen3-1.7B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/11-sft-8gpu

# PLatour-GRPO, Qwen3.
platour-exps train --experiment 17 \
  --sender /path/to/Qwen3-8B --receiver /path/to/Qwen3-1.7B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/17-grpo-8gpu

# PLatour-GRPO, Qwen3.5.
platour-exps train --experiment 19 \
  --sender /path/to/Qwen3.5-35B-A3B --receiver /path/to/Qwen3.5-4B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/19-grpo-8gpu

# PACT inference only; --n-gpu 1 also works for 8B→1.7B.
platour-exps pact-check --experiment 17 --controls DT IT \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/17-dt-it-8gpu

# PACT SC/TR training and their tests, with collective validation.
platour-exps pact-check --experiment 17 --controls SC TR \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/17-sc-tr-8gpu
```

Portable training splits the global batch across replicas and sums gradients
before each shared update. SFT loss remains normalized by global response tokens;
GRPO retains the global question/sample denominator, including short final batches.
Validation and the final TEST now use every training replica. PACT DT calibration,
DT/IT inference, and SC/TR SFT or GRPO training/validation/test also participate
collectively. Controls still run in sequence, each using the available replicas.
Only rank zero writes shared reports/checkpoints; checkpoints contain every rank's
RNG. Resume with the same replica count and GPUs per replica.

The default group size is 2 GPUs for a 235B sender, 4 for a 397B sender, and 1
otherwise; `--gpus-per-replica` overrides it and must divide N. PACT defaults use
the experiment ID or sender model name; for a renamed large-model directory,
specify the group size explicitly. PACT SC/TR-only runs default to one GPU per
replica because they do not load the external sender. Mixed DT/IT/SC/TR runs keep
one group layout throughout; SC/TR use the first GPU in each group.
These are placement defaults, not a guarantee that a particular training batch
fits in memory. Portable grouped training is separate from the frozen native
backends: historical SFT keeps four ranks/eight GPUs, and native 397B training
keeps its two-host requirement. Use the portable backend for single-GPU 8B runs.

GPU replica count is not a measured speedup. Portable GRPO still starts a fresh
vLLM worker per update; active-request limits are unchanged. Floating-point
reduction and decode batching can produce numerical differences across GPU counts.

## PACT-Check

PACT-Check combines four controls to assess the contribution of sender information:

| Control | What it tests |
| --- | --- |
| Dummy Transfer (DT) | Replace question-dependent sender information with fixed template-level features. |
| Irrelevant Transfer (IT) | Supply sender representations from an unrelated question. |
| Self-Communication (SC) | Train communication modules using the receiver itself as the sender. |
| Trained Receiver (TR) | Train the receiver directly with full-parameter fine-tuning (default) or LoRA. |

Use `--controls DT IT SC TR` to choose any subset of the four sanity checks. The entry reports each selected control independently; it does not evaluate a PACT baseline or calculate PACT-Score.

For Qwen3 experiments **07–12 and 16–18**, the default is to run all four
controls for the selected experiment. DT and IT require no training; SC and TR
train independently for three epochs. SC/TR default to SFT for 07–12 and GRPO
for 16–18, following the checkpoint rules above. TR defaults to full-parameter
training; use `--tr-method lora` for LoRA (default rank=128, alpha=128).

IT uses fixed HF-hosted pairings and length-matched supplements. Task, template,
mapping and test sampling come from the pinned profile. The examples below use
**experiment 17: Qwen3-8B → Qwen3-1.7B, GSM8K, GRPO**.

```sh
# Validate data for DT, IT, SC and TR without loading models.
platour-exps pact-check --experiment 17 --prepare-only

# Run DT, IT, SC and TR; SC/TR each train for three epochs.
platour-exps pact-check --experiment 17 \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/17-pact-check

# Evaluate IT only, without SC/TR training.
platour-exps pact-check --experiment 17 --controls IT \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/17-it

# Evaluate DT and IT, without SC/TR training.
platour-exps pact-check --experiment 17 --controls DT IT \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/17-dt-it
```

Qwen3.5 uses the `--family qwen35` interface; its SC/TR training budget is one epoch. See the [Qwen3 PACT-Check guide](qwen3/README.md#frozen-it-evaluation-0712-1618)
for data preparation, offline use and custom Qwen3 inputs.

## Checkpoints on Hugging Face

The [checkpoint index](models/README.md#downloadable-checkpoints) lists every supported HF repository and its experiment ID. All 30 checkpoints are supported by this repository's evaluation loader.

| Experiments | Released checkpoints |
|---|---|
| 01–12 | Qwen3 PLatour-SFT: 235B→1.7B and 8B→1.7B on ARC-C, MMLU-Redux, Chemistry without/with CoT, GSM8K and MATH500 |
| 13–18 | Qwen3 PLatour (GRPO): 235B→1.7B and 8B→1.7B on Chemistry with CoT, GSM8K and MATH500 |
| 19–24 | Qwen3.5 DAPO: 35B→4B, 397B-FP8→4B, Self-Communication, Virtual Sender, Full-GRPO and LoRA-GRPO |
| 25–30 | Qwen3.5 DeepCoder: 397B-FP8→4B, 35B→4B, Self-Communication, Virtual Sender, Full-GRPO and LoRA-GRPO |

Run `evaluate` to score a published checkpoint directly, without training.
The experiment ID selects the task and TEST protocol. When `--weights` is
omitted, the loader downloads and verifies that experiment's released checkpoint;
you can also specify its HF repository explicitly.

```sh
# Qwen3 PLatour-SFT: evaluate 8B→1.7B GSM8K on one GPU.
platour-exps evaluate --experiment 11 \
  --sender Qwen/Qwen3-8B --receiver Qwen/Qwen3-1.7B \
  --n-gpu 1 --vllm-python /path/to/vllm/bin/python --output ./outputs/11-release-eval

# Qwen3 PLatour (GRPO): released GSM8K checkpoint, eight one-GPU replicas.
platour-exps evaluate --experiment 17 \
  --weights zen-E/PLatour-Qwen3-8B-to-1.7B-GSM8K \
  --sender Qwen/Qwen3-8B --receiver Qwen/Qwen3-1.7B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/17-release-eval

# Qwen3.5 PLatour (GRPO): 35B→4B DAPO, eight one-GPU replicas.
platour-exps evaluate --experiment 19 \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python --output ./outputs/19-release-eval

# Optional: download only the released bridge checkpoint, without evaluating.
platour-exps download --experiment 17
```

For Qwen3.5, add `--eval-suites aime` to evaluate only AIME-135 and its
Clean80 subset with the original sampling and token budget. A partial-suite run
reports `three_suite_macro: null`; omitting the option still evaluates all three suites.
Add `--response-cap 16384` for an explicitly shorter Qwen3.5 evaluation;
the selected cap is recorded in `summary.json` and differs from the default 32K
DAPO test protocol.

### GPU defaults for standalone evaluation

Standalone `evaluate` accepts the **optional** `--n-gpu N` (alias `--n_gpu N`),
unset by default. It uses N GPUs in total and splits questions among
N / G independent replicas, where G is `--gpus-per-replica` or its experiment
default. `--n-gpu 1` requires G = 1 and enough memory for the workload.
Omitting `--n-gpu` keeps one process with automatic placement over visible GPUs;
it neither forces single-GPU placement nor starts data-parallel replicas.
The launcher respects `CUDA_VISIBLE_DEVICES`, using the first N visible devices.
The eight-GPU column below is an example; other valid totals use the same rule.

| Released model / experiments | Default GPUs per replica | Replicas with `--n-gpu 8` | Minimum `--n-gpu` with defaults |
|---|---:|---:|---:|
| Qwen3-8B → 1.7B (07–12, 16–18) | 1 | 8 | 1 |
| Qwen3-235B-A22B-FP8 → 1.7B (01–06, 13–15) | 2 | 4 | 2 |
| Qwen3.5-35B-A3B → 4B (19, 26) | 1 | 8 | 1 |
| Qwen3.5-397B-A17B-FP8 → 4B (20, 25) | 4 | 2 | 4 |
| Qwen3.5 self/virtual/full/LoRA controls (21–24, 27–30) | 1 | 8 | 1 |

These defaults are selected by experiment ID, not by detecting available GPU
memory. They are intended for GPUs with enough memory for the assigned models
and decoder cache; the GPU count alone does not guarantee a fit.
`--gpus-per-replica G` requires `--n-gpu N` and overrides this size; N must be
divisible by G and cannot exceed the visible GPU count. Invalid counts fail
before loading models. The default 235B and 397B configurations reject `--n-gpu 1`.
For example, `CUDA_VISIBLE_DEVICES=4,5,6,7` with `--n-gpu 4` uses only those
four physical GPUs. Do not wrap this launcher in `torchrun`. Training and PACT
use the same flag names through their own collective launcher, described above.

Each replica keeps its receiver and TP=1 vLLM decoder on its first GPU,
and partitions larger senders by layer within its group. Existing decode
batch sizes and active-sequence limits are unchanged. Original question IDs,
sample counts and per-answer seeds are retained; batch-dependent numerical
variation means outputs need not be bitwise identical across GPU counts.
The launcher verifies disjoint complete coverage and aggregates counts into
`summary.json`; per-replica outputs and logs remain under the output directory.
`parallel.json` records placement; `evaluation_wall_seconds` includes replica
startup, model loading, generation and scoring but excludes artifact downloads.
Model-specific GPU execution and throughput must be measured before claiming
an acceleration factor; replica count alone is not a benchmark.
Questions are statically assigned as `rows[rank::replicas]`, with no work stealing;
some GPUs can finish earlier when response lengths differ. The implementation
is shared in [`qwen3/eval_parallel.py`](qwen3/eval_parallel.py), with model loading
and scoring integrated into the two families' evaluation entry points.

Replace the vLLM Python path with your generation environment. Evaluation accepts
HF IDs or downloaded local directories for `--sender` and `--receiver`, and
downloads missing backbones and signed TEST data automatically. Use a fresh
`--output` directory for each evaluation; generated responses and scores are
saved there. The `--weights` repository must match the selected experiment.

Full checkpoints (23/29) support standard Transformers loading; LoRA checkpoints (24/30) support PEFT loading. Bridge and Virtual Sender checkpoints use the project loader. See the [loading formats](models/README.md#loading-formats) and [dataset downloads](data/README.md). DeepCoder experiments (25–30) additionally require the [coding judge setup](AGENT.md#coding-judge-ids-2530).

## Reproducing the paper results

This repository contains a refactored version of the code used for the paper’s
experiments. The release may still contain bugs or implementation differences
that affect reproducibility.

The core ideas behind PLatour and PLatour-SFT are straightforward, and the
paper describes both methods. If your results differ from those reported in the
paper, we encourage you to use this repository as a starting point and adapt
the training or evaluation code as needed, using the paper’s experimental
setup as a reference.

If you encounter a bug or a reproduction discrepancy, please open a
[GitHub issue](https://github.com/zhenyi4/llm_understanding/issues). Including
the experiment ID, commit hash, command, and expected and observed results will
help us investigate. Pull requests with fixes are also welcome.

## Citation

If you use PLatour in your research, please cite:

```bibtex
@misc{pact_platour,
  title={On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer},
  author={Zhenyi Shen and Junru Lu and Wei Liu and Jiangnan Ye and Jiazheng Li and Di Yin and Xing Sun and Tameem Adel and Yali Du and Lin Gui and Yulan He},
  year={2026},
  eprint={2611.06976},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2611.06976},
}
```

## License

Original project code is provided under the [Apache License 2.0](LICENSE). Bundled third-party components, downloaded model backbones/checkpoints, and datasets retain their own licenses; this file does not relicense them.
