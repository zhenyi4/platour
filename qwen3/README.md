# Qwen3: PLatour and PLatour-SFT

The Qwen3 experiments pair a frozen Qwen3-235B-A22B-FP8 or Qwen3-8B sender with a frozen Qwen3-1.7B receiver. A per-layer FP32 linear bridge modifies the receiver only during prompt prefill; the receiver generates on its own. IDs 01–12 train the bridge with SFT (PLatour-SFT); IDs 13–18 use group-reward optimization (PLatour). The paper's broader PACT-Check comparisons are outside these 18 recipes.

## Release scope

The inference export does not establish historical checkpoint-selection provenance or unbiased benchmark estimates. The validation-based procedures below describe new runs with the current runner. Independent evaluation and provenance review are required before making new comparative research claims.

## Inputs and tasks

Each experiment has a signed `data/qwen3/<experiment-tag>/` directory containing `train.jsonl`, `validation.jsonl`, `test.jsonl` and `manifest.json`. SFT also provides its `split.json`. The rows retain question indices, prompt token IDs and gold labels; SFT train/validation also retain the sender-generated or labeled response token IDs. `qwen3.cli` checks input hashes before loading models.

Git retains the manifests and split indices. Missing JSONL inputs are downloaded from the fixed public HF revision in [`data/hf_manifest.json`](../data/hf_manifest.json), then checked against both download and split hashes. Data paths default to the selected experiment: evaluation needs no `--test-data`; training defaults to signed TRAIN and validation (legacy GRPO pins retain their historical holdout policy). SFT final testing remains opt-in with `--test-data`. `platour-exps download-data --experiment <ID>` restores all three splits in advance; see [`data/README.md`](../data/README.md) for offline and custom-root use.

## Standalone evaluation GPU defaults

`platour-exps evaluate` accepts the **optional** `--n-gpu N` (alias `--n_gpu N`),
unset by default. N is the total GPU count on this host. With G GPUs per replica,
the launcher starts N / G replicas, splits questions among them and aggregates
the final counts after checking coverage. Without the flag, evaluation keeps one
process with automatic model placement over visible GPUs; it neither forces
single-GPU placement nor starts data-parallel replicas.

The default `--gpus-per-replica` is **1 for 8B→1.7B** (07–12, 16–18) and
**2 for 235B-FP8→1.7B** (01–06, 13–15). Thus `--n-gpu 8` runs eight or four
replicas respectively. `--n-gpu 1` supports 8B→1.7B; the 235B default requires
at least two GPUs. Replica sizes are fixed by experiment, not by GPU memory
detection. `--gpus-per-replica G` requires `--n-gpu N`. Both counts must be
positive, N must be divisible by G and cannot exceed the visible GPU count,
and each replica's workload must fit. GPU totals other than 1 and 8 follow the
same rules.

```sh
# MATH500: eight one-GPU replicas. Use --n-gpu 1 for one GPU.
platour-exps evaluate --experiment 18 \
  --sender Qwen/Qwen3-8B --receiver Qwen/Qwen3-1.7B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python \
  --output ./outputs/18-math500-8gpu

# MATH500: four two-GPU replicas, with the FP8 sender split by layer.
platour-exps evaluate --experiment 15 \
  --sender Qwen/Qwen3-235B-A22B-FP8 --receiver Qwen/Qwen3-1.7B \
  --n-gpu 8 --vllm-python /path/to/vllm/bin/python \
  --output ./outputs/15-math500-8gpu
```

Use `CUDA_VISIBLE_DEVICES` to select physical GPUs; the launcher uses the first
N visible devices. Each replica's receiver and TP=1 vLLM decoder use its first
GPU. Sampling counts, global question IDs, seeds and active-request limits are
preserved. Do not combine this launcher with `torchrun`. `train` and `pact-check`
also accept these flags through the [training/PACT launcher](../README.md#single-gpu-and-eight-gpu-training--pact-check).
See the [shared evaluation guide](../README.md#gpu-defaults-for-standalone-evaluation)
for output files, constraints and timing semantics.

## Frozen IT evaluation (07–12, 16–18)

The default Qwen3 `pact-check` entry runs **DT, IT, SC and TR** using the
complete archived Qwen3-8B→1.7B TEST rows. It restores fixed IT pairings and
length-matched supplements from the pinned HF dataset automatically. This profile
supports experiments 07–12 and 16–18; other IDs require custom inputs.
Use `--controls` to select any nonempty subset of DT, IT, SC and TR. The entry
does not evaluate the PACT baseline or calculate PACT-Score.

```sh
# Download/verify full-suite TRAIN, VAL and IT inputs without loading models.
platour-exps pact-check --experiment 16 --prepare-only

# Full suite: both SC and TR train for three complete epochs.
platour-exps pact-check --experiment 16 \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/16-pact-check

# Explicit evaluation-only modes, without SC/TR training.
platour-exps pact-check --experiment 16 --controls IT \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/16-it
platour-exps pact-check --experiment 16 --controls DT IT \
  --vllm-python /path/to/vllm/bin/python --output ./outputs/16-dt-it
```

Experiments 07–12 use `last_aligned`; 16–18 use `uniform_endpoints`. The pinned
input config supplies task, sampling, sample count and response cap for **all selected
TEST evaluations**. Default backbones are pinned Qwen/Qwen3-8B and Qwen/Qwen3-1.7B
snapshots. `--sender` and `--receiver` can point to local copies; `--bridge` accepts
the matching released checkpoint and checks its SHA-256. The backend is vLLM.
CUDA/driver compatibility remains an environment prerequisite.

Full-suite preparation uses checksum-verified published TRAIN/VAL data. ARC-C
and MMLU TRAIN/VAL get the exact archived instruction suffix; question and answer
tokens are preserved. GRPO TRAIN/VAL are read directly from the dataset, without runtime splitting.
The corrected dataset assigns Chemistry 210 validation questions and GSM8K/MATH
500 each from original TRAIN, using seed 42 once at publication. Legacy files
whose validation aliases TEST are rejected by this full PACT entry. All TRAIN/VAL/TEST prompt sets must be disjoint. Repeated signed rows within
a split retain distinct indices, including all 5,632 MMLU test rows. DT samples
500 distinct TRAIN prompts by default. Supplements remain IT donors only.

GSM8K PLatour-GRPO and both GRPO SC/TR controls validate with avg@4 (500 × 4 answers).
Their final TEST protocol remains avg@2 (1,319 × 2); validation and TEST sampling
budgets are recorded separately.

SC and TR each train for three complete epochs:
`3 * ceil(TRAIN questions / batch_size)`
updates, including a short final batch in each epoch. SC/TR use SFT for 07–12 and
GRPO for 16–18 unless `--control-objective` is specified. Other controls include
`--batch-size`, `--tr-method full|lora` (default `full`), `--tr-lora-rank` (default
128; alpha equals rank), and `--dt-samples` (default 500). The Qwen3 entry rejects
`--control-steps`; the budget is derived from the published TRAIN size. Results are saved in
`pact_check.json`; `pact_protocol.json` seals split hashes, template transforms,
IT provenance and control budgets. An incomplete full run can resume with the same
arguments and output directory; a completed run requires a fresh directory.

The default effective batch follows the published training recipe: SFT uses 64
questions; GRPO uses 32 for chemistry and 64 for GSM8K/MATH. `--batch-size`
overrides it. SC and TR SFT use padded micro-batches with gradient accumulation.
`--micro-batch` defaults to 8 for ARC-C, 4 for GSM8K, and 1 for MATH.
SC/TR-only runs load only their receiver/control models; the original sender and
released bridge are loaded only when DT or IT is requested. Before a trained
control starts TEST, gradients are cleared and idle CUDA allocator blocks are
released. Both bridge and receiver-only vLLM workers also release idle cache
before startup and record `memory_before_worker.json` with free/total and
allocated/reserved bytes. This does not change checkpoint choice or sampling.

GSM8K uses a smaller micro-batch to reduce long-sequence activation memory;
the effective batch remains 64. This reduces OOM risk but does not guarantee
that every sequence fits on every device.
TR normalizes by the entire batch's response-token count,
and updates once per effective batch. The short last batch uses its actual token
count. SFT SC/TR measure token-averaged held-out NLL at step zero and each
epoch boundary using HF teacher forcing, with no answer generation for VAL.
Initial and epoch-end VAL loss are monitoring only; TEST always uses the final
training checkpoint (epoch 3 in formal SFT), with vLLM generation. Existing runs must retain their signed batch/protocol;
use a new output directory for this training configuration.

Checkpoints default to epoch-end SFT validation points, GRPO selection points, and the final step.
Reference-only step-zero VAL is recorded without a checkpoint: three-epoch
ARC-C SFT saves steps 18/36/54 and GSM8K saves 116/232/348;
the ten-step GRPO smoke saves only step 10. `--checkpoint-every N` adds recovery
points without changing validation or selection (`0`, the default, adds none).
Every checkpoint contains model, optimizer and RNG state. The TR writer hashes
new files once, avoids rehashing previous payloads during each save/retirement,
and still fully verifies payloads when locating/restoring checkpoints. Retention
keeps the two newest committed states; GRPO additionally keeps its validation-best
checkpoint and step zero when eligible for selection. An interrupted run replays updates after its last
saved state and archives uncommitted metric lines. Before the first checkpoint,
restart with a fresh session; training and initial VAL are repeated from scratch.
Micro-batch and checkpoint settings are signed into the training protocol; old
training attempts require a fresh output with the new settings. Preserve previous
outputs when starting a run under a different protocol.

`--it-data-root` relocates the IT inputs; `--data-root` relocates the published data
root containing `qwen3/<experiment-tag>/`. `--local-files-only` uses cached HF data
and backbones; offline execution also requires `--bridge`. `--prepare-only`
validates the selected full or evaluation-only mode without loading models.

Every selected control reports its own TEST result in `pact_check.json` and
`test/<control>/`. Unselected controls do not train, calibrate, or evaluate.
For example, `--controls SC TR --tr-method full` runs the two trained controls
with full-parameter TR; use a separate output with `--tr-method lora` for LoRA TR.

For a bounded GRPO code check, use `--controls SC` or `--controls TR` with
`--grpo-smoke-steps 10`. This runs the full initial VAL, then ten training steps
with vLLM rollouts. It performs no subsequent VAL and no TEST. The final training
checkpoint is retained with `selection_performed: false`; the run is explicitly
marked smoke-only. This option requires GRPO and only SC/TR controls. Formal
training defaults remain three epochs with validation-based selection.

TR GRPO rollouts use vLLM with a fresh current-weight snapshot per update,
eight responses per TRAIN question, and the worker's token log probabilities.
There is no HF rollout fallback for TR. SC GRPO training rollouts also use vLLM;
GRPO validation and all TEST generation use vLLM. For GRPO SC/TR, validation runs at step zero and every half epoch; training always completes its three epochs. The highest validation
accuracy selects the checkpoint (earliest step on ties); TEST is evaluated only
after loading that checkpoint. GSM8K GRPO PLatour/SC/TR use 500-question VAL avg@4 and 1,319-question TEST avg@2.
`SC/selection.json` and `TR/selection.json` record all candidates and each selected
weight hash. Recovery uses the latest optimizer state; completed training restores
the selected state. The selected checkpoints are retained along with step zero
and the latest two optimizer states. Signed `test/SC-summary.json` and
`test/TR-summary.json` results are reused if a later PACT stage is interrupted.
Changing the budget or selection/rollout protocol requires a new run directory.

Custom Qwen3 runs use `--it-source custom` plus the legacy `--sender`, `--receiver`,
`--bridge`, `--dataset`, `--task`, `--prefix-ids`, `--suffix-ids` and
`--semantic-group-field` arguments. This is an explicit opt-out from the frozen
profile; default runs never silently fall back to selecting new donors. Qwen3.5
continues to use the existing `--family qwen35` interface unchanged.

Archived prompts can differ from the release's normal test templates. All five
settings use those same archived receiver prompts and original gold answers;
donors are never attached to a different test by index or length alone. See the
[IT data notes](../data/README.md#frozen-it-supplements-and-pairings) for MATH
provenance and duplicate-prompt behavior.

| Task | IDs: SFT / GRPO | Public source | Evaluation |
|---|---|---|---|
| ARC-C | 01, 07 / — | ARC-Challenge | 1,172 questions, greedy |
| MMLU-Redux | 02, 08 / — | MMLU training data; MMLU-Redux test | 5,632 questions, greedy |
| Chemistry without CoT | 03, 09 / — | SciKnowEval Chemistry | 210 questions, greedy |
| Chemistry with CoT | 04, 10 / 13, 16 | SciKnowEval Chemistry | 210 × 8 sampled answers |
| GSM8K | 05, 11 / 14, 17 | GSM8K | 1,319 × 2 sampled answers |
| MATH500 | 06, 12 / 15, 18 | MATH training data; MATH-500 test | 500 × 4 sampled answers |

The reported metric counts correct **individual responses** over the complete denominator, not pass@k. The sampled Qwen3 test protocol uses temperature `0.7`, `top_p=0.8`, `top_k=20`; task-specific caps are in `experiments.json`. SFT uses its signed train/held-out split; corrected GRPO releases store their TRAIN-derived validation split in the dataset (below). The signed prompt token IDs, rather than a re-rendered template, determine what the models see.

## Evaluate the released weights

Supply the matching public sender and receiver backbones as local directories or HF repository IDs. Evaluation downloads HF backbones, including their tokenizer files, into the normal Hub cache and passes the local snapshots to both generation backends. To select a particular backbone revision, pass a separately downloaded local snapshot of that revision. Training still requires local backbone directories. The per-experiment repositories and 28 projection tensors for each Qwen3 experiment are identified in [`models/manifest.json`](../models/manifest.json). See [`models/README.md`](../models/README.md) for paper accuracies.

`--weights` accepts an HF repository ID or a local checkpoint file. Omit it to use the manifest's release repository. Hub inputs verify the configuration and weight against the selected experiment's pinned SHA-256 identities. Private repositories require an authenticated HF account with access.

```sh
TAG=01_qwen3_235b_to_1p7b_sft_arc_c
platour-exps evaluate --experiment 1 \
  --sender <public-sender> --receiver <public-receiver> \
  --test-data "data/qwen3/$TAG/test.jsonl" \
  --output <new-evaluation-dir>
```

The evaluator uses the signed test rows and the task-specific answer checker (`qwen3.scoring.evaluation_score`, separate from the clipped training reward). By default, the HF sender/receiver computes the bridged prompt cache and native vLLM 0.17.1 batches the remaining decoding using a signed KV handoff. HF answer generation is not available through the public CLI/API. Pass `--vllm-python <vllm-env>/bin/python` when vLLM lives in a separate environment; `--parity-report` additionally records HF/vLLM log-probability differences. Models are placed automatically over the visible GPUs; for 8B→1.7B a single visible GPU is sufficient, provided it has room for both backbones and the vLLM decoder. `--run <training-dir>` evaluates the validation-selected checkpoint of a training run instead of the released weights.

## Train a bridge

```sh
platour-exps train --experiment 1 \
  --sender <public-sender> --receiver <public-receiver> \
  --train-data "data/qwen3/$TAG/train.jsonl" \
  --validation-data "data/qwen3/$TAG/validation.jsonl" \
  --test-data "data/qwen3/$TAG/test.jsonl" \
  --output <new-training-dir>

# Data-parallel training (8B sender): eight GPUs on one machine
TAG7=07_qwen3_8b_to_1p7b_sft_arc_c
torchrun --nnodes=1 --nproc-per-node=8 -m platour_exps train --experiment 7 \
  --sender <public-8b-sender> --receiver <public-1.7b-receiver> \
  --train-data "data/qwen3/$TAG7/train.jsonl" \
  --validation-data "data/qwen3/$TAG7/validation.jsonl" \
  --test-data "data/qwen3/$TAG7/test.jsonl" \
  --vllm-python <vllm-env>/bin/python --output <new-training-dir>
```

Every new bridge projection starts at exact FP32 zero. SFT follows the reference training recipe: each 64-question batch is split over four strided shares, each share is sorted longest-first and cut into padded micro-batches, gradients are summed as one flattened vector and clipped to global norm 1, and TF32, reduced-precision bf16 matmul and fused attention kernels are off. These micro-batches are the same for any number of GPUs. With the historical micro-batch unchanged, four ranks (`torchrun --nproc-per-node=4`, or `--train-backend historical-sft` below) reproduce the reference SFT runs bit for bit; other GPU counts use the same micro-batches and arithmetic but sum gradients in a different floating-point order, so later steps may differ in their last bits. Held-out token NLL (HF forward pass, same padded micro-batches) is recorded before training (`initial_validation.json`) and at every epoch boundary. With `--test-data`, the signed test is generated once (native vLLM) at the **final training checkpoint**, regardless of VAL loss. `selection.json` and `test/summary.json` record the step and the selection policy.

Portable SFT caps GSM8K micro-batches at 4 (8B recipe: previously 8), keeping effective batch 64 and three epochs. Other recipe micro-batches remain unchanged. An explicit `--micro-batch N` on portable SFT changes only the padded micro-batch size, preserving the effective batch, learning-rate schedule and training budget. For example, Chemistry-CoT ID 10 exceeded the available memory on one GPU with its default 8; using micro-batch 2 completed full training and TEST. This override is rejected for GRPO and historical backends. Use the same value when resuming. MATH SFT IDs 06/12 and MATH SC/TR already use micro-batch 1. Historical native recipes remain unchanged; changing micro-batch can change floating-point accumulation, so this override is not a bitwise historical reproduction. The portable SFT protocol now signs micro-batch size: old runs without this field, or runs with a different value, require a fresh training root; their completed checkpoints remain usable for standalone evaluation. Before final TEST, portable SFT clears gradients and moves Adam state to CPU; the signed terminal model/optimizer/RNG are restored after evaluation even on failure. The shared vLLM startup releases idle cache and records `memory_before_worker.json`. Long sequences and the required sender/receiver models can still exceed available GPU memory; these safeguards are not a GPU qualification.

Portable training and PACT accept the same optional `--n-gpu N` flag. Omitting
it preserves automatic device placement for a direct invocation, or the existing
explicit `torchrun` topology; visible GPU count alone does not enable replicas.
Use `--n-gpu 1` for single-GPU 8B→1.7B training or `--n-gpu 8` for eight replicas.
With `--n-gpu 8`, the portable 235B default uses four two-GPU replicas; each
group splits sender layers and keeps the receiver on its first GPU.
`--gpus-per-replica` can override this
layout when memory permits. Existing explicit `torchrun` remains supported for
one-GPU-per-rank models, including multiple hosts with shared output storage.
Do not combine it with `--n-gpu`. These flexible GPU-count rules apply to the
portable backend; historical native backends retain their fixed topologies.

Every rank processes a disjoint training slice and synchronizes gradients;
SFT NLL, GRPO validation, and final vLLM TEST are collective. Rank zero writes
shared reports and checkpoints containing every rank's RNG. Resume with
`--resume <output>/checkpoints/step-XXXX` and the same topology. These rules also
apply to PACT SC/TR, whose API training keeps the same global token/sample
normalization. PACT DT calibration and DT/IT inference are sharded too.
`platour-exps evaluate ... --run <training-dir>` tests the selected checkpoint
of a finished run. Portable GRPO still recreates its vLLM worker per update.

For new GRPO training, provide the signed train and test files:

```sh
TAG=14_qwen3_235b_to_1p7b_grpo_gsm8k
platour-exps train --experiment 14 \
  --sender <public-sender> --receiver <public-receiver> \
  --train-data "data/qwen3/$TAG/train.jsonl" --test-data "data/qwen3/$TAG/test.jsonl" \
  --output <new-training-dir>
```

| GRPO task | Train / validation / test questions | Validation / final-test sampling |
|---|---|---|
| Chemistry with CoT | 1,680 / 210 / 210 | avg@8 / avg@8 |
| GSM8K | 6,973 / 500 / 1,319 | avg@4 / avg@2 |
| MATH500 | 7,000 / 500 / 500 | avg@4 / avg@4 |

These are the corrected `published_train_holdout_v2` dataset sizes. Validation is partitioned once from original TRAIN with seed 42 and saved in HF; both portable and native training read it directly. Older pinned datasets keep the historical runtime holdout policy until the corrected release is published and installed. See [publishing the fixed splits](../data/README.md#fixed-qwen3-grpo-validation-splits). Validation runs before training and periodically thereafter (native vLLM). The best validation accuracy selects the checkpoint (earliest on a tie); the full test runs once in `test/`. Outputs include the train/validation index split, the selection and the tested checkpoint. `qwen3/checkpoints.py` saves bridge, optimizer and RNG before each scheduled validation; `--resume` retries a missing score from the signed checkpoint.

### Portable and native GRPO

`--train-backend portable` remains the default. It now uses vLLM for GRPO
rollouts as well as generated VAL/TEST answers; it exports the current policy
at each update and consumes token-level rollout log-probabilities. HF/PyTorch
still performs bridge prefill, loss computation and backward. The native
`historical-grpo` backend retains its bundled trainer and eight-GPU topology.
These implementations are not guaranteed to produce identical numerical trajectories.

Formal Qwen3 training uses three complete epochs, including short final batches.
PLatour-SFT selects minimum held-out NLL; GRPO and PACT SC/TR select maximum
VAL accuracy (earliest on ties), then evaluate the selected checkpoint on TEST.
Python API `max_steps` overrides require `smoke=True`. A changed backend or
budget requires a new run directory; previous HF-rollout runs cannot be resumed
under the new protocol. GPU regression of the new portable rollout path is pending.

### Native eight-GPU backends

The 235B-FP8 sender is not replicated per GPU; its recipes use the native backends.

- `--train-backend historical-sft`: four paired-GPU ranks on eight GPUs with truly padded SFT micro-batches. Launch with `torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=<free-port> -m qwen3.cli train` plus the same signed SFT arguments. Bridge, FP32 AdamW, the linear scheduler and both GPU RNG states of each rank are signed at step zero, every 16 updates and at each epoch boundary; continue an interrupted run with the same `--output` and `--resume <output>/checkpoints/step-XXXX`. FP8 sender training requires vLLM 0.17.1 in the **same** training Python for its expert kernel; use Torch 2.10, Transformers 5.3.0 and vLLM 0.17.1.
  With `--test-data`, historical SFT runs final TEST collectively using `--vllm-python`, after releasing gradients and optimizer GPU memory. `selection.json` and `test/summary.json` record the final checkpoint. The run is marked complete only after TEST succeeds. If decoding fails, resume with the same arguments and `--resume <output>/checkpoints/step-XXXX` pointing to the final training step; no optimizer updates are repeated. Partial decoder output is archived and completed TEST is reused. TEST data/settings cannot change during resume. Without `--test-data`, no TEST is run. Smoke TEST summaries are explicitly labelled. The collective timeout defaults to 24 hours (`PLATOUR_DIST_TIMEOUT_HOURS`).
- `--train-backend historical-grpo`: the bundled, SHA-checked native trainer with native vLLM rollouts/log-probabilities and the group reward. Supply the signed `--train-data`, `--validation-data` (for the corrected dataset) and `--test-data`; it consumes the published split and runs the single final test as above. Use eight GPUs, vLLM 0.17.1 in the training Python and an isolated `PLATOUR_SCORING_PYTHON` with MathVerify for GSM8K/MATH. Bridge, FP32 AdamW and per-rank CPU/CUDA RNG are signed at every validation boundary; resume with `--resume <output>/train/checkpoints/step-XXXX`.

Both native backends accept `--smoke` (with a fresh output directory) for a bounded few-update check; resume a smoke with `--resume <output>/smoke/checkpoints/step-0001 --smoke`. For smoke runs on GPUs with less memory, `PLATOUR_SFT_FP8_SMOKE_MICRO_BATCH`, `PLATOUR_NATIVE_VLLM_FRACTION` and `PLATOUR_NATIVE_VLLM_GPU_PLACEMENT=secondary` reduce the per-GPU footprint while keeping the global batch; full-budget runs reject these overrides.

Chemistry CoT scoring requires a complete `<answer>...</answer>` field and uses the final field; a missing or unclosed field is incorrect. GSM8K scoring is unchanged. Full-GRPO TR defaults to constant LR `1e-6`; bridge/SC/LoRA GRPO use `1e-5`.

New historical-GRPO runs apply the same strict Chemistry answer-field extraction to a signed private evaluator copy. Archived sources remain unchanged; old roots using permissive extraction require a fresh run.
