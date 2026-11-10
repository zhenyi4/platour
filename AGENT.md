# Agent guide for PLatour

This is the complete usage guide, written for coding agents. [`README.md`](README.md) is the short overview for people; recipe-level details are in [`qwen3/README.md`](qwen3/README.md) and [`qwen35/README.md`](qwen35/README.md).

## Ground rules

- This is the public code/data interface for 30 *different* experiments, not a claim that every historical GPU trajectory has been reproduced.
- Do not treat two-step/three-step smokes as full-budget runs or as a completed PACT-Check. CPU mocks do not establish GPU training, Docker readiness or published test accuracy.
- Use a new private output directory for each run and leave unsuccessful runs unchanged. Do not commit model weights, private outputs, machine paths, credentials or other users' processes.
- Report numerical equality only after comparing actual tensors/gradients with an uninterrupted control in the same environment.

## Where things are

- `platour-exps` (`platour_exps.py`): one command for all 30 experiments; IDs 01–18 route to `qwen3/`, 19–30 to `qwen35/`.
- `models/README.md`, `models/manifest.json`: artifact descriptions and the filenames, sizes and SHA-256 identities of the released inference weights, mapped to separate Hugging Face model repositories by the manifest. Do not add the `.safetensors` files to a code commit. Sender/receiver backbones must be downloaded separately under their respective model licenses.
- `data/README.md`, `data/qwen35/manifest.json`: small split protocols and pinned HF payload identities (no Git LFS required), plus fixed upstream identities for rebuilding programming cases.
- `platour_api.py`: independent sender/bridge/receiver API. `pact_check.py`: PACT-Check library.
- `qwen3/native_reference/`, `qwen35/native_reference/`: bundled native training code used by the `historical-*`/native backends, checked file by file against the `*_source_sha256.json` manifests. They match the original training code except that internal identifiers, environment-variable prefixes, cache paths, W&B names, hardware comments, dates and example addresses were renamed for publication; training, rollout, reward and evaluation logic is unchanged.

## Setup

```sh
pip install -e . "torch==2.10.0" "transformers==5.17.0" "fla-core==0.5.2" "kernels==0.16.0"
```

- Run from this checkout: the signed inputs (`data/`) and the weight manifest (`models/manifest.json`) are read from the repository, so keep the editable install rather than a standalone wheel.
- The command above installs the tested HF environment: Python 3.11 with Torch 2.10.0, Transformers 5.17.0, fla-core 0.5.2, TileLang 0.1.15 and kernels 0.16.0; plain `pip install -e .` may resolve newer, untested fla-core/kernels releases. On Hopper GPUs, fla's gated-delta backward needs TileLang (installed by default) or Triton ≥ 3.7.1. For Qwen3.5 GPU training, run `python scripts/install_qwen35_extras.py` after the base install. It adds the packages in `requirements/qwen35-extras.txt` and a SHA-verified upstream `causal-conv1d==1.6.2.post1` wheel matched to Linux x86_64/Python 3.11/Torch 2.10/CUDA 12/CXX11 ABI. It preserves core package versions, refreshes editable imports, and checks the convolution's CUDA forward/backward. Without causal-conv1d, Transformers falls back to a correct but much slower PyTorch convolution. Other ABI combinations need their own matching wheel or source build; do not silently replace Torch or the system CUDA Toolkit. The installer leaves the separate vLLM environment unchanged. The native two-host backend additionally needs the `native-train` extra (`pip install -e ".[native-train]"` with the same pins).
- GRPO training rollouts and validation/test answer generation require native **vLLM 0.17.1** for both families, including portable training and PACT SC/TR. Keep vLLM in a separate generation Python and pass `--vllm-python <vllm-env>/bin/python`. HF answer-generation options are rejected; SFT held-out NLL remains an HF forward pass. vLLM's package metadata declares `transformers<5`, which conflicts with the Qwen3.5 HF runtime.
- The native Qwen3 GRPO trainer and the 235B-FP8 SFT expert kernel need vLLM 0.17.1 **in the HF training Python**; prepare that combination deliberately rather than relying on pip's dependency resolution. The native FP8 SFT backend uses Torch 2.10, Transformers 5.3.0 and vLLM 0.17.1.
- FP8 senders (Qwen3-235B, Qwen3.5-397B) load a pinned expert kernel from the Hugging Face Hub each time they start, so run them with Hub access (`HF_HUB_OFFLINE` unset) and with `DO_NOT_TRACK` / `HF_HUB_DISABLE_TELEMETRY` unset—kernels 0.16 otherwise sends an invalid request header. The two-host native backend instead caches the complete pinned kernel once in its run root (see [`qwen35/README.md`](qwen35/README.md#native-backends)).
- `platour-exps download-data --experiment 1` restores all signed inputs for one experiment; `--all` restores all 68 payloads. Training/evaluation hydrate missing inputs automatically, using pinned HF revisions and SHA-256. Local modified inputs fail verification. See [`data/README.md`](data/README.md) for custom roots and offline caches. Qwen3 data paths default to the selected experiment; SFT final testing remains optional through `--test-data`.
- `platour-exps download --experiment 1` retrieves and verifies a released weight; evaluation does this automatically if it is not already present.

## Evaluation

Inference packaging does not certify historical checkpoint selection or unbiased evaluation. Current validation-based training code describes new runs; review provenance and obtain independent evaluation before making new comparative research claims.

```sh
TAG=01_qwen3_235b_to_1p7b_sft_arc_c
platour-exps evaluate --experiment 1 \
  --sender <public-sender> --receiver <public-receiver> \
  --test-data "data/qwen3/$TAG/test.jsonl" \
  --output <new-evaluation-dir>

platour-exps evaluate --experiment 19 \
  --weights zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --output <new-evaluation-dir>

# Test the validation-best checkpoint of your own training run instead of the release:
platour-exps evaluate --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --run <training-dir> --output <new-evaluation-dir>
```

- Without `--weights`/`--run`, evaluation downloads and verifies the released checkpoint. `--run <training-dir>` instead loads the step recorded in that run's TEST-blind `selection.json`, the `best_step` of a native Qwen3.5 run, or—for a native Qwen3 SFT run—the final signed training checkpoint. The summary records `checkpoint_source`, the selected step, the selection policy and the checkpoint SHA-256; new SFT runs always use the final checkpoint; GRPO uses validation to select it. Existing selection records retain their historical policy. This describes new runs only and does not certify selection of historical releases.
- Inference artifact descriptions and full-denominator test protocols are in [`models/README.md`](models/README.md).
- `evaluate`, portable `train`, and `pact-check` accept `--n-gpu N` (alias `--n_gpu`) on the first N visible GPUs. Default GPUs per replica: Qwen3-235B-FP8 IDs 01–06/13–15 use 2; Qwen3.5-397B-FP8 IDs 20/25 use 4; all others use 1. PACT infers the size from its experiment/sender name, with SC/TR-only defaulting to 1; renamed large-model directories need an explicit override. Eight GPUs therefore run 4, 2 or 8 replicas. `--gpus-per-replica` overrides the size; N must be divisible by it and each replica must fit. `--n-gpu 1` supports 8B→1.7B SFT/GRPO/PACT and the other one-GPU defaults. Omitting the flag retains existing placement/torchrun behavior. Use `CUDA_VISIBLE_DEVICES` for device selection; leave Qwen3.5 `--device=cuda:0` and do not combine `--n-gpu` with `torchrun`. See [training/PACT usage](README.md#single-gpu-and-eight-gpu-training--pact-check) and [evaluation defaults](README.md#gpu-defaults-for-standalone-evaluation).
- Qwen3.5 standalone evaluation accepts `--eval-suites aime` (or other task suites); partial suites report no three-suite macro. `--response-cap 16384` explicitly shortens the published DAPO 32K test cap and records it in the summary. Do not describe a shortened run as the unchanged published protocol. Sampling counts and active-request limits are unchanged.
- Full IDs 23/29 use standard Transformers weights; LoRA IDs 24/30 use standard PEFT adapters. The download/evaluation loader verifies their pinned configuration and weight hashes and keeps generic filenames in separate experiment caches. It also accepts the older experiment-tagged local files. Standard PEFT names are mapped back to the project actor schema; vLLM conversion retains its required fused-module mapping. Full loading preserves independently stored input embeddings and output head.
- `--weights` accepts an HF repository ID or a local checkpoint file. Hub inputs verify the configuration and weight against the selected experiment's pinned SHA-256 identities; `--experiment` still selects the data and scoring protocol. Evaluation also accepts HF IDs for `--sender` and `--receiver`: it downloads backbone weights, configuration and tokenizer into the normal HF cache, then gives both HF and vLLM loaders the local snapshot. Local model directories remain supported. Private repositories require an authenticated HF account with access. Training commands still take local backbone directories.
- Coding experiments (IDs 25–30) also need the coding judge; see [Coding judge](#coding-judge-ids-2530).

## Training

```sh
platour-exps train --experiment 1 \
  --sender <public-sender> --receiver <public-receiver> \
  --train-data "data/qwen3/$TAG/train.jsonl" \
  --validation-data "data/qwen3/$TAG/validation.jsonl" \
  --test-data "data/qwen3/$TAG/test.jsonl" \
  --output <new-training-dir>

platour-exps train --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --output <new-training-dir>
```

The commands above use single-process automatic placement. Add `--n-gpu 8` for
one-host data parallelism, or `--n-gpu 1` for single-GPU 8B→1.7B. Alternatively,
use explicit `torchrun` for one-GPU-per-rank models as below. Grouped portable
training splits larger senders within each group; each replica takes a deterministic
slice of the global batch and sums gradients before the shared update. Global
batch, learning rate and sampling seeds stay fixed:

```sh
# One machine, eight GPUs
torchrun --nnodes=1 --nproc-per-node=8 -m platour_exps train --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --vllm-python <vllm-env>/bin/python --output <new-training-dir>

# N machines x eight GPUs: run on every node with its own --node-rank (0..N-1)
torchrun --nnodes=<N> --node-rank=<i> --nproc-per-node=8 \
  --master-addr=<node-0-address> --master-port=29500 \
  -m platour_exps train --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --vllm-python <vllm-env>/bin/python --output <shared-training-dir>
```

- The output root must be on storage shared by all nodes. Rank 0 writes signed checkpoints containing every rank's RNG. All ranks now generate their validation/TEST shards with vLLM and aggregate counts, preserving question IDs and sampling seeds. PACT DT calibration, DT/IT inference, and SC/TR training/validation/test are collective too. Shared metadata has one writer. New multi-rank protocols sign replica count and group size; changing topology or resuming an older unsigned-topology run requires a fresh root. Portable GRPO still recreates vLLM per update; active-request limits are unchanged.
- **API validation cadence.** `validation_schedule="auto"` defaults to epoch-end SFT validation, half-epoch Qwen3 GRPO validation, and recipe-based Qwen3.5 GRPO validation. DAPO uses every 33 optimizer updates; DeepCoder uses every 42, with final-step validation and the published 294/295 exception (only IDs 26/28 include 294 at the 295-step budget). Custom budgets retain the task interval and always include the final step. All include initial validation at step zero. PACT SC/TR use these defaults. Explicit `validation_schedule="step"` is available for debugging; Qwen3 initial-only GRPO smoke remains supported. Explicit `validation_schedule="final"` is available only for GRPO smoke runs in both families: it validates the terminal step once, with no step-zero or intermediate measurement, and selects that sole candidate. Schedules are part of the signed protocol and cannot change during resume.
- **Historical SFT final TEST.** `--test-data` is optional; when supplied, validate the generation runtime before loading models and run collective vLLM TEST at the final signed checkpoint. TEST data/settings are bound to `protocol.json`. A TEST failure leaves the run incomplete; resume from the final signed training step with identical arguments to retry without additional optimizer updates. Completed TEST summaries are reused. Historical SFT still requires four ranks/eight GPUs; CPU contract tests do not qualify that GPU topology.
- **SFT optimizer defaults.** Qwen3 and Qwen3.5 bridge/SC/Virtual Sender/LoRA SFT use peak LR `1e-4`; full-parameter SFT uses `1e-5`. AdamW uses betas `(0.9, 0.999)` and zero weight decay. All SFT paths use linear warmup then linear decay to zero; the first update uses LR zero. APIs default to `ceil(0.1 * total_steps)` warmup steps and accept explicit `learning_rate` / `warmup_steps` overrides. Qwen3 formal recipes keep their pinned warmup counts, which PACT SC/TR reuse for matching task budgets. The schedule is signed in new SFT protocols; old constant-LR runs cannot resume under the new schedule (use a fresh output root). GRPO uses constant LR `1e-6` for full-parameter training and `1e-5` for other methods, without warmup.
- **SFT uses the final checkpoint; GRPO uses validation-best.** Qwen3 SFT records held-out NLL at each epoch for monitoring; with `--test-data`, the signed test is scored once at the final training checkpoint, regardless of VAL loss. Qwen3 GRPO reads the corrected published validation split (legacy data pins retain their seed-42 runtime holdout), selects the highest validation accuracy (earliest step on ties) and evaluates the signed test once. Qwen3.5 GRPO trains one complete epoch (no early stopping) and then selects among step 0 and every validation point the checkpoint with the highest validation accuracy (earliest step on ties) without a clipping-rate eligibility guard; it writes `selection.json` (all candidates included) and tests the selected checkpoint once in `<output>/test/`; `--no-final-test` skips that test. Every selection records `test_used_for_selection: false`.
- **Qwen3 portable SFT memory.** GSM8K uses micro-batch at most 4 with effective batch 64; MATH remains 1. Portable SFT accepts an explicit `--micro-batch N` without changing effective batch or epochs. Chemistry-CoT retains its recipe default of 8; Using micro-batch 2 passed the full single-GPU run after 8 exceeded memory. The override is rejected for GRPO and historical backends. Historical native recipes are unchanged. Before final TEST, clear gradients and move optimizer tensors to CPU; restore the signed terminal state after success or failure. The SFT protocol signs micro-batch size, so older unsigned layouts require a fresh training root (standalone checkpoint evaluation remains available). Long-sequence GPU safety still requires an actual run.
- **Checkpoints and resume.** Qwen3.5 signs model/optimizer/RNG state after every update, before validation, and keeps the two newest steps (`n`, `n+1`) together with step 0 and every validation checkpoint. Rerun the same command on the same output root **with the same number of ranks** to resume from the latest verified step and retry an interrupted validation; a finished run with its final test missing runs only that test. Qwen3 runs resume with `--resume <output>/checkpoints/step-XXXX`. A resume must point to the same run and its newest complete signed checkpoint; never resume from inference-only `.safetensors`.
- Qwen3 portable protocol v3 binds the output root, full recipe, TRAIN/VAL/TEST and frozen backbone file hashes to each checkpoint. Rank 0 hashes local backbone files once at startup, including on resume. Older protocols cannot resume in place; preserve those runs and use a fresh output root. Standalone evaluation remains available for verified old checkpoints: optimizer checkpoints require their complete marker and weight hash, native GRPO evaluation snapshots require their checkpoint metadata and completed validation summary, and final-SFT evaluation requires the recorded terminal budget to be reached.
- Every new bridge projection is exact FP32 zero for every Qwen3/Qwen3.5 training recipe, including the native runners (IDs 13 and 19 run SHA-verified zero-initialized private copies of their bundled sources). A signed resume restores its actual bridge weights, optimizer and RNG—**never zero again**. The Virtual Sender's virtual states are not projection matrices and keep their own initialization.
- The portable runner and the `historical-*` opt-in native runners are **not** numerically interchangeable. In Qwen3 SFT, held-out NLL is HF; GRPO response validation uses vLLM by default. Native backends have their own checkpoint and topology contracts: [`qwen3/README.md`](qwen3/README.md#native-eight-gpu-backends), [`qwen35/README.md`](qwen35/README.md#native-backends).

## GPUs: one card, one machine, several machines

- Portable training, released-weight evaluation and `PLatourSession.load(...)` use automatic placement over the reader's visible GPUs by default; do **not** require a single-GPU flag or force a multi-GPU reader onto one card. `device_map='auto'` may use a subset when the model fits; it is not a promise of multi-GPU data parallelism.
- Combinations of 8B and smaller (e.g. Qwen3-8B→1.7B) also run on one visible GPU, which then hosts both HF backbones and the vLLM decoder. Check free GPU memory first: a single-card decoder/optimizer can still run out of memory, and that is not an accuracy score.
- Large senders use grouped portable replicas with `--n-gpu`, or the opt-in native backends. Native backends keep their original topology: historical SFT uses four ranks/eight GPUs, and native IDs 20/25 require two eight-GPU hosts through `--train-backend native-multinode`. Grouped portable training is a separate implementation, not certification of a native trajectory. GPU capacity and actual backward execution must be verified before claiming a large-model training run works.

## Independent sender/bridge/receiver API

The experiment-number CLI fixes the published datasets and parameters. `platour_api` instead accepts the caller's fully rendered tokenized data and model paths; it does not pick an experiment or a released checkpoint on its own:

```python
from platour_api import PLatourSession, load_dataset

system = PLatourSession.load("<Qwen3-8B>", "<Qwen3-1.7B>",
                            bridge="<compatible-projection.safetensors>",
                            mapping="last_aligned")  # automatic placement on visible GPUs
answer = system.generate(messages=[{"role": "user", "content": "Compute 2+2."}],
                         backend="vllm", output="<new-private-output>", max_new_tokens=32,
                         vllm_python="<vllm-env>/bin/python")
# Token IDs can also be supplied with generate(prompt_ids=[...], backend="vllm", ...).
train = load_dataset("<independent-train.jsonl>")
validation = load_dataset("<independent-validation.jsonl>")
receipt = system.fit(train, validation, objective="sft", task="gsm8k",
                     output="<new-private-training-root>", epochs=3,
                     validation_backend="hf")
# Repeat with the *same* inputs, model identity and output root after an interruption.
```

- `platour_api.PLatourSession`: Qwen3 sender/bridge/receiver; `load`, `load_dataset`, `generate(prompt_ids=... | messages=...)`, `evaluate` and `fit`. A trained result is not a new paper checkpoint. `fit` measures teacher-forced held-out NLL for SFT and vLLM accuracy for GRPO, uses three epochs and restores the final SFT or validation-best GRPO checkpoint, and resumes from the newest verified same-run signed step. All SFT validation requires supervised response IDs and uses HF NLL, including when `task` is supplied; answer-generation validation is rejected for SFT. GRPO also accepts an inspectable fixed `reward_fn`; use training-only inputs and do not tune it on test data. Checkpoint files from `fit` are private restart state; use `system.save_bridge(...)` for a Qwen3 projection-only inference file.
- One loader for both families: `from platour_api import load_session, load_dataset`; choose `family="qwen3"`, `method="sft" | "grpo" | "self" | "full" | "lora"`, or `family="qwen35"`, `task="dapo" | "deepcoder"`, `method="bridge" | "self" | "random" | "full" | "lora"`. SFT/GRPO select the training objective and its layer mapping; trained-receiver controls have no external sender. Example: `load_session(family="qwen35", sender="<public-35b>", bridge="<compatible-safetensors>", receiver="<public-4b>", task="dapo", method="bridge")`; `generate(prompt_ids=[...])` or `generate(messages=[...])` then uses native vLLM by default.
- `platour_api.Qwen35Session.load(sender=..., bridge=..., receiver=..., task="dapo" | "deepcoder", method="bridge" | "random" | "full" | "lora", template=...)`: Qwen3.5 text backbones and inference weights, plus TRAIN-only virtual-sender statistics for a fresh `random` run. It provides `generate(prompt_ids=... | messages=..., backend="vllm")`, `evaluate(rows, ...)` and `fit(train, validation, objective="grpo" | "sft", output=<private-root>, ...)`; published Qwen3.5 experiments are GRPO only; training defaults to one epoch, batch 64, a 16,384-token TRAIN/VAL cap and vLLM accuracy (DAPO avg@4, DeepCoder avg@16), and the final SFT or best VAL GRPO checkpoint is restored before TEST. Rolling model/optimizer/RNG checkpoints resume interrupted training from the last signed step. The receiver is the 4B architecture and real senders use the audited 32/40/60-layer mappings in `qwen35.catalog`. The Virtual Sender (`method="random"`) needs prompts rendered with the task's fixed template, so free-form `messages` without that template are rejected. DeepCoder also requires the isolated code judge and per-question case hashes. Do not treat this API as generic 397B distributed native training.
- `prompt_ids` and `messages` are mutually exclusive. The Qwen3 bridge API and the Qwen3.5 text-only 4B receiver API require model internals compatible with their projection/cache contracts.
- Self-Communication (SC): load a separate bridge with the receiver as both sender and receiver, then call `fit(..., control="SC", objective="grpo" | "sft")` on TRAIN/VAL only.
- Trained Receiver (TR): `platour_api.ReceiverSession.load("<receiver>", method="full" | "lora")` trains the receiver alone with **GRPO or SFT** and same-root signed model/optimizer/RNG checkpoints. Qwen3 SC/TR save at scheduled VAL/final steps by default; reference-only step zero in epoch SFT and initial-only GRPO smoke has no checkpoint. `checkpoint_interval` adds recovery points, and the two newest saved states plus best (and selection-eligible initial) are retained. SFT TR supports real padded `micro_batch` accumulation with whole-batch response-token normalization. Its native vLLM path exports the current full or merged LoRA weights into a SHA-verified private snapshot, so all selected PACT-Check controls use the same decoder backend and sampling; HF answer generation is rejected. Qwen3 TR GRPO uses this vLLM path for rollouts too. TR defaults to three epochs and returns the final SFT or validation-best GRPO model; its API accepts an explicit `max_steps` only with `smoke=True` for bounded checks. This export and its structural tests do not prove numerical equivalence with the historical native trainer.
- `dummy_sender=True` loads only the sender's configuration plus the trained bridge for DT inference from precomputed TRAIN-only means (non-FP8 Qwen3 architectures).

## PACT-Check

PACT-Check compares the PLatour system against four controls on the same test denominator: two **inference-only** sender interventions—**DT** (Dummy Transfer: fixed TRAIN-averaged sender features at template positions) and **IT** (Irrelevant Transfer: real sender features of an exact-length unrelated TRAIN prompt)—and two **trained** controls—**SC** (Self-Communication: a separately trained bridge with the receiver as its own sender) and **TR** (Trained Receiver: the receiver alone, full or LoRA). Select any nonempty subset with `--controls DT IT SC TR` (default: all four). The entry does not evaluate a PACT baseline or calculate PACT-Score; it writes per-control results to `pact_check.json`.

Qwen3 defaults to the pinned `archived-it-v1` full suite for 8B→1.7B experiments
07–12 and 16–18. It restores HF IT pairings/supplements and published training
inputs, aligns the ARC/MMLU instruction suffix, and reads the published TRAIN/VAL
without runtime splitting. Corrected GRPO data provides Chemistry VAL 210 and
GSM8K/MATH VAL 500, materialized once from TRAIN with seed 42. Every setting uses the complete archived TEST rows and fixed
sampling. Duplicate MMLU test prompts retain their indices and historical effective
donors; DT calibrates on distinct TRAIN prompts. See [the Qwen3 guide](qwen3/README.md#frozen-it-evaluation-0712-1618).

```sh
platour-exps pact-check --experiment 16 --prepare-only
# SC and TR each train for three complete epochs.
platour-exps pact-check --experiment 16 \
  --vllm-python <vllm-env>/bin/python --output <new-private-pact-run>
# IT-only regression remains explicit:
platour-exps pact-check --experiment 16 --controls IT \
  --vllm-python <vllm-env>/bin/python --output <new-private-it-run>

# Custom Qwen3 inputs explicitly opt out of the pinned default:
platour-exps pact-check --it-source custom --sender <public-sender> --bridge <trained-bridge.safetensors> \
  --receiver <public-receiver> --dataset <split-directory> --task gsm8k \
  --prefix-ids '[...]' --suffix-ids '[...]' \
  --semantic-group-field semantic_group --tr-method lora \
  --output <new-private-pact-run>

# Qwen3.5: a Qwen3.5 sender, its bridge and the 4B receiver (dapo or deepcoder)
platour-exps pact-check --family qwen35 --sender <public-35b-sender> \
  --bridge <trained-bridge.safetensors> --receiver <public-4b-receiver> \
  --dataset <split-directory> --task dapo \
  --prefix-ids '[...]' --suffix-ids '[...]' \
  --semantic-group-field semantic_group --tr-method lora \
  --vllm-python <vllm-env>/bin/python --output <new-private-pact-run>
```

- **Custom/Qwen3.5 dataset.** A directory with disjoint `train.jsonl`, `validation.jsonl` and `test.jsonl` (fully rendered `prompt_token_ids`, TRAIN/VAL/TEST `gold`; TRAIN/VAL `response_token_ids` additionally required for `--control-objective sft`). TRAIN and TEST must carry an audited `semantic_group`: differing group labels must exclude duplicates and paraphrases. Audit these groups independently; do not guess them from token inequality.
- **Custom/Qwen3.5 template.** `--prefix-ids` are the fixed tokens every prompt starts with, and `--suffix-ids` the fixed tokens that end every prompt **excluding its final token** (the final prompt token goes only to the receiver, so DT replaces sender features up to it). The arrays must match the prompts exactly.
- **DT.** `pact_check.build_dummy_statistics` selects 500 distinct TRAIN prompts with seed 42 (`--dt-samples`) and averages the communicated block-input features separately at prefix/suffix template positions, never on validation/test; DT inference does not call a sender. A smaller sample count is a structural smoke, not a paper-equivalent calibration.
- **IT.** Default Qwen3 uses the pinned complete HF pairings and supplements, checking hashes, exact token lengths and provenance. Custom Qwen3 and legacy Qwen3.5 use `pact_check.select_irrelevant_donors`, which gives each TEST question an unrelated TRAIN donor with the same fully rendered prompt-token length, drawn with seed 42 (with replacement) under an explicit duplicate/paraphrase exclusion callback. If no such donor exists the call fails rather than padding the question or borrowing its answer; as in the paper, rephrase only a donor's wording to the required length, without access to the test problem or its answer (the code never rephrases automatically).
- **SC/TR.** Each is trained separately on TRAIN/VAL in its own signed private root with its own receipt and checkpoints. TR defaults to full-parameter training; select LoRA explicitly with `--tr-method lora`. With `--family qwen35`, SC is a 4B→4B self-communication bridge and TR a 4B full/LoRA receiver; DT/IT replace the sender features in both HF and vLLM prefill. A run restarted with the same arguments and output reuses completed SC/TR training roots.
- **Qwen3 throughput.** SFT SC/TR default to micro-batch 8 for ARC-C and 4 for GSM8K, accumulating to effective batch 64. GSM8K is reduced from the recipe micro-batch to lower long-sequence activation memory. `--micro-batch` overrides it. Full checkpoints default to scheduled VAL/final points (no step-zero checkpoint for SFT or initial-only GRPO smoke); `--checkpoint-every N` adds recovery points. SC/TR-only Qwen3 runs do not load the original sender/bridge. TEST preparation clears gradients and releases idle allocator cache; Qwen3 vLLM startup repeats cache release and writes `memory_before_worker.json` for diagnosis. Each new TR payload is hashed once, with full verification on restore. Unsaved updates are replayed after interruption; before the first checkpoint, restart with a fresh session and repeat initial VAL. Old micro-batch/checkpoint protocols cannot be resumed in place; use a fresh output directory and preserve previous runs.
- **Backend and objective.** The CLI defaults to native vLLM (`--vllm-python <decoder-python>` if separate). Frozen Qwen3 uses SFT for 07–12 and GRPO for 16–18, and trains both SC and TR for three full epochs (`3 * ceil(TRAIN / batch_size)` updates). Qwen3 SFT uses batch 64, measures token-averaged VAL NLL with HF teacher forcing initially and at each epoch boundary, and always tests the final epoch checkpoint, regardless of VAL loss; step zero is reference-only. Qwen3 GRPO selects highest validation accuracy. TEST generation uses vLLM. Qwen3 TR uses vLLM GRPO rollouts. Both families reject `--control-steps`; Qwen3.5 SC/TR train one full epoch and restore the final SFT checkpoint or highest-accuracy GRPO VAL checkpoint (earliest on ties). GSM8K PLatour-GRPO and both GRPO SC/TR controls validate with avg@4; final TEST remains avg@2. Custom/Qwen3.5 retain the GRPO objective default and require vLLM generation. `--control-objective` overrides the control objective. Selected DT/IT/SC/TR controls use one scorer, decoder, sampling protocol and per-question sampling budget on precisely the same TEST questions. `pact_check.json` is written only after all selected controls have complete equal-denominator test results, and records the objective, backend and calibration count. No CLI setting by itself certifies paper-equivalent evaluation.
- **DeepCoder** additionally takes `--code-cases`, `--official-checker`, `--docker-image` (and `--allow-rebuilt-image` for a locally built judge image).
- **Library.** `pact_check.PACTCheck(bridge, train_rows, validation_rows, test_rows, evaluator=...)` accepts an evaluator using the same decoder/sampling/scorer for all settings. Its `run(controls=("DT", "IT", "SC", "TR"), dummy=..., donors=..., train_sc=..., train_tr=..., samples_per_question=...)` executes only selected controls, checking each selected training receipt and testing each trained control once before releasing it. DT statistics and IT donors are required only when the respective check is selected. It returns per-control accuracies and raw results, with no baseline evaluation or aggregate score. Training-control factories must supply independent checkpoints with matching TRAIN **and VAL** digests, a signed training receipt and a test-blind selection rule.

## Coding judge (IDs 25–30)

- Coding evaluation and training need the LiveCodeBench checker, per-question unit tests and an isolated Docker judge **in addition to model weights**. Follow the [judge download, preparation and image instructions](qwen35/README.md#obtain-and-install-the-coding-judge-ids-2530). `--code-cases` takes the prepared root's `cases/` subdirectory.
- `python -m qwen35.judge_cases --fetch-public-lcb-v6 --checker <pinned-LCB-checker> --output <fresh-private-root>` derives **all 175 LCB v6 TEST case files** from the revision-pinned public HF `test6.jsonl` (no pickle execution) and rebuilds the 19,074 TRAIN/VAL cases from pinned public DeepCoder sources and checks every original per-question SHA. `--lcb-test6 <already-downloaded-raw>` works offline; `--lcb-cases` remains for existing originals. The checker and **extractor** come from a pinned public LiveCodeBench commit.
- The original `lcb-checker:local` image has no public registry address. The bundled `qwen35/lcb_sandbox.Dockerfile` builds a **labelled rebuilt** local image for the direct CLI, not the original image; `Qwen35Session.load(..., allow_rebuilt_image=True)` is required and its SHA becomes part of the run identity. Full native ID 25 and the shared CPU judge demand the original image's exact ID. Only that image is eligible for the optional `python -m qwen35.judge_service` on another trusted CPU host/shared mount; it pins the judge sources for IDs 25/26 (same source) and 29 (separate source) and fails on a changed heartbeat, image, cases or checker.
- Generated code must **never** be executed on the host. An unavailable Docker service is not permission to fake a score or substitute a different checker/image. A read-only preflight does not train a model.

## Verification status

- Portable training and evaluation have been exercised with one GPU, two and four ranks on one host, and two hosts × eight GPUs (16 ranks over a shared output root), including whole-job kill/resume and the rejection of a resume with a different rank count.
- Qwen3 SFT ran its full 54-update ARC-C recipe (four ranks reproduce the reference training run bit for bit). Qwen3 GRPO and Qwen3.5 training were exercised with short budgets (a few updates, sampled validation/test). Full-budget portable GRPO/Qwen3.5 runs have not been verified, so do not report them as such.
- On two eight-GPU hosts, ID 20 has passed the two-host probe, the four-rank 16K smoke, formal step-0 validation (500 × 4) and five signed updates with whole-job kills, `--resume-latest` and rolling checkpoint retention. The full 264-update run and any two-host ID 25 run have not been executed, so do not report them.
- Fresh native 23/24 full 264-step sources and `n`/`n+1` signed optimizer/RNG resume are available; prior GPU evidence is smoke-only, not formal numerical parity. Only 26/29 have the **portable, smoke-only** two-step TRAIN + small VAL/three-suite TEST coding flow; its outputs cannot be relabeled as full 295-update native training or 6,712-answer TEST.

## Workflow checklist

1. Install as in [Setup](#setup) and pin the model-family versions documented by each recipe.
2. Before launching, verify the data/weight manifests, model IDs, split roles, Docker judge provenance (for coding), device topology and free memory.
3. Use a new private output directory per run. An interrupted validation is retried from its signed step without repeating a signed update.
4. Inspect `metrics.jsonl`, complete checkpoint markers, the validation denominator and summary, rank failures and independent TEST coverage.
5. After code edits, run `python -m unittest discover -s tests -p 'test_*.py' -q` and `git diff --check`.

## License

The repository root `LICENSE` is Apache-2.0 for original project code. Bundled third-party sources retain their own license notices. Model and dataset usage is governed separately by upstream terms; do not infer that Apache-2.0 relicenses those assets.
