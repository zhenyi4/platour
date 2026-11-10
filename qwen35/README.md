# Qwen3.5: mathematics, coding, and adaptation controls

These twelve Qwen3.5-4B receiver experiments train separately on mathematical reasoning (IDs 19–24) and programming (IDs 25–30). `qwen35.catalog` maps each ID to its task, optimization protocol, evaluation schedule, and sender–receiver layer correspondence.

| Paper role | Mathematics | Coding | Trainable state |
|---|---:|---:|---|
| PLatour, real 35B-A3B sender | 19 | 26 | 32 FP32 bridge projections |
| PLatour, real 397B-A17B-FP8 sender | 20 | 25 | 32 FP32 bridge projections |
| Self-Communication, independent frozen 4B sender | 21 | 27 | 32 FP32 bridge projections |
| Virtual Sender, train-only template statistics | 22 | 28 | virtual features and 32 projections |
| Trained Receiver, full-parameter GRPO | 23 | 29 | receiver model |
| Trained Receiver, rank-128 LoRA-GRPO | 24 | 30 | receiver adapters |

The real-sender and Self-Communication experiments freeze both model copies during bridge optimization. Full-GRPO and LoRA-GRPO train the receiver instead. These roles separate communication from adaptation in the paper’s PACT-Check analysis.

## Complete the GPU training environment

After the [base installation](../README.md#getting-started), use the training
virtualenv's Python to install the supplemental Qwen3.5 dependencies:

```sh
python scripts/install_qwen35_extras.py --dry-run
python scripts/install_qwen35_extras.py
```

The supplemental [requirements](../requirements/qwen35-extras.txt) include
`flash-linear-attention`, PEFT, datasets, pandas, W&B and build tools. Existing
Torch, Transformers, FLA core, TileLang, kernel-loader and CUDA-library versions
are constrained during installation. The installer also refreshes the editable
package registration so newly added project modules can be imported.

For Linux x86_64, Python 3.11 and Torch 2.10 with CUDA 12/CXX11 ABI, it installs
the matching `causal-conv1d==1.6.2.post1` wheel from the
[official release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.2.post1),
verifying the published SHA-256. This avoids a local CUDA compilation.
Other ABI combinations must use a matching wheel or
a compatible source-build toolchain.

Finish jobs using the training environment before installing. Version snapshots
and the completion receipt are written to `outputs/qwen35-env-install-*`.
Successful installation includes import checks and a small CUDA convolution
forward/backward check. `--imports-only` skips the GPU check; that receipt does
not certify CUDA execution. Repeat verification with `--check-only`.
The separate vLLM environment and its version choices are not changed.

The full/LoRA path in vLLM 0.17.1 can JIT-compile
FlashInfer GDN prefill kernels. An old system CUDA Toolkit can fail here even
when PyTorch reports CUDA 12.8 and the HF convolution extension works. Errors
such as missing `cuda::ptx::tensormap_replace_global_dim` or
`fence_proxy_tensormap_generic` indicate incompatible compiler/CCCL headers.
Install a private compiler and headers, then export them before launching:

```sh
python scripts/install_cuda_toolchain.py
source outputs/cuda-12.8/env.sh
```

This downloads SHA-verified NVIDIA CUDA 12.8.1 NVCC, cudart and CCCL components,
leaves the system Toolkit and driver unchanged, and compiles a probe for those
PTX interfaces. Its activation script sets a toolchain-specific
`FLASHINFER_WORKSPACE_BASE`, because FlashInfer 0.6.4 otherwise reuses an existing
`build.ninja` with the old compiler/header paths even after `CUDA_HOME` changes.
Rerunning the installer refreshes this activation script for existing installs.
It does not certify GPU execution; rerun the affected
vLLM prefill/evaluation after installation. Preserve any driver compatibility
library settings already required by your working GPU environment.

## Data and metrics

`data/qwen35/dapo/` contains 16,891 DAPO-17k training questions, 500 fixed validation questions, and frozen AIME-135, HMMT-60 and IMO-AnswerBench v2 test inputs. The AIME-80 mask excludes 55 identified overlaps; AIME-80 is computed from the same 135 × 16 generated answers, not a second sample. HMMT uses 60 × 16 and IMO uses 400 × 4 responses. Math Avg. is the unweighted mean of AIME-80, HMMT and IMO.

`data/qwen35/deepcoder/` contains 18,873 decontaminated training problems, 201 validation problems and the LCB v6 (175), HumanEval (164) and MBPP (500) test questions. Validation uses 16 responses per question, and tests use eight. Coding Avg. is the unweighted mean of all three full-denominator accuracies. The Virtual Sender’s HumanEval/MBPP inputs are separately named `*_template_adapted.jsonl`; their reported combined average includes those adapted prompts and the LCB prompts.

The public data retain prompt token IDs, gold labels and grading contracts. The 19,074 referenced training/validation programming test cases are rebuilt from a fixed public DeepCoder revision and checked against their original per-question SHA. `python -m qwen35.data --output <new-train-validation-case-directory>` downloads the pinned upstream TRAIN parquet files and reconstructs **only** these cases; for training and all three coding test suites, use the combined case preparation below. Data provenance and format details are in [`data/README.md`](../data/README.md).

## Evaluate

Supply the public Qwen3.5-4B receiver and the corresponding public sender for a PLatour/SC bridge as local directories or HF repository IDs; evaluation downloads HF backbones to its normal Hub cache. The Virtual Sender and receiver-training controls do not use an external sender. The per-experiment model repositories and byte-level identities are listed in [`models/manifest.json`](../models/manifest.json), with paper metrics in [`models/README.md`](../models/README.md).

```sh
platour-exps evaluate --experiment 19 \
  --weights zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --output <new-evaluation-dir>
```

`--weights` accepts an HF repository ID or a local checkpoint file. Omit it to use the repository assigned to the experiment in the manifest. Hub inputs verify both configuration and weight SHA-256 against that experiment; a different experiment's repository is rejected. Private repositories require an authenticated HF account with access. HF backbone IDs download configuration, safetensors and tokenizer files before local loading in both generation backends. To use a particular backbone revision, download that revision separately and pass its local snapshot directory. Training and initialization commands still require local backbone directories.

Evaluation uses native vLLM 0.17.1 by default: real and virtual bridges import the HF prompt's hybrid GDN/KV memory and first sampled token; full receiver and LoRA controls use native vLLM prefill. Use `--vllm-python <vllm-env>/bin/python` for a separate generation runtime, and put `<vllm-env>/bin` on `PATH` so vLLM can invoke its matching `ninja` executable for CUDA kernels. HF answer generation is not available through the public CLI/API. `--run <training-dir>` evaluates the validation-selected checkpoint of a training run instead of the released weights. Programming evaluation additionally takes `--code-cases`, `--official-checker` and `--docker-image`; generated code runs only inside the isolated checker. DeepCoder training also stops *before loading any model* unless the training cases, pinned checker and isolated Docker daemon/image pass preflight. Mathematics uses the final-answer equivalence checker.

### Standalone evaluation GPU defaults

Standalone `evaluate` accepts the **optional** `--n-gpu N` (alias `--n_gpu N`),
unset by default. N is the total GPU count on this host; with G GPUs per replica,
the launcher starts N / G independent replicas. Without the flag, evaluation
retains one process with automatic sender placement over visible GPUs; it
neither forces single-GPU placement nor starts data-parallel replicas.
The default `--gpus-per-replica` is **4 for 397B-FP8** (20/25) and **1 for all
other Qwen3.5 experiments**, including 35B (19/26) and the controls. Therefore
`--n-gpu 8` starts two or eight replicas respectively. `--n-gpu 1` supports the
one-GPU configurations; the 397B default requires at least four GPUs.

```sh
# AIME-135 and its Clean80 subset: eight one-GPU replicas, published 32K cap.
platour-exps evaluate --experiment 19 \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --n-gpu 8 --eval-suites aime --vllm-python /path/to/vllm/bin/python \
  --output ./outputs/19-aime-8gpu

# The 397B-FP8 sender: two four-GPU replicas.
platour-exps evaluate --experiment 20 \
  --sender Qwen/Qwen3.5-397B-A17B-FP8 --receiver Qwen/Qwen3.5-4B \
  --n-gpu 8 --eval-suites aime --vllm-python /path/to/vllm/bin/python \
  --output ./outputs/20-aime-8gpu
```

Use `CUDA_VISIBLE_DEVICES` to select physical GPUs and leave `--device=cuda:0`.
Each replica places its receiver and TP=1 vLLM decoder on its first GPU. The
397B sender's 60 layers are split 12/16/16/16 within each four-GPU group.
These are fixed experiment defaults, not GPU-memory detection. Override with
`--gpus-per-replica G` together with `--n-gpu N` only when the workload fits.
Both counts must be positive, N must be divisible by G and cannot exceed the
visible GPU count. Counts other than 1 and 8 follow the same rules. Do not combine
this launcher with `torchrun`.

Omit `--eval-suites` for all three task suites. A subset reports
`three_suite_macro: null`. DAPO retains its published 32,768-token response cap;
`--response-cap 16384` explicitly shortens it and records the change in
`summary.json`. Samples per question and active-request limits remain unchanged.
AIME Clean80 uses the same generated answers, aggregated across shards.
Suite/cap overrides apply to standalone evaluation. `train` and `pact-check`
also accept `--n-gpu`, and their validation/final tests run collectively.
The two-host native **training** requirements below do not apply to this
one-host evaluation launcher. See the
[shared evaluation guide](../README.md#gpu-defaults-for-standalone-evaluation)
for output files, constraints and timing semantics.

## Train and resume

```sh
platour-exps train --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --output <new-training-dir>

# Data parallel: eight GPUs on one machine (use --nnodes/--node-rank/--master-addr for N machines)
torchrun --nnodes=1 --nproc-per-node=8 -m platour_exps train --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --vllm-python <vllm-env>/bin/python --output <new-training-dir>

# Later: test the validation-best checkpoint of that run again in a new directory
platour-exps evaluate --experiment 19 \
  --sender <public-35b-sender> --receiver <public-4b-receiver> \
  --run <new-training-dir> --output <new-evaluation-dir>
```

Portable training and PACT accept the same optional `--n-gpu N` flag. Omitting
it preserves the existing launch behavior: a direct invocation places the
receiver on `--device` (default `cuda:0`) and automatically places the sender
over visible GPUs; explicit `torchrun` retains its existing topology. Visible
GPU count alone does not enable replicas. For portable training, `--n-gpu 8`
starts eight one-GPU 35B replicas, or two four-GPU 397B-FP8 replicas. Each group partitions
sender layers and places the receiver on its first GPU. `--n-gpu 1` supports
the smaller configurations if they fit. Plain `torchrun` still uses one complete
replica per local GPU, and rejects a 397B sender; use grouped `--n-gpu` for that
portable model. Native backends retain their fixed topologies, including the
two-host contract for frozen native 397B training; `--n-gpu` does not relax them.

Training shards each global batch and sums gradients before one clipped AdamW
update. Validation and final TEST now generate on all replicas and aggregate
counts. PACT `--family qwen35` uses the same collective behavior for DT/IT and
SC/TR, including API SFT controls. Only rank zero writes shared metadata; each
rank's RNG is checkpointed. Resume with the same topology. Multi-host torchrun
needs shared output storage. Do not combine torchrun with `--n-gpu`.
Portable GRPO still recreates vLLM per update; active-request limits are unchanged.
Neither numerical identity nor a speedup factor follows from replica count alone.

Formal training always completes one epoch; the step budget is derived from TRAIN and batch size. Step 0 is the validation baseline, and every scheduled validation point is a candidate: a point becomes the best checkpoint when its accuracy strictly exceeds the best so far (ties keep the earlier step). After the last update, `selection.json` records the validation-best step and every candidate (`test_used_for_selection: false`), and that checkpoint is tested once on the full three-suite benchmark into `<output>/test/summary.json`; `--no-final-test` skips this. The selection is recomputed from the signed validation files, so a resumed run selects the same checkpoint.

Each completed update writes a signed model/bridge/adapter tensor state plus optimizer and RNG state **before validation**. Once a successor passes its own signature check, the non-validation checkpoint two steps behind is retired, so the newest signed `n`/`n+1` pair is kept together with step 0 and every validation checkpoint. Restarting the same output root resumes the most recent complete update and reruns any missing validation before the next update; a finished run with its final test missing runs only that test. The public `.safetensors` files contain inference tensors only.

Every new bridge projection `P` starts at exact FP32 zero; a resume restores the signed state instead. For IDs 22 and 28, first run `platour-exps init-random` with the experiment ID, receiver and a new output path; it derives the virtual features from 500 **training-only** prompts. Pass that output with `--template-init` to the trainer.

### Obtain and install the coding judge (IDs 25–30)

The Hugging Face model checkpoints do not contain judge assets. The 19,074 TRAIN/VAL cases are reconstructed from `agentica-org/DeepCoder-Preview-Dataset` revision `177913a7bd43791646ef6a43645caa3c871ab3db`; the release retains the exact selected question keys, test counts and per-case SHA values in its frozen train/validation JSONL. No case archives need to be downloaded from this repository. The 175 LCB v6 TEST JSON files are derived from [`livecodebench/code_generation_lite` `test6.jsonl` at revision `0fe84c3…`](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl) (the v6 increment to v5), SHA-256 `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`. That source, the [LiveCodeBench code revision](https://github.com/LiveCodeBench/LiveCodeBench/tree/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24) and all 175 derived JSON SHA-256 values are pinned in `qwen35/lcb_v6_cases_sha256.json`. The preparation tool parses the public/private tests without running untrusted pickle operations or contestant code. HumanEval and MBPP grading specs are already in the signed test rows. Respect the upstream dataset's terms.

The judge image is built from `qwen35/lcb_sandbox.Dockerfile` (a digest-pinned `python:3.10-slim` base and `numpy==1.26.4`). The direct `train`/`evaluate` commands accept any image built from it; training binds the image ID into the run identity, so a resume cannot switch judges. The independent API requires `allow_rebuilt_image=True` for an image ID other than `qwen35.judge_service.ORIGINAL_IMAGE`. The shared judge service and the native ID25 trainer use exactly the `ORIGINAL_IMAGE` ID, distributed by the maintainers as a `docker save` archive.

1. Install Docker with a working local daemon on the machine that will execute the **isolated** judge. Run `pip install -e .` from the repository root and retrieve the signed prompts with `platour-exps download-data --experiment 26` (the reconstruction and evaluation commands also download missing prompts automatically). The model and Docker can run on the same machine; a second host is not required for evaluation.
2. Check out the pinned checker/extractor from the [LiveCodeBench repository](https://github.com/LiveCodeBench/LiveCodeBench). The commit and the SHA of **both** `testing_util.py` and the host-side `extraction_utils.py` are pinned in `qwen35/lcb_v6_cases_sha256.json`. From the repository root:

   ```sh
   export LCB_REPO=/absolute/path/to/fresh-LiveCodeBench-checkout
   git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_REPO"
   git -C "$LCB_REPO" checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
   export LCB_CHECKER="$LCB_REPO/lcb_runner/evaluation"
   export PYTHONPATH="$LCB_REPO${PYTHONPATH:+:$PYTHONPATH}"
   python - <<'PY'
   import json
   import os
   from pathlib import Path
   from qwen35.checkpoints import sha256
   pinned = json.loads(Path('qwen35/lcb_v6_cases_sha256.json').read_text())
   checker = Path(os.environ['LCB_CHECKER'])
   assert sha256(checker / 'testing_util.py') == pinned['checker_sha256']
   assert sha256(checker.parent / 'utils/extraction_utils.py') == pinned['extractor_sha256']
   assert len(pinned['case_sha256']) == 175
   from lcb_runner.utils.extraction_utils import extract_code
   print('Official checker and host-side code extractor available')
   PY
   ```

3. Prepare the cases. `--fetch-public-lcb-v6` downloads the fixed-revision `test6.jsonl` from Hugging Face; for offline use, download [this revision's `test6.jsonl`](https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl) beforehand and pass `--lcb-test6 /path/to/test6.jsonl` instead. The tool derives the 175 per-question JSON files with a restricted JSON-string-pickle parser (**no arbitrary pickle execution**), checks **every case SHA**, then downloads the pinned DeepCoder TRAIN parquet files and reconstructs all 19,074 TRAIN/VAL cases with per-file and per-question SHA checks. The output root must not already exist; allow approximately 5.42 GB for the cached upstream files and 8.00 GB for reconstructed cases. For offline preparation, use `--lcb-test6 <fixed-test6.jsonl> --deepcoder-source-root <DeepCoder-snapshot-root>`; `--cache-dir` selects a shared download cache, and `--local-files-only` forbids network downloads. No program is run during preparation.

   ```sh
   export JUDGE_ROOT=/absolute/path/to/new-private-judge-root
   python -m qwen35.judge_cases --data-root data/qwen35 \
     --fetch-public-lcb-v6 --checker "$LCB_CHECKER" --output "$JUDGE_ROOT"
   python - <<'PY'
   import json
   import os
   from pathlib import Path
   proof = json.loads((Path(os.environ['JUDGE_ROOT']) / 'complete.json').read_text())
   assert proof['complete'] and proof['per_question_sha256_verified']
   assert (proof['training_validation_cases'], proof['lcb_v6_cases']) == (19074, 175)
   print('TRAIN/VAL and LCB v6 case bytes verified')
   PY
   ```

4. Prepare the judge image on your own Docker daemon (`docker info` must succeed).

   - **Build locally:**

     ```sh
     docker info >/dev/null
     docker build --platform linux/amd64 -f qwen35/lcb_sandbox.Dockerfile -t platour-lcb-rebuilt:local .
     export JUDGE_IMAGE=platour-lcb-rebuilt:local
     docker image inspect --format '{{.Id}}' "$JUDGE_IMAGE"
     docker run --rm --network none --read-only --user 65534:65534 \
       "$JUDGE_IMAGE" python -c 'import sys, numpy; assert sys.version_info[:2] == (3, 10) and numpy.__version__ == "1.26.4"'
     ```

   - **Load the maintainer-provided image** (needed for the shared judge service and native ID25):

     ```sh
     docker info >/dev/null
     docker load -i /absolute/path/to/lcb-checker-image.tar
     ORIGINAL_IMAGE_ID=$(python -c 'from qwen35.judge_service import ORIGINAL_IMAGE; print(ORIGINAL_IMAGE)')
     docker image inspect "$ORIGINAL_IMAGE_ID" >/dev/null
     docker image tag "$ORIGINAL_IMAGE_ID" lcb-checker:local
     export JUDGE_IMAGE=lcb-checker:local
     test "$(docker image inspect --format '{{.Id}}' "$JUDGE_IMAGE")" = "$ORIGINAL_IMAGE_ID"
     ```

   The runner later executes generated answers only inside restricted, networkless containers of this image.

5. Point the evaluator **to `$JUDGE_ROOT/cases`, not `$JUDGE_ROOT`**, plus the checker directory and image tag. The command below runs a quick **smoke check** of the released ID26 checkpoint (two questions × two answers per suite, 512-token cap); remove `--smoke` for the full test denominators. Download the sender/receiver from their upstream model repositories; the checkpoint is fetched automatically (`platour-exps download --experiment 26`) or given with `--weights`.

   ```sh
   platour-exps evaluate --experiment 26 \
     --sender /absolute/path/to/Qwen3.5-35B-A3B \
     --receiver /absolute/path/to/Qwen3.5-4B \
     --code-cases "$JUDGE_ROOT/cases" --official-checker "$LCB_CHECKER" \
     --docker-image "$JUDGE_IMAGE" --backend vllm \
     --vllm-python /absolute/path/to/vllm-env/bin/python \
     --smoke --smoke-cap 512 --output /absolute/path/to/new-private-evaluation-root
   ```

Use the same three judge flags for `platour-exps train` on coding experiments and for full coding `evaluate` calls. The judge runs generated programs **only inside** disposable, unprivileged, read-only, network-disabled Docker containers, for both the vLLM and HF backends. A failed SHA/daemon/image check stops the run before any score is written.

**Optional separate Docker host:** if the GPU host has no Docker and both hosts share the same verified files, run `python -m qwen35.judge_service --root <fresh-private-service-root> --cases "$JUDGE_ROOT/cases" --checker "$LCB_CHECKER"` on the Docker host. On GPU clients set `PLATOUR_CODE_JUDGE_SERVICE=<private-service-root>` and `PLATOUR_CODE_JUDGE_EXPERIMENT=25`, `26` or `29`. With one Docker-capable machine, use the direct local mode above.

**Workflow smoke:** `platour-exps train --experiment 26 ... --smoke --backend vllm --vllm-python <generation-python> --output <new-private-root>` (or ID29 without an external sender) uses two TRAIN questions × eight sampled responses for each of two signed optimizer steps, and validates two VAL questions × two responses at steps 0, 1 and 2. `platour-exps evaluate --experiment 26 ... --smoke ...` scores two TEST questions × two in each of LCB/HumanEval/MBPP. Smoke outputs are labelled `smoke_only` and record the generated-token cap (`--smoke-cap`, default 512; full runs use 16,384). Full runs use all 18,873 TRAIN/201 VAL questions and the full test denominators.

## Independent API

For external datasets (rather than a fixed experiment ID), `from platour_api import Qwen35Session, load_dataset` exposes `Qwen35Session.load(sender=..., bridge=..., receiver=..., method="bridge" | "random" | "full" | "lora", task="dapo" | "deepcoder")`, `generate(prompt_ids=... | messages=..., backend="vllm")`, `evaluate(...)` and `fit(train, validation, objective="grpo" | "sft", output=<private-root>, ...)`. It supports the 4B text receiver and the Qwen3.5 sender layer maps in `qwen35.catalog`; `random` needs a TRAIN-only virtual template. DeepCoder additionally needs the SHA-verified per-question tests, the pinned checker **and extractor**, and an isolated Docker image; for a locally built image pass `docker_image="platour-lcb-rebuilt:local", allow_rebuilt_image=True`. Reports and the resume protocol record the actual judge image ID.

## Native backends

### IDs 19, 21–24: one eight-GPU machine

`--train-backend historical-native` runs the bundled native Qwen3.5 GRPO trainers (264 updates, native vLLM rollouts). Install the `native-train` optional dependencies (W&B uploads stay disabled) and supply separate HF training, vLLM 0.17.1 (`--vllm-python`) and MathVerify (`--scoring-python`) runtimes. All use the 16,891 signed training questions, the 500 signed validation questions (500 × 4 per validation point, scored by each native trainer's bundled validation protocol), a fresh run directory and native FP32 optimizer/per-rank RNG checkpoints. Each run is supervised in its own process group; a rank failure stops only that run.

- **ID 19**: eight single-GPU 35B-A3B → 4B PLatour replicas, monotone layer mapping, group-of-eight vLLM GRPO and the final-`Answer:` metric.
- **ID 21**: four paired-GPU Self-Communication sender/receiver copies.
- **ID 22**: eight single-GPU receiver replicas with **no real sender**. `--statistics-root` supplies `dapo/statistics.pt` and `dapo/template.json` derived from the same 500 seed-42 training questions; create it with `init-random --experiment 22 --native-statistics-root <root>`, which also writes the portable initial V. `--template-init` alone is not accepted by the native trainer.
- **ID 23**: eight BF16 full-model actor replicas with eight-way sharded FP32 master AdamW and native vLLM weight synchronization.
- **ID 24**: rank-128/alpha-128 LoRA-GRPO on a frozen 4B receiver with native adapter synchronization.

IDs 19, 21 and 22 resume with `--resume-step <signed-validation-step>`, the same `--output` and signed inputs; source, validation history, bridge SHA, optimizer step and every rank's CPU/CUDA RNG are checked before GPUs start. IDs 23 and 24 sign model-or-adapter, FP32 AdamW and all eight ranks' RNG after every update, keep the latest `n`/`n+1` plus validation checkpoints, and resume with `--resume-latest` (or `--resume-step`).

```sh
# Bounded smokes in fresh output directories
platour-exps train --experiment 19 --sender <public-35b-model> --receiver <public-4b-model> \
  --train-backend historical-native --smoke --vllm-python <vllm-env>/bin/python \
  --scoring-python <math-scoring-env>/bin/python --output <fresh-35b-smoke-dir>

platour-exps train --experiment 21 --sender <public-4b-model> --receiver <public-4b-model> \
  --train-backend historical-native --smoke --vllm-python <vllm-env>/bin/python \
  --scoring-python <math-scoring-env>/bin/python --output <fresh-smoke-dir>

platour-exps init-random --experiment 22 --receiver <public-4b-model> \
  --output <fresh-initial-V.safetensors> --native-statistics-root <fresh-train-only-stats-root>
platour-exps train --experiment 22 --receiver <public-4b-model> \
  --train-backend historical-native --statistics-root <fresh-train-only-stats-root> \
  --smoke --vllm-python <vllm-env>/bin/python \
  --scoring-python <math-scoring-env>/bin/python --output <another-fresh-smoke-dir>

# Full 264-update run and resume (ID 23; use 24 with a different output root for LoRA)
platour-exps train --experiment 23 --receiver /models/Qwen3.5-4B \
  --train-backend historical-native --vllm-python /venvs/vllm/bin/python \
  --scoring-python /venvs/math-verify/bin/python --output /private/new-id23
platour-exps train --experiment 23 --receiver /models/Qwen3.5-4B \
  --train-backend historical-native --vllm-python /venvs/vllm/bin/python \
  --scoring-python /venvs/math-verify/bin/python --resume-latest --output /private/new-id23
```

Never resume from an inference-only checkpoint or a completed smoke.

### IDs 20 and 25: two eight-GPU hosts

The 397B-A17B-FP8 sender runs with teacher TP4 over four distributed ranks on two eight-GPU hosts that share the same absolute output and model paths (264 mathematics updates for ID 20, 295 DeepCoder updates for ID 25). `--native-stage plan` is read-only and prints both host-local torchrun commands. On the first host only, `--native-stage prepare` writes the run's source copy, signed TRAIN/VAL inputs and source/protocol SHA receipts (ID 25 also takes `--judge-service-root` from a running `qwen35.judge_service` with the `ORIGINAL_IMAGE` judge). Then, on **both** hosts with `--node-rank 0` / `1`, the same routable node-0 `--master-addr`, the same shared root and the same attempt label, run `probe` (NCCL and shared-storage check), `smoke` and `train`, each with a new attempt label. Per-step bridge, FP32 AdamW and four-rank RNG are signed; `--resume-latest` with a new attempt label continues from the newest complete step, keeping `n`/`n+1` plus validation checkpoints. Install the `native-train` extra in the training Python (`pip install -e ".[native-train]"` with the pins of the main README). The ranks run offline: the first `probe` on node 0 needs Hugging Face Hub access once to cache the complete pinned FP8 expert kernel in `<RUN_ROOT>/runtime_cache/fp8_kernel`, which later stages and node 1 reuse. Before its first update, ID 25 requires a signed two-host probe and smoke and validates all 201 × 16 held-out responses.

```sh
# Replace these illustrative paths/host name with identical accessible paths on both hosts.
export SENDER=/shared/models/Qwen3.5-397B-A17B-FP8
export RECEIVER=/shared/models/Qwen3.5-4B
export RUN_ROOT=/shared/private/id20-fresh
export MASTER_ADDR=node0.example.internal
export VLLM_PYTHON=/shared/venvs/vllm/bin/python
export SCORE_PYTHON=/shared/venvs/math-verify/bin/python
platour-exps train --experiment 20 --sender "$SENDER" --receiver "$RECEIVER" \
  --train-backend native-multinode --native-stage plan --master-addr "$MASTER_ADDR" \
  --output "$RUN_ROOT"
# Node 0, once only; CPU preparation, NOT training:
platour-exps train --experiment 20 --sender "$SENDER" --receiver "$RECEIVER" \
  --train-backend native-multinode --native-stage prepare --output "$RUN_ROOT"
# On node 0 set 0; on node 1 set 1. Launch the same stage concurrently on both hosts:
export NODE_RANK=0
platour-exps train --experiment 20 --sender "$SENDER" --receiver "$RECEIVER" \
  --train-backend native-multinode --native-stage probe --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --native-attempt probe01 --vllm-python "$VLLM_PYTHON" \
  --scoring-python "$SCORE_PYTHON" --output "$RUN_ROOT"
# After BOTH probe commands finish, run on both hosts for a NEW attempt:
platour-exps train --experiment 20 --sender "$SENDER" --receiver "$RECEIVER" \
  --train-backend native-multinode --native-stage smoke --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --native-attempt smoke01 --vllm-python "$VLLM_PYTHON" \
  --scoring-python "$SCORE_PYTHON" --output "$RUN_ROOT"
# Only after both nodes' smoke evidence is signed, use --native-stage train
# with --native-attempt train01 on both hosts; after a failure use
# --native-stage train --resume-latest --native-attempt resume02 and the same root.
# For ID25 use --experiment 25, a separate fresh RUN_ROOT and --judge-service-root
# <shared-running-Docker-judge-service> during prepare; the per-node run flags remain.
```


### Portable training generation and epoch budget

The default `--train-backend portable` uses vLLM for GRPO rollout and VAL/TEST
answer generation, including bridge, virtual sender, full and LoRA policies.
A fresh current-weight snapshot is exported for each update; generated token
log-probabilities feed the rollout importance correction. Bridge/virtual policies
retain the native hybrid handoff: HF computes prefix memory and the first token,
then vLLM decodes the continuation. Training uses temperature 1, top-p 1,
top-k -1 and no presence penalty; evaluation keeps its existing sampling protocol.

Formal training derives one epoch from TRAIN and batch size (currently DAPO
264 updates and DeepCoder 295). Published validation schedules must match this
budget. The ordinary CLI, native launchers and API all select by highest validation accuracy; clipping is diagnostic only.
PACT SC/TR and `Qwen35Session.fit` also default to one epoch and restore the
highest-accuracy VAL checkpoint, choosing the earliest tie, before TEST.
`max_steps` requires explicit API `smoke=True`; PACT rejects `--control-steps`.
New backend/budget identities reject resuming old HF-rollout runs. Archived native source manifests remain unchanged; new native runs use the signed selection adapter described below. GPU regression
of the new portable rollout path is pending.

## Shared formal defaults

All published Qwen3.5 experiments (19–30) use GRPO; there is no published Qwen3.5 SFT experiment. AdamW uses betas `(0.9, 0.999)`, weight decay `0`, gradient clipping `1`, and a constant learning rate without warmup: `1e-6` for full-parameter GRPO and `1e-5` for bridge, SC, Virtual Sender and LoRA. LoRA uses rank/alpha `128/128` and dropout `0`.

The CLI, `Qwen35Session.fit` and PACT SC/TR default to batch 64 and a 16,384-token TRAIN/VAL response limit. DAPO validates 500 questions with avg@4; DeepCoder validates 201 with avg@16. Both train one complete epoch (264/295 updates on the published data). Checkpoint selection uses highest VAL accuracy, with earliest-step ties, including step zero. Clipping rate is recorded but never used to exclude a candidate.

PACT exposes separate TEST and VAL budgets: `--validation-samples` defaults to 4/16 for DAPO/DeepCoder; `--samples` controls TEST only, defaulting to 16 for math and 8 for code. For IMO-AnswerBench use `--samples 4`. `--response-cap` controls TRAIN/VAL (16,384); `--test-response-cap` defaults to 32,768 for math and 16,384 for code. Explicit overrides describe custom experiments. The previous batch=1/cap=256/one-sample API defaults were small general-purpose budgets, not formal recipes.

New native runs use a signed accuracy-only selection adapter while the archived source bundles remain unchanged. Old native roots and portable roots with the clipping-guard policy cannot resume under the new policy; create a fresh root. This does not change released weights or relabel historical checkpoint selection. The custom SFT API, if used outside the published experiments, validates response-token NLL with HF and retains the final checkpoint.
