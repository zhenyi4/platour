# Experiment inputs

Frozen public inputs are hosted in three public Hugging Face datasets: [Qwen3](https://huggingface.co/datasets/zen-E/PLatour-Qwen3-Data), [DAPO](https://huggingface.co/datasets/zen-E/PLatour-DAPO-Data), and [DeepCoder](https://huggingface.co/datasets/zen-E/PLatour-DeepCoder-Data). Git retains the split manifests, split indices, Clean80 mask, reconstruction scripts, and [`hf_manifest.json`](hf_manifest.json), which pins the HF commit, remote filename, byte size and SHA-256 for all 68 payload files. Model backbones are downloaded separately. The paired inference checkpoints and evaluation protocols are described in [`models/README.md`](../models/README.md).

Training and evaluation automatically download the required inputs into the existing `data/qwen3/` and `data/qwen35/` layout. Valid local files are reused; changed files fail verification instead of being overwritten. Downloads use the normal HF cache, per-file process locks and atomic installation. Git LFS is no longer required for the current branch; old commits retain their historical LFS pointers.

```sh
platour-exps download-data --experiment 1   # all splits for one Qwen3 experiment
platour-exps download-data --experiment 19  # complete shared DAPO inputs
platour-exps download-data --experiment 26  # complete shared DeepCoder prompt inputs
platour-exps download-data --all            # all 68 files, deduplicated
```

Use `--output <data-root>` to restore the same layout elsewhere, and `--cache-dir <hf-cache>` to select an HF cache. `--local-files-only` uses only already-cached downloads; `HF_HUB_OFFLINE=1` also applies to automatic downloads. For Qwen3, pass individual `--train-data`, `--validation-data` or `--test-data` paths to use that location; defaults follow the experiment ID. For Qwen3.5, pass `--data-root <data-root>/qwen35`. Protocol JSON files are copied from this checkout when restoring to an empty root. Existing caller-owned custom Qwen3.5 manifests retain their original local audit path. Dataset downloads do not install the coding judge or reconstruct unit tests.

## Qwen3 (01–18)

### Frozen IT supplements and pairings

Qwen3-8B→1.7B experiments **07–12 and 16–18** have frozen `archived-it-v1`
inputs, including 183 archived rephrase records and the complete fixed donor
assignments. [`qwen3/it_release.json`](../qwen3/it_release.json) pins the HF commit
and all payload hashes. No external receipt or historical main checkout is needed.

```sh
platour-exps download-data --experiment 16 --include-it
platour-exps download-data --all --include-it
platour-exps pact-check --experiment 16 --prepare-only
```

The download command restores normal task data plus IT inputs under
`data/it/qwen3/archived-it-v1/`; `--output <data-root>` relocates both. The default
full Qwen3 PACT-Check entry also restores the published TRAIN/VAL
inputs and aligns their instruction template. It reads published GRPO validation
directly and never splits TRAIN at runtime. Explicit `--controls IT` downloads only the chosen
IT configuration plus shared rewrite provenance. `--it-data-root` selects another
IT root. Both support cached offline
data via `--local-files-only`, and changed files fail verification.

Preparation verifies hashes, full index coverage, exact token lengths and the
rewrite record associated with each supplemented donor. It preserves fixed
assignments instead of rerunning random selection. Supplements are IT-only input
artifacts; they are never added to the model training split. These inputs preserve
the archived semantic checks, without claiming a new semantic audit.

Archived prompts can differ from the release's normal test templates. The default
full PACT-Check uses the same archived evaluation prompts and original gold
answers for PACT, DT, IT, SC and TR. MATH experiments 12 and 18 reuse `math_8b_c2c_irrelevant` pairing;
18 uses the release GRPO protocol and is not a historical GRPO IT reproduction.
MMLU's duplicate full prompts retain the historical last-pairing-wins policy,
with counts recorded in the data audit. Original dataset terms apply.

The advanced `platour-exps it-data publish/verify/restore/download` commands
remain available for explicitly publishing or restoring other receipt-pinned
bundles; normal IT evaluation uses the release-pinned registry above.

`data/qwen3/<experiment-tag>/` contains `train.jsonl`, `validation.jsonl`, `test.jsonl` and `manifest.json`; the SFT tasks also include `split.json`. Rows preserve ordered question indices, prompt token IDs and gold labels. SFT training/validation additionally retain the response tokens used for supervision. The manifest records source and split checksums. ARC-C, MMLU-Redux and Chemistry use public source questions; GSM8K and MATH use the task-specific answer format. Details of splits, prompts, budgets and scoring are in [`qwen3/README.md`](../qwen3/README.md).

### Fixed Qwen3 GRPO validation splits

The migration covers both 235B and 8B GRPO groups (13–18). It partitions original
TRAIN once with seed 42 and writes `train.jsonl`, `validation.jsonl`, `split.json`
and updated manifests to the same HF paths. Chemistry uses 210 validation rows
(the TEST size); GSM8K and MATH500 use 500 each. Final TRAIN/VAL/TEST counts are
1,680/210/210, 6,973/500/1,319 and 7,000/500/500, respectively. Prompt/gold fields
are preserved; indices are contiguous within each split and `original_train_index`
records their source. TEST, SFT and `it/v1` files are unchanged. Archived IT donors
retain their original training-pool provenance.

```sh
python -m qwen3.grpo_data prepare --output /path/to/new-grpo-release
python -m qwen3.grpo_data verify --bundle /path/to/new-grpo-release
python -m qwen3.grpo_data publish --bundle /path/to/new-grpo-release
```

Preparation is local and leaves active signed data untouched. Publishing requires
HF write access. It verifies the current remote payloads, atomically commits only
the prepared changes against that parent revision, verifies the new revision,
then installs local TRAIN/VAL, manifests, split records and the pinned HF index.
The immutable receipt is saved in the bundle and in
`docs/qwen3_grpo_split_release.json`. A failed network request does not invent a
revision or switch local inputs. Retry the same publish command to finish remote
verification/local installation after an interrupted publication.

PACT reads the published splits and rejects the old VAL=TEST files. Normal GRPO
training consumes corrected published validation without a second split; old
pinned releases retain their prior training behavior for reproducibility.

## Qwen3.5 (19–30)

`data/qwen35/dapo/` holds DAPO-17k training/validation inputs and the AIME, HMMT and IMO-AnswerBench v2 test inputs. `clean80.json` identifies the fixed 80-question AIME subset. `data/qwen35/deepcoder/` holds the decontaminated coding train/validation splits and LCB v6, HumanEval and MBPP prompts. The Virtual Sender’s adapted HumanEval/MBPP prompts are separately named. The 19,074 TRAIN/VAL programming unit tests are rebuilt from the fixed `agentica-org/DeepCoder-Preview-Dataset` revision recorded in `manifest.json`; the release does not bundle the former eight `train_val_cases_*.tar.zst` archives. Run `python -m qwen35.data --output <new-case-directory>` to download the pinned TRAIN parquet files and reconstruct every case, checking its original per-question SHA. For offline use, pass `--source-root <snapshot-root>` with the `taco/`, `primeintellect/` and `lcbv5/` TRAIN parquet files. Keep the frozen `train.jsonl` and `validation.jsonl`: they define the exact selected questions and checksums. The 175 LCB v6 TEST case files are obtained separately. `qwen35/lcb_v6_cases_sha256.json` pins their per-question SHA values **and the public `livecodebench/code_generation_lite` revision/`test6.jsonl` SHA**; `python -m qwen35.judge_cases --fetch-public-lcb-v6 ...` downloads that v6 increment and derives all TEST case files. The checker comes from a pinned public LiveCodeBench Git commit, and the isolated judge image is built locally from the bundled `qwen35/lcb_sandbox.Dockerfile`. See the [coding-judge installation guide](../qwen35/README.md#obtain-and-install-the-coding-judge-ids-2530). The dataset-wide `data/qwen35/manifest.json` records row counts and source hashes. Task metrics and runnable commands are in [`qwen35/README.md`](../qwen35/README.md).

The JSONL/Parquet payloads and local download locks are ignored by Git. DeepCoder case reconstruction first restores missing frozen TRAIN/VAL prompts from the pinned release, then downloads approximately 5.42 GB of upstream TRAIN parquet files (reusable through the Hugging Face cache) and writes approximately 8.00 GB of verified case JSON; it does not require a GPU or execute test programs. For fully offline reconstruction, provide both cached/retrieved frozen prompts and the upstream TRAIN parquet files. Dataset redistribution follows the terms of the upstream sources (ARC, MMLU, SciKnowEval, GSM8K, MATH, DAPO and DeepCoder).
