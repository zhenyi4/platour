# Inference checkpoints

The 30 inference checkpoints comprise Qwen3 PLatour-SFT/PLatour (01–18) and Qwen3.5 mathematics/coding comparisons (19–30). Each experiment has a public Hugging Face model repository listed in `models/manifest.json` and the [PLatour collection](https://huggingface.co/collections/zen-E/platour-6aee0586243e1c15a5dd2cc4); `models/manifest.json` pins the exact filename, byte size and SHA-256 for every experiment. Run `platour-exps download --experiment <ID>`, or use the evaluation command to download on demand. Task inputs are downloaded separately from pinned HF dataset commits; see [`data/README.md`](../data/README.md). Bridge, Virtual Sender and LoRA evaluation needs the referenced public backbones; evaluation accepts local directories or HF IDs. Standard Full packages include their own configuration and tokenizer.

## Downloadable checkpoints

All 30 checkpoints below are public, with one Hugging Face model repository per experiment. The ID is the `--experiment` value used by both download and evaluation. SC means Self-Communication; VS means Virtual Sender; TR means Trained Receiver. The repository names identify the model pair, method and task; Qwen3-235B senders use the official FP8 backbone.

| ID | Hugging Face checkpoint | Format |
|---:|---|---|
| 01 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-ARC-C](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-ARC-C) | Project bridge |
| 02 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-MMLU-Redux](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-MMLU-Redux) | Project bridge |
| 03 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-Chemistry-NoCoT](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-Chemistry-NoCoT) | Project bridge |
| 04 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-Chemistry-CoT](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-Chemistry-CoT) | Project bridge |
| 05 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-GSM8K](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-GSM8K) | Project bridge |
| 06 | [zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-MATH500](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-235B-to-1.7B-MATH500) | Project bridge |
| 07 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-ARC-C](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-ARC-C) | Project bridge |
| 08 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-MMLU-Redux](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-MMLU-Redux) | Project bridge |
| 09 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-Chemistry-NoCoT](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-Chemistry-NoCoT) | Project bridge |
| 10 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-Chemistry-CoT](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-Chemistry-CoT) | Project bridge |
| 11 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-GSM8K](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-GSM8K) | Project bridge |
| 12 | [zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-MATH500](https://huggingface.co/zen-E/PLatour-SFT-Qwen3-8B-to-1.7B-MATH500) | Project bridge |
| 13 | [zen-E/PLatour-Qwen3-235B-to-1.7B-Chemistry-CoT](https://huggingface.co/zen-E/PLatour-Qwen3-235B-to-1.7B-Chemistry-CoT) | Project bridge |
| 14 | [zen-E/PLatour-Qwen3-235B-to-1.7B-GSM8K](https://huggingface.co/zen-E/PLatour-Qwen3-235B-to-1.7B-GSM8K) | Project bridge |
| 15 | [zen-E/PLatour-Qwen3-235B-to-1.7B-MATH500](https://huggingface.co/zen-E/PLatour-Qwen3-235B-to-1.7B-MATH500) | Project bridge |
| 16 | [zen-E/PLatour-Qwen3-8B-to-1.7B-Chemistry-CoT](https://huggingface.co/zen-E/PLatour-Qwen3-8B-to-1.7B-Chemistry-CoT) | Project bridge |
| 17 | [zen-E/PLatour-Qwen3-8B-to-1.7B-GSM8K](https://huggingface.co/zen-E/PLatour-Qwen3-8B-to-1.7B-GSM8K) | Project bridge |
| 18 | [zen-E/PLatour-Qwen3-8B-to-1.7B-MATH500](https://huggingface.co/zen-E/PLatour-Qwen3-8B-to-1.7B-MATH500) | Project bridge |
| 19 | [zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO](https://huggingface.co/zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO) | Project bridge |
| 20 | [zen-E/PLatour-Qwen3.5-397B-FP8-to-4B-DAPO](https://huggingface.co/zen-E/PLatour-Qwen3.5-397B-FP8-to-4B-DAPO) | Project bridge |
| 21 | [zen-E/PLatour-SC-Qwen3.5-4B-DAPO](https://huggingface.co/zen-E/PLatour-SC-Qwen3.5-4B-DAPO) | Project bridge |
| 22 | [zen-E/PLatour-VS-Qwen3.5-4B-DAPO](https://huggingface.co/zen-E/PLatour-VS-Qwen3.5-4B-DAPO) | Project Virtual Sender |
| 23 | [zen-E/PLatour-TR-FullGRPO-Qwen3.5-4B-DAPO](https://huggingface.co/zen-E/PLatour-TR-FullGRPO-Qwen3.5-4B-DAPO) | Transformers Full |
| 24 | [zen-E/PLatour-TR-LoRA-Qwen3.5-4B-DAPO](https://huggingface.co/zen-E/PLatour-TR-LoRA-Qwen3.5-4B-DAPO) | PEFT LoRA |
| 25 | [zen-E/PLatour-Qwen3.5-397B-FP8-to-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-Qwen3.5-397B-FP8-to-4B-DeepCoder) | Project bridge |
| 26 | [zen-E/PLatour-Qwen3.5-35B-to-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-Qwen3.5-35B-to-4B-DeepCoder) | Project bridge |
| 27 | [zen-E/PLatour-SC-Qwen3.5-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-SC-Qwen3.5-4B-DeepCoder) | Project bridge |
| 28 | [zen-E/PLatour-VS-Qwen3.5-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-VS-Qwen3.5-4B-DeepCoder) | Project Virtual Sender |
| 29 | [zen-E/PLatour-TR-FullGRPO-Qwen3.5-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-TR-FullGRPO-Qwen3.5-4B-DeepCoder) | Transformers Full |
| 30 | [zen-E/PLatour-TR-LoRA-Qwen3.5-4B-DeepCoder](https://huggingface.co/zen-E/PLatour-TR-LoRA-Qwen3.5-4B-DeepCoder) | PEFT LoRA |

### Download and evaluate

```sh
# Download and verify one checkpoint without loading a model.
platour-exps download --experiment 19

# Evaluate the default published checkpoint for this ID.
platour-exps evaluate --experiment 19 \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --output <new-evaluation-dir>

# The same evaluator also accepts the explicit HF repository ID.
platour-exps evaluate --experiment 19 \
  --weights zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --output <another-new-evaluation-dir>
```

The download/evaluation loader checks the configuration and weight hashes in [`manifest.json`](manifest.json). `--experiment` selects the matching data, architecture and scoring protocol; an explicit `--weights` repository must match that experiment. Missing task inputs are downloaded automatically. Coding IDs 25–30 additionally require the [isolated coding judge](../qwen35/README.md#obtain-and-install-the-coding-judge-ids-2530). These are inference checkpoints; resumable training uses the runner's own signed training checkpoints.

### Loading formats

| Checkpoints | Loading interface | Required backbones |
|---|---|---|
| Bridge: 01–21, 25–27 | Project evaluation CLI or PLatour API | Published sender and receiver; SC uses a second receiver as sender |
| Virtual Sender: 22, 28 | Project evaluation CLI or PLatour API | Receiver; learned virtual sender features are included |
| Full: 23, 29 | Standard Transformers `AutoModelForCausalLM.from_pretrained(<HF-ID>)`, or project evaluation CLI | Full receiver weights, configuration and tokenizer are included |
| LoRA: 24, 30 | Standard PEFT `PeftModel.from_pretrained(base_model, <HF-ID>)`, or project evaluation CLI | Official `Qwen/Qwen3.5-4B` receiver plus adapter |

Bridge and Virtual Sender files use the project inference format and load through the project interfaces above. See the experiment-specific commands in [`qwen3/README.md`](../qwen3/README.md) and [`qwen35/README.md`](../qwen35/README.md).

## Qwen3: PLatour-SFT and PLatour

The following scores are **paper-reported results**, transcribed from Table 1 of [the paper (arXiv:2611.06976v1)](https://arxiv.org/html/2611.06976v1). They are reference values, not fresh measurements from this release. All rows use a Qwen3-1.7B receiver. IDs follow the benchmark column order and correspond to the experiment configurations in the [checkpoint index](#downloadable-checkpoints).

### Answer-only tasks: paper-reported accuracy (%)

| IDs (ARC-C / MMLU / Chemistry) | Sender | Method | ARC-C | MMLU-Redux | Chemistry without CoT | Avg. |
|---|---|---|---:|---:|---:|---:|
| 01 / 02 / 03 | Qwen3-235B-A22B-FP8 | PLatour-SFT | 95.56 | 82.90 | 78.57 | 85.68 |
| 07 / 08 / 09 | Qwen3-8B | PLatour-SFT | 89.85 | 71.29 | 52.86 | 71.33 |

### Generation tasks: paper-reported accuracy (%)

| IDs (Chemistry / GSM8K / MATH) | Sender | Method | Chemistry with CoT | GSM8K | MATH500 | Avg. |
|---|---|---|---:|---:|---:|---:|
| 04 / 05 / 06 | Qwen3-235B-A22B-FP8 | PLatour-SFT | 57.80 | 85.63 | 70.65 | 71.36 |
| 10 / 11 / 12 | Qwen3-8B | PLatour-SFT | 38.69 | 83.62 | 72.10 | 64.80 |
| 13 / 14 / 15 | Qwen3-235B-A22B-FP8 | PLatour (GRPO) | 69.29 | 89.35 | 78.20 | 78.95 |
| 16 / 17 / 18 | Qwen3-8B | PLatour (GRPO) | 68.39 | 87.91 | 76.50 | 77.60 |

Avg. is the unweighted mean of the three benchmark accuracies in that row, as reported in the paper.

### Reproduced checkpoint: experiment 18

The released [PLatour (GRPO), Qwen3-8B→Qwen3-1.7B, MATH500 checkpoint](https://huggingface.co/zen-E/PLatour-Qwen3-8B-to-1.7B-MATH500) comes from a subsequent reproduction run using the reproduction code. The authors report **79.35% MATH500 accuracy** for this reproduced checkpoint. This is separate from the original paper's **76.50%** for the same configuration; the paper-results table and its averages above retain the original published values.

ARC-C (1,172), MMLU-Redux (5,632), and Chemistry without CoT (210) use greedy accuracy. Chemistry with CoT uses 210 × 8 sampled responses, GSM8K 1,319 × 2, and MATH500 500 × 4; reported accuracy is the fraction of correct **individual answers**, not pass@k. Load the public sender and receiver frozen and inject 28 FP32 `P.<layer>.weight` tensors of shape `[2048, 4096]` at receiver block inputs during prompt prefill. PLatour-SFT uses `last_aligned` mapping; PLatour uses `uniform_endpoints`. Each model repository includes a public `config.json` and its inference weight. The project checksum manifest maps every experiment to its own repository and validates both configuration and weights. Input and scoring details are in [`qwen3/README.md`](../qwen3/README.md).

## Qwen3.5: mathematics and coding

Scores below are **paper-reported results** from [Table 2 and Table A21 (arXiv:2611.06976v1)](https://arxiv.org/html/2611.06976v1), not fresh measurements from this release.

Full IDs 23/29 publish standard Transformers packages (`model.safetensors`, configuration and tokenizer). LoRA IDs 24/30 publish standard PEFT packages (`adapter_model.safetensors`, `adapter_config.json`) for the referenced official backbone. Both standard formats also work with the experiment-number evaluation loader; generic filenames are cached separately by ID and checked against this release's hashes. Bridge and Virtual Sender files retain the project inference format.

All Qwen3.5 experiments have a Qwen3.5-4B receiver. PLatour uses a frozen real 35B-A3B or 397B-A17B-FP8 sender; Self-Communication uses a second frozen 4B; Virtual Sender uses learned training-only template features instead of an external model; Full-GRPO and LoRA-GRPO update the receiver without an external sender. These methods use independent task-domain training and the original test protocols described in [`qwen35/README.md`](../qwen35/README.md).

### DAPO mathematics

AIME-80 is the fixed 80-question non-overlapping subset ×16 responses of the same 135-question AIME evaluation; HMMT has 60×16 and IMO-AnswerBench v2 has 400×4. Math Avg. is the unweighted mean of **AIME-80, HMMT, and IMO**, excluding AIME-135. All values are percentages.

| ID | Weight stem | Method / sender | AIME-80 / AIME-135 | HMMT | IMO | Math Avg. |
|---:|---|---|---:|---:|---:|---:|
| 19 | `19_qwen35_35b_to_4b_dapo` | PLatour · 35B-A3B | 75.63 / 74.63 | 64.58 | 49.75 | **63.32** |
| 20 | `20_qwen35_397b_fp8_to_4b_dapo` | PLatour · 397B-A17B-FP8 | 71.88 / 72.73 | 61.77 | 49.25 | **60.97** |
| 21 | `21_qwen35_4b_to_4b_dapo` | Self-Communication · 4B | 72.42 / 72.41 | 56.98 | 43.69 | **57.70** |
| 22 | `22_qwen35_random4b_to_4b_dapo` | Virtual Sender · template | 72.66 / 72.73 | 59.79 | 44.56 | **59.00** |
| 23 | `23_qwen35_4b_full_grpo_dapo` | Full-GRPO · 4B | 76.95 / 77.04 | 65.21 | 51.38 | **64.51** |
| 24 | `24_qwen35_4b_lora_grpo_dapo` | LoRA-GRPO · 4B | 72.73 / 73.56 | 59.48 | 47.44 | **59.88** |

### DeepCoder coding

LCB v6 is 175×8, HumanEval is 164×8, and MBPP full is 500×8. Coding Avg. is the unweighted mean of their three full-denominator accuracies.

| ID | Weight stem | Method / sender | LCB v6 | HumanEval | MBPP | Coding Avg. |
|---:|---|---|---:|---:|---:|---:|
| 25 | `25_qwen35_397b_fp8_to_4b_deepcoder` | PLatour · 397B-A17B-FP8 | 735/1400 (52.50%) | 1229/1312 (93.67%) | 3417/4000 (85.43%) | **77.20%** |
| 26 | `26_qwen35_35b_to_4b_deepcoder` | PLatour · 35B-A3B | 760/1400 (54.29%) | 1246/1312 (94.97%) | 3597/4000 (89.93%) | **79.73%** |
| 27 | `27_qwen35_4b_to_4b_deepcoder` | Self-Communication · 4B | 744/1400 (53.14%) | 1252/1312 (95.43%) | 3634/4000 (90.85%) | **79.81%** |
| 28 | `28_qwen35_random4b_to_4b_deepcoder` | Virtual Sender · template | 784/1400 (56.00%) | 1204/1312 (91.77%)* | 3438/4000 (85.95%)* | **77.91%*** |
| 29 | `29_qwen35_4b_full_grpo_deepcoder` | Full-GRPO · 4B | 760/1400 (54.29%) | 1266/1312 (96.49%) | 3655/4000 (91.38%) | **80.72%** |
| 30 | `30_qwen35_4b_lora_grpo_deepcoder` | LoRA-GRPO · 4B | 788/1400 (56.29%) | 1200/1312 (91.46%) | 3313/4000 (82.83%) | **76.86%** |

*The Virtual Sender’s HumanEval/MBPP scores use explicitly labeled template-adapted prompts; the LCB score uses its original prompt.

The Qwen3.5 `safetensors` files hold only the tensors needed for inference: 32 FP32 bridge projections for real-sender/Self-Communication, projections plus `virtual_hidden` and numeric `template_token_ids` for Virtual Sender, rank-128/alpha-128 adapter tensors for LoRA-GRPO, or complete fine-tuned 4B inference weights for Full-GRPO. Download and freeze the separately sourced public backbones when using bridges.
