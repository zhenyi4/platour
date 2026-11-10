# Hugging Face inference release

The 30 model repositories and three dataset repositories (Qwen3, DAPO and DeepCoder) are public in the [PLatour collection](https://huggingface.co/collections/zen-E/platour-6aee0586243e1c15a5dd2cc4). All 236 release files were checked against remote identities, and every repository was verified without authentication. Repository URLs, publication revisions and formats are recorded in [`hf_public_release.json`](hf_public_release.json). Preparation and audit commands themselves do not upload anything.

## Prepare and inspect

Run from the checkout root, using a fresh output directory:

```sh
python scripts/prepare_hf_release.py --source <original-checkpoint-directory> \
  --output outputs/<fresh-release> --namespace <owner> --update-manifest
python scripts/audit_hf_release.py outputs/<fresh-release>
```

The checkpoint directory must contain its original `manifest.json`. Export validates each original file against that manifest and compares the original and exported tensor-payload SHA-256. Only safetensors metadata changes; source files remain untouched. APFS copy-on-write saves disk space while keeping source and export independently writable. Other filesystems need enough space for full copies.

The output contains:

| Location | Purpose | Upload |
|---|---|---|
| `public/models/<name>/` | One weight, `config.json`, model card, project license | That model repository only |
| `public/datasets/<name>/` | Frozen inputs, manifests, dataset card | That dataset repository only |
| `public/models-manifest.json` | Configuration and weight identities for the code release | Copy to code release as `models/manifest.json` |
| `private/` | Original source metadata, original hashes, audit/provenance records | Never |
| `release-plan.json` | Explicit repository-to-directory mapping and file whitelist | Local publication control only |

Never upload the staging root or an original checkpoint folder. Use only the explicit files for each repository in `release-plan.json`. Keep training logs, optimizer/RNG state, paper attachments, old READMEs and source Git history out of these artifact repositories. Review the code repository and paper separately: this artifact audit does not certify their contents.

The public export allowlists inference metadata and omits operational training metadata. Original provenance remains private; omission is not permission to invent a different selection protocol. These cards publish inference artifacts without benchmark scores. Independent evaluation and accurate provenance reporting are required before making new comparative research claims.

## Loading and download statistics

Evaluation accepts an explicit repository ID or a local file through `--weights` in both model families. For example:

```sh
platour-exps evaluate --experiment 19 \
  --weights zen-E/PLatour-Qwen3.5-35B-to-4B-DAPO \
  --sender Qwen/Qwen3.5-35B-A3B --receiver Qwen/Qwen3.5-4B \
  --output <new-evaluation-dir>
```

The experiment selects the signed data and scoring protocol. Hub configuration and weight SHA-256 must match that experiment's release manifest, including when using a mirror. Omit `--weights` to use the manifest's assigned repository. Local file overrides retain the family loader's tensor/schema validation. Evaluation accepts local directories or HF repository IDs for sender and receiver; HF inputs download backbone safetensors, configuration and tokenizer to the normal Hub cache before either generation backend starts. For a particular backbone revision, pass a separately downloaded local snapshot directory. Training still takes local backbone directories. Private repositories require an authenticated HF account with access.

The 26 bridge/Virtual Sender packages have a project `config.json` identifying the experiment, dependencies, tensor schema and inference weight. Full IDs 23/29 instead use standard Transformers `config.json` and `model.safetensors`; LoRA IDs 24/30 use PEFT `adapter_config.json` and `adapter_model.safetensors`. The project manifest pins configuration and weight SHA-256 for every format. `platour-exps download --experiment <ID>` verifies the configuration before downloading the weight, using separate per-experiment caches for generic standard filenames. Evaluation accepts all three formats. Existing legacy project files and single-repository manifests remain supported.

Hugging Face's default model counter uses requests to `config.json`, including GET/HEAD. It is not a count of unique people who successfully downloaded all weights. A weight-only direct link may bypass that counter. The loader's config request serves actual loading and integrity checks; it does not add artificial count requests. See [model download statistics](https://huggingface.co/docs/hub/models-download-stats).

Dataset statistics group requests within a five-minute window per IP and repository. Three dataset repositories give three independent counters. See [dataset download statistics](https://huggingface.co/docs/hub/datasets-download-stats). No download-count experiment has been performed against the planned repositories.

## Evidence already present in this branch

The loading implementation is available; a separate loading implementation is not a missing release dependency:

- `qwen35/model.py::Actor.load_public` and `load_inference_state` load Full and custom LoRA inference states with exact key/shape/dtype checks. Full exports are receiver state dictionaries. LoRA uses the existing decoder-linear wrappers with rank/alpha 128.
- `qwen35/vllm_decode.py::export_lora` converts either legacy project tensors or published PEFT tensors into the native vLLM runtime adapter, including splitting the fused GDN QKV projection. Both inputs produced exactly equal runtime tensors in the real ID24 conversion check; GPU generation is a separate validation item.
- `qwen35/cli.py::_model` loads the receiver and its tokenizer locally. `qwen35/README.md` documents the evaluation entry points. Large-backbone end-to-end inference is a separate validation item, not evidence that the loader is absent.

Some base identities are also recorded:

| Scope | Recorded branch evidence | Remaining gap |
|---|---|---|
| IDs 20 and 25 sender | `native_multinode_protocols.json` records Qwen3.5-397B-A17B-FP8 commit `ea5b4f81096f3901c91dea97f81324302495781d` | This protocol field alone does not establish all receiver/tokenizer identities or a complete historical artifact-to-run link |
| ID 22 source model | `native_protocols.json` records both source model-shard SHA-256 values | Weight-shard identities do not identify tokenizer/config files |
| Other model/tokenizer dependencies | Model names, local loading paths and architecture checks | A complete per-experiment commit or file-hash inventory has not been located in the audited release records |

A revision literal for another bundled recipe is not automatically evidence for a released experiment. In particular, the FP8 kernel commit is a software dependency, not a model or tokenizer revision.

The branch identifies the dataset families, but its code license and bundled third-party code notices are not a source-by-source dataset licensing inventory. The following upstream-card declarations were checked on 2026-11-06; links identify the referenced upstream, not proof that every frozen row came from its current HEAD:

| Referenced source | Upstream card declaration |
|---|---|
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT |
| [MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) | CC-BY-4.0; confirm the frozen collection's Redux version before applying this entry |
| [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval) | MIT |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT |
| [DAPO](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | Apache-2.0 |
| [DeepCoder Preview](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | MIT |
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | MIT |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | CC-BY-4.0 |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | No license declaration found on the inspected card; inspect original MATH terms and source provenance |

MATH training material, AIME, HMMT, IMO-AnswerBench, LCB and the underlying DeepCoder problem sources still need source-specific notice/terms matching. Do not replace attribution or share-alike requirements with the project code license. This table records declarations and outstanding provenance work; it is not a completed redistribution clearance.

## Data and remaining release review

Keep the frozen prompts, split identities and signed input manifests. Native DAPO training also uses the frozen Parquet copies. DeepCoder TRAIN/VAL checker cases are reconstructed by `python -m qwen35.data`; do not upload the old case archives. The project code supplies the reconstruction logic and pinned upstream identities. DeepCoder evaluation additionally needs the separately prepared LCB v6 cases and isolated checker described in `qwen35/README.md`.

The following provenance and evaluation reviews remain open; publication of inference artifacts does not complete them:

- Complete the model/tokenizer identity inventory using the existing evidence above. Current Hub HEAD revisions do not establish which versions produced the original weights.
- Record source-by-source dataset license identifiers, notices and redistribution permissions. The upstream DeepCoder card currently says MIT and DAPO says Apache-2.0; that does not settle the terms of every included benchmark or underlying source problem. Do not blanket-license a mixed dataset under the code license. Sources: [DeepCoder](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset), [DAPO](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
- Check the paper and public results against evaluation provenance. Separate new-run validation procedures from claims about historical artifacts. Avoid portraying packaging checks as model evaluation.
- Verify end-to-end inference in the intended runtime. CPU header/schema checks do not prove successful generation with large backbones or numerical parity with historical runs.
- Keep code, model manifest and dataset releases aligned. The public collection links all 33 repositories; the release record pins their publication revisions.

## Standard Full/LoRA exports

The evaluation loader maps standard PEFT keys back to the project actor schema and converts either PEFT or legacy adapters to the same vLLM runtime layout. Full evaluation preserves independent embeddings and output-head tensors, including when the upstream base configuration ties them. These compatibility checks do not establish benchmark accuracy or GPU runtime parity.

`qwen35.hf_export` supplies the published standard packages for Full IDs 23/29 and LoRA IDs 24/30. Earlier project-format staging packages remain local; the standard packages provide the public `from_pretrained` interface. The conversion never edits the original checkpoints.

```sh
python -m qwen35.hf_export full --checkpoint <full-checkpoint.safetensors> \
  --base <official-base-snapshot> --output <fresh-full-directory>
python -m qwen35.hf_export lora --checkpoint <lora-checkpoint.safetensors> \
  --base <official-base-snapshot> --output <fresh-adapter-directory> \
  --base-model-id Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Full exports contain `model.safetensors`, the matching `qwen3_5_text` configuration, generation settings and tokenizer files. They load with `AutoModelForCausalLM.from_pretrained(...)`. The source saves both input embeddings and output-head tensors, so the export sets `tie_word_embeddings=False` to preserve each tensor independently. No vision encoder is included or required.

LoRA exports contain `adapter_model.safetensors` and `adapter_config.json`. Load the pinned official base using `AutoModelForImageTextToText.from_pretrained(...)`, then use `PeftModel.from_pretrained(base, adapter_directory)`. The official base is multimodal even though the adapter targets only text. All 248 target paths are fully qualified, preventing accidental adapter insertion into vision layers. Original fused QKV matrices stay fused, and original FP32 A/B values are preserved. The existing `qwen35.vllm_decode.export_lora` serves a different vLLM runtime layout and is not the exporter for these HF/PEFT packages.

Actual loading and original-tensor checks can be run locally:

```sh
python scripts/verify_hf_exports.py full --directory <full-export> \
  --original <original-full-checkpoint> --report <fresh-private-report.json>
python scripts/verify_hf_exports.py lora --directory <adapter-export> \
  --original <original-lora-checkpoint> --base <official-base-snapshot> \
  --report <fresh-private-report.json>
```

The verifier performs real standard API loading, exact original tensor comparisons and short synthetic-prompt forward/generation checks. For LoRA it also compares FP32 logits with the project's original `LoRALinear` implementation on the identical frozen backbone. FP32 comparison does not establish identical rounding in BF16 or across different device kernels. These checks are not benchmark evaluation. Verification reports stay outside upload directories.

The conversion/loading environment uses Torch 2.15.0, Transformers 5.22.1 and PEFT 0.21.2. PEFT is needed only for LoRA conversion/loading. The pinned base revision above is the present-day export/verification dependency; it is not automatically evidence of a historical training revision.

All four real checkpoints passed local standard API loading, original tensor comparisons and short synthetic-prompt generation. Both LoRA exports matched the original project forward with maximum absolute FP32 logit difference 0 on the checked input. See [`hf_standard_validation.json`](hf_standard_validation.json) for the recorded evidence and limitations. All four standard packages are now public; [`hf_public_release.json`](hf_public_release.json) records their publication revisions.
