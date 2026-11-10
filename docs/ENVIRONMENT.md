# Environment setup

For the installation commands, see [Getting started](../README.md#getting-started).
Run from an editable checkout on Linux x86_64 with Python 3.11 and CUDA.

## Installation options

The installer creates `.venv` with the training, generation and scoring
dependencies, including the Qwen3.5 acceleration and native-training packages.
The core versions are Torch 2.10.0, Transformers 5.17.0 and vLLM 0.17.1;
the complete pins are in [unified-runtime.txt](../requirements/unified-runtime.txt).
The SHA-verified `causal-conv1d` wheel requires Python 3.11, Torch 2.10,
CUDA 12 and CXX11 ABI.

| Installer option | Purpose |
| --- | --- |
| `--venv /path/to/new-env` | Choose a fresh environment directory; existing environments are preserved. |
| `--dry-run` | Inspect the installation commands without running them. |
| `--imports-only` | Skip GPU kernel checks when installing on a machine without GPUs. |
| `--check-only` | Verify the active Python, or the environment selected with `--venv`. |
| `--audit-only` | Check mandatory package dependencies in the active Python. |

Training, vLLM generation and MathVerify scoring default to the active Python.
The API's `vllm_python` argument and the CLI's `--vllm-python` /
`--scoring-python` flags are optional interpreter overrides. Worker processes
use the selected interpreter's executable directory for kernel build tools.

## Dependency compatibility

vLLM 0.17.1 declares `transformers<5`, while this stack uses Transformers 5.17.0.
The installer resolves this using `uv` and an explicit
[override](../requirements/unified-overrides.txt). Ordinary `pip install -e .`
cannot resolve this combination; use the installer above.

Upstream metadata stays unchanged, so `pip check` reports that conflict.
The installer's dependency audit accepts only this exact version combination
and rejects other missing or incompatible mandatory dependencies. A completion
receipt is written inside the environment only after verification succeeds.

## Runtime checks and CUDA

Repeat the complete check before running on a new GPU machine:

```sh
python scripts/install_environment.py --check-only
```

This verifies package pins and imports, tests causal-conv1d forward/backward,
and executes a small vLLM FlashAttention decode kernel. `--imports-only`
does not establish GPU compatibility. The kernel checks can also run separately:

```sh
python scripts/install_qwen35_extras.py --check-only
python scripts/check_vllm_runtime.py
```

Use the same GPU visibility and library paths as the intended training or
decoder process. These checks load no model and cover only selected kernels.
The vLLM wheel may contain CUDA 12.9 PTX even when Torch reports CUDA 12.8.
For a supported GPU with an older driver, follow NVIDIA's
[forward-compatibility instructions](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
and activate the matching compatibility libraries before starting Python.
The installer does not upgrade the system driver or CUDA Toolkit.

Qwen3.5 full/LoRA decoding may also compile FlashInfer GDN prefill kernels.
If compilation fails because of old CUDA headers or NVCC, follow the
[private toolchain setup](../qwen35/README.md#complete-the-gpu-training-environment).

## Validation coverage

| Path | Completed checks |
| --- | --- |
| Fresh installation | Dependency audit, CUDA kernel checks, 288 unittest cases and 20 persistent-worker pytest cases. |
| Qwen3-8B → 1.7B | Full portable ARC-C/GSM8K SFT and final TEST; bounded GRPO rollout, scoring, updates, selection and TEST; signed resume. |
| Qwen3.5-35B-A3B → 4B | Two-update portable GRPO with finite, nonzero gradients; persistent worker reuse, fixed-checkpoint replay, signed resume and CLI execution. |
| Self-Communication | Short HF/vLLM numerical comparisons for Qwen3-1.7B and Qwen3.5-4B. |

SFT runs used the final checkpoint; GRPO used validation-based selection.
The new-install smokes covered both model pairs and ARC-C/GSM8K SFT without
a decoder interpreter override. Each GRPO fit reused one receiver worker.
For grouped 35B → 4B training, the tested CLI topology uses
`--n-gpu 8 --gpus-per-replica 2`; the single-card check used CPU sender offload.

Short fixed-checkpoint comparisons can match tokens and log-probabilities
without establishing identical training trajectories. Updated parameters and
some sampled answers differed between runs. Long-sequence evidence is limited
to individual batch-one checks; a Qwen3 16K-input attempt exceeded memory.
Full GRPO budgets, FP8 sender models, larger long-context batches and native
Full/LoRA or two-host runs still need validation in this environment.

To compare runtimes with the same models and inputs, use
[compare_runtime_environments.py](../scripts/compare_runtime_environments.py).
Its `run` subcommand records two teacher-forced updates and short native
rollouts; `compare` checks saved gradients, weights, tokens and log-probabilities.
These runs need two visible GPUs. Keep tensors, hardware details and timing
reports in private output directories.
