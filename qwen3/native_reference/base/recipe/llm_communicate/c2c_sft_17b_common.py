"""Pinned original C2C utilities for the 4B/8B -> 1.7B SFT suite."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[2]
C2C = ROOT / 'third_party/c2c'
COMPAT = ROOT / '.c2c_runtime'
PYTHON = ROOT / '.runtime/readme-check/bridge/bin/python'
CPU = ROOT / '.runtime/readme-check/scoring/bin/python'
REVISION = '113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a'
DATA = ROOT / 'outputs/llm_communicate/qwen3_4b_8b_non_thinking_gsm8k_math_train_cap2k4k'
BASELINES = ROOT / 'outputs/llm_communicate/qwen3_17b_4b_8b_native_non_thinking_gsm8k2_math4_cap2k4k_fa2'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def environment(gpus, compat=True):
    paths = [ROOT / 'scripts', ROOT / 'recipe/llm_communicate', ROOT / 'third_party/verl']
    if compat:
        paths = [COMPAT, C2C] + paths
    return dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
        PYTHONPATH=':'.join(map(str, paths)), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', WANDB_MODE='online',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', MATH_VERIFY_BOXED_ONLY='1', MATH_VERIFY_TIMEOUT_S='10',
        PLATOUR_GENERATION_PYTHON=str(ROOT / '.runtime/readme-check/generation/bin/python'),
        PLATOUR_SCORING_PYTHON=str(CPU))


def official_trainer():
    # Original C2C predates the optional Hub-kernels package. It uses native
    # Torch SDPA, so disable discovery of the newer inherited optional package.
    import transformers.utils.import_utils as iu
    iu._kernels_available = False
    sys.modules['kernels'] = None
    import transformers
    import tokenizers
    assert transformers.__version__ == '4.52.4' and tokenizers.__version__ == '0.21.4'
    spec = importlib.util.spec_from_file_location('c2c_original_sft', C2C / 'script/train/SFT_train.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from rosetta.train.dataset_adapters import generate_kv_cache_index
    original_dataset = module.ChatDataset
    class TrajectoryChatDataset(original_dataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            source = self.chat_dataset.dataset[idx]
            if source.get('finish_reason') == 'length':
                # Retokenizing decoded text can reduce its length below the
                # numeric response cap. Still never teach a synthetic EOS on
                # a generation that stopped only because of its length cap.
                start = next(i for i, token in enumerate(item['labels']) if token != -100)
                body = self.tokenizer(source['model_response'], add_special_tokens=False)['input_ids']
                cap = min(len(body), self.max_response_tokens or len(body), self.max_length - start)
                end = start + cap
                item['input_ids'] = item['input_ids'][:end]
                item['labels'] = item['labels'][:end]
                item['kv_cache_index'] = generate_kv_cache_index(start, end)
            return item
    module.ChatDataset = TrajectoryChatDataset
    return module


def freeze_and_verify(model):
    for backbone in model.model_list:
        backbone.requires_grad_(False)
    for projector in model.projector_list:
        projector.requires_grad_(True)
    assert len(model.projector_list) == 28
    assert model.projector_dict[0][1] == {i: [(i + 8, i)] for i in range(28)}
    assert not any(p.requires_grad for b in model.model_list for p in b.parameters())


def projector_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.projector_list.named_parameters():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()
