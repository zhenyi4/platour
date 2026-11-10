"""Shared data and runtime contract for matched ARC-C C2C/hidden SFT."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'scripts'), str(HERE)]
from generate_qwen35_train_suite import save, immutable, read, jsonl

PYTHON = Path(os.environ.get('ARC_SFT_PYTHON',str(ROOT / '.runtime/arc/bin/python')))
BASE = ROOT / 'outputs/llm_communicate/arc_c_answer_sentence_matched_sft_ep3_bs64'
BASELINE = ROOT / 'outputs/llm_communicate/qwen3_native_arc_challenge_answer_sentence_greedy_cap2k'
PAIRS = [('Qwen3-4B', 'Qwen3-0.6B'), ('Qwen3-4B', 'Qwen3-1.7B'), ('Qwen3-8B', 'Qwen3-1.7B')]
GREEDY = dict(temperature=0., top_p=1., top_k=0, min_p=0.,
              presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(compat=False, gpu=True):
    paths = [ROOT / 'scripts', HERE]
    if compat:
        paths = [ROOT / '.c2c_runtime', ROOT / 'third_party/c2c', *paths]
    library = ROOT / '.runtime/cuda-compat-12.9/usr/local/cuda-12.9/compat'
    return dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('ARC_SFT_GPU_IDS','0') if gpu else '',
        PYTHONPATH=':'.join(map(str, paths)),
        LD_LIBRARY_PATH=str(library) + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
        HF_HOME=str(ROOT / '.runtime/huggingface'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', WANDB_MODE=os.environ.get('WANDB_MODE','offline'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
        TRITON_CACHE_DIR=str(ROOT / '.runtime/qwen35-generation-triton'),
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / '.runtime/qwen35-generation-inductor'),
        XDG_CACHE_HOME=str(ROOT / '.runtime/qwen35-generation-cache'))


def exact_arc_trainer():
    """Original C2C loop, with the exact shared ARC token targets and split."""
    from c2c_sft_17b_common import official_trainer
    from rosetta.train.dataset_adapters import generate_kv_cache_index
    trainer = official_trainer()
    parent = trainer.ChatDataset

    class ArcChatDataset(parent):
        def __getitem__(self, index):
            feature = super().__getitem__(index)
            record = self.chat_dataset.dataset[index]
            ids = record['prompt_token_ids'] + record['response_token_ids']
            # The generic chat adapter also appends a newline after im_end.
            # Both ARC methods supervise the same answer + natural im_end only.
            assert feature['input_ids'][:len(ids)] == ids
            prefix = len(record['prompt_token_ids'])
            return dict(input_ids=ids, labels=[-100] * prefix + record['response_token_ids'],
                        kv_cache_index=generate_kv_cache_index(prefix, len(ids)))

    trainer.ChatDataset = ArcChatDataset
    return trainer


def load_hidden(base, checkpoint=None):
    import json
    protocol=json.loads((Path(base)/'protocol.json').read_text())
    if protocol.get('bridge_family')=='qwen3_template_virtual':
        from qwen3_arc_virtual_teacher_sft import load_bridge
        return load_bridge(base,checkpoint)
    if protocol.get('bridge_family')=='c2c_235b':
        from qwen235_c2c_sft import load_bridge
        return load_bridge(base,checkpoint)
    if protocol.get('teacher_family')=='qwen3_235b_fp8':
        from qwen3_235b_trivia_hidden_sft import load_bridge
        return load_bridge(base,checkpoint)
    from qwen3_17b_hidden_sft import load_bridge
    return load_bridge(base, checkpoint)


def hidden_digest(bridge):
    if hasattr(bridge,'virtual_hidden'):
        from qwen35_bridge_opd_core import weight_digest
        return weight_digest(bridge)
    from qwen3_17b_hidden_sft import digest
    return digest(bridge)
