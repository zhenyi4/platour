"""Standard loading and architecture-preservation checks with a tiny real model."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3_5TextConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from qwen35.hf_export import adapter_name, custom_lora_name, export_full
from qwen35.model import Actor
import platour_exps


class StandardHFExportTests(unittest.TestCase):
    def pin_weight(self, root, identifier, path):
        manifest = json.loads((Path(__file__).resolve().parents[1] / 'models/manifest.json').read_text())
        manifest['files'][f'{identifier:02d}'] = [path.name, path.stat().st_size,
                                                  hashlib.sha256(path.read_bytes()).hexdigest()]
        (root / 'manifest.json').write_text(json.dumps(manifest))

    def fixture(self, root):
        base = root / 'base'
        base.mkdir()
        config = Qwen3_5TextConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=1,
            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            tie_word_embeddings=False, eos_token_id=2, pad_token_id=0,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1.})
        torch.manual_seed(123)
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
        state = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
        self.assertFalse(torch.equal(state['lm_head.weight'], state['model.embed_tokens.weight']))
        config.tie_word_embeddings = True
        config.save_pretrained(base)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]')),
            pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
        tokenizer.save_pretrained(base)
        return base, state

    def test_full_standard_load_preserves_both_embedding_and_head(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, state = self.fixture(root)
            source = root / 'original.safetensors'
            save_file(state, source)
            original = source.read_bytes()
            export_full(source, base, root / 'export')
            model, info = AutoModelForCausalLM.from_pretrained(root / 'export',
                local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
            self.assertFalse(model.config.tie_word_embeddings)
            self.assertFalse(info['missing_keys'])
            self.assertFalse(info['unexpected_keys'])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
            self.assertEqual(source.read_bytes(), original)
            self.assertNotIn('_name_or_path', json.loads((root / 'export/config.json').read_text()))

    def test_full_rejects_private_metadata_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, state = self.fixture(root)
            source = root / 'original.safetensors'
            save_file(state, source, metadata={'selected_step': 'synthetic-private-value'})
            with self.assertRaisesRegex(ValueError, 'operational'):
                export_full(source, base, root / 'export')
            self.assertFalse((root / 'export').exists())

    def test_full_rejects_missing_tensor_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, state = self.fixture(root)
            state.pop('lm_head.weight')
            source = root / 'original.safetensors'
            save_file(state, source)
            with self.assertRaisesRegex(ValueError, 'tensor schema'):
                export_full(source, base, root / 'export')
            self.assertFalse((root / 'export').exists())

    def test_project_full_actor_loads_standard_weight_and_rejects_swapped_experiment(self):
        from qwen35.catalog import select
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, state = self.fixture(root)
            path = root / 'model.safetensors'
            save_file(state, path)
            self.pin_weight(root, 23, path)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(base)
            config.tie_word_embeddings = True
            model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
            self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
            actor = Actor(select(23), model)
            with patch.object(platour_exps, 'MODEL_DIR', root):
                actor.load_public(path)
                self.assertFalse(model.config.tie_word_embeddings)
                self.assertIsNot(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
                for key, value in actor.inference_state().items():
                    self.assertTrue(torch.equal(value, state[key]), key)
                with self.assertRaisesRegex(ValueError, 'size differs'):
                    Actor(select(29), model).load_public(path)

    def test_native_full_decoder_disables_tied_embeddings_in_nested_config(self):
        from qwen35.vllm_worker import untied_full_config
        config = SimpleNamespace(tie_word_embeddings=True,
                                 text_config=SimpleNamespace(tie_word_embeddings=True))
        self.assertIs(untied_full_config(config), config)
        self.assertFalse(config.tie_word_embeddings)
        self.assertFalse(config.text_config.tie_word_embeddings)

    def test_native_full_worker_requires_audited_weights_before_ready(self):
        from qwen35 import vllm_worker
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                digest = 'a' * 64
                config = {'method': 'full', 'model': '/base', 'checkpoint': '/full/model.safetensors',
                          'checkpoint_sha256': digest, 'max_model_len': 128,
                          'max_num_seqs': 1, 'memory_fraction': .2}
                (root / 'config.json').write_text(json.dumps(config))
                def factory(**kwargs):
                    self.assertEqual(kwargs['load_format'], 'dummy')
                    self.assertIs(kwargs['hf_overrides'], vllm_worker.untied_full_config)
                    def rpc(method, *, args):
                        self.assertEqual((method, args), ('load_full', (config['checkpoint'], digest)))
                        return [{'full_model_sha256': digest if valid else 'b' * 64}]
                    return SimpleNamespace(collective_rpc=rpc,
                        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: None)))
                modules = {'vllm': SimpleNamespace(LLM=factory, SamplingParams=object),
                           'vllm.lora.request': SimpleNamespace(LoRARequest=object)}
                with patch.dict(sys.modules, modules), patch.object(sys, 'argv',
                        ['worker', '--directory', str(root)]), patch.object(sys, 'stdin', io.StringIO('')), \
                     patch('qwen35.vllm_worker.importlib.metadata.version', return_value='0.17.1'):
                    if valid:
                        vllm_worker.main()
                        self.assertTrue(json.loads((root / 'ready.json').read_text())['complete'])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'did not load'):
                            vllm_worker.main()
                        self.assertFalse((root / 'ready.json').exists())

    def test_project_lora_actor_preserves_standard_peft_tensors(self):
        from transformers import AutoConfig
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, _ = self.fixture(root)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base)).to(torch.bfloat16)
            targets = [name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear) and name.startswith('model.layers.')]
            case = SimpleNamespace(identifier=24, tag='24_tiny', method='lora', lora_target_count=len(targets))
            actor = Actor(case, model)
            expected = {key: torch.randn_like(value) for key, value in actor.inference_state().items()}
            path = root / 'adapter_model.safetensors'
            save_file({adapter_name(key): value for key, value in expected.items()}, path, metadata={'format': 'pt'})
            self.pin_weight(root, 24, path)
            with patch.object(platour_exps, 'MODEL_DIR', root):
                actor.load_public(path)
            for key, value in actor.inference_state().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
                for multimodal in (True, False):
                    self.assertEqual(custom_lora_name(adapter_name(key, official_multimodal=multimodal)), key)
            with self.assertRaises(ValueError):
                custom_lora_name('base_model.model.visual.proj.lora_A.weight')


if __name__ == '__main__':
    unittest.main()
