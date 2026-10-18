"""Offline model selection and numerical HF/native conversion parity."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from olmo.model_configs import LLMS, QWEN25_7B_INSTRUCT
from olmo.nn.llm import Llm, causal_attention_bias
from olmo.nn.peft import LoRALinear
from olmo.torch_util import BufferCache
from olmo.experiment_config import load_experiment, build_experiment
from scripts.prepare_pretrained_model import convert_state_dict_qwen25, CONVERT_FNS, LLM_HF_SOURCES, get_default_load_path


class Qwen25Tests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, MOLMO_DATA_DIR='/tmp/trajvlm-qwen25')
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_official_config_and_registry(self):
        cfg = LLMS['qwen2.5_7b_instruct']
        self.assertIs(cfg, QWEN25_7B_INSTRUCT)
        self.assertEqual((cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim),
                         (3584, 28, 28, 4, 128))
        self.assertEqual((cfg.mlp_hidden_size, cfg.embedding_size, cfg.vocab_size), (37888, 152064, 152064))
        self.assertEqual(cfg.rope_theta, 1000000.0)
        self.assertEqual(cfg.max_sequence_length, 32768)
        self.assertTrue(cfg.qkv_bias)
        self.assertFalse(cfg.attention_layer_norm)
        self.assertFalse(cfg.weight_tying)
        self.assertEqual(cfg.tokenizer.identifier, 'Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(LLM_HF_SOURCES['qwen2.5_7b_instruct'], cfg.tokenizer.identifier)
        self.assertIs(CONVERT_FNS['qwen2.5_7b_instruct'], convert_state_dict_qwen25)
        self.assertEqual(get_default_load_path(cfg), 'pretrained_llms/qwen2.5-7b-instruct.pt')

    def tiny_models(self):
        hf_config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               max_position_embeddings=64, rope_theta=1000000.0,
                               rms_norm_eps=1e-6, attention_dropout=0.0, tie_word_embeddings=False,
                               use_sliding_window=False)
        hf_config._attn_implementation = 'eager'
        torch.manual_seed(15)
        hf = Qwen2ForCausalLM(hf_config).eval()
        cfg = replace(QWEN25_7B_INSTRUCT, d_model=32, mlp_hidden_size=128, n_layers=2,
                      n_heads=4, n_kv_heads=2, head_dim=8, embedding_size=64, vocab_size=64,
                      additional_vocab_size=4, max_sequence_length=64, init_path=None)
        native = Llm(cfg, BufferCache(), device='cpu').eval()
        converted = convert_state_dict_qwen25(hf.state_dict(), cfg, cfg.block_type)
        native_state = {key.removeprefix('transformer.'): value for key, value in converted.items()}
        errors = native.load_state_dict(native_state, strict=False)
        self.assertEqual(errors.missing_keys, ['wte.new_embedding'])
        self.assertEqual(errors.unexpected_keys, [])
        return hf, native, converted

    def test_converted_logits_match_huggingface(self):
        hf, native, converted = self.tiny_models()
        ids = torch.tensor([[3, 7, 1, 11, 4], [8, 9, 17, 6, 2]])
        with torch.no_grad():
            expected = hf(ids).logits
            x = native.wte(ids)
            positions = torch.arange(ids.shape[1]).unsqueeze(0).expand(ids.shape[0], -1)
            bias = causal_attention_bias(ids.shape[1], torch.device('cpu'))
            for block in native.blocks:
                x, _ = block(x, position_ids=positions, attention_bias=bias)
            actual = native.ff_out(native.ln_f(x))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertIn('transformer.blocks.0.att_proj.bias', converted)
        self.assertIn('transformer.ff_out.weight', converted)
        self.assertFalse(any('q_norm' in key or 'k_norm' in key for key in converted))

    def test_pretrained_load_and_lora(self):
        _, native, converted = self.tiny_models()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'qwen25.pt'
            torch.save(converted, path)
            from olmo.nn.peft import LoRAConfig
            cfg = replace(native.config, init_path=str(path), lora=LoRAConfig(rank=2))
            adapted = Llm(cfg, BufferCache(), device='cpu')
            adapted.reset_with_pretrained_weights()
            self.assertEqual(sum(isinstance(module, LoRALinear) for module in adapted.modules()), 4)
            for name, value in converted.items():
                torch.testing.assert_close(adapted.state_dict()[name.removeprefix('transformer.')], value)
            for name, parameter in adapted.named_parameters():
                if name.endswith('lora_B'):
                    self.assertEqual(torch.count_nonzero(parameter).item(), 0)

    def test_reject_unexpected_parameters(self):
        hf, native, _ = self.tiny_models()
        source = dict(hf.state_dict())
        source['unexpected.weight'] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, 'Unused Qwen2.5'):
            convert_state_dict_qwen25(source, native.config, native.config.block_type)
        source = dict(hf.state_dict())
        source.pop('model.layers.0.self_attn.q_proj.bias')
        with self.assertRaises(KeyError):
            convert_state_dict_qwen25(source, native.config, native.config.block_type)

    def test_stage1_and_stage2_config_inheritance(self):
        path = ROOT / 'configs/experiments/stage1_pixmo_cap_qwen25_7b.yaml'
        stage1 = build_experiment(load_experiment(path))
        self.assertEqual(stage1.model.llm.d_model, 3584)
        self.assertEqual(stage1.model.llm.tokenizer.identifier, 'Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(stage1.peft.mode, 'connector_only')
        default = ROOT / 'configs/experiments/stage1_pixmo_cap.yaml'
        previous = build_experiment(load_experiment(default, ['model.llm=qwen2.5_7b_instruct']))
        self.assertEqual(OmegaConf.to_container(OmegaConf.structured(stage1.optimizer)),
                         OmegaConf.to_container(OmegaConf.structured(previous.optimizer)))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'step22347'
            checkpoint.mkdir()
            OmegaConf.save(OmegaConf.structured(stage1), checkpoint / 'config.yaml', resolve=True)
            stage2path = ROOT / 'configs/experiments/stage2_videoqa_qwen25_7b.yaml'
            stage2 = build_experiment(load_experiment(stage2path, [f'checkpoint={checkpoint}']))
            self.assertEqual(stage2.model.llm.d_model, 3584)
            self.assertEqual(stage2.model.llm.tokenizer.identifier, 'Qwen/Qwen2.5-7B-Instruct')
            self.assertFalse(stage2.model.llm.weight_tying)
            self.assertTrue(stage2.model.llm.qkv_bias)
            self.assertIsNotNone(stage2.model.llm.lora)
            self.assertEqual(stage2.peft.mode, 'connector_lora')
            self.assertEqual(stage2.data.sequence_length, 16384)


if __name__ == '__main__':
    unittest.main()
