"""Trainable scopes, LoRA gradients and checkpoint transitions on CPU."""
from dataclasses import replace
from itertools import chain
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from olmo.nn.peft import (LoRAConfig, PeftConfig, LoRALinear, apply_peft_policy,
                          install_lora, reset_lora_parameters, is_lora_key,
                          checkpoint_lora_warm_start, check_missing_lora_keys)
from olmo.experiment_config import load_experiment, build_experiment
from olmo.model_configs import QWEN3_4B_INSTRUCT
from olmo.nn.llm import Llm
from olmo.torch_util import BufferCache
from olmo.train.optim import OptimizerConfig
from olmo.train.distributed_checkpointing import save_model_and_optim_state, load_model_and_optim_state
from olmo.train.checkpointer import load_model_state_unsharded


class Decoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.att_proj = nn.Linear(dim, dim * 3)
        for name in ('attn_out', 'ff_out'):
            setattr(self, name, nn.Linear(dim, dim))

    def forward(self, x):
        q, k, v = self.att_proj(x).chunk(3, dim=-1)
        return self.ff_out(self.attn_out(torch.tanh(q + k + v)))


class TinyModel(nn.Module):
    def __init__(self, lora=None, device=None):
        super().__init__()
        with torch.device(device or 'cpu'):
            self.vision_backbone = nn.Module()
            for name in ('image_vit', 'segmenter', 'query_proj', 'cross_attn', 'image_projector'):
                setattr(self.vision_backbone, name, nn.Linear(8, 8))
            self.transformer = nn.Module()
            self.transformer.blocks = nn.ModuleList([Decoder()])
            self.transformer.wte = nn.Embedding(10, 8)
            self.transformer.ln_f = nn.LayerNorm(8)
            if lora is not None:
                install_lora(self.transformer.blocks, lora)

    def forward(self, x):
        v = self.vision_backbone
        x = v.image_vit(x) + v.segmenter(x)
        x = v.image_projector(v.cross_attn(v.query_proj(x)))
        return self.transformer.ln_f(self.transformer.blocks[0](x))

    def get_llm_parameters(self):
        return self.transformer.parameters()

    def get_connector_parameters(self):
        return chain(*(getattr(self.vision_backbone, n).parameters()
                       for n in ('segmenter', 'query_proj', 'cross_attn', 'image_projector')))

    def get_vit_parameters(self):
        return self.vision_backbone.image_vit.parameters()

    def get_frame_selection_parameters(self):
        return []

    def get_non_weight_decay_params(self):
        return []

    def get_legacy_key_mapping(self):
        return None


class PeftTests(unittest.TestCase):
    def test_stage1_trainable_scope_and_update(self):
        self.check_updates('connector_only')

    def test_stage2_trainable_scope_and_update(self):
        self.check_updates('connector_lora')

    def check_updates(self, mode):
        lora = LoRAConfig(rank=2)
        model = TinyModel(lora if mode == 'connector_lora' else None)
        selected = apply_peft_policy(model, PeftConfig(mode, lora))
        expected = {name for name, _ in model.named_parameters()
                    if name.startswith(('vision_backbone.query_proj.', 'vision_backbone.cross_attn.',
                                        'vision_backbone.image_projector.'))
                    or (mode == 'connector_lora' and is_lora_key(name))}
        self.assertEqual(set(selected), expected)
        groups = OptimizerConfig().get_param_groups(1, None, model)
        self.assertEqual({id(p) for group in groups for p in group['params']},
                         {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({g['group_name'] for g in groups},
                         {'connector', 'llm'} if mode == 'connector_lora' else {'connector'})
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optim = torch.optim.AdamW(groups)
        for _ in range(2):
            optim.zero_grad(set_to_none=True)
            model(torch.randn(3, 8)).square().mean().backward()
            for name, p in model.named_parameters():
                self.assertEqual(p.grad is not None, name in expected, name)
            optim.step()
        for name, p in model.named_parameters():
            if name not in expected:
                self.assertTrue(torch.equal(before[name], p), name)
        for scope in ('vision_backbone.query_proj.', 'vision_backbone.cross_attn.', 'vision_backbone.image_projector.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(scope)))
        if mode == 'connector_lora':
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if is_lora_key(n)))

    def test_zero_delta_preserved_base_keys_and_strict_roundtrip(self):
        base = nn.Linear(7, 5)
        x = torch.randn(3, 7)
        expected = base(x).detach()
        lora = LoRALinear(base, LoRAConfig(rank=2))
        self.assertIs(lora.weight, base.weight)
        torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)
        self.assertEqual(set(lora.state_dict()), {'weight', 'bias', 'lora_A', 'lora_B'})
        lora.lora_B.data.normal_()
        clone = LoRALinear(nn.Linear(7, 5), LoRAConfig(rank=2))
        clone.load_state_dict(lora.state_dict(), strict=True)
        torch.testing.assert_close(clone(x), lora(x))

    def test_meta_materialization_initializes_adapters(self):
        model = TinyModel(LoRAConfig(rank=2), device='meta')
        model.to_empty(device='cpu')
        reset_lora_parameters(model)
        for name, p in model.named_parameters():
            if name.endswith('lora_B'):
                self.assertEqual(torch.count_nonzero(p).item(), 0)
            elif name.endswith('lora_A'):
                self.assertTrue(torch.isfinite(p).all())
                self.assertGreater(torch.count_nonzero(p).item(), 0)

    def test_real_qwen_blocks_and_reset(self):
        config = replace(QWEN3_4B_INSTRUCT, d_model=16, n_heads=2, n_kv_heads=1,
                         head_dim=8, n_layers=1, mlp_hidden_size=32, vocab_size=20,
                         embedding_size=20, additional_vocab_size=2, init_path=None,
                         lora=LoRAConfig(rank=2))
        model = Llm(config, BufferCache(), device='cpu')
        self.assertEqual(sum(isinstance(m, LoRALinear) for m in model.modules()), 2)
        self.assertFalse(any(is_lora_key(n) for n, _ in model.wte.named_parameters()))
        model.reset_with_pretrained_weights()
        for name, p in model.named_parameters():
            if is_lora_key(name):
                self.assertTrue(torch.isfinite(p).all())
                if name.endswith('lora_B'):
                    self.assertEqual(torch.count_nonzero(p).item(), 0)

    def test_invalid_config_and_missing_targets(self):
        for config in (LoRAConfig(rank=0), LoRAConfig(alpha=-1), LoRAConfig(dropout=1),
                       LoRAConfig(target_modules=[]), LoRAConfig(target_modules=['q_proj', 'q_proj']),
                       LoRAConfig(target_modules=['unknown'])):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        with self.assertRaisesRegex(ValueError, 'absent'):
            install_lora(nn.ModuleList([nn.Linear(8, 8)]), LoRAConfig())
        with self.assertRaisesRegex(ValueError, 'cannot contain'):
            apply_peft_policy(TinyModel(LoRAConfig()), PeftConfig('connector_only'))

    def test_checkpoint_policy_validation(self):
        config = PeftConfig('connector_lora')
        stage1 = OmegaConf.create({'model': {'llm': {'lora': None}}, 'peft': {'mode': 'connector_only'}})
        self.assertTrue(checkpoint_lora_warm_start(config, config.lora, stage1, resuming=False))
        with self.assertRaises(ValueError):
            checkpoint_lora_warm_start(config, config.lora, stage1, resuming=True)
        stage2 = OmegaConf.create({'model': {'llm': {'lora': OmegaConf.structured(config.lora)}},
                                  'peft': OmegaConf.structured(config)})
        self.assertFalse(checkpoint_lora_warm_start(config, config.lora, stage2, resuming=True))
        with self.assertRaises(ValueError):
            checkpoint_lora_warm_start(PeftConfig('connector_lora', LoRAConfig(rank=4)),
                                       LoRAConfig(rank=4), stage2, resuming=True)
        with self.assertRaises(RuntimeError):
            check_missing_lora_keys(['transformer.blocks.0.att_proj.weight'], allow_missing_lora=True)
        with self.assertRaises(RuntimeError):
            check_missing_lora_keys(['transformer.blocks.0.att_proj.lora_A'], allow_missing_lora=False)

    def test_recipe_changes_only_peft_fields(self):
        with patch.dict(os.environ, MOLMO_DATA_DIR='/tmp/trajvlm-test-data'):
            for stage, mode in (('stage1_pixmo_cap', 'connector_only'), ('stage2_videoqa', 'connector_lora')):
                args = [] if mode == 'connector_only' else [f'checkpoint={ROOT}/tests/fixtures/pretrain_checkpoint']
                path = ROOT / f'configs/experiments/{stage}.yaml'
                new = build_experiment(load_experiment(path, args))
                previous = build_experiment(load_experiment(path, [*args, 'peft=null']))
                a = OmegaConf.to_container(OmegaConf.structured(new), resolve=True)
                b = OmegaConf.to_container(OmegaConf.structured(previous), resolve=True)
                for key in ('peft', 'ft_llm', 'ft_vit', 'ft_connector'):
                    a.pop(key); b.pop(key)
                a['model']['llm'].pop('lora', None); b['model']['llm'].pop('lora', None)
                self.assertEqual(a, b)
                self.assertFalse(new.ft_vit)
                self.assertEqual(new.ft_llm, mode == 'connector_lora')
                self.assertEqual(new.peft.mode, mode)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.group_file = Path(cls.tmp.name) / 'group'
        torch.distributed.init_process_group('gloo', init_method=cls.group_file.as_uri(), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()
        cls.tmp.cleanup()

    def test_sharded_stage_transition_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1 = TinyModel()
            path = str(Path(tmp) / 'stage1')
            save_model_and_optim_state(path, stage1)
            stage2 = TinyModel(LoRAConfig(rank=2))
            apply_peft_policy(stage2, PeftConfig('connector_lora', LoRAConfig(rank=2)))
            reset_lora_parameters(stage2)
            with self.assertRaisesRegex(RuntimeError, 'missing required'):
                load_model_and_optim_state(path, stage2)
            load_model_and_optim_state(path, stage2, allow_missing_lora=True)
            for name, value in stage1.state_dict().items():
                torch.testing.assert_close(stage2.state_dict()[name], value)
            for name, p in stage2.named_parameters():
                if name.endswith('lora_B'):
                    self.assertEqual(torch.count_nonzero(p).item(), 0)
            groups = OptimizerConfig().get_param_groups(1, None, stage2)
            opt = torch.optim.AdamW(groups)
            stage2(torch.randn(2, 8)).square().mean().backward()
            opt.step()
            resume = str(Path(tmp) / 'stage2')
            save_model_and_optim_state(resume, stage2, opt)
            restored = TinyModel(LoRAConfig(rank=2))
            apply_peft_policy(restored, PeftConfig('connector_lora', LoRAConfig(rank=2)))
            opt2 = torch.optim.AdamW(OptimizerConfig().get_param_groups(1, None, restored))
            load_model_and_optim_state(resume, restored, opt2)
            for name, value in stage2.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)
            self.assertEqual(len(opt.state), len(opt2.state))
            self.assertEqual(len(opt.param_groups), len(opt2.param_groups))

    def test_unsharded_stage_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1 = TinyModel()
            torch.save(stage1.state_dict(), Path(tmp) / 'model.pt')
            stage2 = TinyModel(LoRAConfig(rank=2))
            reset_lora_parameters(stage2)
            load_model_state_unsharded(tmp, stage2, allow_missing_lora=True)
            for name, value in stage1.state_dict().items():
                torch.testing.assert_close(stage2.state_dict()[name], value)
            corrupt = stage1.state_dict()
            corrupt.pop('transformer.blocks.0.att_proj.weight')
            torch.save(corrupt, Path(tmp) / 'model.pt')
            with self.assertRaises((RuntimeError, ValueError)):
                load_model_state_unsharded(tmp, stage2, allow_missing_lora=True)


if __name__ == '__main__':
    unittest.main()
