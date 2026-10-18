"""Native LoRA for the custom Molmo LLM and explicit TrajVLM tuning scopes.

Base Linear weight/bias names are preserved for Stage1 -> Stage2 checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from olmo.config import BaseConfig


@dataclass
class LoRAConfig(BaseConfig):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['att_proj', 'attn_out'])

    def validate(self):
        if self.rank <= 0 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive and finite')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('LoRA dropout must be in [0, 1)')
        allowed = {'att_proj', 'q_proj', 'k_proj', 'v_proj', 'attn_out', 'ff_proj', 'ff_proj1', 'ff_proj2', 'ff_out'}
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError('LoRA target_modules must be nonempty and unique')
        if set(self.target_modules) - allowed:
            raise ValueError(f'Unsupported LoRA target_modules: {set(self.target_modules) - allowed}')


@dataclass
class PeftConfig(BaseConfig):
    mode: str = 'connector_only'
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def validate(self):
        if self.mode not in ('connector_only', 'connector_lora'):
            raise ValueError('peft.mode must be connector_only or connector_lora')
        self.lora.validate()


def is_lora_key(name):
    return name.rsplit('.', 1)[-1] in ('lora_A', 'lora_B')


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        # Do not allocate/copy/reinitialize the base weight (also works on meta).
        nn.Module.__init__(self)
        config.validate()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.scaling = config.alpha / config.rank
        self.lora_dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(config.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_empty(base.out_features, config.rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(
            F.linear(self.lora_dropout(x), self.lora_A), self.lora_B
        ) * self.scaling


def install_lora(blocks, config):
    """Only decoder blocks are visited: embeddings and LM head stay unchanged."""
    config.validate()
    targets = config.target_modules
    matches = {target: [] for target in targets}
    for name, module in list(blocks.named_modules()):
        if name.rsplit('.', 1)[-1] in targets:
            if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
                raise ValueError(f'LoRA target is not a plain Linear: {name}')
            matches[name.rsplit('.', 1)[-1]].append((name, module))
    absent = [target for target, found in matches.items() if not found]
    if absent:
        raise ValueError(f'LoRA targets absent from this LLM: {absent}')
    for found in matches.values():
        for name, module in found:
            parent_name, _, leaf = name.rpartition('.')
            parent = blocks.get_submodule(parent_name) if parent_name else blocks
            setattr(parent, leaf, LoRALinear(module, config))


def reset_lora_parameters(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_lora_parameters()


def apply_peft_policy(model, config):
    """Freeze everything, then enable only the explicitly named tuning scopes."""
    config.validate()
    backbone = model.vision_backbone
    if not all(hasattr(backbone, name) for name in ('cross_attn', 'image_projector', 'segmenter', 'query_proj')):
        raise ValueError('PEFT connector policy requires the SigLip2 TrajVLM backbone')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (backbone.cross_attn, backbone.image_projector):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    adapters = [(name, parameter) for name, parameter in model.transformer.named_parameters() if is_lora_key(name)]
    if config.mode == 'connector_lora':
        if not adapters:
            raise ValueError('connector_lora requires LoRA modules in model.llm.lora')
        for _, parameter in adapters:
            parameter.requires_grad_(True)
    elif adapters:
        raise ValueError('connector_only cannot contain LLM LoRA modules')
    return {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}


def checkpoint_lora_warm_start(config, current_lora, saved_config, *, resuming):
    """Only a new Stage2 run may initialize adapters absent from Stage1."""
    from omegaconf import OmegaConf
    saved_lora = OmegaConf.select(saved_config, 'model.llm.lora')
    current = OmegaConf.structured(current_lora) if current_lora is not None else None
    if resuming:
        saved_peft = OmegaConf.select(saved_config, 'peft')
        if saved_peft is None or OmegaConf.merge(OmegaConf.structured(PeftConfig), saved_peft) != OmegaConf.structured(config):
            raise ValueError('Cannot change PEFT policy when resuming; start a new experiment')
    if saved_lora is None:
        if current is not None and resuming:
            raise ValueError('Cannot resume LoRA from a checkpoint without adapters')
        return current is not None
    if current is None or OmegaConf.merge(OmegaConf.structured(LoRAConfig), saved_lora) != current:
        raise ValueError('Checkpoint LoRA configuration differs; rank/alpha/dropout/targets must match')
    return False


def check_missing_lora_keys(missing_keys, *, allow_missing_lora):
    invalid = [key for key in missing_keys if not (allow_missing_lora and is_lora_key(key))]
    if invalid:
        raise RuntimeError(f'Checkpoint is missing required model keys: {invalid}')
