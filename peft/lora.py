import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from .utils import _get_submodules


# Below code is based on https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py
# and modified to work with nano-GPT models

#  ------------------------------------------------------------------------------------------
#  Copyright 2023-present the HuggingFace Inc. team.
#  Licensed under the Apache License, Version 2.0 (the "License").
#  ------------------------------------------------------------------------------------------


@dataclass
class LoraConfig:
    """
    This is the configuration class to store the configuration of a [`LoraModel`].
    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )


class LoraModel(nn.Module):
    """
    Creates LoRA model from an existing nanoGPT model (in practice this model
    will always be pretrained). For now we modify just attention weights.
    This means we modify .attn.c_attn

    TODO: Change the model and replace targeted modules (linear attention layers) with lora layers
    """

    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.config = config
        # quick hack while testing lora code. Eventually config 
        # param will be properly initialized
        if self.config is None:
            self.config = LoraConfig(lora_alpha=8, lora_dropout=0.02)
        blocks = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "CausalSelfAttention":
                blocks.append(module)
        
        for block in blocks:
            parent, target, target_name = _get_submodules(block, 'c_attn')
            self._replace_module(parent, target, target_name)
        
        

    # replace linear layer with lora linear layer
    def _replace_module(self, parent: nn.Module, target: nn.Linear, target_name: str):
        in_features = target.in_features
        out_features = target.out_features
        r = self.config.r
        alpha = self.config.lora_alpha
        dropout = self.config.lora_dropout
        new_module = LoraLinear(in_features, out_features, r, alpha, dropout)

        setattr(parent, target_name, new_module)
        new_module.base_linear.weight = target.weight

        # put module on same device
        new_module.to(target.weight.device)

        # I don't understand why this is necessary... But I can't see how it hurts so gonna leave
        # it in just in case and debug later
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(target.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------



class LoraLinear(nn.Module):
    
    def __init__(self, in_features, out_features, r, alpha, dropout) -> None:
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

        self.lora_A = nn.Linear(in_features=in_features, out_features=r, bias=False)
        self.lora_B = nn.Linear(in_features=r, out_features=out_features, bias=False)
        # initialize B to zeros as in paper
        with torch.no_grad():
            self.lora_B.weight.data = torch.zeros_like(self.lora_B.weight.data)

        self.lora_dropout = nn.Dropout(p=self.dropout)

        self.merged = False
        self.scaling = self.alpha / self.r

    def forward(self, x: torch.Tensor):
        if self.merged:
            return self.base_linear(x)
        else:
            base = self.base_linear(x)
            lora_x = self.lora_B(self.lora_A(self.lora_dropout(x)))
            lora_x = lora_x * self.scaling
            return base + lora_x


    def merge(self):
        if not self.merged:
            return
        with torch.no_grad():
            self.base_linear.weight += self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.merged = True
    
    def unmerge(self):
        if self.merged:
            return
        with torch.no_grad():
            self.base_linear.weight -= self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.merged = False
    
    

# had to adapt it for `lora_only` to work
# TODO: Update this -- this will likely be wrong for my implementation
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
