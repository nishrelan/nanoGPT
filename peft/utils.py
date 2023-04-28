import torch
import torch.nn as nn




def _get_submodules(model: nn.Module, key: str):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name