"""
LoRA (Low-Rank Adaptation) utilities for fine-tuning the Wan DiT backbone.

Provides:
    - LoRALinear: Drop-in nn.Linear replacement with low-rank trainable adapter.
    - inject_lora_into_wan_model: Inject LoRA adapters into WanModel attention layers.
    - prepare_lora_training: Freeze base weights, unfreeze only LoRA parameters.
    - enable_gradient_checkpointing: Trade compute for memory via activation checkpointing.
    - save_lora_weights / load_lora_weights: Checkpoint only LoRA parameters.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    Effective weight: W_frozen + (lora_B @ lora_A) * scaling
    Only lora_A and lora_B are trainable; the original weight is frozen.

    Args:
        original: The nn.Linear layer to wrap.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor (effective scaling = alpha / rank).
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.register_buffer("frozen_weight", original.weight.data.clone(), persistent=False)
        if original.bias is not None:
            self.register_buffer("frozen_bias", original.bias.data.clone(), persistent=False)
        else:
            self.frozen_bias = None

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.frozen_weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.frozen_bias)


def inject_lora_into_wan_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """Inject LoRA adapters into the DiT backbone's attention layers.

    Replaces specified nn.Linear sub-modules in each WanAttentionBlock with
    LoRALinear wrappers.  Both self_attn and cross_attn are targeted.

    Args:
        model: WanModel instance (must have .blocks attribute).
        rank: LoRA rank (lower = fewer params, higher = more capacity).
        alpha: LoRA scaling factor.
        target_modules: Attribute names to wrap inside each attention module.
            Defaults to ["q", "k", "v", "o"].

    Returns:
        Total number of LoRA parameters added.
    """
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]

    lora_params = 0
    for block in model.blocks:
        for attn_name in ["self_attn", "cross_attn"]:
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for target in target_modules:
                original = getattr(attn, target, None)
                if original is None or not isinstance(original, nn.Linear):
                    continue
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                setattr(attn, target, lora_layer)
                lora_params += rank * (original.in_features + original.out_features)

    return lora_params


def prepare_lora_training(model: nn.Module) -> int:
    """Freeze all parameters except LoRA adapters.

    Args:
        model: WanModel with injected LoRA layers.

    Returns:
        Total number of trainable LoRA parameters.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    lora_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            lora_count += param.numel()

    return lora_count


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on all transformer blocks.

    Trades ~30% extra compute for ~60% memory reduction — essential for
    large models (14B) and helpful for 1.3B when batch size is constrained.

    Args:
        model: WanModel whose blocks should use activation checkpointing.
    """
    from torch.utils.checkpoint import checkpoint

    original_forward = model.blocks[0].__class__.forward

    def checkpointed_forward(self, *args, **kwargs):
        return checkpoint(original_forward, self, *args, use_reentrant=False, **kwargs)

    for block in model.blocks:
        block.forward = checkpointed_forward.__get__(block, block.__class__)


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights to disk.

    Args:
        model: Model with injected LoRA layers.
        path: File path (typically .pt or .pth).
    """
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA adapter weights from disk.

    Args:
        model: Model with injected LoRA layers (same architecture as when saved).
        path: File path saved by save_lora_weights().
    """
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(lora_state, strict=False)
