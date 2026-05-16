"""
compressor/pruner.py
────────────────────
Structured channel pruning using PyTorch's built-in pruning API.

Why structured (not unstructured)?
  Unstructured pruning zeros individual weights → the weight matrix stays
  the same shape → no actual speedup on real hardware.
  Structured pruning removes entire convolutional CHANNELS → the resulting
  tensors are genuinely smaller → real FLOPs reduction.

Usage:
  python -m compressor.pruner
"""

from __future__ import annotations
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm

from configs.settings import cfg

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_teacher() -> nn.Module:
    """
    Load the Teacher model.
    Supports torchvision pretrained weights or a custom .pth checkpoint.
    """
    arch = cfg.model.teacher_arch
    weights_cfg = cfg.model.teacher_weights
    num_classes = cfg.model.num_classes

    logger.info("Loading teacher architecture: %s", arch)

    if weights_cfg == "pretrained":
        # Use torchvision V2 weights API
        weights_enum = {
            "resnet50":    tvm.ResNet50_Weights.IMAGENET1K_V2,
            "resnet34":    tvm.ResNet34_Weights.DEFAULT,
            "efficientnet_b3": tvm.EfficientNet_B3_Weights.DEFAULT,
        }
        if arch not in weights_enum:
            raise ValueError(f"Unknown teacher arch '{arch}'. Add it to weights_enum.")

        model = tvm.get_model(arch, weights=weights_enum[arch])

        # Replace the classifier head with our task's num_classes
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        # Custom checkpoint path
        model = tvm.get_model(arch, num_classes=num_classes)
        state = torch.load(weights_cfg, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    return model


def _collect_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """Return a flat list of (name, module) for all Conv2d layers."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── Main pruning routine ──────────────────────────────────────────────────────

def prune_teacher(save_path: Path | None = None) -> nn.Module:
    """
    Apply L1-norm structured channel pruning across all Conv2d layers.

    Steps:
      1. Load teacher
      2. Apply pruning masks (weights are zeroed, but not yet removed)
      3. Make pruning permanent (remove pruning reparametrization)
      4. Save pruned model

    Returns the pruned model.
    """
    ratio = cfg.compression.pruning_ratio
    save_path = save_path or (cfg.output_dir / "teacher_pruned.pth")

    model = _load_teacher()
    params_before = _count_parameters(model)
    logger.info("Parameters before pruning: %d", params_before)

    conv_layers = _collect_conv_layers(model)
    logger.info("Found %d Conv2d layers to prune at ratio %.2f", len(conv_layers), ratio)

    # ── Apply pruning to each conv layer ──────────────────────────────────────
    for name, module in conv_layers:
        # L1 structured pruning zeroes entire output-channel slices (dim=0).
        prune.ln_structured(
            module,
            name="weight",
            amount=ratio,
            n=1,          # L1 norm
            dim=0,        # prune along output channels
        )

        # To maintain consistency, if we prune an output channel in weights,
        # we MUST zero the corresponding bias element.
        if module.bias is not None:
            # We use the mask from the weight pruning to zero the bias
            # since ln_structured doesn't directly support bias.
            mask = getattr(module, "weight_mask")
            # weight_mask shape is [out_channels, in_channels, k, k]
            # If a row in weight_mask is all zeros, the channel is pruned.
            # We can take the max along dims 1,2,3 to get a [out_channels] mask.
            bias_mask = (mask.sum(dim=(1, 2, 3)) != 0).to(module.bias.dtype)
            module.bias.data *= bias_mask

    # ── Make pruning permanent ─────────────────────────────────────────────────
    # Without this step the pruning masks are stored separately from weights.
    # remove() merges mask * weight into the weight tensor permanently.
    for name, module in conv_layers:
        prune.remove(module, "weight")

    params_after = _count_parameters(model)
    sparsity = 1.0 - (params_after / params_before)
    logger.info(
        "Parameters after pruning: %d  |  Effective sparsity: %.1f%%",
        params_after,
        sparsity * 100,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), save_path)
    logger.info("Pruned teacher saved → %s", save_path)

    return model


if __name__ == "__main__":
    logging.basicConfig(level=cfg.logging.level)
    pruned = prune_teacher()
    logger.info("Pruning complete.")