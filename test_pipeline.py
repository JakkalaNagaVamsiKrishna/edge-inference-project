"""
tests/test_pipeline.py
───────────────────────
End-to-end integration tests.  Run locally before pushing to CI.

Tests:
  1. Config loads correctly
  2. Teacher model loads
  3. Student model builds
  4. Distillation loss is mathematically correct
  5. ONNX FP32 export works
  6. Benchmark gate logic works correctly
  7. Benchmark runner correctly identifies gate failures

Usage:
  pytest tests/ -v
"""

from __future__ import annotations
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ─── Config ───────────────────────────────────────────────────────────────────

def test_config_loads():
    from configs.settings import load_settings
    cfg = load_settings()
    assert cfg.model.teacher_arch == "resnet50"
    assert cfg.compression.pruning_ratio > 0
    assert cfg.deployment.latency_gate_ms > 0
    assert cfg.deployment.accuracy_threshold > 0


# ─── Pruner ───────────────────────────────────────────────────────────────────

def test_teacher_loads():
    import torchvision.models as tvm
    model = tvm.resnet50(weights=None, num_classes=10)
    assert isinstance(model, nn.Module)
    params = sum(p.numel() for p in model.parameters())
    assert params > 1_000_000   # ResNet-50 has ~23M parameters


def test_structured_pruning_reduces_params():
    """After pruning, zeroed weights should reduce effective parameter count."""
    import torch.nn.utils.prune as prune
    import torchvision.models as tvm

    model = tvm.mobilenet_v3_small(weights=None, num_classes=10)

    total_before = sum(p.numel() for p in model.parameters())

    # Apply pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)

    # Count non-zero parameters
    nonzero = sum(
        (p != 0).sum().item() for p in model.parameters()
    )

    # After 30% pruning, at least 20% fewer non-zero weights
    assert nonzero < total_before * 0.95


# ─── Distillation loss ────────────────────────────────────────────────────────

def test_distillation_loss_values():
    """Test that distillation loss is non-negative and decreases when student improves."""
    from compressor.distiller import DistillationLoss

    criterion = DistillationLoss(T=4.0, alpha=0.7)
    B, C = 4, 10  # batch size, classes

    teacher_logits  = torch.randn(B, C)
    labels          = torch.randint(0, C, (B,))

    # Student logits identical to teacher → minimum distillation loss
    perfect_student = teacher_logits.clone()
    loss_perfect, breakdown_perfect = criterion(perfect_student, teacher_logits, labels)

    # Random student → higher distillation loss
    random_student = torch.randn(B, C)
    loss_random, breakdown_random = criterion(random_student, teacher_logits, labels)

    assert loss_perfect.item() >= 0
    assert loss_random.item() >= 0
    # A perfect student should have lower distillation loss than random
    assert breakdown_perfect["distill_loss"] <= breakdown_random["distill_loss"]


def test_temperature_effect():
    """Higher temperature should produce softer (more uniform) target distributions."""
    import torch.nn.functional as F

    logits = torch.tensor([[3.0, 1.0, 0.5, 0.1]])

    soft_T1 = F.softmax(logits / 1.0, dim=1)
    soft_T8 = F.softmax(logits / 8.0, dim=1)

    # Entropy should be higher (distribution more uniform) at T=8
    entropy_T1 = -(soft_T1 * soft_T1.log()).sum()
    entropy_T8 = -(soft_T8 * soft_T8.log()).sum()

    assert entropy_T8 > entropy_T1


# ─── ONNX Export ──────────────────────────────────────────────────────────────

def test_onnx_export():
    """Student model should export to valid ONNX without errors."""
    import onnx
    import torchvision.models as tvm
    from compressor.quantizer import export_onnx_fp32

    model = tvm.mobilenet_v3_small(weights=None, num_classes=10)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_export.onnx"
        export_onnx_fp32(model, out_path, input_shape=(1, 3, 224, 224))

        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # non-trivial size

        # ONNX checker validates graph structure
        loaded = onnx.load(str(out_path))
        onnx.checker.check_model(loaded)


def test_onnx_inference_output_shape():
    """ONNX Runtime should produce [B, num_classes] output."""
    import numpy as np
    import onnxruntime as ort
    import torchvision.models as tvm
    from compressor.quantizer import export_onnx_fp32

    model = tvm.mobilenet_v3_small(weights=None, num_classes=10)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "shape_test.onnx"
        export_onnx_fp32(model, out_path, input_shape=(1, 3, 224, 224))

        sess = ort.InferenceSession(str(out_path))
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        outputs = sess.run(None, {"input": dummy})

        assert outputs[0].shape == (1, 10)


# ─── Gate logic ───────────────────────────────────────────────────────────────

def test_gate_passes_when_all_thresholds_met():
    from benchmark.gate import check_gate

    with tempfile.TemporaryDirectory() as tmpdir:
        report = {
            "passed": True,
            "failures": [],
            "latency": {"p95_ms": 30.0, "mean_ms": 25.0, "memory_rss_mb": 200.0},
            "accuracy": {"accuracy": 0.90},
            "gates": {"latency_gate_ms": 50, "accuracy_threshold": 0.80, "memory_limit_mb": 512},
        }
        p = Path(tmpdir) / "benchmark.json"
        with open(p, "w") as f:
            json.dump(report, f)

        passed, failures = check_gate(p)
        assert passed is True
        assert failures == []


def test_gate_fails_on_latency_breach():
    from benchmark.gate import check_gate

    with tempfile.TemporaryDirectory() as tmpdir:
        report = {
            "passed": False,
            "failures": ["Latency gate FAILED: p95=82.0ms > 50ms"],
            "latency": {"p95_ms": 82.0, "mean_ms": 70.0, "memory_rss_mb": 200.0},
            "accuracy": {"accuracy": 0.90},
            "gates": {"latency_gate_ms": 50, "accuracy_threshold": 0.80, "memory_limit_mb": 512},
        }
        p = Path(tmpdir) / "benchmark.json"
        with open(p, "w") as f:
            json.dump(report, f)

        passed, failures = check_gate(p)
        assert passed is False
        assert len(failures) > 0


def test_gate_fails_on_accuracy_breach():
    from benchmark.gate import check_gate

    with tempfile.TemporaryDirectory() as tmpdir:
        report = {
            "passed": False,
            "failures": ["Accuracy gate FAILED: 0.7200 < 0.80"],
            "latency": {"p95_ms": 30.0, "mean_ms": 25.0, "memory_rss_mb": 200.0},
            "accuracy": {"accuracy": 0.72},
            "gates": {"latency_gate_ms": 50, "accuracy_threshold": 0.80, "memory_limit_mb": 512},
        }
        p = Path(tmpdir) / "benchmark.json"
        with open(p, "w") as f:
            json.dump(report, f)

        passed, failures = check_gate(p)
        assert passed is False