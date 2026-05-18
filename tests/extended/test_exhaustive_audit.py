import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

# ─── Internal Modules ────────────────────────────────────────────────────────
from configs.settings import load_settings, cfg
from compressor.pruner import _collect_conv_layers, _count_parameters, prune_teacher
from compressor.distiller import DistillationLoss, _build_student
from compressor.quantizer import export_onnx_fp32, quantize
from benchmark.runner import image_to_pixels, evaluate_gates, wait_for_server
from benchmark.gate import check_gate

# ─────────────────────────────────────────────────────────────────────────────
# 1. API AUDIT: Internal Logic & Parameter Integrity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T", [0.01, 1.0, 5.0, 20.0, 100.0])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_classes", [2, 10, 100])
def test_distillation_loss_exhaustive(T, alpha, batch_size, num_classes):
    """
    AUDIT: DistillationLoss.__init__ and forward()
    Verifies: Temperature scaling, alpha weighting, and dimension compatibility.
    """
    criterion = DistillationLoss(T=T, alpha=alpha)
    s_logits = torch.randn(batch_size, num_classes)
    t_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss, stats = criterion(s_logits, t_logits, labels)
    assert not torch.isnan(loss), f"NaN loss with T={T}, alpha={alpha}"
    assert loss >= 0

# ─── Exhaustive Parameter Matrix for load_settings ───────────────────────────

@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])
@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "mobilenet_v3_small"])
@pytest.mark.parametrize("pr", [0.1, 0.5, 0.9])
def test_config_parameter_permutations(task, arch, pr):
    """
    AUDIT: load_settings parsing and dataclass instantiation.
    """
    raw_mock = {
        "model": {"task": task, "teacher_arch": arch, "teacher_weights": "none", "student_arch": "mobilenet", "dataset": "cifar10", "num_classes": 10, "data_dir": "./data", "epochs": 1, "batch_size": 32, "learning_rate": 0.01},
        "compression": {"pruning_ratio": pr, "quantization_mode": "int8", "calibration_samples": 10, "distillation_temperature": 4.0, "distillation_alpha": 0.5},
        "deployment": {"target_arch": "arm64", "latency_gate_ms": 50, "accuracy_threshold": 0.8, "memory_limit_mb": 512, "server_port": 8080, "input_height": 224, "input_width": 224, "input_channels": 3},
        "logging": {"level": "INFO", "output_dir": "./outputs", "benchmark_json": "./outputs/b.json", "dashboard_port": 5000}
    }
    with patch("yaml.safe_load", return_value=raw_mock), patch("builtins.open", MagicMock()):
        settings = load_settings("dummy.yaml")
        assert settings.model.task == task
        assert settings.compression.pruning_ratio == pr

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTERNAL API COMPATIBILITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("img_size", [(10, 10), (100, 100), (1000, 1000)])
@pytest.mark.parametrize("target_h, target_w", [(224, 224), (32, 32), (640, 640)])
def test_image_to_pixels_compatibility(img_size, target_h, target_w):
    """
    AUDIT: PIL.Image -> NumPy -> List conversion used in benchmark runner.
    """
    img = Image.new("RGB", img_size, color=(255, 255, 255))
    pixels = image_to_pixels(img, target_h, target_w)
    assert len(pixels) == target_h * target_w * 3

@pytest.mark.parametrize("status_code", [200, 400, 404, 500])
def test_requests_post_audit(status_code):
    """
    AUDIT: requests.post argument compatibility and error handling.
    """
    from benchmark.runner import run_latency_benchmark
    with patch("requests.post") as m:
        m.return_value.status_code = status_code
        # Fixed: Included 'memory_rss_mb' in mock response
        m.return_value.json.return_value = {
            "mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0, "fps": 1.0, "memory_rss_mb": 50.0
        }
        if status_code == 200:
            run_latency_benchmark("localhost", 8080, 1)
        else:
            with pytest.raises(Exception):
                run_latency_benchmark("localhost", 8080, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTO-GENERATED MATRIX (Targeting 300+ Cases)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("i", range(100))
def test_gate_evaluation_jitter(i):
    """AUDIT: evaluate_gates logic against probabilistic input noise."""
    l_stats = {"p95_ms": float(10.0 + i), "mean_ms": 5.0} 
    a_stats = {"accuracy": float(0.95 - (i/200.0))}        
    
    with patch("configs.settings.cfg.deployment.latency_gate_ms", 50), \
         patch("configs.settings.cfg.deployment.accuracy_threshold", 0.8):
        passed, failures = evaluate_gates(l_stats, a_stats)
        
        expected_pass = (10.0 + i <= 50) and (0.95 - (i/200.0) >= 0.8)
        assert passed == expected_pass

@pytest.mark.parametrize("depth", range(1, 101))
def test_collect_layers_depth_audit(depth):
    """AUDIT: Recursion limit and structure traversal for torch.nn.Sequential."""
    layers = []
    for _ in range(depth):
        layers.append(nn.Conv2d(3, 3, 3))
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    
    found = _collect_conv_layers(model)
    assert len(found) == depth

@pytest.mark.parametrize("p95", np.linspace(10, 100, 50))
def test_ci_gate_exhaustive(p95):
    """AUDIT: check_gate file I/O and JSON schema validation."""
    # Ensure all values are standard Python types (float/bool) to avoid numpy JSON serialization issues
    is_passed = bool(p95 <= 50)
    report = {
        "latency": {"p95_ms": float(p95)},
        "accuracy": {"accuracy": 0.85},
        "passed": is_passed, 
        "failures": []
    }
    path = Path(f"temp_gate_{p95}.json")
    with open(path, "w") as f: json.dump(report, f)
    try:
        with patch("configs.settings.cfg.deployment.latency_gate_ms", 50), \
             patch("configs.settings.cfg.deployment.accuracy_threshold", 0.8):
            passed, _ = check_gate(path)
            assert passed == is_passed
    finally:
        if path.exists(): path.unlink()
