import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from configs.settings import load_settings, Settings

# ─── Config & Settings Tests ──────────────────────────────────────────────────

@pytest.mark.parametrize("invalid_key", ["model", "compression", "deployment", "logging"])
def test_settings_missing_sections(invalid_key):
    """Test loading settings with missing required top-level sections."""
    raw = {"model": {}, "compression": {}, "deployment": {}, "logging": {}}
    del raw[invalid_key]
    
    with patch("yaml.safe_load", return_value=raw):
        with patch("builtins.open", MagicMock()):
            # The updated load_settings raises specific errors for missing sections/fields
            with pytest.raises((KeyError, TypeError)):
                load_settings("dummy.yaml")

@pytest.mark.parametrize("value, expected_type", [
    (100, int),
    ("INFO", str),
    (0.001, float),
])
def test_settings_type_integrity(value, expected_type):
    """Ensure that loaded settings maintain expected types (basic check)."""
    from configs.settings import cfg
    assert isinstance(cfg.model.epochs, int)
    assert isinstance(cfg.logging.level, str)

def test_settings_env_override():
    """Verify that environment variables can override config paths (if logic exists)."""
    with patch.dict("os.environ", {"EDGE_CV_CONFIG": "nonexistent.yaml"}):
        with pytest.raises(FileNotFoundError):
            load_settings()

# ─── Pruner Edge Case Tests ───────────────────────────────────────────────────

def test_prune_no_conv_layers():
    """Verify pruning logic on a model with no convolutional layers."""
    from compressor.pruner import prune_teacher
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
    
    with patch("compressor.pruner._load_teacher", return_value=model):
        pruned = prune_teacher()
        # Should not crash, and model should remain functional
        dummy = torch.randn(1, 10)
        assert pruned(dummy).shape == (1, 2)

@pytest.mark.parametrize("ratio", [0.0, 0.99])
def test_prune_extreme_ratios(ratio):
    """Test pruning with extreme ratios (0% and 99%)."""
    from compressor.pruner import prune_teacher
    with patch("configs.settings.cfg.compression.pruning_ratio", ratio):
        model = nn.Sequential(nn.Conv2d(3, 16, 3))
        with patch("compressor.pruner._load_teacher", return_value=model):
            pruned = prune_teacher()
            assert isinstance(pruned, nn.Module)

# ─── Distiller Edge Case Tests ────────────────────────────────────────────────

@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("T", [0.1, 50.0])
def test_distillation_loss_edge_params(alpha, T):
    """Test DistillationLoss with boundary values for alpha and Temperature."""
    from compressor.distiller import DistillationLoss
    criterion = DistillationLoss(T=T, alpha=alpha)
    
    s_logits = torch.randn(4, 10)
    t_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    
    loss, stats = criterion(s_logits, t_logits, labels)
    assert loss >= 0
    assert "distill_loss" in stats
    assert "task_loss" in stats

def test_distiller_mismatched_logits():
    """Verify behavior when student and teacher have different output dimensions."""
    from compressor.distiller import DistillationLoss
    criterion = DistillationLoss()
    
    s_logits = torch.randn(4, 5)  # 5 classes
    t_logits = torch.randn(4, 10) # 10 classes
    labels = torch.randint(0, 5, (4,))
    
    with pytest.raises(RuntimeError): # KL Div or Softmax will fail on dim mismatch
        criterion(s_logits, t_logits, labels)

# ─── Quantizer Edge Case Tests ────────────────────────────────────────────────

def test_export_onnx_empty_shape():
    """Test ONNX export with an invalid (empty) input shape."""
    from compressor.quantizer import export_onnx_fp32
    model = nn.Linear(10, 2)
    with pytest.raises(Exception): # Should fail during dummy tensor creation or export
        export_onnx_fp32(model, Path("test.onnx"), input_shape=())

# ─── Benchmark Runner Edge Case Tests ─────────────────────────────────────────

def test_wait_for_server_immediate_fail():
    """Test wait_for_server when the host is unreachable."""
    from benchmark.runner import wait_for_server
    # Use a non-routable IP
    assert wait_for_server("192.0.2.0", 8080, timeout=1) == False

def test_run_latency_benchmark_server_error():
    """Test latency benchmark when the server returns a 500 error."""
    from benchmark.runner import run_latency_benchmark
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 500
        # Our updated run_latency_benchmark now raises an Exception on status != 200
        with pytest.raises(Exception):
            run_latency_benchmark("localhost", 8080, n=10)

# ─── Dashboard Tests ──────────────────────────────────────────────────────────

def test_dashboard_api_missing_file():
    """Test dashboard API when benchmark.json is missing."""
    from dashboard.app import app
    client = app.test_client()
    with patch("pathlib.Path.exists", return_value=False):
        response = client.get("/api/benchmark")
        assert response.status_code == 200
        assert response.json == {}

def test_dashboard_api_malformed_json():
    """Test dashboard API when benchmark.json is malformed."""
    from dashboard.app import app
    client = app.test_client()
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", side_effect=ValueError("Malformed")):
                response = client.get("/api/benchmark")
                assert response.status_code == 500
                assert "error" in response.json
