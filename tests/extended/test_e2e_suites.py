import pytest
import os
import shutil
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Core system orchestrators
from compressor.pipeline import run
from benchmark.runner import main as run_benchmark
from benchmark.gate import check_gate

@pytest.fixture
def e2e_workspace():
    """Sets up a clean temporary workspace for E2E runs."""
    tmpdir = tempfile.mkdtemp()
    data_dir = Path(tmpdir) / "data"
    output_dir = Path(tmpdir) / "outputs"
    data_dir.mkdir()
    output_dir.mkdir()
    config_path = Path(tmpdir) / "test_config.yaml"
    yield {"root": Path(tmpdir), "data": data_dir, "outputs": output_dir, "config": config_path}
    shutil.rmtree(tmpdir)

def create_mock_config(workspace, arch="resnet50", prune_ratio=0.1, gate_ms=100, acc_gate=0.5, weights="pretrained"):
    config = f"""
model:
  task: classification
  teacher_arch: {arch}
  teacher_weights: {weights}
  student_arch: mobilenet_v3_small
  dataset: cifar10
  num_classes: 10
  data_dir: {workspace['data']}
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
compression:
  pruning_ratio: {prune_ratio}
  quantization_mode: int8_ptq
  calibration_samples: 2
  distillation_temperature: 4.0
  distillation_alpha: 0.5
deployment:
  target_arch: x86
  latency_gate_ms: {gate_ms}
  accuracy_threshold: {acc_gate}
  memory_limit_mb: 512
  server_port: 8080
  input_height: 224
  input_width: 224
  input_channels: 3
logging:
  level: INFO
  output_dir: {workspace['outputs']}
  benchmark_json: {workspace['outputs']}/benchmark.json
  dashboard_port: 5000
"""
    with open(workspace['config'], "w") as f: f.write(config)
    return workspace['config']

# 🛡️ GROUP 1: Happy Path
def test_e2e_full_pipeline_standard_resnet(e2e_workspace):
    conf = create_mock_config(e2e_workspace, arch="resnet50", weights="pretrained")
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.pruner.prune_teacher"), \
         patch("compressor.distiller.distill"), \
         patch("compressor.quantizer.quantize") as m_q:
        m_q.return_value = Path("dummy.onnx")
        results = run()
        assert results["stages"]["quantization"] == "success"

def test_e2e_full_pipeline_mobilenet(e2e_workspace):
    conf = create_mock_config(e2e_workspace, arch="resnet50")
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.pruner.prune_teacher"), \
         patch("compressor.distiller.distill"), \
         patch("compressor.quantizer.quantize"):
        results = run()
        assert "stages" in results

def test_e2e_pipeline_no_pruning(e2e_workspace):
    conf = create_mock_config(e2e_workspace)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.distiller.distill"), \
         patch("compressor.quantizer.quantize"), \
         patch("compressor.pruner.prune_teacher") as m_p:
        run(skip_prune=True)
        assert not m_p.called

def test_e2e_pipeline_no_distillation(e2e_workspace):
    conf = create_mock_config(e2e_workspace)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.pruner.prune_teacher"), \
         patch("compressor.quantizer.quantize"), \
         patch("compressor.distiller.distill") as m_d:
        run(skip_distill=True)
        assert not m_d.called

def test_e2e_benchmark_only_workflow(e2e_workspace):
    conf = create_mock_config(e2e_workspace, gate_ms=500, acc_gate=0.1)
    report = {"passed": True, "latency": {"p95_ms": 10}, "accuracy": {"accuracy": 0.5}, "failures": []}
    with open(Path(e2e_workspace['outputs']) / "benchmark.json", "w") as f:
        json.dump(report, f)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}):
        passed, _ = check_gate(Path(e2e_workspace['outputs']) / "benchmark.json")
        assert passed is True

# ☣️ GROUP 2: Stress Path
def test_e2e_stress_extreme_pruning(e2e_workspace):
    conf = create_mock_config(e2e_workspace, prune_ratio=0.99)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.pruner.prune_teacher"), \
         patch("compressor.distiller.distill"), \
         patch("compressor.quantizer.quantize"):
        results = run()
        assert results["stages"]["quantization"] == "success"

def test_e2e_failure_broken_config(e2e_workspace):
    with open(e2e_workspace['config'], "w") as f: f.write("invalid: [")
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(e2e_workspace['config'])}):
        with pytest.raises(Exception):
            run()

def test_e2e_failure_unreachable_server(e2e_workspace):
    conf = create_mock_config(e2e_workspace)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("benchmark.runner.wait_for_server", return_value=False):
        assert run_benchmark("localhost", 8080, n=1) == 1

def test_e2e_gate_failure_low_accuracy(e2e_workspace):
    conf = create_mock_config(e2e_workspace, acc_gate=0.99)
    report = {"passed": False, "latency": {"p95_ms": 10}, "accuracy": {"accuracy": 0.1}, "failures": ["Fail"]}
    with open(Path(e2e_workspace['outputs']) / "benchmark.json", "w") as f: json.dump(report, f)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}):
        passed, _ = check_gate(Path(e2e_workspace['outputs']) / "benchmark.json")
        assert passed is False

def test_e2e_pipeline_crash_simulation(e2e_workspace):
    conf = create_mock_config(e2e_workspace)
    with patch.dict(os.environ, {"EDGE_CV_CONFIG": str(conf)}), \
         patch("compressor.pruner.prune_teacher", side_effect=RuntimeError("Crash")):
        with pytest.raises(RuntimeError, match="Crash"):
            run()
