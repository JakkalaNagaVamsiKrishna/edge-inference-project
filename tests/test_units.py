import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np

# ─── Pruner Tests ─────────────────────────────────────────────────────────────

def test_collect_conv_layers():
    from compressor.pruner import _collect_conv_layers
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3)
    )
    layers = _collect_conv_layers(model)
    assert len(layers) == 2
    assert all(isinstance(layer[1], nn.Conv2d) for layer in layers)

def test_count_parameters():
    from compressor.pruner import _count_parameters
    model = nn.Linear(10, 5) # 10*5 weights + 5 bias = 55 params
    assert _count_parameters(model) == 55

# ─── Distiller Tests ──────────────────────────────────────────────────────────

def test_build_student():
    from compressor.distiller import _build_student
    from unittest.mock import patch
    
    with patch("configs.settings.cfg.model.student_arch", "mobilenet_v3_small"):
        model = _build_student()
        assert isinstance(model, nn.Module)

# ─── Benchmark Runner Tests ───────────────────────────────────────────────────

def test_image_to_pixels():
    from benchmark.runner import image_to_pixels
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    pixels = image_to_pixels(img, 224, 224)
    
    # Expected size: 224 * 224 * 3 channels
    assert len(pixels) == 224 * 224 * 3
    # Check first pixel is red (225, 0, 0)
    assert pixels[0] == 255
    assert pixels[1] == 0
    assert pixels[2] == 0

# ─── Compatibility Tests (Python-ONNX) ────────────────────────────────────────

def test_onnx_python_compatibility():
    """Verify that a model exported via Python can be correctly loaded and run by ONNX Runtime."""
    import onnxruntime as ort
    from compressor.quantizer import export_onnx_fp32
    import tempfile
    
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 224 * 224, 10)
    )
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "compat_test.onnx"
        export_onnx_fp32(model, out_path, input_shape=(1, 3, 224, 224))
        
        # This checks if the ONNX Runtime (which the C++ engine uses) can load the model
        sess = ort.InferenceSession(str(out_path))
        input_meta = sess.get_inputs()[0]
        assert input_meta.name == "input"
        assert input_meta.shape == ['batch_size', 3, 224, 224]
        
        output_meta = sess.get_outputs()[0]
        assert output_meta.name == "output"
        assert output_meta.shape == ['batch_size', 10]
        
        # Test inference
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = sess.run(None, {"input": dummy_input})
        assert outputs[0].shape == (1, 10)
