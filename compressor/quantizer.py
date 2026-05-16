# compressor/quantizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Model Quantization Module
#
# Techniques:
#   1. Static Post-Training Quantization (PTQ)
#   2. Accuracy Verification using ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from configs.settings import cfg

logger = logging.getLogger(__name__)


# ─── Data Utilities ───────────────────────────────────────────────────────────

def _build_calibration_loader() -> DataLoader:
    """
    Build a small DataLoader for PTQ calibration.
    Uses images from the validation set as representative samples.
    """
    h, w = cfg.deployment.input_height, cfg.deployment.input_width
    tf = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name = cfg.model.dataset.lower()
    data_dir = cfg.model.data_dir

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    else:
        # Standard ImageFolder for custom datasets
        ds = torchvision.datasets.ImageFolder(f"{data_dir}/val", transform=tf)

    # Use subset for calibration
    indices = torch.randperm(len(ds))[:cfg.compression.calibration_samples]
    subset = torch.utils.data.Subset(ds, indices)

    return DataLoader(subset, batch_size=1, shuffle=False)


# ─── FP32 Export ──────────────────────────────────────────────────────────────

def export_onnx_fp32(model: torch.nn.Module, save_path: Path, input_shape: tuple | None = None) -> Path:
    """
    Export the PyTorch model to ONNX FP32.
    This is the intermediate step before INT8 quantization.
    """
    if input_shape is None:
        c, h, w = cfg.deployment.input_channels, cfg.deployment.input_height, cfg.deployment.input_width
        input_shape = (1, c, h, w)   # batch size 1

    dummy = torch.zeros(*input_shape)
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        str(save_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    size_mb = save_path.stat().st_size / 1e6
    logger.info("ONNX FP32 exported → %s  (%.1f MB)", save_path, size_mb)
    return save_path


# ─── INT8 Quantization via ONNX Runtime ───────────────────────────────────────

def quantize_to_int8(fp32_onnx_path: Path, save_path: Path) -> Path:
    """
    Apply static PTQ using ONNX Runtime's quantization toolkit.

    Static quantization:
      - Runs calibration data through the FP32 model
      - Computes per-layer scale + zero_point from observed activation ranges
      - Rewrites the ONNX graph replacing FP32 ops with INT8 equivalents
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantFormat,
            QuantType,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        logger.error(
            "onnxruntime-tools not installed. Run:\n"
            "  pip install onnxruntime onnxruntime-tools"
        )
        raise

    # ── Calibration data reader ────────────────────────────────────────────────
    class _DataReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._loader = iter(_build_calibration_loader())
            self._done = False

        def get_next(self) -> dict | None:
            if self._done:
                return None
            try:
                images, _ = next(self._loader)
                return {"input": images.numpy()}
            except StopIteration:
                self._done = True
                return None

    # ── Pre-process ────────────────────────────────────────────────────────────
    logger.info("Pre-processing FP32 ONNX model...")
    preprocessed_path = str(fp32_onnx_path).replace(".onnx", "_infer.onnx")
    # Set skip_symbolic_shape=True to avoid 'Incomplete symbolic shape inference' errors
    quant_pre_process(str(fp32_onnx_path), preprocessed_path, skip_symbolic_shape=True)

    # ── Quantize ───────────────────────────────────────────────────────────────
    logger.info("Running PTQ calibration on %d samples…", cfg.compression.calibration_samples)
    quantize_static(
        model_input=preprocessed_path,
        model_output=str(save_path),
        calibration_data_reader=_DataReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    before_mb = fp32_onnx_path.stat().st_size / 1e6
    after_mb  = save_path.stat().st_size / 1e6
    logger.info(
        "INT8 quantization complete.\n"
        "  FP32: %.1f MB  →  INT8: %.1f MB  (%.1fx smaller)",
        before_mb, after_mb, before_mb / (after_mb if after_mb > 0 else 1.0),
    )
    return save_path


# ─── Accuracy verification after quantization ─────────────────────────────────

def verify_accuracy(int8_onnx_path: Path) -> float:
    """
    Run the INT8 ONNX model through ONNX Runtime and measure Top-1 accuracy.
    Returns float in [0, 1].
    """
    import onnxruntime as ort

    h, w = cfg.deployment.input_height, cfg.deployment.input_width
    tf = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_name = cfg.model.dataset.lower()
    data_dir = cfg.model.data_dir

    if dataset_name == "cifar10":
        val_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif dataset_name == "cifar100":
        val_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    else:
        val_ds = torchvision.datasets.ImageFolder(f"{data_dir}/val", transform=tf)

    import os
    num_workers = min(os.cpu_count() or 1, 4)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers)

    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(int8_onnx_path), sess_opts, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    correct = 0
    total = 0

    logger.info("Verifying accuracy on validation set...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # ONNX Runtime expectations: numpy array [B, C, H, W]
            outputs = sess.run(None, {input_name: images.numpy()})[0]
            preds = outputs.argmax(axis=1)
            correct += (preds == labels.numpy()).sum()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                logger.info("  Batch %d: Accuracy = %.2f%%", i + 1, 100 * correct / total)
            
            # Limit verification for speed in CI/Local
            if total >= 1000:
                break

    acc = correct / total
    logger.info("Final INT8 Accuracy: %.2f%%", 100 * acc)
    return acc


# ─── Main Entry ───────────────────────────────────────────────────────────────

def quantize(student_model: torch.nn.Module | None = None) -> Path:
    """
    Full quantization pipeline:
      1. Load student model (if not provided)
      2. Export to FP32 ONNX
      3. Static PTQ → INT8 ONNX
      4. Verify accuracy
    """
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / "student_fp32.onnx"
    int8_path = output_dir / "student_int8.onnx"

    # 1. Load model
    if student_model is None:
        from compressor.distiller import _build_student
        student_model = _build_student()
        # In a real scenario, you'd load weights here
        # student_model.load_state_dict(torch.load("..."))

    # 2. Export
    export_onnx_fp32(student_model, fp32_path)

    # 3. Quantize
    quantize_to_int8(fp32_path, int8_path)

    # 4. Verify
    acc = verify_accuracy(int8_path)

    # 5. Check accuracy gate
    if acc < cfg.deployment.accuracy_threshold:
        logger.warning(
            "Quantized model accuracy (%.2f) below threshold (%.2f).",
            acc, cfg.deployment.accuracy_threshold
        )
    
    return int8_path
