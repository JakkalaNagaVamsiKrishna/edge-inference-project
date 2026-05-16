"""
compressor/quantizer.py
────────────────────────
Post-Training Quantization (PTQ): converts FP32 Student → INT8 ONNX.

How PTQ works:
  FP32 weights store values as 32-bit floats.  INT8 maps those values to
  the range [-128, 127] using a per-layer SCALE and ZERO_POINT:
      x_int8 = round(x_fp32 / scale) + zero_point

  We need real data (the "calibration set") to compute scale/zero_point
  accurately — we pass ~512 images through the model and observe the
  actual activation ranges per layer.

Why ONNX?
  ONNX (Open Neural Network Exchange) is a language-agnostic format.
  We produce it in Python and load it in C++.  ONNX Runtime then uses
  hardware-specific backends (XNNPACK on ARM, TensorRT on Nvidia) to
  execute it at native speed.

Usage:
  python -m compressor.quantizer
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from configs.settings import cfg

logger = logging.getLogger(__name__)


# ─── Calibration dataloader ────────────────────────────────────────────────────

def _build_calibration_loader() -> DataLoader:
    """
    A small DataLoader (~512 images) used to observe activation ranges.
    Only needs to be representative — does NOT need to be the full dataset.
    """
    h, w = cfg.deployment.input_height, cfg.deployment.input_width
    tf = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name = cfg.model.dataset.lower()
    data_dir = cfg.model.data_dir
    n = cfg.compression.calibration_samples

    if dataset_name == "cifar10":
        full = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    elif dataset_name == "cifar100":
        full = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=tf)
    else:
        full = torchvision.datasets.ImageFolder(f"{data_dir}/train", transform=tf)

    import os
    num_workers = min(os.cpu_count() or 1, 4)
    indices = list(range(min(n, len(full))))
    subset = Subset(full, indices)
    return DataLoader(subset, batch_size=32, shuffle=False, num_workers=num_workers)


# ─── Export to ONNX (FP32 first) ──────────────────────────────────────────────

def export_onnx_fp32(
    model: nn.Module,
    save_path: Path,
    input_shape: tuple[int, ...] | None = None,
) -> Path:
    """
    Export the PyTorch model to ONNX FP32.
    This is the intermediate step before INT8 quantization.
    """
    if input_shape is None:
        c, h, w = cfg.input_shape
        input_shape = (1, c, h, w)   # batch size 1

    dummy = torch.zeros(*input_shape)
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        str(save_path),
        export_params=True,
        opset_version=17,
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

    # ── Quantize ───────────────────────────────────────────────────────────────
    logger.info("Running PTQ calibration on %d samples…", cfg.compression.calibration_samples)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(save_path),
        calibration_data_reader=_DataReader(),
        quant_format=QuantFormat.QOperator,    # fused INT8 operators
        per_channel=True,                      # per-channel is more accurate
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    before_mb = fp32_onnx_path.stat().st_size / 1e6
    after_mb  = save_path.stat().st_size / 1e6
    logger.info(
        "INT8 quantization complete.\n"
        "  FP32: %.1f MB  →  INT8: %.1f MB  (%.1fx smaller)",
        before_mb, after_mb, before_mb / after_mb,
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
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(int8_onnx_path), sess_opts)
    input_name = sess.get_inputs()[0].name

    correct, total = 0, 0
    for images, labels in loader:
        outputs = sess.run(None, {input_name: images.numpy()})
        preds = np.argmax(outputs[0], axis=1)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)

    acc = correct / total
    logger.info("INT8 ONNX Top-1 accuracy on validation set: %.4f", acc)
    return float(acc)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def quantize(student_model: nn.Module | None = None) -> Path:
    """
    Full quantization pipeline:
      1. Load student (or accept it as argument)
      2. Export FP32 ONNX
      3. Apply INT8 PTQ
      4. Verify accuracy
      5. Return path to INT8 ONNX file
    """
    from compressor.distiller import _build_student

    if student_model is None:
        student_model = _build_student()
        ckpt = cfg.student_checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Student checkpoint not found at {ckpt}. "
                "Run distiller.py first."
            )
        student_model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    student_model.eval()

    fp32_path = cfg.output_dir / "student_fp32.onnx"
    int8_path = cfg.onnx_path

    export_onnx_fp32(student_model, fp32_path)

    mode = cfg.compression.quantization_mode
    if mode == "int8_ptq":
        quantize_to_int8(fp32_path, int8_path)
    elif mode == "fp16":
        # FP16 via onnxconverter (lighter than INT8, useful for Jetson)
        try:
            import onnxconverter_common as occ
            import onnx
            model_fp32 = onnx.load(str(fp32_path))
            model_fp16 = occ.float16.convert_float_to_float16(model_fp32)
            onnx.save(model_fp16, str(int8_path))
            logger.info("FP16 ONNX exported → %s", int8_path)
        except ImportError:
            logger.warning("onnxconverter-common not installed. Falling back to FP32.")
            import shutil
            shutil.copy(fp32_path, int8_path)
    else:
        # mode == "none" — just use FP32
        import shutil
        shutil.copy(fp32_path, int8_path)

    acc = verify_accuracy(int8_path)
    threshold = cfg.deployment.accuracy_threshold
    if acc < threshold:
        logger.warning(
            "⚠  Quantized model accuracy %.4f is below threshold %.4f. "
            "Consider reducing pruning ratio or using QAT.",
            acc, threshold,
        )
    else:
        logger.info("✓ Accuracy %.4f meets threshold %.4f", acc, threshold)

    return int8_path


if __name__ == "__main__":
    logging.basicConfig(level=cfg.logging.level)
    path = quantize()
    logger.info("Final INT8 ONNX model: %s", path)