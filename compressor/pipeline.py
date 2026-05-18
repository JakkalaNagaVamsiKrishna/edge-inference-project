"""
compressor/pipeline.py
───────────────────────
Master orchestrator for the full ML compression pipeline.

Run this single script to go from Teacher → INT8 ONNX Student.

Usage:
  python -m compressor.pipeline
  python -m compressor.pipeline --skip-prune    # if teacher already pruned
  python -m compressor.pipeline --skip-distill  # if student already trained
"""

from __future__ import annotations
import argparse
import json
import logging
import time

from configs.settings import cfg

logger = logging.getLogger(__name__)


def run(skip_prune: bool = False, skip_distill: bool = False) -> dict:
    """
    Full pipeline:
      Stage 1: Prune Teacher
      Stage 2: Distill into Student
      Stage 3: Quantize to INT8 ONNX
      Stage 4: Write summary JSON
    """
    results: dict = {
        "stages": {},
        "output_onnx": str(cfg.onnx_path),
    }

    # ── Stage 1: Prune ────────────────────────────────────────────────────────
    if not skip_prune:
        logger.info("=" * 60)
        logger.info("STAGE 1 — Pruning Teacher")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from compressor.pruner import prune_teacher
        prune_teacher()
        results["stages"]["pruning"] = {
            "duration_s": round(time.perf_counter() - t0, 2),
            "ratio": cfg.compression.pruning_ratio,
        }
    else:
        logger.info("Skipping pruning (--skip-prune).")

    # ── Stage 2: Distill ──────────────────────────────────────────────────────
    if not skip_distill:
        logger.info("=" * 60)
        logger.info("STAGE 2 — Knowledge Distillation")
        logger.info("=" * 60)
        t0 = time.perf_counter()
        from compressor.distiller import distill
        student = distill()
        results["stages"]["distillation"] = {
            "duration_s": round(time.perf_counter() - t0, 2),
            "student_arch": cfg.model.student_arch,
            "epochs": cfg.model.epochs,
            "temperature": cfg.compression.distillation_temperature,
            "alpha": cfg.compression.distillation_alpha,
        }
    else:
        logger.info("Skipping distillation (--skip-distill).")
        student = None

    # ── Stage 3: Quantize ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — Quantization → INT8 ONNX")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    from compressor.quantizer import quantize
    onnx_path = quantize(student_model=student)
    size_mb = onnx_path.stat().st_size / 1e6
    results["stages"]["quantization"] = {
        "duration_s": round(time.perf_counter() - t0, 2),
        "mode": cfg.compression.quantization_mode,
        "output_size_mb": round(size_mb, 2),
        "onnx_path": str(onnx_path),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = cfg.output_dir / "compression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  ONNX model: %s  (%.1f MB)", onnx_path, size_mb)
    logger.info("  Summary:    %s", summary_path)
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Edge-CV-Hub compression pipeline")
    parser.add_argument("--skip-prune",   action="store_true", help="Skip pruning stage")
    parser.add_argument("--skip-distill", action="store_true", help="Skip distillation stage")
    args = parser.parse_args()

    run(skip_prune=args.skip_prune, skip_distill=args.skip_distill)