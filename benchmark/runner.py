"""
benchmark/runner.py
────────────────────
Hardware-in-the-Loop (HIL) benchmark suite.

This script:
  1. Waits for the C++ inference server to be healthy
  2. Runs N inference calls and collects latency + memory stats
  3. Runs the full validation set for accuracy measurement
  4. Writes benchmark.json with all results
  5. Exits 0 (pass) or 1 (fail) based on CI gate thresholds

The CI/CD pipeline reads the exit code and the JSON.

Usage:
  python -m benchmark.runner
  python -m benchmark.runner --n 200 --host localhost --port 8080
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torchvision
import torchvision.transforms as T
from PIL import Image

from configs.settings import cfg

logger = logging.getLogger(__name__)


# ─── Server utilities ──────────────────────────────────────────────────────────

def wait_for_server(host: str, port: int, timeout: int = 60) -> bool:
    """Poll /health until the C++ server is ready."""
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                logger.info("Server ready: %s", url)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    logger.error("Server did not become ready within %ds", timeout)
    return False


def image_to_pixels(img: Image.Image, height: int, width: int) -> list[int]:
    """
    Resize an image and convert to a flat RGB pixel list.
    The C++ server's /predict endpoint expects raw uint8 pixels.
    Normalization happens inside the C++ preprocess() function.
    """
    img = img.convert("RGB").resize((width, height), Image.BILINEAR)
    return list(img.tobytes())


# ─── Latency benchmark ────────────────────────────────────────────────────────

def run_latency_benchmark(host: str, port: int, n: int) -> dict:
    """
    POST to /benchmark — the C++ server runs N dummy inferences internally
    and returns latency statistics.  This is the most accurate measurement
    because it avoids Python/network overhead in the per-call timing.
    """
    url = f"http://{host}:{port}/benchmark"
    resp = requests.post(url, json={"n": n}, timeout=n * 0.5)
    if resp.status_code != 200:
        raise Exception(f"Server error: {resp.status_code}")
    resp.raise_for_status()
    stats = resp.json()
    logger.info(
        "Latency  mean=%.2fms  p50=%.2fms  p95=%.2fms  p99=%.2fms  FPS=%.1f",
        stats["mean_ms"], stats["p50_ms"], stats["p95_ms"],
        stats["p99_ms"], stats["fps"],
    )
    logger.info("Memory RSS: %.1f MB", stats.get("memory_rss_mb", 0.0))

    return stats


# ─── Accuracy benchmark ───────────────────────────────────────────────────────

def run_accuracy_benchmark(host: str, port: int, max_images: int = 500) -> dict:
    """
    Evaluate Top-1 accuracy by sending validation images to /predict.
    We cap at max_images for CI speed — 500 images is statistically sufficient.
    """
    url = f"http://{host}:{port}/predict"
    h, w = cfg.deployment.input_height, cfg.deployment.input_width

    dataset_name = cfg.model.dataset.lower()
    data_dir = cfg.model.data_dir

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)
    else:
        ds = torchvision.datasets.ImageFolder(f"{data_dir}/val")

    correct, total = 0, 0
    end_idx = min(max_images, len(ds))

    for i in range(end_idx):
        img, label = ds[i]
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)

        pixels = image_to_pixels(img, h, w)
        payload = {"pixels": pixels, "height": h, "width": w}

        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            if result["top_class"] == label:
                correct += 1
        except Exception as e:
            logger.warning("Predict failed for image %d: %s", i, e)

        total += 1

        if (i + 1) % 50 == 0:
            logger.info("  Accuracy progress: %d/%d  (%.3f)", correct, total, correct/total)

    accuracy = correct / total if total > 0 else 0.0
    logger.info("Accuracy on %d validation images: %.4f", total, accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ─── Precision / Recall analysis ──────────────────────────────────────────────

def run_threshold_analysis(host: str, port: int, n_images: int = 200) -> dict:
    """
    Compute precision and recall at different confidence thresholds.
    On edge devices (e.g. security cameras) Recall matters more than Precision.

    Returns a dict with threshold → {precision, recall} for the top class.
    """
    url = f"http://{host}:{port}/predict"
    h, w = cfg.deployment.input_height, cfg.deployment.input_width
    data_dir = cfg.model.data_dir
    dataset_name = cfg.model.dataset.lower()

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)
    else:
        ds = torchvision.datasets.ImageFolder(f"{data_dir}/val")

    # Collect (true_label, predicted_class, score) for each image
    records = []
    end_idx = min(n_images, len(ds))

    for i in range(end_idx):
        img, label = ds[i]
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)
        pixels = image_to_pixels(img, h, w)

        try:
            resp = requests.post(url, json={"pixels": pixels, "height": h, "width": w}, timeout=5)
            data = resp.json()
            records.append((label, data["top_class"], data["top_score"]))
        except Exception:
            pass

    # Sweep thresholds from 0.1 to 0.95
    threshold_results = {}
    for t in np.arange(0.1, 1.0, 0.05):
        t = round(float(t), 2)
        tp = sum(1 for (gt, pred, score) in records if score >= t and pred == gt)
        fp = sum(1 for (gt, pred, score) in records if score >= t and pred != gt)
        fn = sum(1 for (gt, pred, score) in records if score <  t and pred == gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        threshold_results[t] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(2 * precision * recall / (precision + recall + 1e-9), 4),
        }

    logger.info("Threshold analysis complete (%d points)", len(threshold_results))
    return threshold_results


# ─── CI Gate ──────────────────────────────────────────────────────────────────

def evaluate_gates(latency_stats: dict, accuracy_stats: dict) -> tuple[bool, list[str]]:
    """
    Check all CI gate conditions.
    Returns (passed: bool, failure_reasons: list[str])
    """
    failures = []

    # Gate 1: p95 latency (not mean — we care about worst-case tail)
    lat_gate = cfg.deployment.latency_gate_ms
    if latency_stats["p95_ms"] > lat_gate:
        failures.append(
            f"Latency gate FAILED: p95={latency_stats['p95_ms']:.1f}ms > {lat_gate}ms"
        )

    # Gate 2: Accuracy
    acc_gate = cfg.deployment.accuracy_threshold
    if accuracy_stats["accuracy"] < acc_gate:
        failures.append(
            f"Accuracy gate FAILED: {accuracy_stats['accuracy']:.4f} < {acc_gate}"
        )

    # Gate 4: Memory
    mem_gate = cfg.deployment.memory_limit_mb
    if latency_stats.get("memory_rss_mb", 0) > mem_gate:
        failures.append(
            f"Memory gate FAILED: {latency_stats['memory_rss_mb']:.1f}MB > {mem_gate}MB"
        )

    return len(failures) == 0, failures


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(host: str, port: int, n: int) -> int:
    # Increase timeout for QEMU emulation startup
    if not wait_for_server(host, port, timeout=180):
        return 1

    logger.info("─" * 50)
    logger.info("Running latency benchmark (%d iterations)…", n)
    latency_stats = run_latency_benchmark(host, port, n)

    logger.info("─" * 50)
    logger.info("Running accuracy benchmark…")
    accuracy_stats = run_accuracy_benchmark(host, port)

    logger.info("─" * 50)
    logger.info("Running threshold analysis…")
    threshold_results = run_threshold_analysis(host, port)

    passed, failures = evaluate_gates(latency_stats, accuracy_stats)

    # Assemble full report
    report = {
        "passed":     passed,
        "failures":   failures,
        "latency":    latency_stats,
        "accuracy":   accuracy_stats,
        "thresholds": threshold_results,
        "gates": {
            "latency_gate_ms":    cfg.deployment.latency_gate_ms,
            "accuracy_threshold": cfg.deployment.accuracy_threshold,
            "memory_limit_mb":    cfg.deployment.memory_limit_mb,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Add warnings for secondary checks
    warnings = []
    lat_gate = cfg.deployment.latency_gate_ms
    if latency_stats["mean_ms"] > lat_gate * 0.8:
        warnings.append(
            f"Latency warning: mean={latency_stats['mean_ms']:.1f}ms > "
            f"{lat_gate * 0.8:.1f}ms (80% of gate)"
        )
    report["warnings"] = warnings

    out_path = Path(cfg.logging.benchmark_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("─" * 50)
    if passed:
        logger.info("✅  ALL GATES PASSED  — benchmark.json written to %s", out_path)
        return 0
    else:
        logger.error("❌  GATE FAILURES:")
        for reason in failures:
            logger.error("    %s", reason)
        logger.error("benchmark.json written to %s", out_path)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.logging.level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="HIL Benchmark Runner")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=cfg.deployment.server_port)
    parser.add_argument("--n",    type=int, default=100, help="Number of latency iterations")
    args = parser.parse_args()

    sys.exit(main(args.host, args.port, args.n))