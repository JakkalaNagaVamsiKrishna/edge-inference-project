# Edge-CV-Hub: Full Code Audit Report

---

## 1. Critical Compatibility Error — ONNX Runtime (Root Cause + Fix)

### The Problem

In `compressor/quantizer.py`, the `quantize_to_int8()` function imports from `onnxruntime.quantization`:

```python
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
```

And `requirements.txt` lists **both**:

```
onnxruntime>=1.18.0
onnxruntime-tools>=1.7.0     # quantize_static API
```

**This is the compatibility error.** Starting from `onnxruntime >= 1.16`, the `onnxruntime.quantization` module (including `quantize_static`, `CalibrationDataReader`, `QuantFormat`, `QuantType`, and `quant_pre_process`) was **fully merged into the main `onnxruntime` package**. `onnxruntime-tools` is an **obsolete, abandoned package** that is incompatible with `onnxruntime >= 1.16`:

- It pins to an older API surface and will conflict with or shadow the built-in quantization module.
- On Python 3.11 (which this project uses), `onnxruntime-tools` **fails to install** entirely because it has no published wheel for Python 3.11.
- The error comment in the `except ImportError` block incorrectly tells users to install `onnxruntime-tools`, which will not fix the problem.

**Resolution — remove `onnxruntime-tools` from `requirements.txt`:**

```diff
- onnxruntime-tools>=1.7.0     # quantize_static API
```

The entire quantization API used in the project (`quantize_static`, `CalibrationDataReader`, `QuantFormat`, `QuantType`, `quant_pre_process`) is available directly in `onnxruntime >= 1.16.0` with no additional package needed. The existing import code is already correct — only the requirements entry and the error message need fixing.

**Also fix the misleading error message in `quantizer.py`:**

```python
# BEFORE (misleading):
logger.error(
    "onnxruntime-tools not installed. Run:\n"
    "  pip install onnxruntime onnxruntime-tools"
)

# AFTER (correct):
logger.error(
    "onnxruntime quantization module not found.\n"
    "  Ensure onnxruntime >= 1.16.0 is installed:\n"
    "  pip install 'onnxruntime>=1.16.0'"
)
```

---

## 2. Bugs and Errors

### 2.1 `InferenceEngine` — XNNPACK Registration API Broken (C++ / `main.cpp`)

```cpp
OrtSessionOptionsAppendExecutionProvider_XNNPACK(opts, {});
```

This is **incorrect**. The XNNPACK execution provider is registered via the standard ORT C++ API:

```cpp
// Correct way:
opts.AppendExecutionProvider("XNNPACK", {});
```

The function `OrtSessionOptionsAppendExecutionProvider_XNNPACK` does not exist in the ORT 1.18 C++ API. This will cause a **linker error or runtime crash** on ARM64. Additionally, XNNPACK must be included as a build option in the ONNX Runtime prebuilt — the standard ARM64 prebuilt tarball from GitHub does **not** include it. The safest fix is to remove the XNNPACK call entirely and rely on the default CPU provider, which already uses NEON on ARM64:

```cpp
// Remove this line:
// OrtSessionOptionsAppendExecutionProvider_XNNPACK(opts, {});
```

### 2.2 `MappedModel` — Missing Default Constructor (C++ / `main.cpp`)

```cpp
MappedModel(MappedModel&&) = default;
MappedModel() = default;
```

`MappedModel() = default;` is placed **after** `MappedModel(MappedModel&&) = default;` but the member initializers (`addr = nullptr`, `size = 0`, `fd = -1`) are inline, so the default constructor is fine. However, `operator=(MappedModel&&)` is missing. Since `MappedModel` is used as a local variable with `MappedModel mapped = MappedModel::load(MODEL_PATH)`, copy elision (NRVO) should apply in C++17, but the move assignment omission is still a latent defect.

### 2.3 `distiller.py` — `torch.load()` Missing `weights_only` (Security / FutureWarning)

Both in `distiller.py` and `pruner.py`, `torch.load()` is called without `weights_only=True`:

```python
teacher.load_state_dict(torch.load(pruned_path, map_location="cpu"))  # distiller.py line ~1579
student.load_state_dict(torch.load(save_path, map_location="cpu"))     # distiller.py line ~1660
state = torch.load(weights_cfg, map_location="cpu")                    # pruner.py
```

In PyTorch >= 2.4, this raises a `FutureWarning` and will become an error in a future version. Fix:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

### 2.4 `benchmark/runner.py` — Mean Latency Gate is a Warning, Not a Gate Failure

```python
# Gate 2: Mean latency as secondary check
if latency_stats["mean_ms"] > lat_gate * 0.8:
    failures.append(
        f"Latency warning: mean={latency_stats['mean_ms']:.1f}ms > ..."
    )
```

This appends to `failures`, which causes the CI gate to **fail** the build even though it is labeled a "warning". This will trigger false failures in CI whenever mean latency is > 40ms (80% of the 50ms gate). Either rename this list entry to something that doesn't contribute to a hard fail, or use a separate `warnings` list in the report JSON.

### 2.5 `configs/settings.py` — Module-Level `cfg` Instantiation Side Effect

```python
cfg: Settings = load_settings()
```

This runs `load_settings()` at **import time**, meaning that importing `configs.settings` in any test will immediately try to open `configs/config.yaml`. Tests that mock the config path or run from a different working directory will fail with a `FileNotFoundError`. The singleton should be lazily initialized.

### 2.6 `Dockerfile.inference` — Syntax Error in `RUN` Layer

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \    
    && rm -rf /var/lib/apt/lists/*
```

There is trailing whitespace after the backslash on the `libgomp1` line (`\    `). This will cause a **Docker build failure** because the shell sees `libgomp1 \` followed by spaces and then a newline, breaking the line continuation. Fix:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

Also, the runtime image is missing `curl`, which is required by the `HEALTHCHECK`:

```dockerfile
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD curl -f http://localhost:8080/health || exit 1
```

`curl` is not installed in the runtime stage, so the healthcheck will always fail. Add it:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

### 2.7 `compressor/quantizer.py` — Preprocessed ONNX File Left on Disk

```python
preprocessed_path = str(fp32_onnx_path).replace(".onnx", "_infer.onnx")
quant_pre_process(str(fp32_onnx_path), preprocessed_path, skip_symbolic_shape=True)
```

The intermediate `_infer.onnx` file is never deleted after quantization. In a CI environment with `*.onnx` in `.gitignore`, this is harmless but wasteful. Wrap in a `try/finally` to clean up, or use `tempfile.NamedTemporaryFile`.

### 2.8 `benchmark/runner.py` — `wait_for_server` Timeout of 90s May Not Be Enough for QEMU

In the CI QEMU environment, the ARM64 emulated container can take longer than 90 seconds to become responsive due to emulation overhead. The `timeout-minutes: 30` job timeout is generous, but the Python-level `wait_for_server(..., timeout=90)` will give up after 90s and return exit code 1 before the container is ready. The timeout should be increased to at least 120–180 seconds for QEMU.

---

## 3. Incompatibilities

### 3.1 `onnxruntime-tools` vs Python 3.11 (Confirmed)

As detailed in Section 1, `onnxruntime-tools` has no Python 3.11 wheel. The CI job uses `PYTHON_VERSION: "3.11"`. Installing `requirements.txt` will **fail** in the `compress-model` job. This is the most critical blocking issue.

### 3.2 `opset_version=18` vs Older ONNX Runtime Deployments

```python
torch.onnx.export(
    ...
    opset_version=18,
    ...
)
```

ONNX opset 18 requires `onnxruntime >= 1.14`. While `onnxruntime >= 1.18.0` is specified in requirements, the Dockerfile downloads `ORT_VERSION=1.18.0` for the C++ engine — this is consistent. However, if someone attempts to run this model on an older device or ORT version, it will silently fail. This should be documented in the README.

### 3.3 `onnx>=1.16.0` vs Opset 18

ONNX opset 18 requires `onnx >= 1.13.0`, and `onnx >= 1.16.0` is specified. This is compatible, but worth noting that opset 18 support for all MobileNetV3 and EfficientNet operators was only stabilized in ORT 1.15+.

### 3.4 `activation_type=QuantType.QInt8` — CPU Compatibility Issue

```python
quantize_static(
    ...
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)
```

Using **signed INT8 activations** (`QInt8`) with QDQ format is the correct choice for ARM/NEON hardware and the ONNX Runtime CPU execution provider. However, many ORT CPU kernels for operations like `Conv` with `QDQ` nodes require `QUInt8` (unsigned) for activations to trigger the optimized kernel paths. With `QInt8` activations, ORT may fall back to slower reference implementations on x86 CPUs. For ARM64 this is generally fine, but consider adding a note or allowing this to be configured.

### 3.5 `Colab_Runner.ipynb` — Conflict Between `onnxruntime` and `onnxruntime-gpu`

```python
!pip install -r requirements.txt --quiet
!pip install onnxruntime-gpu --quiet
```

`onnxruntime` (CPU) and `onnxruntime-gpu` are **mutually exclusive packages** — installing both will cause one to overwrite the other. The CPU package from `requirements.txt` will be installed first, then `onnxruntime-gpu` will replace it. This is actually the intended behavior for Colab (you want GPU), but it means `requirements.txt` and the Colab notebook are inconsistent. The correct approach is to install only `onnxruntime-gpu` in Colab and skip the CPU package. The Colab cell should be:

```python
!pip install -r requirements.txt --quiet --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
# OR: replace onnxruntime in requirements with onnxruntime-gpu for GPU envs:
!grep -v onnxruntime requirements.txt | pip install -r /dev/stdin --quiet
!pip install onnxruntime-gpu --quiet
```

---

## 4. Improvements

### 4.1 Quantization: Add `reduce_range=True` for Better ARM Compatibility

Some ARM CPUs have issues with the full INT8 range. Adding `reduce_range=True` improves portability:

```python
quantize_static(
    ...
    reduce_range=True,   # Use 7-bit range instead of 8-bit for safer ARM inference
)
```

### 4.2 `quant_pre_process` — Remove `skip_symbolic_shape=True`

```python
quant_pre_process(str(fp32_onnx_path), preprocessed_path, skip_symbolic_shape=True)
```

Skipping symbolic shape inference prevents ORT from optimizing the quantized graph properly. The `dynamic_axes` in the ONNX export (`batch_size`) cause this to be set, but the workaround should instead be to use a fixed batch size of 1 during pre-processing:

```python
# Better: run shape inference with a fixed batch size hint
quant_pre_process(str(fp32_onnx_path), preprocessed_path)
```

If symbolic shape inference still fails, the correct fix is to run `onnx.shape_inference.infer_shapes_path()` with `check_type=True` before calling `quant_pre_process`.

### 4.3 `configs/settings.py` — Use Lazy Loading for `cfg` Singleton

```python
# Instead of module-level:
cfg: Settings = load_settings()

# Use a lazy accessor:
_cfg: Settings | None = None

def get_cfg() -> Settings:
    global _cfg
    if _cfg is None:
        _cfg = load_settings()
    return _cfg
```

This prevents import-time failures in tests and CI environments where `config.yaml` may not be in the expected path.

### 4.4 `compressor/quantizer.py` — `verify_accuracy` Runs Full Validation Set in CI

```python
# Limit verification for speed in CI/Local
if total >= 1000:
    break
```

1000 images at batch size 64 is 16 batches of full ORT inference, which can be slow in CI. This should be configurable via the config YAML (e.g., `accuracy_verification_samples: 200`) rather than hardcoded.

### 4.5 `distiller.py` — Teacher Loaded but Its Classifier Head Not Replaced When Loading from Checkpoint

When loading the pruned teacher from disk:

```python
teacher = tvm2.get_model(cfg.model.teacher_arch, num_classes=cfg.model.num_classes)
teacher.load_state_dict(torch.load(pruned_path, map_location="cpu"))
```

This works correctly. But the pretrained teacher loaded in `pruner.py` via `_load_teacher()` replaces the classifier head (`model.fc = nn.Linear(in_features, num_classes)`), and then saves the state dict. The state dict keys will include the modified head. This is consistent, but it means the pruned `.pth` file is not transferable to a different `num_classes` without re-running pruning. This should be documented.

### 4.6 CI Workflow — `compress-model` Job Downloads Dataset Every Run

```yaml
- name: Cache dataset
  uses: actions/cache@v4
  with:
    path: ./data
    key: dataset-${{ hashFiles('configs/config.yaml') }}
```

The cache key is based on the config file hash, which is correct. However, CIFAR-10 (~170MB) will be re-downloaded whenever `config.yaml` changes for any reason (including comment edits). Split the cache key:

```yaml
key: dataset-cifar10-v1
```

### 4.7 `benchmark/runner.py` — No Retry Logic on HTTP Failures

The accuracy benchmark loop catches all exceptions and continues:

```python
except Exception as e:
    logger.warning("Predict failed for image %d: %s", i, e)
```

Under QEMU emulation, transient timeouts are common. A simple retry with backoff would prevent inaccurate accuracy measurements:

```python
for attempt in range(3):
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        ...
        break
    except Exception:
        if attempt == 2:
            logger.warning("Predict failed after 3 attempts for image %d", i)
```

### 4.8 `inference/CMakeLists.txt` — `-march=native` Is a Cross-Compilation Hazard

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
```

The comment in the file says "For ARM cross-compilation the Dockerfile overrides -march to armv8-a+simd". But for local builds (`cmake -B build && cmake --build build` as shown in the README), there is no `-march` set at all — GCC will default to a conservative baseline. Add a sensible fallback:

```cmake
if(NOT CMAKE_CROSSCOMPILING)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()
```

### 4.9 `test_units.py` — Missing `import torch` at the Top

```python
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
```

`torch` itself is not directly imported, which works because `torch.nn` imports the parent. However, tests like `test_build_student` mutate `cfg.model.student_arch` directly — on a frozen dataclass this would fail. `Settings` uses regular `@dataclass` (not `frozen=True`), so mutation is technically allowed, but it is not thread-safe if tests ever run in parallel (e.g., with `pytest-xdist`). Tests should restore config state using `unittest.mock.patch` instead of direct attribute mutation.

### 4.10 `dashboard/app.py` — Chart.js Loaded from CDN in Production

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

The dashboard is described as running inside an edge container with no network access. This CDN dependency will cause the dashboard to load with a blank chart area in offline/airgapped deployments. Bundle Chart.js as a static file or use a self-hosted copy.

---

## 5. Summary Table

| # | File | Severity | Type | Issue |
|---|------|----------|------|-------|
| 1 | `requirements.txt` | 🔴 Critical | Compatibility | `onnxruntime-tools` incompatible with Python 3.11 and ORT ≥ 1.16 — blocks CI |
| 2 | `inference/src/main.cpp` | 🔴 Critical | Bug | `OrtSessionOptionsAppendExecutionProvider_XNNPACK` API does not exist — linker/runtime crash |
| 3 | `docker/Dockerfile.inference` | 🔴 Critical | Bug | Trailing whitespace after `\` breaks Docker build; missing `curl` breaks HEALTHCHECK |
| 4 | `compressor/quantizer.py` | 🟠 High | Compatibility | Error message tells users to install obsolete `onnxruntime-tools` |
| 5 | `Colab_Runner.ipynb` | 🟠 High | Compatibility | Installs both `onnxruntime` (CPU) and `onnxruntime-gpu` — conflict |
| 6 | `compressor/distiller.py`, `pruner.py` | 🟡 Medium | Bug | `torch.load()` without `weights_only=True` — FutureWarning / security risk |
| 7 | `benchmark/runner.py` | 🟡 Medium | Bug | Mean latency "warning" appended to `failures` causes false CI gate failures |
| 8 | `configs/settings.py` | 🟡 Medium | Bug | `cfg` instantiated at import time — breaks tests and CI with wrong cwd |
| 9 | `quantizer.py` | 🟡 Medium | Improvement | `skip_symbolic_shape=True` degrades quantization graph optimization |
| 10 | `benchmark/runner.py` | 🟡 Medium | Improvement | `wait_for_server` timeout too short for QEMU emulation (90s → 180s) |
| 11 | `quantizer.py` | 🟢 Low | Improvement | Intermediate `_infer.onnx` file not cleaned up |
| 12 | `CMakeLists.txt` | 🟢 Low | Improvement | No `-march` flag for local (non-cross) builds |
| 13 | `dashboard/app.py` | 🟢 Low | Improvement | Chart.js from CDN — fails in offline/edge deployments |
| 14 | CI `hil-test.yml` | 🟢 Low | Improvement | Dataset cache key invalidates on any config change |
| 15 | `test_units.py` | 🟢 Low | Improvement | Config mutation in tests not thread-safe; use `mock.patch` |
