"""
configs/settings.py
Loads config.yaml and exposes a typed Settings object.
Every Python module imports from here — never reads YAML directly.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    task: str
    teacher_arch: str
    teacher_weights: str
    student_arch: str
    dataset: str
    num_classes: int
    data_dir: str
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class CompressionConfig:
    pruning_ratio: float
    quantization_mode: str
    calibration_samples: int
    distillation_temperature: float
    distillation_alpha: float


@dataclass
class DeploymentConfig:
    target_arch: str
    latency_gate_ms: float
    accuracy_threshold: float
    memory_limit_mb: int
    server_port: int
    input_height: int
    input_width: int
    input_channels: int


@dataclass
class LoggingConfig:
    level: str
    output_dir: str
    benchmark_json: str
    dashboard_port: int


@dataclass
class Settings:
    model: ModelConfig
    compression: CompressionConfig
    deployment: DeploymentConfig
    logging: LoggingConfig

    # Convenience helpers
    @property
    def input_shape(self) -> tuple[int, int, int]:
        d = self.deployment
        return (d.input_channels, d.input_height, d.input_width)

    @property
    def output_dir(self) -> Path:
        p = Path(self.logging.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def onnx_path(self) -> Path:
        return self.output_dir / "student_int8.onnx"

    @property
    def student_checkpoint(self) -> Path:
        return self.output_dir / "student_distilled.pth"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Load settings from YAML.  Searches in order:
      1. Explicit argument
      2. EDGE_CV_CONFIG env var
      3. <project_root>/configs/config.yaml
    """
    if config_path is None:
        config_path = os.environ.get(
            "EDGE_CV_CONFIG",
            Path(__file__).parent / "config.yaml",
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Malformed config at {config_path}: expected dictionary.")

    try:
        return Settings(
            model=ModelConfig(**raw["model"]),
            compression=CompressionConfig(**raw["compression"]),
            deployment=DeploymentConfig(**raw["deployment"]),
            logging=LoggingConfig(**raw["logging"]),
        )
    except KeyError as e:
        raise KeyError(f"Missing required config section: {e}") from e
    except TypeError as e:
        raise TypeError(f"Invalid or missing fields in config: {e}") from e


# Module-level singleton — import this instead of calling load_settings()
_cfg: Settings | None = None


def get_cfg() -> Settings:
    global _cfg
    if _cfg is None:
        _cfg = load_settings()
    return _cfg


# For backward compatibility with existing imports
class _CfgProxy:
    def __getattr__(self, name):
        return getattr(get_cfg(), name)


cfg: Settings = _CfgProxy()  # type: ignore