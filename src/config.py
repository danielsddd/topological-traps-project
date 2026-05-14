"""
Config loader for Directional Topological Traps.

Usage (from any script):
    from src.config import load_config
    cfg = load_config()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from configs.config_schema import (
    Config,
    DataConfig,
    RobotConfig,
    OracleConfig,
    ModelConfig,
    LossConfig,
    TrainingConfig,
    EvaluationConfig,
    LoggingConfig,
    DiscoPyGalConfig,
    PathConfig,
    ConfigError,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("configs/config.yaml")


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load and validate project configuration.

    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml.

    Returns:
        Fully populated Config dataclass.

    Raises:
        ConfigError: If the config file is missing or malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG

    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path.resolve()}\n"
            f"Run from the project root, or pass an explicit path."
        )

    logger.info(f"Loading config from: {path.resolve()}")

    with open(path, "r") as f:
        raw: dict = yaml.safe_load(f)

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}")

    cfg = _build_config(raw)
    _apply_env_overrides(cfg)
    _validate(cfg)

    logger.info("Config loaded successfully.")
    return cfg


def _build_config(raw: dict) -> Config:
    """Construct Config from parsed YAML dict."""
    cfg = Config()

    if "data" in raw:
        d = raw["data"]
        cfg.data = DataConfig(
            resolution=d.get("resolution", cfg.data.resolution),
            num_maps=d.get("num_maps", cfg.data.num_maps),
            train_split=d.get("split_ratios", {}).get("train", cfg.data.train_split),
            val_split=d.get("split_ratios", {}).get("val", cfg.data.val_split),
            test_split=d.get("split_ratios", {}).get("test", cfg.data.test_split),
            random_seed=d.get("random_seed", cfg.data.random_seed),
        )

    if "robot" in raw:
        r = raw["robot"]
        cfg.robot = RobotConfig(
            train_sizes=[tuple(s) for s in r.get("train_sizes", [[20,15],[30,20],[40,25]])],
            test_only_sizes=[tuple(s) for s in r.get("test_only_sizes", [[25,18]])],
        )

    if "oracle" in raw:
        o = raw["oracle"]
        cfg.oracle = OracleConfig(
            num_workers=o.get("num_workers", cfg.oracle.num_workers),
            directions=o.get("directions", cfg.oracle.directions),
        )

    if "model" in raw:
        m = raw["model"]
        cfg.model = ModelConfig(
            encoder=m.get("encoder", cfg.model.encoder),
            encoder_weights=m.get("encoder_weights", cfg.model.encoder_weights),
            in_channels=m.get("in_channels", cfg.model.in_channels),
            out_channels=m.get("out_channels", cfg.model.out_channels),
            activation=m.get("activation", cfg.model.activation),
        )

    if "training" in raw:
        t = raw["training"]
        loss_raw = t.get("loss", {})
        cfg.training = TrainingConfig(
            batch_size=t.get("batch_size", cfg.training.batch_size),
            epochs=t.get("epochs", cfg.training.epochs),
            learning_rate=t.get("learning_rate", cfg.training.learning_rate),
            weight_decay=t.get("weight_decay", cfg.training.weight_decay),
            optimizer=t.get("optimizer", cfg.training.optimizer),
            scheduler=t.get("scheduler", cfg.training.scheduler),
            warmup_epochs=t.get("warmup_epochs", cfg.training.warmup_epochs),
            mixed_precision=t.get("mixed_precision", cfg.training.mixed_precision),
            num_workers=t.get("num_workers", cfg.training.num_workers),
            pin_memory=t.get("pin_memory", cfg.training.pin_memory),
            gradient_clip=t.get("gradient_clip", cfg.training.gradient_clip),
            early_stopping_patience=t.get("early_stopping_patience", cfg.training.early_stopping_patience),
            resume=t.get("resume", cfg.training.resume),
            loss=LossConfig(
                bce_weight=loss_raw.get("bce_weight", 0.5),
                dice_weight=loss_raw.get("dice_weight", 0.5),
            ),
        )

    if "evaluation" in raw:
        e = raw["evaluation"]
        cfg.evaluation = EvaluationConfig(
            threshold=e.get("threshold", cfg.evaluation.threshold),
            speed_benchmark_maps=e.get("speed_benchmark_maps", cfg.evaluation.speed_benchmark_maps),
            speed_benchmark_repeats=e.get("speed_benchmark_repeats", cfg.evaluation.speed_benchmark_repeats),
        )

    if "logging" in raw:
        lg = raw["logging"]
        cfg.logging = LoggingConfig(
            use_wandb=lg.get("use_wandb", cfg.logging.use_wandb),
            wandb_project=lg.get("wandb_project", cfg.logging.wandb_project),
            wandb_entity=lg.get("wandb_entity", cfg.logging.wandb_entity),
            log_interval=lg.get("log_interval", cfg.logging.log_interval),
        )

    if "discopygal" in raw:
        dp = raw["discopygal"]
        cfg.discopygal = DiscoPyGalConfig(
            num_landmarks=dp.get("num_landmarks", cfg.discopygal.num_landmarks),
            k_nn=dp.get("k_nn", cfg.discopygal.k_nn),
            viability_threshold=dp.get("viability_threshold", cfg.discopygal.viability_threshold),
            trap_penalty=dp.get("trap_penalty", cfg.discopygal.trap_penalty),
        )

    if "paths" in raw:
        p = raw["paths"]
        cfg.paths = PathConfig(
            project_root=Path(p.get("project_root", ".")),
            raw_maps_dir=Path(p.get("raw_maps_dir", "data/raw_maps")),
            processed_dir=Path(p.get("processed_dir", "data/processed")),
            labels_dir=Path(p.get("labels_dir", "data/labels")),
            manifest_path=Path(p.get("manifest_path", "data/manifest.csv")),
            checkpoint_dir=Path(p.get("checkpoint_dir", "checkpoints")),
            log_dir=Path(p.get("log_dir", "logs")),
            output_dir=Path(p.get("output_dir", "outputs")),
            figures_dir=Path(p.get("figures_dir", "outputs/figures")),
            results_dir=Path(p.get("results_dir", "outputs/results")),
        )

    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    """Override path fields from environment variables."""
    env_map = {
        "PROJECT_DIR":    "project_root",
        "RAW_MAPS_DIR":   "raw_maps_dir",
        "PROCESSED_DIR":  "processed_dir",
        "LABELS_DIR":     "labels_dir",
        "MANIFEST_PATH":  "manifest_path",
        "CHECKPOINT_DIR": "checkpoint_dir",
        "LOG_DIR":        "log_dir",
        "OUTPUT_DIR":     "output_dir",
        "FIGURES_DIR":    "figures_dir",
        "RESULTS_DIR":    "results_dir",
    }
    for env_var, attr in env_map.items():
        val = os.environ.get(env_var)
        if val:
            setattr(cfg.paths, attr, Path(val))

    if wandb_entity := os.environ.get("WANDB_ENTITY"):
        cfg.logging.wandb_entity = wandb_entity
    if wandb_project := os.environ.get("WANDB_PROJECT"):
        cfg.logging.wandb_project = wandb_project


def _validate(cfg: Config) -> None:
    """Sanity-check the loaded config."""
    errors = []
    if cfg.model.in_channels != 3:
        errors.append(f"model.in_channels must be 3, got {cfg.model.in_channels}")
    if cfg.model.out_channels != 4:
        errors.append(f"model.out_channels must be 4, got {cfg.model.out_channels}")
    if cfg.training.batch_size < 1:
        errors.append("training.batch_size must be >= 1")
    if not cfg.robot.train_sizes:
        errors.append("robot.train_sizes is empty")
    loss = cfg.training.loss
    if abs(loss.bce_weight + loss.dice_weight - 1.0) > 1e-6:
        errors.append(f"loss weights must sum to 1.0, got {loss.bce_weight + loss.dice_weight}")
    if errors:
        raise ConfigError("Config validation failed:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = load_config()
        print("✓ Config loaded successfully")
        print(f"  train sizes : {cfg.robot.train_sizes}")
        print(f"  test sizes  : {cfg.robot.test_only_sizes}")
        print(f"  epochs      : {cfg.training.epochs}")
        print(f"  project_root: {cfg.paths.project_root}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)