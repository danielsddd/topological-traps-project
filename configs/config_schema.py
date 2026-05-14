"""
Configuration schema and dataclasses for Directional Topological Traps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    resolution: int = 512
    num_maps: int = 800
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    def __post_init__(self):
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class RobotConfig:
    train_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (20, 15),
        (30, 20),
        (40, 25),
    ])
    test_only_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (25, 18),
    ])

    @property
    def all_sizes(self) -> List[Tuple[int, int]]:
        return self.train_sizes + self.test_only_sizes

    def get_size_tag(self, length: int, width: int) -> str:
        return f"robot_{length}x{width}"

    def is_train_size(self, length: int, width: int) -> bool:
        return (length, width) in self.train_sizes

    def is_test_only_size(self, length: int, width: int) -> bool:
        return (length, width) in self.test_only_sizes


@dataclass
class OracleConfig:
    num_workers: int = 16
    directions: List[str] = field(default_factory=lambda: ["N", "S", "E", "W"])


@dataclass
class ModelConfig:
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    out_channels: int = 4
    activation: Optional[str] = None


@dataclass
class LossConfig:
    bce_weight: float = 0.5
    dice_weight: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    resume: bool = False
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    speed_benchmark_maps: int = 50
    speed_benchmark_repeats: int = 5


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "topological-traps"
    wandb_entity: Optional[str] = None
    log_interval: int = 10


@dataclass
class DiscoPyGalConfig:
    num_landmarks: int = 1000
    k_nn: int = 15
    viability_threshold: float = 0.5
    trap_penalty: float = 10.0


@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: Path("."))
    raw_maps_dir: Path = field(default_factory=lambda: Path("data/raw_maps"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    labels_dir: Path = field(default_factory=lambda: Path("data/labels"))
    manifest_path: Path = field(default_factory=lambda: Path("data/manifest.csv"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    results_dir: Path = field(default_factory=lambda: Path("outputs/results"))

    def label_dir_for(self, length: int, width: int) -> Path:
        return self.labels_dir / f"robot_{length}x{width}"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    discopygal: DiscoPyGalConfig = field(default_factory=DiscoPyGalConfig)
    paths: PathConfig = field(default_factory=PathConfig)


class Direction:
    NORTH = 0
    SOUTH = 1
    EAST  = 2
    WEST  = 3
    NAMES       = ["North", "South", "East", "West"]
    SHORT_NAMES = ["N", "S", "E", "W"]
    VECTORS     = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    ROTATE_90_CW  = [2, 3, 1, 0]
    ROTATE_180    = [1, 0, 3, 2]
    ROTATE_270_CW = [3, 2, 0, 1]
    FLIP_H        = [0, 1, 3, 2]
    FLIP_V        = [1, 0, 2, 3]

    @classmethod
    def vector(cls, d: int) -> Tuple[int, int]:
        return cls.VECTORS[d]

    @classmethod
    def name(cls, d: int) -> str:
        return cls.NAMES[d]

    @classmethod
    def short(cls, d: int) -> str:
        return cls.SHORT_NAMES[d]


# Exceptions
class TopologicalTrapsError(Exception): pass
class MapLoadError(TopologicalTrapsError): pass
class CorruptedMapError(TopologicalTrapsError): pass
class ConfigError(TopologicalTrapsError): pass
class DataError(TopologicalTrapsError): pass