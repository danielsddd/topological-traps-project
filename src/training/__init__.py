"""
Training Module - Training Pipeline and Callbacks.

This module provides:
- Trainer class for orchestrating training
- Mixed precision (AMP) support
- Callbacks for early stopping, checkpointing, logging
- Learning rate scheduling
"""

from .trainer import (
    Trainer,
    create_optimizer,
    create_scheduler,
)

from .callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LRSchedulerCallback,
    TensorBoardCallback,
    ProgressCallback,
)

__all__ = [
    # Trainer
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    # Callbacks
    "Callback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LRSchedulerCallback",
    "TensorBoardCallback",
    "ProgressCallback",
]
