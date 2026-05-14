"""
Evaluation Module - Comprehensive Model Evaluation and Benchmarking.

This module provides:
- Overall metrics computation
- Per-robot-size evaluation
- Generalization testing (seen vs unseen sizes)
- Speed benchmarking (Oracle vs Neural Network)
- Visualization of predictions
"""

from .evaluator import (
    Evaluator,
    evaluate_model,
    evaluate_per_robot_size,
    benchmark_speed,
)

__all__ = [
    "Evaluator",
    "evaluate_model",
    "evaluate_per_robot_size",
    "benchmark_speed",
]
