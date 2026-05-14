"""
Visualization Module - Plotting and Figure Generation.

This module provides visualization tools for:
- Training curves (loss, metrics)
- Predictions vs ground truth
- Per-direction viability maps
- Robot size comparisons
- Publication-quality figures
"""

from .plotting import (
    plot_training_curves,
    plot_predictions,
    plot_viability_comparison,
    plot_per_direction_viability,
    plot_robot_size_comparison,
    create_figure_grid,
)

__all__ = [
    "plot_training_curves",
    "plot_predictions",
    "plot_viability_comparison",
    "plot_per_direction_viability",
    "plot_robot_size_comparison",
    "create_figure_grid",
]
