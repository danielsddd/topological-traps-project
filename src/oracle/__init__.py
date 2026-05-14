"""
Oracle — Ground truth directional viability label generation.

Pipeline per (map, robot_size):
    1. Rotation-safe mask  — circular erosion (can robot rotate here?)
    2. Translation masks   — rectangular erosion × 4 directions (can robot fit here?)
    3. Reverse BFS         — flood-fill from safe seeds (can robot escape?)
"""
from .rotation_check       import compute_rotation_safe_mask, visualize_rotation_safe
from .translation_check    import compute_translation_safe_mask, compute_all_translation_masks
from .directional_viability import (
    compute_viability_single_direction,
    compute_viability_all_directions,
    generate_labels_for_map,
    generate_labels_all_robots,
)
from .generator import LabelGenerator, DatasetVerifier, run_oracle_dataset

__all__ = [
    "compute_rotation_safe_mask",
    "visualize_rotation_safe",
    "compute_translation_safe_mask",
    "compute_all_translation_masks",
    "compute_viability_single_direction",
    "compute_viability_all_directions",
    "generate_labels_for_map",
    "generate_labels_all_robots",
    "LabelGenerator",
    "DatasetVerifier",
    "run_oracle_dataset",
]