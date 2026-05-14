"""
DiscoPyGal Integration Module.

This module provides integration with the DiscoPyGal motion planning library:
- Scene conversion from HouseExpo to DiscoPyGal format
- Oracle validation against DiscoPyGal collision detection
- TrapAwarePRM: Viability-aware sampling-based planner
"""

from .scene_converter import (
    houseexpo_to_discopygal,
    save_discopygal_scene,
    batch_convert_houseexpo,
)

from .trap_aware_prm import (
    TrapAwarePRM,
)

__all__ = [
    "houseexpo_to_discopygal",
    "save_discopygal_scene",
    "batch_convert_houseexpo",
    "TrapAwarePRM",
]
