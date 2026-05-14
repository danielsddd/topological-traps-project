"""
src/planning/__init__.py

Motion planning module: Dynamic Window Approach (DWA) local planner
with optional viability cost injection via the trained continuous-angle
U-Net model.
"""

from .dwa_planner import DWAPlanner, DWAConfig, DWAState

__all__ = ["DWAPlanner", "DWAConfig", "DWAState"]