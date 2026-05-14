"""
Utility Module - Common Utilities and Helper Functions.

This module provides:
- Setup verification
- Logging utilities
- Device management
- Reproducibility helpers
"""

from .verify_setup import verify_setup
from .helpers import (
    set_seed,
    get_device,
    count_parameters,
    format_time,
)

__all__ = [
    "verify_setup",
    "set_seed",
    "get_device",
    "count_parameters",
    "format_time",
]
