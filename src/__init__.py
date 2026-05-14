"""
Directional Topological Traps - Source Package

This package contains all source code for predicting heading-dependent
viability maps for non-holonomic robots using deep learning.

Modules:
    data: Data loading, preprocessing, and dataset management
    oracle: Ground truth generation using morphological operations and BFS
    models: Neural network architectures (U-Net) and loss functions
    training: Training pipeline with mixed precision and callbacks
    evaluation: Metrics computation and benchmarking
    visualization: Plotting and figure generation
    discopygal_integration: Integration with DiscoPyGal motion planning
    utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Daniel Simanovsky"
__description__ = "Deep learning for directional topological trap prediction"
