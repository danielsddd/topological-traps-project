"""
General Helper Utilities.

Common utilities used throughout the project:
- Random seed management
- Device selection
- Parameter counting
- Time formatting
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_gpu: bool = True) -> str:
    """
    Get appropriate device for computation.
    
    Args:
        prefer_gpu: Prefer GPU if available
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format time duration for display.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2h 30m 15s" or "45.3s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_number(n: int) -> str:
    """
    Format large numbers for display.
    
    Args:
        n: Number to format
    
    Returns:
        Formatted string (e.g., "24.5M" for 24,500,000)
    """
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Size in MB (assuming float32)
    """
    params = sum(p.numel() for p in model.parameters())
    return params * 4 / (1024 ** 2)


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 512, 512)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Example input shape
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"  Total parameters:     {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"  Non-trainable:        {format_number(total_params - trainable_params)}")
    print(f"  Model size:           {get_model_size_mb(model):.1f} MB")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Input shape:          {list(input_size)}")
        print(f"  Output shape:         {list(output.shape)}")
    except Exception as e:
        print(f"  Forward pass failed:  {e}")
    
    print("=" * 60)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        The same path (for chaining)
    """
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str) -> dict:
    """Load JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: str, indent: int = 2):
    """Save dictionary to JSON file."""
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save dictionary to YAML file."""
    import yaml
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


class Timer:
    """
    Simple timer context manager.
    
    Usage:
        with Timer("Training"):
            train()
        # Prints: "Training took 2m 30s"
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"{self.name} took {format_time(self.elapsed)}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Usage:
        meter = AverageMeter()
        for batch in dataloader:
            loss = ...
            meter.update(loss.item(), batch_size)
        print(f"Average loss: {meter.avg}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


if __name__ == "__main__":
    # Test utilities
    print("Testing helper utilities...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random number after seed: {random.random():.6f}")
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test time formatting
    print(f"45 seconds: {format_time(45)}")
    print(f"125 seconds: {format_time(125)}")
    print(f"7325 seconds: {format_time(7325)}")
    
    # Test number formatting
    print(f"1234: {format_number(1234)}")
    print(f"24500000: {format_number(24500000)}")
    
    # Test timer
    import time
    with Timer("Sleep test"):
        time.sleep(0.1)
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average of 0-9: {meter.avg}")
    
    print("\n✓ All helper tests passed!")
