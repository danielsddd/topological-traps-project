"""
Setup Verification Utility.

Verifies that the environment is correctly configured for training:
- Python packages installed
- CUDA available
- Directories exist
- Config file valid
- Sufficient disk space
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def verify_setup(
    config_path: str = "configs/config.yaml",
    verbose: bool = True
) -> bool:
    """
    Verify environment setup for training.
    
    Checks:
    1. Required Python packages
    2. CUDA availability
    3. Directory structure
    4. Configuration file
    5. Disk space
    6. HouseExpo data (optional)
    
    Args:
        config_path: Path to configuration file
        verbose: Print detailed output
    
    Returns:
        True if all checks pass
    """
    results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0,
    }
    
    if verbose:
        print("=" * 60)
        print("ENVIRONMENT VERIFICATION")
        print("=" * 60)
    
    # 1. Check Python packages
    if verbose:
        print("\n=== Checking Python Packages ===")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
        ("segmentation_models_pytorch", "segmentation_models_pytorch"),
    ]
    
    optional_packages = [
        ("tensorboard", "TensorBoard"),
    ]
    
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            if verbose:
                print(f"  [✓] {display_name}")
            results["passed"] += 1
        except ImportError:
            if verbose:
                print(f"  [✗] {display_name} - NOT INSTALLED")
            results["failed"] += 1
    
    for module_name, display_name in optional_packages:
        try:
            __import__(module_name)
            if verbose:
                print(f"  [✓] {display_name} (optional)")
            results["passed"] += 1
        except ImportError:
            if verbose:
                print(f"  [!] {display_name} (optional) - not installed")
            results["warnings"] += 1
    
    # 2. Check CUDA
    if verbose:
        print("\n=== Checking CUDA ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if verbose:
                print(f"  [✓] CUDA available: {device_name}")
                print(f"  [✓] GPU Memory: {device_mem:.1f} GB")
            results["passed"] += 1
        else:
            if verbose:
                print("  [!] CUDA not available - will use CPU")
            results["warnings"] += 1
    except Exception as e:
        if verbose:
            print(f"  [✗] Error checking CUDA: {e}")
        results["failed"] += 1
    
    # 3. Check directories
    if verbose:
        print("\n=== Checking Directories ===")
    
    required_dirs = [
        "configs",
        "data",
        "data/raw_maps",
        "data/processed",
        "data/labels",
        "src",
        "scripts",
        "outputs",
        "logs",
    ]
    
    # Robot size directories
    robot_sizes = [(6, 4), (10, 6), (14, 9), (18, 11), (22, 14)]
    for length, width in robot_sizes:
        required_dirs.append(f"data/labels/robot_{length}x{width}")
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            if verbose:
                print(f"  [✓] {dir_path}/")
            results["passed"] += 1
        else:
            if verbose:
                print(f"  [!] {dir_path}/ - missing (will be created)")
            results["warnings"] += 1
            # Try to create it
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    
    # 4. Check config file
    if verbose:
        print("\n=== Checking Configuration ===")
    
    config_path = Path(config_path)
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ["paths", "data", "robot_sizes", "model", "training"]
            missing_keys = [k for k in required_keys if k not in config]
            
            if not missing_keys:
                if verbose:
                    print(f"  [✓] Config file valid: {config_path}")
                results["passed"] += 1
            else:
                if verbose:
                    print(f"  [!] Config missing keys: {missing_keys}")
                results["warnings"] += 1
        except Exception as e:
            if verbose:
                print(f"  [✗] Config file error: {e}")
            results["failed"] += 1
    else:
        if verbose:
            print(f"  [!] Config file not found: {config_path}")
        results["warnings"] += 1
    
    # 5. Check disk space
    if verbose:
        print("\n=== Checking Disk Space ===")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if verbose:
            print(f"  Free space: {free_gb:.1f} GB")
        
        if free_gb >= 50:
            if verbose:
                print(f"  [✓] Sufficient space (need ~50GB)")
            results["passed"] += 1
        else:
            if verbose:
                print(f"  [!] Low space - may need ~50GB for full dataset")
            results["warnings"] += 1
    except Exception as e:
        if verbose:
            print(f"  [!] Could not check disk space: {e}")
        results["warnings"] += 1
    
    # 6. Check for data (optional)
    if verbose:
        print("\n=== Checking Data ===")
    
    raw_maps = Path("data/raw_maps")
    if raw_maps.exists():
        json_files = list(raw_maps.glob("*.json"))
        if json_files:
            if verbose:
                print(f"  [✓] Found {len(json_files)} raw map files")
            results["passed"] += 1
        else:
            if verbose:
                print(f"  [!] No JSON files in data/raw_maps/")
            results["warnings"] += 1
    else:
        if verbose:
            print(f"  [!] data/raw_maps/ directory missing")
        results["warnings"] += 1
    
    processed_maps = Path("data/processed")
    if processed_maps.exists():
        npy_files = list(processed_maps.glob("*.npy"))
        if npy_files:
            if verbose:
                print(f"  [✓] Found {len(npy_files)} processed maps")
            results["passed"] += 1
        else:
            if verbose:
                print(f"  [!] No processed maps yet")
            results["warnings"] += 1
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Passed:   {results['passed']}")
        print(f"  Failed:   {results['failed']}")
        print(f"  Warnings: {results['warnings']}")
    
    success = results["failed"] == 0
    
    if verbose:
        if success:
            print("\n✓ Setup verification PASSED")
        else:
            print("\n✗ Setup verification FAILED - fix errors above")
    
    return success


def check_gpu_memory() -> Dict:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory stats
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        cached = torch.cuda.memory_reserved(device) / (1024**3)
        free = total - allocated
        
        return {
            "available": True,
            "device_name": props.name,
            "total_gb": total,
            "allocated_gb": allocated,
            "cached_gb": cached,
            "free_gb": free,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_system_info() -> Dict:
    """
    Get system information.
    
    Returns:
        Dictionary with system info
    """
    import platform
    
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    # PyTorch version
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
    except ImportError:
        info["pytorch_version"] = "Not installed"
    
    # NumPy version
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = "Not installed"
    
    return info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify setup")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    success = verify_setup(args.config, verbose=not args.quiet)
    
    print("\n=== System Info ===")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")
    
    print("\n=== GPU Info ===")
    for key, value in check_gpu_memory().items():
        print(f"  {key}: {value}")
    
    sys.exit(0 if success else 1)
