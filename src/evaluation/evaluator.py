"""
Comprehensive Evaluation for Viability Prediction Model.

This module provides evaluation tools including:
- Overall metrics (IoU, Dice, Accuracy)
- Per-direction metrics (N, S, E, W)
- Per-robot-size evaluation
- Generalization testing (seen vs unseen sizes)
- Speed benchmarking (Oracle vs Neural Network)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json
from pathlib import Path

from ..models.metrics import (
    MetricTracker,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    compute_per_channel_metrics,
)
from ..data.dataset import RobotSpecificDataset


class Evaluator:
    """
    Comprehensive evaluation for viability prediction model.
    
    Provides methods for:
    - Overall evaluation on a dataset
    - Per-robot-size evaluation
    - Generalization testing
    - Speed benchmarking
    
    Args:
        model: Trained model
        device: Evaluation device
        threshold: Threshold for binary predictions
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            verbose: Show progress bar
        
        Returns:
            Dictionary with all metrics
        """
        tracker = MetricTracker(threshold=self.threshold)
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        for batch in iterator:
            inputs, labels, metadata = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            tracker.update(outputs, labels)
        
        return tracker.compute()
    
    @torch.no_grad()
    def evaluate_robot_size(
        self,
        map_dir: str,
        label_dir: str,
        robot_length: int,
        robot_width: int,
        file_list: List[str] = None,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Dict:
        """
        Evaluate model on a specific robot size.
        
        Args:
            map_dir: Directory with occupancy grids
            label_dir: Directory with labels for this robot size
            robot_length: Robot length
            robot_width: Robot width
            file_list: Specific files to evaluate (None = all)
            batch_size: Batch size
            num_workers: Data loading workers
        
        Returns:
            Metrics dictionary for this robot size
        """
        dataset = RobotSpecificDataset(
            map_dir=map_dir,
            label_dir=label_dir,
            robot_length=robot_length,
            robot_width=robot_width,
            file_list=file_list
        )
        
        if len(dataset) == 0:
            return {"error": "No data found"}
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return self.evaluate_dataset(dataloader, verbose=False)
    
    def evaluate_all_robot_sizes(
        self,
        map_dir: str,
        label_base_dir: str,
        robot_sizes: List[Tuple[int, int]],
        file_list: List[str] = None,
        batch_size: int = 16
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Evaluate on all robot sizes.
        
        Args:
            map_dir: Directory with occupancy grids
            label_base_dir: Base directory for labels
            robot_sizes: List of (length, width) tuples
            file_list: Files to evaluate
            batch_size: Batch size
        
        Returns:
            Dictionary mapping robot size to metrics
        """
        results = {}
        
        for length, width in tqdm(robot_sizes, desc="Robot sizes"):
            label_dir = Path(label_base_dir) / f"robot_{length}x{width}"
            
            metrics = self.evaluate_robot_size(
                map_dir=map_dir,
                label_dir=str(label_dir),
                robot_length=length,
                robot_width=width,
                file_list=file_list,
                batch_size=batch_size
            )
            
            results[(length, width)] = metrics
        
        return results
    
    def evaluate_generalization(
        self,
        map_dir: str,
        label_base_dir: str,
        train_sizes: List[Tuple[int, int]],
        test_only_sizes: List[Tuple[int, int]],
        file_list: List[str] = None,
        batch_size: int = 16
    ) -> Dict:
        """
        Evaluate generalization to unseen robot sizes.
        
        Compares performance on:
        - Seen sizes (used during training)
        - Unseen sizes (held out for testing)
        
        Args:
            map_dir: Directory with occupancy grids
            label_base_dir: Base directory for labels
            train_sizes: Sizes used during training
            test_only_sizes: Sizes held out for testing
            file_list: Files to evaluate
            batch_size: Batch size
        
        Returns:
            Dictionary with seen/unseen comparison
        """
        # Evaluate seen sizes
        seen_results = {}
        for length, width in train_sizes:
            label_dir = Path(label_base_dir) / f"robot_{length}x{width}"
            metrics = self.evaluate_robot_size(
                map_dir=map_dir,
                label_dir=str(label_dir),
                robot_length=length,
                robot_width=width,
                file_list=file_list,
                batch_size=batch_size
            )
            seen_results[(length, width)] = metrics
        
        # Evaluate unseen sizes
        unseen_results = {}
        for length, width in test_only_sizes:
            label_dir = Path(label_base_dir) / f"robot_{length}x{width}"
            metrics = self.evaluate_robot_size(
                map_dir=map_dir,
                label_dir=str(label_dir),
                robot_length=length,
                robot_width=width,
                file_list=file_list,
                batch_size=batch_size
            )
            unseen_results[(length, width)] = metrics
        
        # Compute averages
        def avg_metric(results, key):
            values = [r[key] for r in results.values() if key in r]
            return sum(values) / len(values) if values else 0
        
        return {
            "seen_sizes": seen_results,
            "unseen_sizes": unseen_results,
            "summary": {
                "seen_avg_iou": avg_metric(seen_results, "iou"),
                "seen_avg_dice": avg_metric(seen_results, "dice"),
                "unseen_avg_iou": avg_metric(unseen_results, "iou"),
                "unseen_avg_dice": avg_metric(unseen_results, "dice"),
                "generalization_gap_iou": (
                    avg_metric(seen_results, "iou") - avg_metric(unseen_results, "iou")
                ),
            }
        }
    
    @torch.no_grad()
    def predict_single(
        self,
        occupancy: np.ndarray,
        robot_length: int,
        robot_width: int,
        resolution: int = 512
    ) -> np.ndarray:
        """
        Predict viability for a single map and robot size.
        
        Args:
            occupancy: Occupancy grid (H, W)
            robot_length: Robot length
            robot_width: Robot width
            resolution: Resolution for normalization
        
        Returns:
            Predictions array (4, H, W) with probabilities
        """
        # Construct input
        H, W = occupancy.shape
        input_tensor = np.zeros((1, 3, H, W), dtype=np.float32)
        input_tensor[0, 0] = occupancy.astype(np.float32)
        input_tensor[0, 1] = robot_length / resolution
        input_tensor[0, 2] = robot_width / resolution
        
        input_tensor = torch.from_numpy(input_tensor).to(self.device)
        
        # Predict
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        
        return probs[0].cpu().numpy()
    
    def benchmark_inference_speed(
        self,
        dataloader: DataLoader,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict:
        """
        Benchmark model inference speed.
        
        Args:
            dataloader: DataLoader for benchmarking
            num_warmup: Warmup iterations
            num_iterations: Timed iterations
        
        Returns:
            Dictionary with timing statistics
        """
        data_iter = iter(dataloader)
        
        # Get sample batch
        try:
            inputs, _, _ = next(data_iter)
        except StopIteration:
            return {"error": "No data"}
        
        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = self.model(inputs)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Timed iterations
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(inputs)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        total_samples = batch_size * num_iterations
        ms_per_sample = (elapsed / total_samples) * 1000
        samples_per_second = total_samples / elapsed
        
        return {
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "total_time_s": elapsed,
            "ms_per_sample": ms_per_sample,
            "samples_per_second": samples_per_second,
            "ms_per_batch": (elapsed / num_iterations) * 1000,
        }


def evaluate_model(
    model_path: str,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict:
    """
    Convenience function to evaluate a saved model.
    
    Args:
        model_path: Path to model checkpoint
        dataloader: DataLoader for evaluation
        device: Device
    
    Returns:
        Metrics dictionary
    """
    from ..models.unet import MultiRobotViabilityUNet
    
    model = MultiRobotViabilityUNet.from_checkpoint(model_path, device=device)
    evaluator = Evaluator(model, device=device)
    
    return evaluator.evaluate_dataset(dataloader)


def evaluate_per_robot_size(
    model_path: str,
    map_dir: str,
    label_base_dir: str,
    robot_sizes: List[Tuple[int, int]],
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on each robot size separately.
    
    Args:
        model_path: Path to model checkpoint
        map_dir: Directory with occupancy grids
        label_base_dir: Base directory for labels
        robot_sizes: List of robot sizes
        device: Device
    
    Returns:
        Dictionary mapping robot size to metrics
    """
    from ..models.unet import MultiRobotViabilityUNet
    
    model = MultiRobotViabilityUNet.from_checkpoint(model_path, device=device)
    evaluator = Evaluator(model, device=device)
    
    return evaluator.evaluate_all_robot_sizes(
        map_dir=map_dir,
        label_base_dir=label_base_dir,
        robot_sizes=robot_sizes
    )


def benchmark_speed(
    model_path: str,
    oracle_func,
    map_dir: str,
    robot_length: int = 14,
    robot_width: int = 9,
    num_maps: int = 100,
    device: str = "cuda"
) -> Dict:
    """
    Benchmark Neural Network vs Oracle speed.
    
    Args:
        model_path: Path to model checkpoint
        oracle_func: Oracle function (occupancy, length, width) -> labels
        map_dir: Directory with occupancy grids
        robot_length: Robot length for testing
        robot_width: Robot width for testing
        num_maps: Number of maps to test
        device: Device
    
    Returns:
        Speed comparison dictionary
    """
    from ..models.unet import MultiRobotViabilityUNet
    
    model = MultiRobotViabilityUNet.from_checkpoint(model_path, device=device)
    model.eval()
    
    # Load maps
    map_dir = Path(map_dir)
    map_files = sorted(map_dir.glob("*.npy"))[:num_maps]
    
    if not map_files:
        return {"error": "No maps found"}
    
    occupancies = [np.load(f) for f in map_files]
    
    # Benchmark Oracle
    oracle_times = []
    for occ in tqdm(occupancies[:min(10, len(occupancies))], desc="Oracle"):
        start = time.time()
        _ = oracle_func(occ, robot_length, robot_width)
        oracle_times.append(time.time() - start)
    
    avg_oracle_ms = np.mean(oracle_times) * 1000
    
    # Benchmark Neural Network
    nn_times = []
    
    for occ in tqdm(occupancies, desc="Neural Network"):
        H, W = occ.shape
        input_tensor = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)
        input_tensor[0, 0] = torch.from_numpy(occ.astype(np.float32))
        input_tensor[0, 1] = robot_length / 512
        input_tensor[0, 2] = robot_width / 512
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        nn_times.append(time.time() - start)
    
    avg_nn_ms = np.mean(nn_times) * 1000
    
    return {
        "oracle_avg_ms": avg_oracle_ms,
        "nn_avg_ms": avg_nn_ms,
        "speedup": avg_oracle_ms / avg_nn_ms if avg_nn_ms > 0 else float('inf'),
        "num_maps": len(occupancies),
    }


def save_evaluation_results(
    results: Dict,
    output_path: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    # Convert tuple keys to strings
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, tuple) else k: convert_keys(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj
    
    converted = convert_keys(results)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)
def benchmark_fleet_scaling(
    model,
    map_dir: str,
    robot_sizes_pool: List[Tuple[int, int]],
    fleet_sizes: List[int] = None,
    num_maps: int = 10,
    device: str = "cuda",
    resolution: int = 512,
) -> Dict:
    """
    Benchmark Oracle vs Model as fleet size (number of robot sizes) grows.

    Demonstrates the model's architectural advantage:
      - Oracle cost scales linearly with fleet size (must rerun per size)
      - Model sequential: also linear but much faster per size
      - Model batched: nearly constant (all sizes in one GPU forward pass)

    Args:
        model:            Trained model.
        map_dir:          Directory with .npy occupancy grids.
        robot_sizes_pool: All available robot sizes to sample fleet from.
        fleet_sizes:      List of fleet sizes to benchmark. Default [1,2,3,5,10].
        num_maps:         Maps to average over.
        device:           cuda or cpu.
        resolution:       Grid resolution for input normalization.

    Returns:
        Dict with timing results per fleet size and per method.
    """
    from src.oracle.directional_viability import generate_labels_for_map

    if fleet_sizes is None:
        fleet_sizes = [1, 2, 3, 5, 10]

    # Clamp fleet sizes to available robot sizes
    fleet_sizes = [f for f in fleet_sizes if f <= len(robot_sizes_pool)]

    # Load maps
    map_files  = sorted(Path(map_dir).glob("*.npy"))[:num_maps]
    occupancies = [np.load(f) for f in map_files]
    if not occupancies:
        return {"error": "No maps found"}

    results = {}

    for fleet_size in fleet_sizes:
        # Pick first fleet_size robot sizes from pool
        fleet = robot_sizes_pool[:fleet_size]

        oracle_times_total    = []
        model_seq_times_total = []
        model_batch_times_total = []

        for occ in occupancies:
            H, W = occ.shape

            # ---- Oracle: run once per robot size, sequentially ----
            t0 = time.time()
            for L, W_r in fleet:
                _ = generate_labels_for_map(occ, L, W_r)
            oracle_times_total.append(time.time() - t0)

            # ---- Model sequential: one forward pass per size ----
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for L, W_r in fleet:
                inp = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)
                inp[0, 0] = torch.from_numpy(occ.astype(np.float32))
                inp[0, 1] = L   / resolution
                inp[0, 2] = W_r / resolution
                with torch.no_grad():
                    _ = model(inp)
            if device == "cuda":
                torch.cuda.synchronize()
            model_seq_times_total.append(time.time() - t0)

            # ---- Model batched: all sizes in ONE forward pass ----
            batch = torch.zeros(fleet_size, 3, H, W, dtype=torch.float32, device=device)
            occ_tensor = torch.from_numpy(occ.astype(np.float32))
            for idx, (L, W_r) in enumerate(fleet):
                batch[idx, 0] = occ_tensor
                batch[idx, 1] = L   / resolution
                batch[idx, 2] = W_r / resolution

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                _ = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            model_batch_times_total.append(time.time() - t0)

        results[fleet_size] = {
            "fleet":              [f"{L}x{W}" for L, W in fleet],
            "oracle_ms":          round(np.mean(oracle_times_total) * 1000, 2),
            "model_seq_ms":       round(np.mean(model_seq_times_total) * 1000, 2),
            "model_batch_ms":     round(np.mean(model_batch_times_total) * 1000, 2),
            "speedup_vs_oracle_seq":   round(
                np.mean(oracle_times_total) / np.mean(model_seq_times_total), 1),
            "speedup_vs_oracle_batch": round(
                np.mean(oracle_times_total) / np.mean(model_batch_times_total), 1),
        }

    return results


def plot_fleet_scaling(
    fleet_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot Oracle vs Model scaling as fleet size grows.

    Generates two panels:
      Left:  Absolute time (ms) vs fleet size for all three methods
      Right: Speedup (Oracle / Model) vs fleet size
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    sizes       = sorted(fleet_results.keys())
    oracle_ms   = [fleet_results[s]["oracle_ms"]      for s in sizes]
    seq_ms      = [fleet_results[s]["model_seq_ms"]   for s in sizes]
    batch_ms    = [fleet_results[s]["model_batch_ms"] for s in sizes]
    speedup_seq = [fleet_results[s]["speedup_vs_oracle_seq"]   for s in sizes]
    speedup_bat = [fleet_results[s]["speedup_vs_oracle_batch"] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: absolute time ----
    ax1.plot(sizes, oracle_ms,  "o-", color="crimson",    linewidth=2.5,
             label="Oracle (rerun per size)", markersize=8)
    ax1.plot(sizes, seq_ms,     "s-", color="steelblue",  linewidth=2.5,
             label="Model sequential",        markersize=8)
    ax1.plot(sizes, batch_ms,   "^-", color="seagreen",   linewidth=2.5,
             label="Model batched (1 pass)",  markersize=8)

    for i, s in enumerate(sizes):
        ax1.annotate(f"{oracle_ms[i]:.0f}ms",
                     (s, oracle_ms[i]),  textcoords="offset points",
                     xytext=(0, 8),  ha="center", fontsize=9, color="crimson")
        ax1.annotate(f"{batch_ms[i]:.0f}ms",
                     (s, batch_ms[i]), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9, color="seagreen")

    ax1.set_xlabel("Fleet size (number of robot sizes queried)", fontsize=12)
    ax1.set_ylabel("Total inference time (ms)", fontsize=12)
    ax1.set_title("Inference Time vs Fleet Size", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xticks(sizes)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ---- Right: speedup ----
    ax2.plot(sizes, speedup_seq, "s-", color="steelblue", linewidth=2.5,
             label="Oracle / Model sequential", markersize=8)
    ax2.plot(sizes, speedup_bat, "^-", color="seagreen",  linewidth=2.5,
             label="Oracle / Model batched",    markersize=8)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Baseline (1×)")

    for i, s in enumerate(sizes):
        ax2.annotate(f"{speedup_bat[i]:.0f}×",
                     (s, speedup_bat[i]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=10, color="seagreen",
                     fontweight="bold")

    ax2.set_xlabel("Fleet size (number of robot sizes queried)", fontsize=12)
    ax2.set_ylabel("Speedup over Oracle", fontsize=12)
    ax2.set_title("Speedup vs Fleet Size\n(higher = better)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xticks(sizes)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.suptitle(
        "Neural Network Fleet Scaling Advantage\n"
        "Oracle cost scales linearly with fleet size — Model batched cost is nearly constant",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Fleet scaling figure saved to {save_path}")
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Use Evaluator class or convenience functions for evaluation")
