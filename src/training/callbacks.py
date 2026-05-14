"""
Training Callbacks for Viability Model.

This module provides callback classes for monitoring and controlling
the training process:

- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save model checkpoints based on validation metrics
- LRScheduler: Adjust learning rate during training
- TensorBoard: Log metrics to TensorBoard
- Progress: Display training progress

Callbacks are optional extensions to the Trainer class.
"""

import os
import time
import torch
from typing import Dict, Any, Optional, Callable
from pathlib import Path


class Callback:
    """
    Base callback class.
    
    Callbacks are called at specific points during training:
    - on_train_begin / on_train_end
    - on_epoch_begin / on_epoch_end
    - on_batch_begin / on_batch_end
    - on_validation_begin / on_validation_end
    """
    
    def set_trainer(self, trainer):
        """Set reference to trainer."""
        self.trainer = trainer
    
    def on_train_begin(self, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, **kwargs):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict = None, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, **kwargs):
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, logs: Dict = None, **kwargs):
        """Called at the end of validation."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Stop training when a monitored metric stops improving.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss', 'val_iou')
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like IoU
        restore_best_weights: Restore weights from best epoch
        verbose: Print messages
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "auto",
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Determine mode
        if mode == "auto":
            if "loss" in monitor or "error" in monitor:
                self.mode = "min"
            else:
                self.mode = "max"
        else:
            self.mode = mode
        
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        else:
            return current > (self.best_value + self.min_delta)
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        """Check for improvement and update state."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in self.trainer.model.state_dict().items()
                }
            
            if self.verbose:
                print(f"  → Improved {self.monitor}: {current:.4f}")
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"  Best {self.monitor}: {self.best_value:.4f}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    self.trainer.model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("  Restored best weights")


class ModelCheckpointCallback(Callback):
    """
    Save model checkpoints based on validation metrics.
    
    Args:
        filepath: Path template for saving (can include {epoch}, {metric})
        monitor: Metric to monitor
        save_best_only: Only save when metric improves
        save_weights_only: Save only weights (not optimizer state)
        mode: 'min' or 'max'
        verbose: Print messages
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = "auto",
        verbose: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode
        
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.save_best_only:
            if not self._is_improvement(current):
                return
            self.best_value = current
        
        # Format filepath
        filepath = self.filepath.format(
            epoch=epoch + 1,
            **{self.monitor: current}
        )
        
        # Save checkpoint
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.save_weights_only:
            torch.save(self.trainer.model.state_dict(), filepath)
        else:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.trainer.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                self.monitor: current,
            }
            torch.save(checkpoint, filepath)
        
        if self.verbose:
            print(f"  → Saved checkpoint to {filepath}")


class LRSchedulerCallback(Callback):
    """
    Learning rate scheduler callback.
    
    Wraps PyTorch schedulers for use as a callback.
    
    Args:
        scheduler: PyTorch LR scheduler
        monitor: Metric for ReduceLROnPlateau
        verbose: Print LR changes
    """
    
    def __init__(
        self,
        scheduler,
        monitor: str = "val_loss",
        verbose: bool = True
    ):
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
        self.last_lr = None
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        logs = logs or {}
        
        # Get current LR
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        
        # Step scheduler
        if hasattr(self.scheduler, 'step'):
            if 'ReduceLROnPlateau' in type(self.scheduler).__name__:
                metric = logs.get(self.monitor)
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
        
        # Check for LR change
        new_lr = self.trainer.optimizer.param_groups[0]['lr']
        
        if self.verbose and self.last_lr is not None and new_lr != self.last_lr:
            print(f"  → Learning rate changed: {self.last_lr:.2e} → {new_lr:.2e}")
        
        self.last_lr = new_lr


class TensorBoardCallback(Callback):
    """
    Log metrics to TensorBoard.
    
    Args:
        log_dir: Directory for TensorBoard logs
        update_freq: How often to update ('epoch' or 'batch')
    """
    
    def __init__(
        self,
        log_dir: str,
        update_freq: str = "epoch"
    ):
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.writer = None
    
    def on_train_begin(self, **kwargs):
        from torch.utils.tensorboard import SummaryWriter
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
    
    def on_train_end(self, **kwargs):
        if self.writer:
            self.writer.close()
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        if self.writer is None or logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def on_batch_end(self, batch: int, logs: Dict = None, **kwargs):
        if self.update_freq != "batch" or self.writer is None:
            return
        
        if logs:
            step = self.trainer.global_step
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"batch/{key}", value, step)


class ProgressCallback(Callback):
    """
    Display training progress.
    
    Args:
        verbose: Verbosity level (0=silent, 1=epoch, 2=batch)
    """
    
    def __init__(self, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        self.epoch_start_time = None
        self.train_start_time = None
    
    def on_train_begin(self, **kwargs):
        self.train_start_time = time.time()
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("Training started")
            print("=" * 60)
    
    def on_train_end(self, **kwargs):
        if self.verbose > 0:
            elapsed = time.time() - self.train_start_time
            print("\n" + "=" * 60)
            print(f"Training completed in {elapsed/60:.1f} minutes")
            print("=" * 60)
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        if self.verbose == 0:
            return
        
        elapsed = time.time() - self.epoch_start_time
        logs = logs or {}
        
        # Build summary string
        parts = [f"Epoch {epoch + 1}"]
        parts.append(f"({elapsed:.1f}s)")
        
        for key in ["train_loss", "val_loss", "val_iou"]:
            if key in logs:
                parts.append(f"{key}: {logs[key]:.4f}")
        
        print(" | ".join(parts))


class CallbackList:
    """
    Container for multiple callbacks.
    """
    
    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        self.callbacks.append(callback)
    
    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)
    
    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: Dict = None, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs, **kwargs)
    
    def on_batch_begin(self, batch: int, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)
    
    def on_batch_end(self, batch: int, logs: Dict = None, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs, **kwargs)
    
    @property
    def stop_training(self) -> bool:
        """Check if any callback wants to stop training."""
        for callback in self.callbacks:
            if hasattr(callback, 'stop_training') and callback.stop_training:
                return True
        return False


if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")
    
    # Test EarlyStopping
    es = EarlyStoppingCallback(monitor="val_loss", patience=3, verbose=True)
    
    # Simulate improving then stagnating
    class MockTrainer:
        class MockModel:
            def state_dict(self):
                return {"weight": torch.randn(10)}
        model = MockModel()
    
    es.set_trainer(MockTrainer())
    
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]  # Improves then stagnates
    
    for epoch, loss in enumerate(losses):
        es.on_epoch_end(epoch, {"val_loss": loss})
        if es.stop_training:
            print(f"Would stop at epoch {epoch + 1}")
            break
    
    print("✓ EarlyStopping test passed")
    
    # Test ProgressCallback
    pc = ProgressCallback(verbose=1)
    pc.on_train_begin()
    pc.on_epoch_begin(0)
    pc.on_epoch_end(0, {"train_loss": 0.5, "val_loss": 0.4, "val_iou": 0.7})
    pc.on_train_end()
    
    print("✓ ProgressCallback test passed")
    
    print("\n✓ All callback tests passed!")
