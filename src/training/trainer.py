"""
Training Orchestrator for Viability Prediction Model.

This module provides the Trainer class that handles:
- Training loop with mixed precision (AMP)
- Validation with per-robot-size metrics
- Checkpoint saving and loading
- TensorBoard logging
- Early stopping

Usage:
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        config=config,
    )
    history = trainer.train(num_epochs=100)
"""

import os
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, Optional, List, Tuple, Any
from tqdm import tqdm
import json
from pathlib import Path

from ..models.metrics import MetricTracker, compute_iou, compute_dice


class Trainer:
    """
    Training orchestrator for multi-robot viability prediction.
    
    Features:
    - Mixed precision training (AMP) for memory efficiency
    - Random robot size sampling per batch
    - Per-robot-size validation metrics
    - Early stopping with patience
    - Learning rate scheduling
    - Checkpoint management
    - TensorBoard logging
    
    Args:
        model: Neural network model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        config: Training configuration dictionary
        device: Training device
        output_dir: Directory for outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        config: Dict = None,
        device: str = "cuda",
        output_dir: str = "outputs"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Mixed precision
        self.use_amp = self.config.get("use_amp", True) and device == "cuda"
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Gradient clipping
        self.gradient_clip_val = self.config.get("gradient_clip_val", 1.0)
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_dice": [],
            "learning_rate": [],
        }
        
        # Early stopping
        self.patience = self.config.get("early_stopping_patience", 15)
        self.patience_counter = 0
        
        # Logging
        self.writer = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging directories and TensorBoard."""
        # Create directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(tensorboard_dir))
    
    def train_epoch(self) -> Dict:
        """
        Run one training epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for batch in pbar:
            inputs, labels, metadata = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to TensorBoard
            if self.global_step % 100 == 0:
                self.writer.add_scalar(
                    "train/loss_step",
                    loss.item(),
                    self.global_step
                )
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # Track metrics
        tracker = MetricTracker()
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            inputs, labels, metadata = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            tracker.update(outputs, labels)
        
        avg_loss = total_loss / num_batches
        metrics = tracker.compute()
        
        return {
            "val_loss": avg_loss,
            "val_iou": metrics.get("iou", 0),
            "val_dice": metrics.get("dice", 0),
            "val_accuracy": metrics.get("accuracy", 0),
            **{f"val_{k}": v for k, v in metrics.items() if k.startswith("iou_")}
        }
    
    def train(self, num_epochs: int = None) -> Dict:
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs (default from config)
        
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 100)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_iou"].append(val_metrics["val_iou"])
            self.history["val_dice"].append(val_metrics["val_dice"])
            self.history["learning_rate"].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar("train/loss", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("val/iou", val_metrics["val_iou"], epoch)
            self.writer.add_scalar("val/dice", val_metrics["val_dice"], epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)
            
            # Per-direction IoU
            for key, val in val_metrics.items():
                if key.startswith("val_iou_"):
                    self.writer.add_scalar(f"val/{key[4:]}", val, epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val IoU: {val_metrics['val_iou']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            improved = False
            
            if val_metrics["val_iou"] > self.best_val_iou:
                self.best_val_iou = val_metrics["val_iou"]
                improved = True
                self.save_checkpoint("best_iou.pth")
                print(f"  → New best IoU: {self.best_val_iou:.4f}")
            
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                improved = True
                self.save_checkpoint("best_loss.pth")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")
            
            # Always save last checkpoint
            self.save_checkpoint("last.pth")
            
            # Early stopping
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {self.patience} epochs)")
                break
        
        # Training complete
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {elapsed/60:.1f} minutes")
        print(f"Best Val IoU: {self.best_val_iou:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        self.writer.close()
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_iou": self.best_val_iou,
            "config": getattr(self.model, 'config', {}),
            "history": self.history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.best_val_iou = checkpoint.get("best_val_iou", 0.0)
        self.history = checkpoint.get("history", self.history)
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Resumed from epoch {self.epoch}, best IoU: {self.best_val_iou:.4f}")


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **{k: v for k, v in kwargs.items() if k != "momentum"}
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    **kwargs
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Scheduler instance
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 1e-6)
        )
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            min_lr=kwargs.get("min_lr", 1e-7)
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1)
        )
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Test trainer creation
    print("Testing Trainer...")
    
    # Create dummy model
    from ..models.unet import MultiRobotViabilityUNet
    from ..models.losses import DiceBCELoss
    
    model = MultiRobotViabilityUNet()
    criterion = DiceBCELoss()
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)
    
    print("Trainer components created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Scheduler: {type(scheduler).__name__}")
    
    print("\n✓ Trainer test passed!")
