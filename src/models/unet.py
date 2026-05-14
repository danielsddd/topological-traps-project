"""
U-Net Model for Multi-Robot Viability Prediction.

This module implements the neural network architecture for predicting
directional viability maps. The model takes a 3-channel input:
- Channel 0: Occupancy grid (binary map)
- Channel 1: Normalized robot length (constant, length/512)
- Channel 2: Normalized robot width (constant, width/512)

And outputs 4 channels:
- Channel 0: North viability logits
- Channel 1: South viability logits
- Channel 2: East viability logits
- Channel 3: West viability logits

Architecture:
- U-Net with ResNet34 encoder (pretrained on ImageNet)
- ~24M parameters
- Skip connections preserve spatial detail
- Encoder extracts multi-scale features
- Decoder upsamples with skip connections
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available")


class MultiRobotViabilityUNet(nn.Module):
    """
    U-Net for predicting directional viability maps.
    
    Takes 3-channel input (occupancy + robot dimensions) and
    outputs 4-channel viability predictions (N, S, E, W).
    
    The robot dimensions are encoded as constant feature maps,
    allowing the network to learn size-dependent viability.
    This is known as "Global Conditioning" - the constant channels
    bias the convolutional filters based on robot size.
    
    Architecture:
        - Encoder: ResNet34 (pretrained on ImageNet)
        - Decoder: U-Net decoder with skip connections
        - Output: 4 channels of logits (apply sigmoid for probabilities)
    
    Example:
        >>> model = MultiRobotViabilityUNet()
        >>> x = torch.randn(4, 3, 512, 512)  # Batch of 4
        >>> logits = model(x)  # Shape: (4, 4, 512, 512)
        >>> probs = model.predict(x)  # Probabilities in [0, 1]
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 4,
        activation: Optional[str] = None,
        encoder_depth: int = 5,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        """
        Initialize the model.
        
        Args:
            encoder_name: Name of encoder backbone (default: resnet34)
            encoder_weights: Pretrained weights (default: imagenet)
            in_channels: Number of input channels (default: 3)
            classes: Number of output classes/channels (default: 4)
            activation: Output activation (None = logits, 'sigmoid', 'softmax')
            encoder_depth: Number of encoder stages (default: 5)
            decoder_channels: Number of channels in decoder stages
        """
        super().__init__()
        
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        # Build U-Net using segmentation_models_pytorch
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )
        
        # Store config for serialization
        self.config = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": classes,
            "activation": activation,
            "encoder_depth": encoder_depth,
            "decoder_channels": decoder_channels,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
               - x[:, 0]: Occupancy grid (0 or 1)
               - x[:, 1]: Normalized robot length (constant)
               - x[:, 2]: Normalized robot width (constant)
        
        Returns:
            Logits tensor (B, 4, H, W)
        """
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference with sigmoid activation.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Probabilities tensor (B, 4, H, W) in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict_binary(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Inference with thresholding.
        
        Args:
            x: Input tensor (B, 3, H, W)
            threshold: Probability threshold for binary prediction
        
        Returns:
            Binary predictions tensor (B, 4, H, W) in {0, 1}
        """
        probs = self.predict(x)
        return (probs > threshold).float()
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: Count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_encoder_parameters(self):
        """Get encoder parameters (for differential learning rates)."""
        return self.model.encoder.parameters()
    
    def get_decoder_parameters(self):
        """Get decoder parameters (for differential learning rates)."""
        return self.model.decoder.parameters()
    
    def freeze_encoder(self):
        """Freeze encoder weights (for fine-tuning)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_config(cls, config: Dict) -> "MultiRobotViabilityUNet":
        """
        Create model from config dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Model instance
        """
        return cls(**config)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu"
    ) -> "MultiRobotViabilityUNet":
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint .pth file
            device: Device to load model to
        
        Returns:
            Model instance with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model from config
        config = checkpoint.get("config", checkpoint.get("model_config", {}))
        model = cls.from_config(config)
        
        # Load weights
        state_dict_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_dict_key])
        
        return model
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Output path for checkpoint
            optimizer: Optional optimizer to save
            epoch: Current epoch number
            metrics: Optional metrics dictionary
            **kwargs: Additional data to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)


def create_model(
    config: Dict = None,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    device: str = "cuda"
) -> MultiRobotViabilityUNet:
    """
    Factory function to create model.
    
    Args:
        config: Optional configuration dictionary
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        device: Device to create model on
    
    Returns:
        Model instance
    """
    if config is not None:
        model = MultiRobotViabilityUNet.from_config(config)
    else:
        model = MultiRobotViabilityUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
        )
    
    return model.to(device)


def get_model_summary(model: nn.Module) -> Dict:
    """
    Get model summary information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 ** 2),  # Assuming float32
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing MultiRobotViabilityUNet...")
    
    # Create model
    model = MultiRobotViabilityUNet()
    print(f"Model created successfully")
    
    # Print summary
    summary = get_model_summary(model)
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Model size: {summary['model_size_mb']:.1f} MB")
    
    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
        probs = model.predict(x)
        binary = model.predict_binary(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Binary shape: {binary.shape}")
    print(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # Test checkpoint save/load
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pth")
        model.save_checkpoint(ckpt_path, epoch=10)
        
        loaded_model = MultiRobotViabilityUNet.from_checkpoint(ckpt_path, device=device)
        
        # Verify same output
        with torch.no_grad():
            logits2 = loaded_model(x)
        
        assert torch.allclose(logits, logits2), "Loaded model should produce same output"
        print("\n✓ Checkpoint save/load test passed")
    
    print("\n✓ All model tests passed!")
