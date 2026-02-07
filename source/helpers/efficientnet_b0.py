"""
EfficientNet-B0 Implementation from Scratch
Based on: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
Mingxing Tan, Quoc V. Le (2019)

This implementation provides a detailed, educational version of EfficientNet-B0
with extensive documentation for understanding the architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class Swish(nn.Module):
    """
    Swish Activation Function (also known as SiLU - Sigmoid Linear Unit)
    
    Formula: f(x) = x · σ(x) where σ(x) is the sigmoid function
    
    Swish is a self-gated activation function that has been shown to work
    better than ReLU for deeper models. It's smooth and non-monotonic.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    SE blocks adaptively recalibrate channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    
    Process:
    1. Squeeze: Global average pooling to get channel descriptors
    2. Excitation: Two FC layers with sigmoid to get channel weights
    3. Scale: Multiply input features by the learned channel weights
    
    Args:
        in_channels: Number of input channels
        se_ratio: Reduction ratio for the SE bottleneck (default: 0.25)
    """
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) Block
    
    This is the fundamental building block of EfficientNet, based on
    MobileNetV2's inverted residual structure with several enhancements:
    
    Architecture:
    1. Expansion: 1x1 conv to expand channels (if expand_ratio > 1)
    2. Depthwise: Depthwise conv with kernel_size k
    3. Squeeze-Excitation: Channel attention mechanism
    4. Projection: 1x1 conv to project back to output channels
    5. Skip Connection: If stride=1 and in_channels=out_channels
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise convolution
        stride: Stride for depthwise convolution
        expand_ratio: Channel expansion ratio (e.g., 6 means 6x expansion)
        se_ratio: Squeeze-excitation reduction ratio
        drop_connect_rate: Dropout rate for stochastic depth
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        
        # Depthwise convolution phase
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=padding, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        # Squeeze-and-Excitation phase
        self.se = SqueezeExcitation(expanded_channels, se_ratio)
        
        # Output projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def drop_connect(self, x, training: bool):
        """
        Stochastic Depth (Drop Connect)
        
        Randomly drops the entire residual branch during training.
        This acts as a form of regularization and helps with gradient flow.
        """
        if not training or self.drop_connect_rate == 0:
            return x
        
        keep_prob = 1 - self.drop_connect_rate
        random_tensor = keep_prob + torch.rand(
            (x.size(0), 1, 1, 1), dtype=x.dtype, device=x.device
        )
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand is not None:
            x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Output projection
        x = self.project(x)
        
        # Skip connection with stochastic depth
        if self.use_residual:
            x = self.drop_connect(x, self.training)
            x = x + identity
        
        return x


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 Architecture
    
    EfficientNet uses compound scaling to uniformly scale network width,
    depth, and resolution with a set of fixed scaling coefficients.
    
    For B0 (baseline):
    - Width multiplier (α): 1.0
    - Depth multiplier (β): 1.0
    - Resolution: 224x224
    
    Architecture follows the pattern:
    Stage 1: MBConv1, k3x3, 32→16, 1 block
    Stage 2: MBConv6, k3x3, 16→24, 2 blocks
    Stage 3: MBConv6, k5x5, 24→40, 2 blocks
    Stage 4: MBConv6, k3x3, 40→80, 3 blocks
    Stage 5: MBConv6, k5x5, 80→112, 3 blocks
    Stage 6: MBConv6, k5x5, 112→192, 4 blocks
    Stage 7: MBConv6, k3x3, 192→320, 1 block
    
    Args:
        num_classes: Number of output classes
        drop_connect_rate: Maximum drop connect rate (increases per block)
        dropout_rate: Dropout rate before final classifier
    """
    
    # EfficientNet-B0 configuration
    # Each tuple: (expand_ratio, channels, num_blocks, kernel_size, stride)
    BLOCK_CONFIG = [
        # Stage 1
        (1, 16, 1, 3, 1),
        # Stage 2
        (6, 24, 2, 3, 2),
        # Stage 3
        (6, 40, 2, 5, 2),
        # Stage 4
        (6, 80, 3, 3, 2),
        # Stage 5
        (6, 112, 3, 5, 1),
        # Stage 6
        (6, 192, 4, 5, 2),
        # Stage 7
        (6, 320, 1, 3, 1),
    ]
    
    def __init__(
        self,
        num_classes: int = 1000,
        drop_connect_rate: float = 0.2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        total_blocks = sum(config[2] for config in self.BLOCK_CONFIG)
        block_idx = 0
        in_channels = 32
        
        for expand_ratio, out_channels, num_blocks, kernel_size, stride in self.BLOCK_CONFIG:
            for i in range(num_blocks):
                # Calculate drop connect rate for this block
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                # First block of each stage uses specified stride
                block_stride = stride if i == 0 else 1
                
                block = MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25,
                    drop_connect_rate=drop_rate
                )
                
                self.blocks.append(block)
                in_channels = out_channels
                block_idx += 1
        
        # Head: Final convolution and pooling
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1280, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Weight initialization following the paper's recommendations.
        Uses fan-out mode for conv layers and normal distribution for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Stem
        x = self.stem(x)  # (B, 32, 112, 112)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)  # Final: (B, 320, 7, 7)
        
        # Head
        x = self.head(x)  # (B, 1280, 7, 7)
        x = self.avgpool(x)  # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1280)
        x = self.dropout(x)
        x = self.classifier(x)  # (B, num_classes)
        
        return x
    
    def get_feature_extractor(self):
        """
        Returns the model without the classification head.
        Useful for transfer learning and feature extraction.
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.stem = parent.stem
                self.blocks = parent.blocks
                self.head = parent.head
                self.avgpool = parent.avgpool
            
            def forward(self, x):
                x = self.stem(x)
                for block in self.blocks:
                    x = block(x)
                x = self.head(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
        
        return FeatureExtractor(self)


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """
    Print a detailed summary of the model architecture.
    """
    print(f"{'='*80}")
    print(f"EfficientNet-B0 Model Summary")
    print(f"{'='*80}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Input Size: {input_size}")
    print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    # Create model
    model = EfficientNetB0(num_classes=1000)
    
    # Print summary
    get_model_summary(model)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get feature extractor
    feature_extractor = model.get_feature_extractor()
    features = feature_extractor(x)
    print(f"Feature shape: {features.shape}")
    
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")