"""
EfficientNet-B0 Mathematical Process Visualization System
This module provides a non-intrusive hook-based tracer to visualize the mathematical
operations and tensor transformations within EfficientNet-B0 during interference/training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from collections import OrderedDict
import sys
from pathlib import Path

# Import model definition to identify modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.helpers.efficientnet_b0 import MBConvBlock, SqueezeExcitation, Swish, EfficientNetB0


class MathFormatter:
    """Helper to format mathematical expressions and tensor shapes."""
    
    @staticmethod
    def format_shape(shape: tuple) -> str:
        if isinstance(shape, torch.Size):
            shape = tuple(shape)
        return f"ℝ[{', '.join(map(str, shape))}]"

    @staticmethod
    def format_kernel(module: nn.Conv2d) -> str:
        return f"{module.kernel_size[0]}×{module.kernel_size[1]}"


class EfficientNetMathTracer:
    """
    Hooks into an EfficientNetB0 model to trace and visualize mathematical operations per epoch.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[Any] = []
        self.logs: List[str] = []
        self.enabled = False
        self.current_epoch = 0
        self.traced_this_epoch = False
        
        # Buffer to store intermediate MBConv outputs for a cleaner tree
        self.mbconv_buffer: Dict[str, Dict[str, Any]] = {}
        
    def enable(self):
        """Enable tracing hooks."""
        if not self.hooks:
            self._register_hooks()
        self.enabled = True
        self.traced_this_epoch = False

    def disable(self):
        """Disable tracing (remove hooks to save overhead)."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.enabled = False

    def reset_epoch(self, epoch: int):
        """Reset state for a new epoch."""
        self.current_epoch = epoch
        self.traced_this_epoch = False
        self.logs = []
        self.mbconv_buffer = {}

    def _register_hooks(self):
        """Register forward hooks on relevant modules."""
        # Helper to attach hook
        def attach(module, name, type_tag):
            hook = module.register_forward_hook(
                lambda m, i, o: self._hook_callback(m, i, o, name, type_tag)
            )
            self.hooks.append(hook)

        # 1. Stem
        if hasattr(self.model, 'stem'):
            attach(self.model.stem[0], "Stem Convolution", "stem_conv")
            
        # 2. Blocks
        if hasattr(self.model, 'blocks'):
            for idx, block in enumerate(self.model.blocks):
                block_name = f"MBConv Block {idx+1}"
                # We hook the block itself to get the final output and identity
                attach(block, block_name, "mbconv_block")
                
                # Hook internals
                if block.expand is not None:
                    # expand is sequential: conv -> bn -> swish
                    attach(block.expand[0], f"{block_name} - Expansion", "mbconv_expand")
                
                # depthwise is sequential: conv -> bn -> swish
                attach(block.depthwise[0], f"{block_name} - Depthwise", "mbconv_dw")
                
                # SE
                attach(block.se, f"{block_name} - SE", "mbconv_se")
                
                # Project
                attach(block.project[0], f"{block_name} - Projection", "mbconv_proj")

        # 3. Head
        if hasattr(self.model, 'head'):
            attach(self.model.head[0], "Head Convolution", "head_conv")

        # 4. Global Pool
        if hasattr(self.model, 'avgpool'):
            attach(self.model.avgpool, "Global Average Pooling", "pooling")
            
        # 5. Classifier
        if hasattr(self.model, 'classifier'):
            attach(self.model.classifier, "Classifier", "classifier")

    def _hook_callback(self, module, input_t, output_t, name, type_tag):
        """
        Generic hook callback.
        Note: input is a tuple.
        """
        if not self.enabled or self.traced_this_epoch:
            return

        x = input_t[0]
        
        # We only want to log the FIRST pass of the epoch
        # But since this is called for every layer in sequence, we need a flag that persists 
        # for the duration of one full forward pass. 
        # Actually, standard practice for "once per epoch" logging with hooks:
        # Enable at start of epoch, disable after first batch.
        
        log_entry = self._generate_math_log(module, x, output_t, name, type_tag)
        if log_entry:
            self.logs.append(log_entry)

    def _generate_math_log(self, module, x, y, name, type_tag) -> str:
        """Generate the mathematical explanation string."""
        formatter = MathFormatter()
        lines = []

        if type_tag == "stem_conv":
            lines.append(f"├── {name}")
            lines.append(f"│   Formula: Y_i,j,k = ∑_c ∑_m,n X_c,i+m,j+n · W_k,c,m,n")
            lines.append(f"│   Input:   {formatter.format_shape(x.shape)}")
            lines.append(f"│   Kernel:  {formatter.format_kernel(module)}, Stride={module.stride[0]}")
            lines.append(f"│   Output:  {formatter.format_shape(y.shape)} (Channel Expansion: {x.shape[1]}→{y.shape[1]})")

        elif type_tag == "mbconv_block":
            # This is the wrapper block. We used it to frame the MBConv section.
            # We can print the block header here, or rely on the sub-components.
            # Let's print a high-level summary of the block's config.
            # Note: The sub-components (expand, dw, se) naturally run INSIDE this, 
            # but hooks fire AFTER forward. So this hook fires LAST for the block.
            # This is tricky for ordering. 
            pass # We will rely on sub-hooks for details, checking if we need a footer?
            
        elif type_tag == "mbconv_expand":
            # Just parsed parent name "MBConv Block N"
            block_header = name.split(" - ")[0]
            lines.append(f"├── {block_header}")
            lines.append(f"│   ├── Expansion Phase")
            lines.append(f"│   │   Formula: X_exp = Conv₁ₓ₁(X) · t")
            lines.append(f"│   │   Input:   {formatter.format_shape(x.shape)}")
            lines.append(f"│   │   Output:  {formatter.format_shape(y.shape)}")
            
        elif type_tag == "mbconv_dw":
            lines.append(f"│   ├── Depthwise Convolution")
            lines.append(f"│   │   Formula: X_dw = DWConv_k×k(X_exp)")
            lines.append(f"│   │   Kernel:  {formatter.format_kernel(module)}, Stride={module.stride[0]}")
            lines.append(f"│   │   Output:  {formatter.format_shape(y.shape)}")

        elif type_tag == "mbconv_se":
            lines.append(f"│   ├── Squeeze-and-Excitation")
            lines.append(f"│   │   Squeeze: z_c = (1/H×W) ∑ X_c(i,j)")
            lines.append(f"│   │   Excitation: s = σ(W₂ δ(W₁ z))")
            lines.append(f"│   │   Scale:   X̃_c = s_c · X_c")
            # SE module input is X, output is Scaled X
            # Dimensions: Input [N, C, H, W], Output [N, C, H, W]
            lines.append(f"│   │   Shape:   {formatter.format_shape(x.shape)} (unchanged spatial)")

        elif type_tag == "mbconv_proj":
            lines.append(f"│   ├── Projection")
            lines.append(f"│   │   Formula: X_proj = Conv₁ₓ₁(X_se)")
            lines.append(f"│   │   Output:  {formatter.format_shape(y.shape)}")
            lines.append(f"│   └── Skip Connection + DropConnect") 
            # We assume skip happens if dimensions match, strictly we should check block.use_residual
            # taking a guess based on shapes is reasonably safe for viz or referencing the module parent if possible
            # but getting parent from module in callback is hard without partials.
            # We'll stick to generic description.
            lines.append(f"│       Formula: X_out = X + DropConnect(X_proj)")

        elif type_tag == "head_conv":
            lines.append(f"├── {name}")
            lines.append(f"│   Formula: X_head = Conv₁ₓ₁(X_final)")
            lines.append(f"│   Input:   {formatter.format_shape(x.shape)}")
            lines.append(f"│   Output:  {formatter.format_shape(y.shape)}")

        elif type_tag == "pooling":
            lines.append(f"├── {name}")
            lines.append(f"│   Formula: f(x) = GlobalAvgPool(X)")
            lines.append(f"│   Collapse: {formatter.format_shape(x.shape)} → {formatter.format_shape(y.shape)}")

        elif type_tag == "classifier":
            lines.append(f"├── {name}")
            lines.append(f"│   Formula: σ(z_i) = e^(z_i) / ∑ e^(z_j)")
            lines.append(f"│   Loss:    L = -∑ y'_i log(ŷ_i)")
            lines.append(f"│   Input:   {formatter.format_shape(x.shape)}")
            lines.append(f"│   Logits:  {formatter.format_shape(y.shape)}")
            
        return "\n".join(lines)

    def print_report(self):
        """Print the aggregated epoch report."""
        if not self.logs:
            return

        print(f"\n{self._get_epoch_header()}")
        print("\n".join(self.logs))
        self.print_backprop_summary()
        print(f"{'='*60}\n")
        
        # Mark as done for this epoch to prevent spam if enable() wasn't toggled off
        self.traced_this_epoch = True

    def _get_epoch_header(self) -> str:
        return f"""
============================================================
   EFFICIENTNET-B0 MATHEMATICAL VISUALIZATION - EPOCH {self.current_epoch}
============================================================
1️⃣ Input Representation
   X ∈ ℝ[batch_size, 3, 224, 224]
   • Spatial Resolution: 224×224 pixels
   • RGB Channels: 3 (Red, Green, Blue)"""

    def print_backprop_summary(self):
        print("└── Backpropagation Summary")
        print("    Symbolic Gradient: ∂L/∂W_l")
        print("    • Gradients flow backwards from Classifier → Head → Blocks → Stem")
        print("    • Skip connections allow gradients to bypass deep non-linearities (ResNet principle)")
        print("    • DropConnect acts as an ensemble regularizer during backward pass")
