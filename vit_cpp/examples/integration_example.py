#!/usr/bin/env python3
"""
Integration example showing how to use the C++ ViT implementation with PyTorch.
This demonstrates the bridge between PyTorch training and C++ HLS deployment.
"""

import torch
import numpy as np
from vit_pytorch import ViT
import subprocess
import os
import tempfile

def create_test_model():
    """Create a simple ViT model for testing."""
    return ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )

def pytorch_inference(model, image):
    """Run inference using PyTorch."""
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

def cpp_inference(image_np, cpp_executable="./build/vit_test_sim"):
    """Run inference using the C++ implementation."""
    # This is a simplified example - in practice, you would need to:
    # 1. Save the image to a binary file
    # 2. Call the C++ executable with the image file
    # 3. Read the results from the output file
    
    # For now, we'll just call the test executable
    try:
        result = subprocess.run([cpp_executable], 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Timeout"
    except FileNotFoundError:
        return "C++ executable not found. Please run 'make' first."

def compare_implementations():
    """Compare PyTorch and C++ implementations."""
    print("=== ViT Implementation Comparison ===")
    
    # Create test model and data
    model = create_test_model()
    test_image = torch.randn(1, 3, 224, 224)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # PyTorch inference
    print("\n1. PyTorch Implementation:")
    pytorch_output = pytorch_inference(model, test_image)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
    print(f"   First 5 predictions: {pytorch_output[0, :5].tolist()}")
    
    # C++ inference
    print("\n2. C++ HLS Implementation:")
    cpp_output = cpp_inference(test_image.numpy())
    print(f"   C++ output:\n{cpp_output}")
    
    print("\n=== Implementation Details ===")
    print("PyTorch:")
    print("  - Uses float32 precision")
    print("  - Dynamic computation graph")
    print("  - GPU acceleration available")
    print("  - Easy to train and fine-tune")
    
    print("\nC++ HLS:")
    print("  - Uses fixed-point arithmetic")
    print("  - Static, optimized for hardware")
    print("  - FPGA acceleration")
    print("  - Optimized for inference")
    print("  - Lower power consumption")
    
    print("\n=== Use Cases ===")
    print("PyTorch: Research, training, prototyping")
    print("C++ HLS: Production deployment, edge devices, low-power applications")

def generate_hls_config():
    """Generate HLS configuration for different FPGA targets."""
    configs = {
        "xcu280": {
            "name": "Xilinx Alveo U280",
            "resources": {
                "DSP": 9024,
                "BRAM": 4032,
                "URAM": 960,
                "LUT": 1303680,
                "FF": 2607360
            },
            "clock_freq": 250,  # MHz
            "recommended_config": {
                "embed_dim": 768,
                "num_heads": 12,
                "batch_size": 1,
                "parallelism": "medium"
            }
        },
        "xcu250": {
            "name": "Xilinx Alveo U250",
            "resources": {
                "DSP": 6840,
                "BRAM": 2688,
                "URAM": 640,
                "LUT": 1172800,
                "FF": 2345600
            },
            "clock_freq": 300,  # MHz
            "recommended_config": {
                "embed_dim": 512,
                "num_heads": 8,
                "batch_size": 1,
                "parallelism": "low"
            }
        },
        "zcu104": {
            "name": "Xilinx Zynq UltraScale+ ZCU104",
            "resources": {
                "DSP": 1728,
                "BRAM": 312,
                "URAM": 96,
                "LUT": 230400,
                "FF": 460800
            },
            "clock_freq": 100,  # MHz
            "recommended_config": {
                "embed_dim": 256,
                "num_heads": 4,
                "batch_size": 1,
                "parallelism": "very_low"
            }
        }
    }
    
    print("\n=== FPGA Configuration Recommendations ===")
    for part, config in configs.items():
        print(f"\n{config['name']} ({part}):")
        print(f"  Resources: DSP={config['resources']['DSP']}, BRAM={config['resources']['BRAM']}")
        print(f"  Clock: {config['clock_freq']} MHz")
        print(f"  Recommended ViT config:")
        print(f"    - Embedding dim: {config['recommended_config']['embed_dim']}")
        print(f"    - Number of heads: {config['recommended_config']['num_heads']}")
        print(f"    - Parallelism: {config['recommended_config']['parallelism']}")

def estimate_performance():
    """Estimate performance characteristics."""
    print("\n=== Performance Estimates ===")
    
    # Configuration
    embed_dim = 768
    num_patches = 196
    num_heads = 12
    mlp_dim = 3072
    num_layers = 12
    
    # Compute estimates
    patch_ops = num_patches * embed_dim * 768  # Patch embedding
    attention_ops = num_layers * num_heads * num_patches * num_patches * (embed_dim // num_heads)
    mlp_ops = num_layers * num_patches * (embed_dim * mlp_dim * 2)
    
    total_ops = patch_ops + attention_ops + mlp_ops
    
    print(f"Operation count estimates:")
    print(f"  Patch embedding: {patch_ops:,} ops")
    print(f"  Attention: {attention_ops:,} ops")
    print(f"  MLP: {mlp_ops:,} ops")
    print(f"  Total: {total_ops:,} ops")
    
    # Performance estimates
    clock_freq = 250e6  # 250 MHz
    parallel_factor = 64  # Estimated parallelism
    
    cycles_per_inference = total_ops / parallel_factor
    inference_time = cycles_per_inference / clock_freq
    throughput = 1 / inference_time
    
    print(f"\nPerformance estimates (250MHz, {parallel_factor}x parallelism):")
    print(f"  Cycles per inference: {cycles_per_inference:,.0f}")
    print(f"  Inference time: {inference_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.1f} inferences/second")
    
    # Power estimates
    power_per_dsp = 0.5  # mW per DSP
    power_per_bram = 2.0  # mW per BRAM
    estimated_dsps = min(1000, total_ops // 1000)
    estimated_brams = min(500, embed_dim * num_patches // 1000)
    
    estimated_power = estimated_dsps * power_per_dsp + estimated_brams * power_per_bram
    
    print(f"\nPower estimates:")
    print(f"  Estimated DSPs used: {estimated_dsps}")
    print(f"  Estimated BRAMs used: {estimated_brams}")
    print(f"  Estimated power: {estimated_power:.1f} mW")

def main():
    """Main function to run all comparisons and estimates."""
    print("ViT PyTorch to C++ HLS Integration Example")
    print("=" * 50)
    
    # Check if C++ implementation is built
    cpp_path = "./build/vit_test_sim"
    if not os.path.exists(cpp_path):
        print(f"C++ executable not found at {cpp_path}")
        print("Please run 'make' in the vit_cpp directory first.")
        return
    
    # Run comparisons
    compare_implementations()
    generate_hls_config()
    estimate_performance()
    
    print("\n" + "=" * 50)
    print("Integration complete!")
    print("\nNext steps:")
    print("1. Train a ViT model in PyTorch")
    print("2. Export weights using tools/convert_weights.py")
    print("3. Run HLS synthesis with 'make hls'")
    print("4. Deploy to FPGA hardware")

if __name__ == "__main__":
    main()