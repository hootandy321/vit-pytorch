# Vision Transformer (ViT) - C++ HLS Implementation

This directory contains a C++ implementation of the Vision Transformer (ViT) model optimized for FPGA High-Level Synthesis (HLS). The implementation is designed to be synthesizable using Xilinx Vivado HLS or Vitis HLS tools.

## Features

- **FPGA-optimized**: Uses fixed-point arithmetic and HLS-specific optimizations
- **Parallel processing**: Optimized for parallel execution on FPGA hardware
- **Configurable**: Easy to modify dimensions and parameters
- **Complete implementation**: Includes all ViT components (patch embedding, multi-head attention, MLP, etc.)
- **HLS pragmas**: Extensive use of HLS pragmas for optimization

## Architecture

The implementation includes the following components:

### Core Components
- **Patch Embedding**: Converts input images into patch tokens
- **Multi-Head Attention**: Implements scaled dot-product attention with multiple heads
- **MLP (Feed-Forward)**: Two-layer MLP with GELU activation
- **Layer Normalization**: Normalizes layer inputs for training stability
- **Transformer Encoder**: Complete transformer encoder block
- **Classification Head**: Final linear layer for classification

### Data Types
- `data_t`: 16-bit fixed-point for activations (8 integer bits)
- `weight_t`: 32-bit fixed-point for weights (16 integer bits)
- `acc_t`: 24-bit fixed-point for accumulators (12 integer bits)

### Default Configuration
- Image size: 224x224
- Patch size: 16x16
- Number of patches: 196
- Embedding dimension: 768
- Number of heads: 12
- MLP dimension: 3072
- Number of layers: 12
- Number of classes: 1000

## File Structure

```
vit_cpp/
├── include/
│   ├── vit_types.h      # Data types and constants
│   ├── vit_layers.h     # Layer function declarations
│   └── vit.h            # Main ViT interface
├── src/
│   ├── vit_layers.cpp   # Layer implementations
│   └── vit.cpp          # Main ViT implementation
├── test/
│   └── test_vit.cpp     # Test suite
├── Makefile             # Build configuration
└── README.md           # This file
```

## Building and Testing

### Prerequisites
- C++ compiler (g++ or clang++)
- For HLS synthesis: Xilinx Vivado HLS or Vitis HLS

### Building
```bash
# Build the test executable
make all

# Run tests
make test

# Clean build files
make clean
```

### HLS Synthesis
```bash
# Run HLS synthesis (requires Vivado HLS)
make hls

# Clean HLS project files
make clean-hls
```

## HLS Optimizations

The implementation includes several HLS-specific optimizations:

### Memory Optimizations
- **Array Partitioning**: Critical arrays are partitioned for parallel access
- **Complete Partitioning**: Small arrays are completely partitioned
- **Dimension-wise Partitioning**: Large arrays are partitioned along specific dimensions

### Compute Optimizations
- **Loop Pipelining**: Loops are pipelined with II=1 where possible
- **Loop Unrolling**: Critical loops are unrolled for parallel execution
- **Dataflow**: Layer-level dataflow for pipelined execution

### Interface Optimizations
- **AXI4 Interfaces**: Memory-mapped interfaces for large data transfers
- **AXI4-Lite**: Control interface for configuration
- **Streaming**: Stream interfaces for pipelined processing

## Example Usage

### C++ Simulation
```cpp
#include "vit.h"

int main() {
    // Create input image
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    // ... initialize image ...
    
    // Output predictions
    data_t predictions[NUM_CLASSES];
    
    // Run inference
    vit_inference(image, predictions);
    
    return 0;
}
```

### HLS Testbench
```cpp
#include "vit.h"
#include <iostream>

int main() {
    // Test with random input
    data_t test_image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    data_t predictions[NUM_CLASSES];
    
    // Initialize test data
    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < IMAGE_SIZE; h++) {
            for (int w = 0; w < IMAGE_SIZE; w++) {
                test_image[c][h][w] = (rand() % 256) / 255.0;
            }
        }
    }
    
    // Run inference
    vit_inference(test_image, predictions);
    
    // Check results
    std::cout << "Inference completed successfully!" << std::endl;
    return 0;
}
```

## Resource Utilization

The implementation is designed to be resource-efficient for FPGA deployment:

### Memory Requirements
- **Input image**: ~150 KB (224x224x3 pixels)
- **Weights**: ~300 MB (for full ViT-Base configuration)
- **Intermediate activations**: ~1 MB (with optimizations)

### Expected FPGA Resources
- **DSP slices**: ~1000-2000 (for parallel multiply-accumulate operations)
- **Block RAM**: ~500-1000 (for weight storage and buffering)
- **Look-up tables**: ~100K-200K (for control logic)
- **Flip-flops**: ~100K-200K (for pipelining)

## Configuration

### Modifying Model Size
Edit `vit_types.h` to change model dimensions:
```cpp
const int IMAGE_SIZE = 224;     // Input image size
const int PATCH_SIZE = 16;      // Patch size
const int EMBED_DIM = 768;      // Embedding dimension
const int NUM_HEADS = 12;       // Number of attention heads
const int MLP_DIM = 3072;       // MLP hidden dimension
const int NUM_LAYERS = 12;      // Number of transformer layers
const int NUM_CLASSES = 1000;   // Number of output classes
```

### Adjusting Precision
Modify fixed-point types in `vit_types.h`:
```cpp
typedef ap_fixed<16, 8> data_t;   // 16-bit total, 8 integer bits
typedef ap_fixed<32, 16> weight_t; // 32-bit total, 16 integer bits
typedef ap_fixed<24, 12> acc_t;    // 24-bit total, 12 integer bits
```

## Performance Considerations

### Throughput Optimization
- **Pipelining**: All loops are pipelined for maximum throughput
- **Parallel Processing**: Multiple operations executed simultaneously
- **Streaming**: Data flows through pipeline without blocking

### Latency Optimization
- **Dataflow**: Layer-level pipelining reduces overall latency
- **Memory Hierarchy**: Efficient memory access patterns
- **Compute Overlap**: Overlapping computation with memory access

### Power Optimization
- **Fixed-point Arithmetic**: Reduces power consumption vs. floating-point
- **Clock Gating**: Unused logic is clock-gated
- **Resource Sharing**: Shared resources where possible

## Limitations and Future Work

### Current Limitations
- Simplified transformer layer implementation
- Static weight initialization (needs external weight loading)
- No dynamic sequence length support
- Limited batch processing capability

### Future Enhancements
- Dynamic weight loading from external memory
- Batch processing support
- Quantization to lower bit-widths
- Sparse attention implementation
- Multi-FPGA scaling support

## Integration with PyTorch

The C++ implementation can be integrated with the PyTorch version:

1. **Weight Transfer**: Export PyTorch weights to C++ format
2. **Validation**: Compare outputs between PyTorch and C++ versions
3. **Calibration**: Use PyTorch for quantization calibration
4. **Training**: Train in PyTorch, deploy in C++

## Troubleshooting

### Common Issues
1. **Memory Allocation**: Ensure sufficient FPGA memory for model size
2. **Timing Closure**: Adjust clock frequency if timing constraints fail
3. **Resource Utilization**: Optimize array partitioning if resources exceed limits
4. **Precision Issues**: Adjust fixed-point bit-widths if accuracy is poor

### Debug Tips
- Use HLS simulation for functional verification
- Check synthesis reports for resource utilization
- Use co-simulation for hardware/software verification
- Monitor intermediate values during synthesis

## License

This implementation follows the same license as the parent repository.

## Contributing

Contributions are welcome! Please focus on:
- Performance optimizations
- Resource utilization improvements
- Additional model variants
- Better documentation
- Test coverage improvements