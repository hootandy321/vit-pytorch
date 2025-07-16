#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../include/vit.h"

// Test function to validate basic functionality
void test_vit_basic() {
    std::cout << "Testing basic ViT functionality..." << std::endl;
    
    // Create test image
    data_t test_image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    
    // Initialize with random values
    srand(time(NULL));
    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < IMAGE_SIZE; h++) {
            for (int w = 0; w < IMAGE_SIZE; w++) {
                test_image[c][h][w] = (data_t)(rand() % 256) / 255.0 - 0.5;
            }
        }
    }
    
    // Output predictions
    data_t predictions[NUM_CLASSES];
    
    // Run inference
    vit_inference(test_image, predictions);
    
    // Print first few predictions
    std::cout << "First 10 predictions:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Class " << i << ": " << predictions[i] << std::endl;
    }
    
    std::cout << "Basic ViT test completed!" << std::endl;
}

// Test individual layer functions
void test_layer_functions() {
    std::cout << "Testing individual layer functions..." << std::endl;
    
    // Test layer norm
    data_t input[EMBED_DIM];
    data_t output[EMBED_DIM];
    weight_t gamma[EMBED_DIM];
    weight_t beta[EMBED_DIM];
    
    // Initialize test data
    for (int i = 0; i < EMBED_DIM; i++) {
        input[i] = (data_t)(rand() % 100) / 100.0;
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }
    
    layer_norm(input, output, gamma, beta);
    
    // Check that output has approximately zero mean and unit variance
    data_t mean = 0, var = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        mean += output[i];
    }
    mean /= EMBED_DIM;
    
    for (int i = 0; i < EMBED_DIM; i++) {
        var += (output[i] - mean) * (output[i] - mean);
    }
    var /= EMBED_DIM;
    
    std::cout << "Layer norm test - Mean: " << mean << ", Variance: " << var << std::endl;
    
    // Test GELU activation
    data_t gelu_input = 0.5;
    data_t gelu_output = gelu(gelu_input);
    std::cout << "GELU(0.5) = " << gelu_output << std::endl;
    
    // Test softmax
    data_t softmax_input[NUM_PATCHES + 1];
    data_t softmax_output[NUM_PATCHES + 1];
    
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        softmax_input[i] = (data_t)(rand() % 100) / 100.0;
    }
    
    softmax(softmax_input, softmax_output);
    
    // Check that softmax output sums to 1
    data_t sum = 0;
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        sum += softmax_output[i];
    }
    std::cout << "Softmax sum: " << sum << std::endl;
    
    std::cout << "Layer function tests completed!" << std::endl;
}

// Test patch embedding
void test_patch_embedding() {
    std::cout << "Testing patch embedding..." << std::endl;
    
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    data_t patches[NUM_PATCHES][PATCH_DIM];
    data_t embeddings[NUM_PATCHES][EMBED_DIM];
    weight_t embed_weights[EMBED_DIM][PATCH_DIM];
    weight_t embed_bias[EMBED_DIM];
    
    // Initialize image
    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < IMAGE_SIZE; h++) {
            for (int w = 0; w < IMAGE_SIZE; w++) {
                image[c][h][w] = (data_t)(rand() % 256) / 255.0;
            }
        }
    }
    
    // Initialize weights
    for (int i = 0; i < EMBED_DIM; i++) {
        embed_bias[i] = 0.0;
        for (int j = 0; j < PATCH_DIM; j++) {
            embed_weights[i][j] = (weight_t)(rand() % 100) / 10000.0;
        }
    }
    
    patch_embedding(image, patches, embeddings, embed_weights, embed_bias);
    
    std::cout << "Patch embedding completed. First embedding values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "Patch 0, dim " << i << ": " << embeddings[0][i] << std::endl;
    }
    
    std::cout << "Patch embedding test completed!" << std::endl;
}

// Performance and resource utilization info
void print_resource_info() {
    std::cout << "\n=== ViT HLS Implementation Resource Information ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Image size: " << IMAGE_SIZE << "x" << IMAGE_SIZE << std::endl;
    std::cout << "  Patch size: " << PATCH_SIZE << "x" << PATCH_SIZE << std::endl;
    std::cout << "  Number of patches: " << NUM_PATCHES << std::endl;
    std::cout << "  Embedding dimension: " << EMBED_DIM << std::endl;
    std::cout << "  Number of heads: " << NUM_HEADS << std::endl;
    std::cout << "  Head dimension: " << HEAD_DIM << std::endl;
    std::cout << "  MLP dimension: " << MLP_DIM << std::endl;
    std::cout << "  Number of layers: " << NUM_LAYERS << std::endl;
    std::cout << "  Number of classes: " << NUM_CLASSES << std::endl;
    
    std::cout << "\nMemory requirements (approximate):" << std::endl;
    
    // Calculate memory usage
    size_t image_size = CHANNELS * IMAGE_SIZE * IMAGE_SIZE * sizeof(data_t);
    size_t patch_embed_weights = EMBED_DIM * PATCH_DIM * sizeof(weight_t);
    size_t attention_weights = NUM_LAYERS * 4 * EMBED_DIM * EMBED_DIM * sizeof(weight_t);
    size_t mlp_weights = NUM_LAYERS * (MLP_DIM * EMBED_DIM + EMBED_DIM * MLP_DIM) * sizeof(weight_t);
    size_t head_weights = NUM_CLASSES * EMBED_DIM * sizeof(weight_t);
    size_t pos_embed = (NUM_PATCHES + 1) * EMBED_DIM * sizeof(data_t);
    
    std::cout << "  Input image: " << image_size << " bytes" << std::endl;
    std::cout << "  Patch embedding weights: " << patch_embed_weights << " bytes" << std::endl;
    std::cout << "  Attention weights: " << attention_weights << " bytes" << std::endl;
    std::cout << "  MLP weights: " << mlp_weights << " bytes" << std::endl;
    std::cout << "  Classification head weights: " << head_weights << " bytes" << std::endl;
    std::cout << "  Position embeddings: " << pos_embed << " bytes" << std::endl;
    
    size_t total_weights = patch_embed_weights + attention_weights + mlp_weights + head_weights + pos_embed;
    std::cout << "  Total weight storage: " << total_weights << " bytes (" << total_weights / (1024*1024) << " MB)" << std::endl;
    
    std::cout << "\nHLS Optimizations applied:" << std::endl;
    std::cout << "  - Array partitioning for parallel access" << std::endl;
    std::cout << "  - Loop pipelining for throughput" << std::endl;
    std::cout << "  - Loop unrolling for critical paths" << std::endl;
    std::cout << "  - Dataflow optimization for layer pipelining" << std::endl;
    std::cout << "  - Fixed-point arithmetic for hardware efficiency" << std::endl;
    std::cout << "=================================================" << std::endl;
}

int main() {
    std::cout << "ViT HLS Implementation Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;
    
    print_resource_info();
    
    std::cout << "\nRunning tests..." << std::endl;
    
    // Run individual tests
    test_layer_functions();
    std::cout << std::endl;
    
    test_patch_embedding();
    std::cout << std::endl;
    
    test_vit_basic();
    std::cout << std::endl;
    
    std::cout << "All tests completed!" << std::endl;
    
    return 0;
}