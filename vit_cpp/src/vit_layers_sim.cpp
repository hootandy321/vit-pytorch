#include "vit_layers.h"
#include <algorithm>

// Layer normalization implementation
void layer_norm(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t gamma[EMBED_DIM],
    weight_t beta[EMBED_DIM]
) {
    // Calculate mean
    acc_t mean = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        mean += input[i];
    }
    mean /= EMBED_DIM;
    
    // Calculate variance
    acc_t variance = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        acc_t diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= EMBED_DIM;
    
    // Apply normalization
    data_t inv_std = 1.0 / hls::sqrt(variance + 1e-6);
    for (int i = 0; i < EMBED_DIM; i++) {
        output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    }
}

// Linear transformation layer implementation
void linear_layer(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t weights[EMBED_DIM][EMBED_DIM],
    weight_t bias[EMBED_DIM]
) {
    for (int i = 0; i < EMBED_DIM; i++) {
        acc_t sum = bias[i];
        for (int j = 0; j < EMBED_DIM; j++) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }
}

// Patch embedding implementation
void patch_embedding(
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE],
    data_t patches[NUM_PATCHES][PATCH_DIM],
    data_t embeddings[NUM_PATCHES][EMBED_DIM],
    weight_t embed_weights[EMBED_DIM][PATCH_DIM],
    weight_t embed_bias[EMBED_DIM]
) {
    // Extract patches
    int patch_idx = 0;
    for (int h = 0; h < IMAGE_SIZE; h += PATCH_SIZE) {
        for (int w = 0; w < IMAGE_SIZE; w += PATCH_SIZE) {
            int idx = 0;
            for (int c = 0; c < CHANNELS; c++) {
                for (int ph = 0; ph < PATCH_SIZE; ph++) {
                    for (int pw = 0; pw < PATCH_SIZE; pw++) {
                        patches[patch_idx][idx++] = image[c][h + ph][w + pw];
                    }
                }
            }
            patch_idx++;
        }
    }
    
    // Apply linear projection to each patch
    for (int p = 0; p < NUM_PATCHES; p++) {
        for (int i = 0; i < EMBED_DIM; i++) {
            acc_t sum = embed_bias[i];
            for (int j = 0; j < PATCH_DIM; j++) {
                sum += embed_weights[i][j] * patches[p][j];
            }
            embeddings[p][i] = sum;
        }
    }
}

// GELU activation function
data_t gelu(data_t x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    data_t x3 = x * x * x;
    data_t inner = hls::sqrt(2.0 / 3.14159) * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + hls::tanh(inner));
}

// Softmax function implementation
void softmax(data_t input[NUM_PATCHES + 1], data_t output[NUM_PATCHES + 1]) {
    // Find maximum for numerical stability
    data_t max_val = input[0];
    for (int i = 1; i < NUM_PATCHES + 1; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    data_t exp_sum = 0;
    data_t exp_vals[NUM_PATCHES + 1];
    
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        exp_vals[i] = hls::exp(input[i] - max_val);
        exp_sum += exp_vals[i];
    }
    
    // Normalize
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        output[i] = exp_vals[i] / exp_sum;
    }
}

// Matrix multiplication optimized for HLS
void matrix_mult(
    data_t A[NUM_PATCHES + 1][HEAD_DIM],
    data_t B[HEAD_DIM][NUM_PATCHES + 1],
    data_t C[NUM_PATCHES + 1][NUM_PATCHES + 1]
) {
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        for (int j = 0; j < NUM_PATCHES + 1; j++) {
            acc_t sum = 0;
            for (int k = 0; k < HEAD_DIM; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// MLP layer implementation
void mlp_layer(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t fc1_weights[MLP_DIM][EMBED_DIM],
    weight_t fc1_bias[MLP_DIM],
    weight_t fc2_weights[EMBED_DIM][MLP_DIM],
    weight_t fc2_bias[EMBED_DIM]
) {
    data_t intermediate[MLP_DIM];
    
    // First linear layer
    for (int i = 0; i < MLP_DIM; i++) {
        acc_t sum = fc1_bias[i];
        for (int j = 0; j < EMBED_DIM; j++) {
            sum += fc1_weights[i][j] * input[j];
        }
        intermediate[i] = gelu(sum);
    }
    
    // Second linear layer
    for (int i = 0; i < EMBED_DIM; i++) {
        acc_t sum = fc2_bias[i];
        for (int j = 0; j < MLP_DIM; j++) {
            sum += fc2_weights[i][j] * intermediate[j];
        }
        output[i] = sum;
    }
}

// Simplified multi-head attention implementation
void multi_head_attention(
    data_t input[NUM_PATCHES + 1][EMBED_DIM],
    data_t output[NUM_PATCHES + 1][EMBED_DIM],
    weight_t wq[EMBED_DIM][EMBED_DIM],
    weight_t wk[EMBED_DIM][EMBED_DIM],
    weight_t wv[EMBED_DIM][EMBED_DIM],
    weight_t wo[EMBED_DIM][EMBED_DIM],
    weight_t bias_q[EMBED_DIM],
    weight_t bias_k[EMBED_DIM],
    weight_t bias_v[EMBED_DIM],
    weight_t bias_o[EMBED_DIM]
) {
    // Query, Key, Value projections
    data_t queries[NUM_PATCHES + 1][EMBED_DIM];
    data_t keys[NUM_PATCHES + 1][EMBED_DIM];
    data_t values[NUM_PATCHES + 1][EMBED_DIM];
    
    // Compute Q, K, V for all tokens
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        // Query projection
        for (int j = 0; j < EMBED_DIM; j++) {
            acc_t sum = bias_q[j];
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += wq[j][k] * input[i][k];
            }
            queries[i][j] = sum;
        }
        
        // Key projection
        for (int j = 0; j < EMBED_DIM; j++) {
            acc_t sum = bias_k[j];
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += wk[j][k] * input[i][k];
            }
            keys[i][j] = sum;
        }
        
        // Value projection
        for (int j = 0; j < EMBED_DIM; j++) {
            acc_t sum = bias_v[j];
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += wv[j][k] * input[i][k];
            }
            values[i][j] = sum;
        }
    }
    
    // Simplified attention (single head for demo)
    data_t attention_scores[NUM_PATCHES + 1][NUM_PATCHES + 1];
    data_t scale = 1.0 / hls::sqrt((data_t)HEAD_DIM);
    
    // Q * K^T
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        for (int j = 0; j < NUM_PATCHES + 1; j++) {
            acc_t sum = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += queries[i][k] * keys[j][k];
            }
            attention_scores[i][j] = sum * scale;
        }
        
        // Apply softmax to each row
        softmax(attention_scores[i], attention_scores[i]);
    }
    
    // Apply attention to values
    data_t attention_output[NUM_PATCHES + 1][EMBED_DIM];
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            acc_t sum = 0;
            for (int k = 0; k < NUM_PATCHES + 1; k++) {
                sum += attention_scores[i][k] * values[k][j];
            }
            attention_output[i][j] = sum;
        }
    }
    
    // Apply output projection
    for (int i = 0; i < NUM_PATCHES + 1; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            acc_t sum = bias_o[j];
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += wo[j][k] * attention_output[i][k];
            }
            output[i][j] = sum;
        }
    }
}