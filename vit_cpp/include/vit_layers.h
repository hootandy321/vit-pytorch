#ifndef VIT_LAYERS_H
#define VIT_LAYERS_H

#ifdef __SYNTHESIS__
    #include "vit_types.h"
    #include <hls_math.h>
#else
    #include "vit_types_sim.h"
    #include <cmath>
#endif

// Layer normalization function
void layer_norm(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t gamma[EMBED_DIM],
    weight_t beta[EMBED_DIM]
);

// Linear transformation layer
void linear_layer(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t weights[EMBED_DIM][EMBED_DIM],
    weight_t bias[EMBED_DIM]
);

// Patch embedding layer
void patch_embedding(
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE],
    data_t patches[NUM_PATCHES][PATCH_DIM],
    data_t embeddings[NUM_PATCHES][EMBED_DIM],
    weight_t embed_weights[EMBED_DIM][PATCH_DIM],
    weight_t embed_bias[EMBED_DIM]
);

// Multi-head attention layer
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
);

// MLP (feedforward) layer
void mlp_layer(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t fc1_weights[MLP_DIM][EMBED_DIM],
    weight_t fc1_bias[MLP_DIM],
    weight_t fc2_weights[EMBED_DIM][MLP_DIM],
    weight_t fc2_bias[EMBED_DIM]
);

// GELU activation function
data_t gelu(data_t x);

// Softmax function
void softmax(data_t input[NUM_PATCHES + 1], data_t output[NUM_PATCHES + 1]);

// Matrix multiplication optimized for HLS
void matrix_mult(
    data_t A[NUM_PATCHES + 1][HEAD_DIM],
    data_t B[HEAD_DIM][NUM_PATCHES + 1],
    data_t C[NUM_PATCHES + 1][NUM_PATCHES + 1]
);

#endif // VIT_LAYERS_H