#ifndef VIT_H
#define VIT_H

#ifdef __SYNTHESIS__
    #include "vit_types.h"
#else
    #include "vit_types_sim.h"
#endif

#include "vit_layers.h"

// Transformer encoder block
void transformer_encoder_block(
    data_t input[NUM_PATCHES + 1][EMBED_DIM],
    data_t output[NUM_PATCHES + 1][EMBED_DIM],
    // Attention weights
    weight_t wq[EMBED_DIM][EMBED_DIM],
    weight_t wk[EMBED_DIM][EMBED_DIM],
    weight_t wv[EMBED_DIM][EMBED_DIM],
    weight_t wo[EMBED_DIM][EMBED_DIM],
    weight_t bias_q[EMBED_DIM],
    weight_t bias_k[EMBED_DIM],
    weight_t bias_v[EMBED_DIM],
    weight_t bias_o[EMBED_DIM],
    // MLP weights
    weight_t fc1_weights[MLP_DIM][EMBED_DIM],
    weight_t fc1_bias[MLP_DIM],
    weight_t fc2_weights[EMBED_DIM][MLP_DIM],
    weight_t fc2_bias[EMBED_DIM],
    // Layer norm weights
    weight_t ln1_gamma[EMBED_DIM],
    weight_t ln1_beta[EMBED_DIM],
    weight_t ln2_gamma[EMBED_DIM],
    weight_t ln2_beta[EMBED_DIM]
);

// Complete Vision Transformer
void vision_transformer(
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE],
    data_t output[NUM_CLASSES],
    // Patch embedding weights
    weight_t embed_weights[EMBED_DIM][PATCH_DIM],
    weight_t embed_bias[EMBED_DIM],
    // Position embedding
    pos_encoding_t pos_embed,
    // CLS token
    data_t cls_token[EMBED_DIM],
    // Transformer weights (simplified - in practice would be arrays of weights)
    weight_t transformer_weights[NUM_LAYERS][8][EMBED_DIM][EMBED_DIM], // Simplified representation
    // Classification head weights
    weight_t head_weights[NUM_CLASSES][EMBED_DIM],
    weight_t head_bias[NUM_CLASSES]
);

// Simplified inference function for testing
void vit_inference(
    data_t image[CHANNELS][IMAGE_SIZE][IMAGE_SIZE],
    data_t predictions[NUM_CLASSES]
);

#endif // VIT_H