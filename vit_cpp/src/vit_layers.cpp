#include "vit_layers.h"

// Layer normalization implementation
void layer_norm(
    data_t input[EMBED_DIM],
    data_t output[EMBED_DIM],
    weight_t gamma[EMBED_DIM],
    weight_t beta[EMBED_DIM]
) {
    #ifdef __SYNTHESIS__
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    #endif
    
    // Calculate mean
    acc_t mean = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        #ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
        #endif
        mean += input[i];
    }
    mean /= EMBED_DIM;
    
    // Calculate variance
    acc_t variance = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        #ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
        #endif
        acc_t diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= EMBED_DIM;
    
    // Apply normalization
    data_t inv_std = 1.0 / hls::sqrt(variance + 1e-6);
    for (int i = 0; i < EMBED_DIM; i++) {
        #ifdef __SYNTHESIS__
        #pragma HLS PIPELINE II=1
        #endif
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
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=weights dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=input complete
    #pragma HLS ARRAY_PARTITION variable=output complete
    
    OUTPUT_LOOP: for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1
        acc_t sum = bias[i];
        INPUT_LOOP: for (int j = 0; j < EMBED_DIM; j++) {
            #pragma HLS UNROLL
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
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=embed_weights dim=2 complete
    
    // Extract patches
    int patch_idx = 0;
    PATCH_H_LOOP: for (int h = 0; h < IMAGE_SIZE; h += PATCH_SIZE) {
        PATCH_W_LOOP: for (int w = 0; w < IMAGE_SIZE; w += PATCH_SIZE) {
            int idx = 0;
            PATCH_CH_LOOP: for (int c = 0; c < CHANNELS; c++) {
                PATCH_PH_LOOP: for (int ph = 0; ph < PATCH_SIZE; ph++) {
                    PATCH_PW_LOOP: for (int pw = 0; pw < PATCH_SIZE; pw++) {
                        #pragma HLS PIPELINE II=1
                        patches[patch_idx][idx++] = image[c][h + ph][w + pw];
                    }
                }
            }
            patch_idx++;
        }
    }
    
    // Apply linear projection to each patch
    EMBED_PATCH_LOOP: for (int p = 0; p < NUM_PATCHES; p++) {
        linear_layer(patches[p], embeddings[p], embed_weights, embed_bias);
    }
}

// GELU activation function
data_t gelu(data_t x) {
    #pragma HLS INLINE
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    data_t x3 = x * x * x;
    data_t inner = hls::sqrt(2.0 / 3.14159) * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + hls::tanh(inner));
}

// Softmax function implementation
void softmax(data_t input[NUM_PATCHES + 1], data_t output[NUM_PATCHES + 1]) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    
    // Find maximum for numerical stability
    data_t max_val = input[0];
    MAX_LOOP: for (int i = 1; i < NUM_PATCHES + 1; i++) {
        #pragma HLS PIPELINE II=1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    data_t exp_sum = 0;
    data_t exp_vals[NUM_PATCHES + 1];
    #pragma HLS ARRAY_PARTITION variable=exp_vals complete
    
    EXP_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
        #pragma HLS PIPELINE II=1
        exp_vals[i] = hls::exp(input[i] - max_val);
        exp_sum += exp_vals[i];
    }
    
    // Normalize
    NORM_SOFTMAX_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = exp_vals[i] / exp_sum;
    }
}

// Matrix multiplication optimized for HLS
void matrix_mult(
    data_t A[NUM_PATCHES + 1][HEAD_DIM],
    data_t B[HEAD_DIM][NUM_PATCHES + 1],
    data_t C[NUM_PATCHES + 1][NUM_PATCHES + 1]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=A dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=B dim=1 complete
    
    MULT_I_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
        MULT_J_LOOP: for (int j = 0; j < NUM_PATCHES + 1; j++) {
            #pragma HLS PIPELINE II=1
            acc_t sum = 0;
            MULT_K_LOOP: for (int k = 0; k < HEAD_DIM; k++) {
                #pragma HLS UNROLL
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
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=fc1_weights dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=fc2_weights dim=2 complete
    
    data_t intermediate[MLP_DIM];
    #pragma HLS ARRAY_PARTITION variable=intermediate complete
    
    // First linear layer
    FC1_LOOP: for (int i = 0; i < MLP_DIM; i++) {
        #pragma HLS PIPELINE II=1
        acc_t sum = fc1_bias[i];
        FC1_INNER_LOOP: for (int j = 0; j < EMBED_DIM; j++) {
            #pragma HLS UNROLL
            sum += fc1_weights[i][j] * input[j];
        }
        intermediate[i] = gelu(sum);
    }
    
    // Second linear layer
    FC2_LOOP: for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1
        acc_t sum = fc2_bias[i];
        FC2_INNER_LOOP: for (int j = 0; j < MLP_DIM; j++) {
            #pragma HLS UNROLL
            sum += fc2_weights[i][j] * intermediate[j];
        }
        output[i] = sum;
    }
}

// Multi-head attention implementation
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
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wq dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wk dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wv dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=wo dim=2 complete
    
    // Query, Key, Value projections
    data_t queries[NUM_PATCHES + 1][EMBED_DIM];
    data_t keys[NUM_PATCHES + 1][EMBED_DIM];
    data_t values[NUM_PATCHES + 1][EMBED_DIM];
    
    #pragma HLS ARRAY_PARTITION variable=queries dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=keys dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=values dim=2 complete
    
    // Compute Q, K, V for all tokens
    QKV_TOKEN_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
        #pragma HLS PIPELINE II=1
        linear_layer(input[i], queries[i], wq, bias_q);
        linear_layer(input[i], keys[i], wk, bias_k);
        linear_layer(input[i], values[i], wv, bias_v);
    }
    
    // Multi-head attention computation
    data_t attention_output[NUM_PATCHES + 1][EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=attention_output dim=2 complete
    
    HEAD_LOOP: for (int h = 0; h < NUM_HEADS; h++) {
        #pragma HLS PIPELINE II=1
        
        // Extract head-specific Q, K, V
        data_t q_head[NUM_PATCHES + 1][HEAD_DIM];
        data_t k_head[NUM_PATCHES + 1][HEAD_DIM];
        data_t v_head[NUM_PATCHES + 1][HEAD_DIM];
        
        #pragma HLS ARRAY_PARTITION variable=q_head dim=2 complete
        #pragma HLS ARRAY_PARTITION variable=k_head dim=2 complete
        #pragma HLS ARRAY_PARTITION variable=v_head dim=2 complete
        
        HEAD_EXTRACT_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
            HEAD_DIM_LOOP: for (int j = 0; j < HEAD_DIM; j++) {
                #pragma HLS UNROLL
                q_head[i][j] = queries[i][h * HEAD_DIM + j];
                k_head[i][j] = keys[i][h * HEAD_DIM + j];
                v_head[i][j] = values[i][h * HEAD_DIM + j];
            }
        }
        
        // Compute attention scores
        data_t attention_scores[NUM_PATCHES + 1][NUM_PATCHES + 1];
        #pragma HLS ARRAY_PARTITION variable=attention_scores complete
        
        // Q * K^T
        data_t k_transpose[HEAD_DIM][NUM_PATCHES + 1];
        #pragma HLS ARRAY_PARTITION variable=k_transpose complete
        
        TRANSPOSE_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                #pragma HLS UNROLL
                k_transpose[j][i] = k_head[i][j];
            }
        }
        
        matrix_mult(q_head, k_transpose, attention_scores);
        
        // Scale and apply softmax
        data_t scale = 1.0 / hls::sqrt((data_t)HEAD_DIM);
        SCALE_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
            SCALE_INNER_LOOP: for (int j = 0; j < NUM_PATCHES + 1; j++) {
                #pragma HLS PIPELINE II=1
                attention_scores[i][j] *= scale;
            }
            softmax(attention_scores[i], attention_scores[i]);
        }
        
        // Apply attention to values
        data_t head_output[NUM_PATCHES + 1][HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=head_output dim=2 complete
        
        ATTN_OUTPUT_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
            HEAD_OUT_DIM_LOOP: for (int j = 0; j < HEAD_DIM; j++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                ATTN_SUM_LOOP: for (int k = 0; k < NUM_PATCHES + 1; k++) {
                    #pragma HLS UNROLL
                    sum += attention_scores[i][k] * v_head[k][j];
                }
                head_output[i][j] = sum;
            }
        }
        
        // Concatenate head outputs
        HEAD_CONCAT_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                #pragma HLS UNROLL
                attention_output[i][h * HEAD_DIM + j] = head_output[i][j];
            }
        }
    }
    
    // Apply output projection
    OUTPUT_PROJ_LOOP: for (int i = 0; i < NUM_PATCHES + 1; i++) {
        #pragma HLS PIPELINE II=1
        linear_layer(attention_output[i], output[i], wo, bias_o);
    }
}