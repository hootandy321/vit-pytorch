#ifndef VIT_TYPES_H
#define VIT_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// Define fixed-point types for FPGA HLS
typedef ap_fixed<16, 8> data_t;   // 16-bit fixed point, 8 integer bits
typedef ap_fixed<24, 12> acc_t;   // 24-bit accumulator, 12 integer bits
typedef ap_fixed<32, 16> weight_t; // 32-bit weight, 16 integer bits

// Define common dimensions as compile-time constants
const int IMAGE_SIZE = 224;
const int PATCH_SIZE = 16;
const int NUM_PATCHES = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE); // 196
const int EMBED_DIM = 768;
const int NUM_HEADS = 12;
const int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 64
const int MLP_DIM = 3072;
const int NUM_LAYERS = 12;
const int NUM_CLASSES = 1000;
const int CHANNELS = 3;
const int PATCH_DIM = CHANNELS * PATCH_SIZE * PATCH_SIZE; // 768

// Stream types for pipelined processing
typedef hls::stream<data_t> data_stream_t;
typedef hls::stream<weight_t> weight_stream_t;

// Matrix types
typedef data_t patch_t[PATCH_DIM];
typedef data_t embedding_t[EMBED_DIM];
typedef data_t attention_t[NUM_HEADS][HEAD_DIM];
typedef weight_t linear_weight_t[EMBED_DIM][EMBED_DIM];
typedef weight_t mlp_weight_t[MLP_DIM][EMBED_DIM];

// Position encoding type
typedef data_t pos_encoding_t[NUM_PATCHES + 1][EMBED_DIM]; // +1 for CLS token

#endif // VIT_TYPES_H