#ifndef VIT_TYPES_SIMULATION_H
#define VIT_TYPES_SIMULATION_H

#include <cstdint>
#include <iostream>
#include <cmath>

// Simulation versions of HLS types for testing without Vivado HLS
#ifdef __SYNTHESIS__
    #include <ap_int.h>
    #include <ap_fixed.h>
    #include <hls_stream.h>
    #include <hls_math.h>
    
    // Use actual HLS types
    typedef ap_fixed<16, 8> data_t;
    typedef ap_fixed<24, 12> acc_t;
    typedef ap_fixed<32, 16> weight_t;
    typedef hls::stream<data_t> data_stream_t;
    
#else
    // Simulation types for regular C++ compilation
    typedef float data_t;
    typedef double acc_t;
    typedef float weight_t;
    
    // Mock stream class for simulation
    template<typename T>
    class mock_stream {
    public:
        void write(const T& data) { buffer = data; }
        T read() { return buffer; }
        bool empty() { return false; }
        bool full() { return false; }
    private:
        T buffer;
    };
    
    typedef mock_stream<data_t> data_stream_t;
    
    // Mock HLS math functions
    namespace hls {
        inline float sqrt(float x) { return std::sqrt(x); }
        inline float exp(float x) { return std::exp(x); }
        inline float tanh(float x) { return std::tanh(x); }
    }
#endif

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

// Matrix types
typedef data_t patch_t[PATCH_DIM];
typedef data_t embedding_t[EMBED_DIM];
typedef data_t attention_t[NUM_HEADS][HEAD_DIM];
typedef weight_t linear_weight_t[EMBED_DIM][EMBED_DIM];
typedef weight_t mlp_weight_t[MLP_DIM][EMBED_DIM];

// Position encoding type
typedef data_t pos_encoding_t[NUM_PATCHES + 1][EMBED_DIM]; // +1 for CLS token

#endif // VIT_TYPES_SIMULATION_H