// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

#include <xmmintrin.h>
#ifdef __APPLE__
#include <cstdlib>
#else
#include <malloc.h>
#endif

#include "quago/core/common.h"  // DATA_ALIGN_FACTOR (single definition site)

namespace quago {

// Fill addr[0..size) with `size` distinct values in [0, N), pseudo-randomly.
// Requires size < N. Used for the skeleton's random initialization / medoid sampling.
inline void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) addr[i] = rng() % (N - size);
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) addr[i] = addr[i - 1] + 1;
    }
    const unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) addr[i] = (addr[i] + off) % N;
}

// Re-pack a row-major [point_num x dim] buffer so the row stride is a multiple
// of DATA_ALIGN_FACTOR, zero-padding each row. `dim` is updated in place to the
// aligned stride. Takes ownership of `data_ori` and frees it (it must have come
// from aligned_malloc / memalign, matching free()).
inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
    const uint64_t pts = static_cast<uint64_t>(point_num);
    const uint64_t d = static_cast<uint64_t>(dim);
    const uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;

#ifdef __APPLE__
    float* data_new = new float[new_dim * pts];
#else
    float* data_new = static_cast<float*>(memalign(DATA_ALIGN_FACTOR * 8, pts * new_dim * sizeof(float)));
#endif

    for (uint64_t i = 0; i < pts; ++i) {
        std::memcpy(data_new + i * new_dim, data_ori + i * d, d * sizeof(float));
        std::memset(data_new + i * new_dim + d, 0, (new_dim - d) * sizeof(float));
    }
    dim = static_cast<unsigned>(new_dim);

#ifdef __APPLE__
    delete[] data_ori;
#else
    free(data_ori);
#endif
    return data_new;
}

// Software-prefetch a vector (vecsize given in floats) in 64-byte lines.
inline void prefetch_vector(const char* vec, size_t vecsize) {
    const size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64) {
        _mm_prefetch(static_cast<const char*>(vec) + d, _MM_HINT_T0);
    }
}

// L2-normalize a vector in place.
template <typename T>
inline void normalize(T* arr, const size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += arr[i] * arr[i];
    sum = std::sqrt(sum);
    for (size_t i = 0; i < dim; ++i) arr[i] = static_cast<T>(arr[i] / sum);
}

}  // namespace quago
