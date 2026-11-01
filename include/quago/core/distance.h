// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
//
// Distance functors. The hot paths use AVX-512 when available and fall back to
// plain scalar loops otherwise, so the project compiles on any x86-64 target.
// All functors return a value where SMALLER == CLOSER:
//   - DistanceL2            -> squared Euclidean distance (non-negative)
//   - DistanceInnerProduct  -> -dot(a, b)  (negated so that larger dot == smaller)
#pragma once

#include <cstddef>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

namespace quago {

enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3, COSINE = 4 };

class Distance {
 public:
  virtual ~Distance() = default;
  virtual float compare(const float* a, const float* b, unsigned length) const = 0;
};

#if defined(__AVX512F__) && defined(__AVX512DQ__)
namespace detail {
// Load up to 3 trailing floats (d in [0,3]) into a zero-padded SSE register.
inline __m128 masked_read(int d, const float* x) {
  __attribute__((__aligned__(16))) float buf[4] = {0.f, 0.f, 0.f, 0.f};
  switch (d) {
    case 3: buf[2] = x[2]; [[fallthrough]];
    case 2: buf[1] = x[1]; [[fallthrough]];
    case 1: buf[0] = x[0]; break;
    default: break;
  }
  return _mm_load_ps(buf);
}
}  // namespace detail
#endif

class DistanceL2 : public Distance {
 public:
  float compare(const float* x, const float* y, unsigned d) const override {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 msum0 = _mm512_setzero_ps();
    while (d >= 16) {
      const __m512 mx = _mm512_loadu_ps(x); x += 16;
      const __m512 my = _mm512_loadu_ps(y); y += 16;
      const __m512 diff = mx - my;
      msum0 += diff * diff;
      d -= 16;
    }
    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 += _mm512_extractf32x8_ps(msum0, 0);
    if (d >= 8) {
      const __m256 mx = _mm256_loadu_ps(x); x += 8;
      const __m256 my = _mm256_loadu_ps(y); y += 8;
      const __m256 diff = mx - my;
      msum1 += diff * diff;
      d -= 8;
    }
    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);
    if (d >= 4) {
      const __m128 mx = _mm_loadu_ps(x); x += 4;
      const __m128 my = _mm_loadu_ps(y); y += 4;
      const __m128 diff = mx - my;
      msum2 += diff * diff;
      d -= 4;
    }
    if (d > 0) {
      const __m128 mx = detail::masked_read(static_cast<int>(d), x);
      const __m128 my = detail::masked_read(static_cast<int>(d), y);
      const __m128 diff = mx - my;
      msum2 += diff * diff;
    }
    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
#else
    float sum = 0.f;
    for (unsigned i = 0; i < d; ++i) {
      const float t = x[i] - y[i];
      sum += t * t;
    }
    return sum;
#endif
  }
};

class DistanceInnerProduct : public Distance {
 public:
  float compare(const float* a, const float* b, unsigned size) const override {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 msum0 = _mm512_setzero_ps();
    while (size >= 16) {
      const __m512 mx = _mm512_loadu_ps(a); a += 16;
      const __m512 my = _mm512_loadu_ps(b); b += 16;
      msum0 = _mm512_add_ps(msum0, _mm512_mul_ps(mx, my));
      size -= 16;
    }
    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 += _mm512_extractf32x8_ps(msum0, 0);
    if (size >= 8) {
      const __m256 mx = _mm256_loadu_ps(a); a += 8;
      const __m256 my = _mm256_loadu_ps(b); b += 8;
      msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
      size -= 8;
    }
    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);
    if (size >= 4) {
      const __m128 mx = _mm_loadu_ps(a); a += 4;
      const __m128 my = _mm_loadu_ps(b); b += 4;
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
      size -= 4;
    }
    if (size > 0) {
      const __m128 mx = detail::masked_read(static_cast<int>(size), a);
      const __m128 my = detail::masked_read(static_cast<int>(size), b);
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
    }
    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return -1.0f * _mm_cvtss_f32(msum2);
#else
    float dot = 0.f;
    for (unsigned i = 0; i < size; ++i) dot += a[i] * b[i];
    return -dot;
#endif
  }
};

// ||a - b||^2 == ||a||^2 - 2*a.b + ||b||^2. When ||b||^2 (`norm`) is
// precomputed this avoids re-squaring b. Inherits compare(a, b, size).
class DistanceFastL2 : public DistanceInnerProduct {
 public:
  using DistanceInnerProduct::compare;

  float compare(const float* a, const float* b, float norm, unsigned size) const {
    return -2.f * DistanceInnerProduct::compare(a, b, size) + norm;
  }

  float norm(const float* a, unsigned size) const {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512 msum0 = _mm512_setzero_ps();
    while (size >= 16) {
      const __m512 mx = _mm512_loadu_ps(a); a += 16;
      msum0 = _mm512_add_ps(msum0, _mm512_mul_ps(mx, mx));
      size -= 16;
    }
    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 += _mm512_extractf32x8_ps(msum0, 0);
    if (size >= 8) {
      const __m256 mx = _mm256_loadu_ps(a); a += 8;
      msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, mx));
      size -= 8;
    }
    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);
    if (size >= 4) {
      const __m128 mx = _mm_loadu_ps(a); a += 4;
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, mx));
      size -= 4;
    }
    if (size > 0) {
      const __m128 mx = detail::masked_read(static_cast<int>(size), a);
      msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, mx));
    }
    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
#else
    float sum = 0.f;
    for (unsigned i = 0; i < size; ++i) sum += a[i] * a[i];
    return sum;
#endif
  }
};

}  // namespace quago
