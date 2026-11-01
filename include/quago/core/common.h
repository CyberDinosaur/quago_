#pragma once

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

// Branch-hint macros. Defined once here for the whole project; do not redefine
// in individual translation units.
#ifndef QUAGO_LIKELY
#define QUAGO_LIKELY(x) __builtin_expect(!!(x), 1)
#define QUAGO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// ============================================================================
//  DATA_ALIGN_FACTOR -- the single definition site for the whole project.
// ----------------------------------------------------------------------------
//  Row alignment granularity (in floats) used when vectors are stored row-major.
//    - load_data_aligned() (file_io.h) uses it to decide the buffer row stride.
//    - data_align()        (util.h)    uses it to decide the copy source stride.
//  The two must agree: callers index with the original dim as the row step
//  (e.g. `query_data + i * q_dim` in quago_search.cpp). If the stride != dim,
//  every vector after row 0 is read at the wrong offset. When dim is an integer
//  multiple of the factor it happens to work (512 % 8 == 0); otherwise it breaks
//  (e.g. dim=200: 200 % 8 != 0 -> stride 200 vs step 200 is fine only because
//  the factor divides the aligned stride consistently).
// ============================================================================
#ifndef DATA_ALIGN_FACTOR
#if defined(__GNUC__) && defined(__AVX__)
#define DATA_ALIGN_FACTOR 8
#elif defined(__GNUC__) && defined(__SSE2__)
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif

namespace quago {

// Simple RAII stopwatch that prints its elapsed time on destruction unless
// auto_print is disabled.
class Timer {
public:
    explicit Timer(std::string name, bool auto_print = true)
        : name_(std::move(name)), auto_print_(auto_print),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        if (auto_print_) print();
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    double elapsed_ms() const { return elapsed_us() / 1000.0; }

    void print() const {
        std::cout << name_ << ": " << elapsed_us() << " us" << std::endl;
    }

private:
    std::string name_;
    bool auto_print_;
    std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace quago
