// Binary file IO for the fbin/ibin vector formats used by ANN benchmarks, plus
// aligned allocation and CSV result output.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/mman.h>

#include "quago/core/common.h"  // DATA_ALIGN_FACTOR (single definition site)

namespace quago {

// Join path parts with a single '/' separator.
inline std::string join_path(const std::vector<std::string>& parts) {
    std::string result;
    for (const auto& part : parts) {
        if (!result.empty() && result.back() != '/') result += '/';
        result += part;
    }
    return result;
}

// Aligned allocation via posix_memalign; throws std::bad_alloc on failure.
// Free with free().
inline void* aligned_malloc(size_t size, size_t alignment = 64) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) throw std::bad_alloc();
    return ptr;
}

// Append one benchmark run (recall/QPS/latency vs. L) to <save_path>/<method>.csv,
// writing a header only when the file is new. method_name is sanitized for the
// filesystem. base_graph/enhancement are optional metadata columns.
inline void SaveResultsToCSV(const std::string& method_name,
                             const std::vector<uint32_t>& L_values,
                             const std::vector<float>& qps_values,
                             const std::vector<float>& avg_hops_values,
                             const std::vector<float>& avg_cmps_values,
                             const std::vector<float>& mean_latency_values,
                             const std::vector<float>& recall_values,
                             uint32_t K,
                             const std::string& save_path = ".",
                             const std::string& base_graph = "",
                             const std::string& enhancement = "") {
    std::string safe_name = method_name;
    std::replace_if(safe_name.begin(), safe_name.end(),
                    [](char c) {
                        return c == '/' || c == '\\' || c == ':' || c == '*' ||
                               c == '?' || c == '"' || c == '<' || c == '>' || c == '|';
                    },
                    '_');

    const std::filesystem::path dir_path(save_path);
    const std::filesystem::path file_path = dir_path / (safe_name + ".csv");

    try {
        if (!std::filesystem::exists(dir_path)) std::filesystem::create_directories(dir_path);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return;
    }

    const bool has_meta = !base_graph.empty() || !enhancement.empty();
    const bool file_exists = std::filesystem::exists(file_path);
    std::ofstream out(file_path.string(), std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open CSV file: " << file_path.string() << std::endl;
        return;
    }
    if (!file_exists) {
        out << "method_name,L_queue,K,QPS,Avg_hops,Avg_cmps,Mean_latency(ms),Recall@" << K;
        if (has_meta) out << ",base_graph,enhancement";
        out << "\n";
    }

    for (size_t i = 0; i < L_values.size(); ++i) {
        out << safe_name << "," << L_values[i] << "," << K << ","
            << std::fixed << std::setprecision(2) << qps_values[i] << ","
            << std::fixed << std::setprecision(2) << avg_hops_values[i] << ","
            << std::fixed << std::setprecision(2) << avg_cmps_values[i] << ","
            << std::fixed << std::setprecision(2) << mean_latency_values[i] << ","
            << std::fixed << std::setprecision(4) << recall_values[i];
        if (has_meta) out << "," << base_graph << "," << enhancement;
        out << "\n";
    }
    std::cout << "Results appended to: " << file_path.string() << std::endl;
}

// Load a fbin file ([u32 n][u32 dim][n*dim values]) into an aligned buffer whose
// row stride is padded to DATA_ALIGN_FACTOR. `data` is allocated here; free with free().
template <typename T>
void load_data_aligned(const char* filename, uint32_t& points_num, uint32_t& dim, T*& data) {
    std::cout << "[Loading] " << filename << std::endl;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("open file error: " + std::string(filename));

    in.read(reinterpret_cast<char*>(&points_num), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    const uint64_t pts = points_num;
    const uint64_t d = dim;
    const uint64_t new_dim = (d + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    std::cout << "  points_num = " << pts << ", dim = " << d << ", aligned_dim = " << new_dim << std::endl;

    data = static_cast<T*>(aligned_malloc(pts * new_dim * sizeof(T), DATA_ALIGN_FACTOR * 4));
    for (uint64_t i = 0; i < pts; ++i) {
        in.read(reinterpret_cast<char*>(data + i * new_dim), d * sizeof(T));
        std::memset(data + i * new_dim + d, 0, (new_dim - d) * sizeof(T));
    }

    const size_t actual = static_cast<size_t>(in.tellg());
    const size_t expected = static_cast<size_t>(pts * d * sizeof(T) + sizeof(uint32_t) * 2);
    if (actual != expected) throw std::runtime_error("Data file size mismatch in load_data_aligned");
}

// Read only the [u32 n][u32 dim] header of a fbin file and report the implied
// point count computed from the file size (a consistency check).
template <typename T>
void load_meta(const char* filename, unsigned& points_num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("open file error: " + std::string(filename));

    in.read(reinterpret_cast<char*>(&points_num), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    in.seekg(0, std::ios::end);
    const size_t fsize = static_cast<size_t>(in.tellg());
    const uint32_t calc_pts = static_cast<uint32_t>((fsize - 8) / (sizeof(T) * dim));
    std::cout << "[Loaded Meta] " << filename << " pts=" << points_num << " dim=" << dim
              << " computed_pts=" << calc_pts << std::endl;
}

// Read the header of a ground-truth ibin file. GT files store two parallel
// blocks (ids then distances), so the implied count must be 2 * points_num.
template <typename T>
void load_gt_meta(const char* filename, unsigned& points_num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("open file error: " + std::string(filename));

    in.read(reinterpret_cast<char*>(&points_num), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    in.seekg(0, std::ios::end);
    const size_t fsize = static_cast<size_t>(in.tellg());
    const uint32_t calc_pts = static_cast<uint32_t>((fsize - 8) / (sizeof(T) * dim));
    if (points_num * 2 != calc_pts) throw std::runtime_error("GT meta file size mismatch");
    std::cout << "[Loaded GT Meta] " << filename << " pts=" << points_num << " dim=" << dim << std::endl;
}

// Load ground-truth ids only. `data` is allocated with new[]; free with delete[].
template <typename T>
void load_gt_data(const char* filename, uint32_t& points_num, uint32_t& dim, T*& data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("open file error: " + std::string(filename));

    in.read(reinterpret_cast<char*>(&points_num), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    data = new T[static_cast<size_t>(points_num) * dim];
    in.read(reinterpret_cast<char*>(data), sizeof(T) * points_num * dim);

    const size_t actual = static_cast<size_t>(in.tellg());
    const size_t expected = sizeof(uint32_t) * 2 + sizeof(T) * points_num * dim;
    if (actual != expected) throw std::runtime_error("GT data file size mismatch");
}

// Load ground-truth ids and distances. Both are allocated with new[]; free with delete[].
template <typename T, typename T2>
void load_gt_data_with_dist(const char* filename, uint32_t& points_num, uint32_t& dim, T*& ids, T2*& dists) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("open file error: " + std::string(filename));

    in.read(reinterpret_cast<char*>(&points_num), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    ids = new T[static_cast<size_t>(points_num) * dim];
    dists = new T2[static_cast<size_t>(points_num) * dim];
    in.read(reinterpret_cast<char*>(ids), sizeof(T) * points_num * dim);
    in.read(reinterpret_cast<char*>(dists), sizeof(T2) * points_num * dim);

    const size_t actual = static_cast<size_t>(in.tellg());
    const size_t expected = sizeof(uint32_t) * 2 + (sizeof(T) + sizeof(T2)) * points_num * dim;
    if (actual != expected) throw std::runtime_error("GT data (with dist) size mismatch");
}

}  // namespace quago
