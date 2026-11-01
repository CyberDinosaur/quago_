// kmeans_run: run k-means++ / Lloyd's clustering on a fbin dataset and write the
// centers (fbin) and per-point assignments (raw u32). A standalone utility built on
// the quago::kmeans helpers (MKL-backed); not part of the enhancement pipeline.
#include <omp.h>

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "quago/core/common.h"
#include "quago/core/file_io.h"
#include "quago/core/math_utils.h"

namespace po = boost::program_options;

namespace {

void save_assignments(const std::string& path, const uint32_t* assignments, size_t N) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file for assignments: " << path << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(assignments), sizeof(uint32_t) * N);
    std::cout << "[Saved] assignments to " << path << std::endl;
}

void save_centers_fbin(const std::string& path, const float* centers, uint32_t num_centers, uint32_t dim) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to write centers file: " << path << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(&num_centers), 4);
    out.write(reinterpret_cast<const char*>(&dim), 4);
    out.write(reinterpret_cast<const char*>(centers), sizeof(float) * num_centers * dim);
    std::cout << "[Saved] centers to " << path << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path, out_centers, out_assign;
    size_t num_centers = 0, max_iters = 0;
    uint32_t num_threads = 0;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("input", po::value<std::string>(&input_path)->required(), "Input vectors, fbin format");
        desc.add_options()("centers", po::value<size_t>(&num_centers)->default_value(4096), "Number of clusters");
        desc.add_options()("iters", po::value<size_t>(&max_iters)->default_value(50), "Max Lloyd iterations");
        desc.add_options()("out_centers", po::value<std::string>(&out_centers)->default_value("km_centers.fbin"),
                           "Output centers file (fbin)");
        desc.add_options()("out_assign", po::value<std::string>(&out_assign)->default_value("km_assignments.bin"),
                           "Output assignments file (raw u32)");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()));

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    omp_set_num_threads(num_threads);

    float* data = nullptr;
    uint32_t N = 0, dim = 0;
    std::cout << "Loading data: " << input_path << std::endl;
    quago::load_data_aligned<float>(input_path.c_str(), N, dim, data);
    if (!data || N == 0 || dim == 0) {
        std::cerr << "Failed to load data or empty dataset." << std::endl;
        return 2;
    }
    std::cout << "Loaded N=" << N << " dim=" << dim << std::endl;

    float* centers = static_cast<float*>(quago::aligned_malloc(sizeof(float) * num_centers * dim, 64));
    std::memset(centers, 0, sizeof(float) * num_centers * dim);

    std::cout << "Selecting pivots with kmeans++ ..." << std::endl;
    quago::kmeans::kmeanspp_selecting_pivots(data, N, dim, centers, num_centers);

    std::cout << "Running Lloyd's kmeans (max_iters=" << max_iters << ") ..." << std::endl;
    uint32_t* assignments = new uint32_t[N];
    auto* closest_docs = new std::vector<size_t>[num_centers];
    const float residual = quago::kmeans::run_lloyds(data, N, dim, centers, num_centers, max_iters, closest_docs, assignments);
    std::cout << "KMeans finished. Residual (sum sq dist) = " << residual << std::endl;

    // Cluster statistics.
    std::vector<size_t> cluster_sizes(num_centers, 0);
    for (size_t i = 0; i < N; ++i) cluster_sizes[assignments[i]]++;
    std::vector<double> cluster_sumdist(num_centers, 0.0);
#pragma omp parallel for schedule(static, 8192)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        const uint32_t cid = assignments[i];
        const float* p = data + static_cast<size_t>(i) * dim;
        const float* c = centers + static_cast<size_t>(cid) * dim;
        float dist = 0.f;
        for (uint32_t d = 0; d < dim; ++d) {
            const float diff = p[d] - c[d];
            dist += diff * diff;
        }
#pragma omp atomic
        cluster_sumdist[cid] += dist;
    }

    std::vector<size_t> idx(num_centers);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return cluster_sizes[a] > cluster_sizes[b]; });
    std::cout << "Top 20 clusters by size:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(20, num_centers); ++i) {
        const size_t cid = idx[i];
        const double avg_dist = cluster_sizes[cid] ? cluster_sumdist[cid] / static_cast<double>(cluster_sizes[cid]) : 0.0;
        std::cout << std::setw(3) << i << ": center_id=" << cid << " size=" << cluster_sizes[cid]
                  << " avg_sqdist=" << avg_dist << std::endl;
    }

    save_centers_fbin(out_centers, centers, static_cast<uint32_t>(num_centers), dim);
    save_assignments(out_assign, assignments, N);

    free(centers);
    free(data);
    delete[] assignments;
    delete[] closest_docs;
    std::cout << "All results saved." << std::endl;
    return 0;
}
