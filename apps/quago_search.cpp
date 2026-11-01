#include <omp.h>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "quago/core/common.h"
#include "quago/core/file_io.h"
#include "quago/core/math_utils.h"
#include "quago/core/parameters.h"
#include "quago/core/util.h"
#include "quago/index_quago.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string base_data_file, query_file, gt_file;
    std::string base_index_path, base_index_format, data_type, dist;
    std::string enhancement_delta_path, method_name, eval_path, frontier;
    std::vector<uint32_t> L_values;
    std::vector<uint32_t> K_list;
    uint32_t num_threads = 0, entry_point_override = 0;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "Data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "Distance function <l2/ip/cosine>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(), "Base vectors, fbin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(), "Query vectors, fbin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(), "Ground-truth file, ibin format");
        desc.add_options()("base_index_path", po::value<std::string>(&base_index_path)->required(), "Base graph file");
        desc.add_options()("base_index_format", po::value<std::string>(&base_index_format)->default_value("hnsw"),
                           "Base graph format: quago/nsg/hnsw");
        desc.add_options()("entry_point", po::value<uint32_t>(&entry_point_override)->default_value(0),
                           "If non-zero, override the search entry point provided by the loader");
        desc.add_options()("enhancement_delta_path", po::value<std::string>(&enhancement_delta_path)->default_value(""),
                           "Optional enhancement delta to overlay onto the base graph");
        desc.add_options()("L_pq", po::value<std::vector<uint32_t>>(&L_values)->multitoken()->required(),
                           "Priority-queue lengths (L) to sweep");
        desc.add_options()("k", po::value<std::vector<uint32_t>>(&K_list)->multitoken(),
                           "One or more k values to evaluate (e.g. --k 10 100). The search runs once "
                           "at max(k) and recall is reported for each; default 10.");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of search threads");
        desc.add_options()("method_name", po::value<std::string>(&method_name)->default_value(""),
                           "Method name used as the CSV filename (default: derived from base graph + delta)");
        desc.add_options()("eval_path", po::value<std::string>(&eval_path)->default_value("."),
                           "Directory to write the evaluation CSV into");
        desc.add_options()("frontier", po::value<std::string>(&frontier)->default_value("sorted"),
                           "Frontier data structure: sorted (shipped sorted array) or heap "
                           "(hnswlib-style two heaps; drops the O(L) memmove per insertion and is "
                           "up to 1.8x faster at L=20000 for the same recall)");

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

    if (frontier != "sorted" && frontier != "heap") {
        std::cerr << "Error: --frontier must be 'sorted' or 'heap', got '" << frontier << "'." << std::endl;
        return -1;
    }

    quago::Parameters parameters;
    parameters.Set<uint32_t>("num_threads", num_threads);
    omp_set_num_threads(num_threads);

    std::cout << "========================= Loading data =========================" << std::endl;
    uint32_t base_num = 0, base_dim = 0;
    quago::load_meta<float>(base_data_file.c_str(), base_num, base_dim);

    uint32_t q_pts = 0, q_dim = 0;
    float* query_data = nullptr;
    quago::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    quago::load_data_aligned<float>(query_file.c_str(), q_pts, q_dim, query_data);

    uint32_t gt_pts = 0, gt_dim = 0;
    uint32_t* gt_ids = nullptr;
    float* gt_dists = nullptr;
    quago::load_gt_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    quago::load_gt_data_with_dist<uint32_t, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids, gt_dists);

    quago::Metric dist_metric;
    if (dist == "l2") {
        dist_metric = quago::L2;
    } else if (dist == "ip") {
        dist_metric = quago::INNER_PRODUCT;
    } else if (dist == "cosine") {
        dist_metric = quago::COSINE;
    } else {
        std::cerr << "Unknown distance type: " << dist << std::endl;
        return -1;
    }
    std::cout << "Distance metric: " << dist << std::endl;

    std::cout << "========================= Loading graph =========================" << std::endl;
    quago::IndexQuago index_quago(base_dim, base_num, dist_metric);
    index_quago.SetHeapFrontier(frontier == "heap");
    std::cout << "[Frontier] " << frontier << std::endl;
    index_quago.LoadBaseGraph(base_index_path, base_index_format);
    if (entry_point_override != 0) {
        std::cout << "[Override] entry_point = " << entry_point_override << std::endl;
        index_quago.SetEntryPoint(entry_point_override);
    }
    index_quago.LoadSearchNeededData(base_data_file.c_str());

    if (!enhancement_delta_path.empty() && std::filesystem::exists(enhancement_delta_path)) {
        std::cout << "[Load] enhancement delta: " << enhancement_delta_path << std::endl;
        index_quago.LoadQuagoDelta(enhancement_delta_path);
        index_quago.apply_quago_delta();
    }

    if (index_quago.need_normalize) {
        std::cout << "Normalizing query data" << std::endl;
        for (uint32_t i = 0; i < q_pts; ++i) quago::normalize<float>(query_data + static_cast<size_t>(i) * q_dim, q_dim);
    }
    index_quago.InitVisitedListPool(num_threads);

    // Derive metadata names for the CSV.
    const std::string base_graph_name = std::filesystem::path(base_index_path).stem().string();
    std::string enhancement_name;
    if (!enhancement_delta_path.empty() && std::filesystem::exists(enhancement_delta_path)) {
        enhancement_name = std::filesystem::path(enhancement_delta_path).stem().string();
    }
    if (method_name.empty()) {
        method_name = base_graph_name;
        if (!enhancement_name.empty()) method_name += "_" + enhancement_name;
    }
    std::cout << "[Meta] base_graph=" << base_graph_name
              << " enhancement=" << (enhancement_name.empty() ? "none" : enhancement_name)
              << " method_name=" << method_name << std::endl;

    // Evaluate on at most 1000 queries, bounded by what the query/GT files actually hold.
    const uint32_t num_searches = std::min<uint32_t>({1000u, q_pts, gt_pts});
    if (num_searches == 0) {
        std::cerr << "No queries to evaluate." << std::endl;
        return -1;
    }

    if (K_list.empty()) K_list = {10};
    std::sort(K_list.begin(), K_list.end());
    K_list.erase(std::unique(K_list.begin(), K_list.end()), K_list.end());
    const uint32_t K_search = K_list.back();  // search once at the largest k; recall@k derives from the top-k

    std::cout << "========================= Searching (" << num_threads << " threads) =========================\n";
    std::cout << std::left << std::setw(10) << "L_queue" << std::setw(12) << "QPS" << std::setw(12) << "Avg_hops"
              << std::setw(12) << "Avg_cmps" << std::setw(15) << "Mean_latency";
    for (uint32_t k : K_list) std::cout << std::setw(12) << ("Recall@" + std::to_string(k));
    std::cout << "\n";

    std::vector<std::vector<uint32_t>> res_indices(num_searches, std::vector<uint32_t>(K_search, 0));
    std::vector<uint32_t> hops_vec(num_searches, 0), cmps_vec(num_searches, 0);
    std::vector<float> qps_values, avg_hops_values, avg_cmps_values, mean_latency_values;
    std::map<uint32_t, std::vector<float>> recall_by_k;
    for (uint32_t k : K_list) recall_by_k[k] = {};

    for (uint32_t L : L_values) {
        if (L < K_search) {
            std::cerr << "Error: L=" << L << " cannot be smaller than k=" << K_search << std::endl;
            return -1;
        }
        parameters.Set<uint32_t>("L_search", L);

        const auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (uint32_t search = 0; search < num_searches; ++search) {
            const auto ret_val = index_quago.Search_normal(query_data + static_cast<size_t>(search) * q_dim, search,
                                                           K_search, parameters, res_indices[search].data());
            hops_vec[search] = ret_val.first;
            cmps_vec[search] = ret_val.second;
        }
        const auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::high_resolution_clock::now() - start)
                                 .count();

        float avg_hops = 0.f, avg_cmps = 0.f;
        for (uint32_t i = 0; i < num_searches; ++i) {
            avg_hops += static_cast<float>(hops_vec[i]);
            avg_cmps += static_cast<float>(cmps_vec[i]);
        }
        avg_hops /= static_cast<float>(num_searches);
        avg_cmps /= static_cast<float>(num_searches);

        const float qps = static_cast<float>(num_searches) / (static_cast<float>(diff_ms) / 1000.0f);
        const float mean_latency = static_cast<float>(diff_ms) / static_cast<float>(num_searches);

        std::cout << std::left << std::setw(10) << L << std::setw(12) << qps << std::setw(12) << avg_hops
                  << std::setw(12) << avg_cmps << std::setw(15) << mean_latency;
        for (uint32_t k : K_list) {
            const float recall = quago::math_utils::ComputeRecall(num_searches, k, gt_dim, res_indices, gt_ids);
            recall_by_k[k].push_back(recall);
            std::cout << std::setw(12) << recall;
        }
        std::cout << "\n";

        qps_values.push_back(qps);
        avg_hops_values.push_back(avg_hops);
        avg_cmps_values.push_back(avg_cmps);
        mean_latency_values.push_back(mean_latency);
    }

    // One CSV per k, named <method_name>_k<k>.csv.
    for (uint32_t k : K_list) {
        quago::SaveResultsToCSV(method_name + "_k" + std::to_string(k), L_values, qps_values, avg_hops_values,
                                avg_cmps_values, mean_latency_values, recall_by_k[k], k, eval_path,
                                base_graph_name, enhancement_name);
    }

    free(query_data);  // allocated by load_data_aligned via posix_memalign
    delete[] gt_ids;
    delete[] gt_dists;
    return 0;
}
