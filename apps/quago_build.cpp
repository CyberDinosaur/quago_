#include <omp.h>

#include <boost/program_options.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "quago/core/common.h"
#include "quago/core/distance.h"
#include "quago/core/file_io.h"
#include "quago/core/parameters.h"
#include "quago/index_quago.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string base_data_file, data_type, dist;
    std::string base_index_file, base_index_format, knng_path;
    std::string quago_strategy, enhance_method, quago_index_save_dir;
    uint32_t M_nhood = 0, num_threads = 0, sel_dirs = 0, skeleton_search_L = 0;
    size_t knn_size = 0;
    float quago_ratio = 0.f, skeleton_alpha = 0.f;
    float sel_wpos = 0.f, sel_wgeo = 0.f, sel_wdir = 0.f, sel_dir_tau = 0.f;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "Data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "Distance function <l2/ip/cosine>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(), "Base vectors, fbin format");
        desc.add_options()("M_nhood", po::value<uint32_t>(&M_nhood)->required(), "Max degree budget for enhancement");
        desc.add_options()("base_index_path", po::value<std::string>(&base_index_file)->required(), "Base graph file");
        desc.add_options()("base_index_format", po::value<std::string>(&base_index_format)->default_value("hnsw"),
                           "Base graph format: quago/nsg/hnsw");
        desc.add_options()("knng_path", po::value<std::string>(&knng_path)->required(), "KNNG file (ids-only, NSG-style format)");

        // Region selection
        desc.add_options()("quago_strategy", po::value<std::string>(&quago_strategy)->default_value("none"),
                           "Region selection strategy: frontier/random/none");
        desc.add_options()("knn_size", po::value<size_t>(&knn_size)->default_value(20),
                           "Neighborhood size k used for region scoring / knn enhancement");
        desc.add_options()("quago_ratio", po::value<float>(&quago_ratio)->default_value(0.1f),
                           "Top fraction of points selected as the QUAGO region");
        desc.add_options()("enhance_method", po::value<std::string>(&enhance_method)->default_value("qmrng"),
                           "Enhancement method: knn/qmrng");

        // Region-skeleton parameters (the skeleton rebuild stage of qmrng; always Set so Get never misses a key)
        desc.add_options()("skeleton_search_L", po::value<uint32_t>(&skeleton_search_L)->default_value(100),
                           "Greedy beam-search width during the skeleton rebuild");
        desc.add_options()("skeleton_alpha", po::value<float>(&skeleton_alpha)->default_value(1.2f),
                           "RobustPrune slack on the skeleton's second pass");

        // frontier selector parameters (quago_strategy=frontier)
        desc.add_options()("sel_dirs", po::value<uint32_t>(&sel_dirs)->default_value(8192),
                           "Number of direction probes M (greedy farthest-point spherical code)");
        // The position sub-weights (0.3*d_c + 0.7*r_k) are fixed in
        // IndexQuago::compute_quago_score_frontier and deliberately not exposed here.
        desc.add_options()("sel_wpos", po::value<float>(&sel_wpos)->default_value(-1.f),
                           "Fusion weight: position (-1 = metric-dependent auto: IP/COS pos0.8+dir0.2, L2 pos0.7+geo0.1+dir0.2)");
        desc.add_options()("sel_wgeo", po::value<float>(&sel_wgeo)->default_value(-1.f), "Fusion weight: PCA anisotropy (-1 = auto; 0 under IP/COSINE)");
        desc.add_options()("sel_wdir", po::value<float>(&sel_wdir)->default_value(-1.f), "Fusion weight: direction-frontier count (-1 = auto)");
        desc.add_options()("sel_dir_tau", po::value<float>(&sel_dir_tau)->default_value(1.0f),
                           "Direction activation temperature: a_hat = expm1(hits/tau); smaller tau = sharper");

        desc.add_options()("quago_index_save_dir", po::value<std::string>(&quago_index_save_dir)->required(),
                           "Directory to write the enhancement delta into");
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

    quago::Metric dist_metric = quago::L2;
    if (dist == "ip") dist_metric = quago::INNER_PRODUCT;
    else if (dist == "cosine") dist_metric = quago::COSINE;

    uint32_t total_num = 0, dim = 0;
    float* base_data = nullptr;
    quago::load_data_aligned<float>(base_data_file.c_str(), total_num, dim, base_data);

    quago::Parameters parameters;
    parameters.Set<std::string>("data_type", data_type);
    parameters.Set<uint32_t>("M_nhood", M_nhood);
    parameters.Set<uint32_t>("num_threads", num_threads);
    parameters.Set<std::string>("quago_strategy", quago_strategy);
    parameters.Set<size_t>("knn_size", knn_size);
    parameters.Set<float>("quago_ratio", quago_ratio);
    parameters.Set<std::string>("enhance_method", enhance_method);
    parameters.Set<uint32_t>("skeleton_search_L", skeleton_search_L);
    parameters.Set<float>("skeleton_alpha", skeleton_alpha);
    parameters.Set<uint32_t>("sel_dirs", sel_dirs);
    parameters.Set<float>("sel_wpos", sel_wpos);
    parameters.Set<float>("sel_wgeo", sel_wgeo);
    parameters.Set<float>("sel_wdir", sel_wdir);
    parameters.Set<float>("sel_dir_tau", sel_dir_tau);
    parameters.Set<std::string>("base_index_file", base_index_file);
    parameters.Set<std::string>("base_index_format", base_index_format);
    parameters.Set<std::string>("knng_path", knng_path);
    parameters.Set<std::string>("quago_index_save_dir", quago_index_save_dir);

    quago::IndexQuago index_quago(dim, total_num, dist_metric);
    omp_set_num_threads(num_threads);

    const auto s = std::chrono::high_resolution_clock::now();
    index_quago.BuildQuagoGraph(base_data, total_num, dim, parameters);
    const std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Indexing time: " << diff.count() << " seconds\n";

    free(base_data);
    return 0;
}
