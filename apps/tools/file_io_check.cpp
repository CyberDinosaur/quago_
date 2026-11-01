// file_io_check: smoke-test the fbin/ibin loaders on real files.
//   usage: file_io_check <base.fbin> [query.fbin] [gt.ibin]
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

#include "quago/core/file_io.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <base.fbin> [query.fbin] [gt.ibin]\n";
        return 1;
    }
    try {
        std::cout << "\n==== Load BASE ====\n";
        uint32_t base_n = 0, base_dim = 0;
        quago::load_meta<float>(argv[1], base_n, base_dim);
        float* base_data = nullptr;
        quago::load_data_aligned<float>(argv[1], base_n, base_dim, base_data);
        std::cout << "[BASE] n=" << base_n << " dim=" << base_dim << "\n  first 5 dims: ";
        for (int i = 0; i < 5 && i < static_cast<int>(base_dim); ++i) std::cout << base_data[i] << " ";
        std::cout << "\n";
        free(base_data);

        if (argc >= 3) {
            std::cout << "\n==== Load QUERY ====\n";
            uint32_t qn = 0, qdim = 0;
            quago::load_meta<float>(argv[2], qn, qdim);
            float* query_data = nullptr;
            quago::load_data_aligned<float>(argv[2], qn, qdim, query_data);
            std::cout << "[QUERY] n=" << qn << " dim=" << qdim << "\n  first 5 dims: ";
            for (int i = 0; i < 5 && i < static_cast<int>(qdim); ++i) std::cout << query_data[i] << " ";
            std::cout << "\n";
            free(query_data);
        }

        if (argc >= 4) {
            std::cout << "\n==== Load GT (ibin) ====\n";
            uint32_t gt_q = 0, gt_k = 0;
            quago::load_gt_meta<uint32_t>(argv[3], gt_q, gt_k);
            uint32_t* gt_ids = nullptr;
            quago::load_gt_data<uint32_t>(argv[3], gt_q, gt_k, gt_ids);
            std::cout << "[GT] queries=" << gt_q << " topK=" << gt_k << "\n  first 10 ids: ";
            for (uint32_t i = 0; i < 10 && i < gt_k; ++i) std::cout << gt_ids[i] << " ";
            std::cout << "\n";
            delete[] gt_ids;
        }

        std::cout << "\n==== IO CHECK PASS ====\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
