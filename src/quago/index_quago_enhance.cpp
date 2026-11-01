#include "quago/index_quago.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "quago/core/parameters.h"

namespace quago {

// Region-selection strategies: "frontier", "random" (anything else disables enhancement).
// Enhancement methods: "knn", "qmrng".
bool IndexQuago::quago_enhancement(const std::string& quago_strategy, const std::string& enhance_method,
                                   const Parameters& parameters) {
    std::vector<uint32_t> quago_ids;
    const float quago_ratio = parameters.Get<float>("quago_ratio");

    if (quago_strategy == "random") {
        quago_ids = select_quago_region_random(quago_ratio);
    } else if (quago_strategy == "frontier") {
        // Three-signal selector: position + k-neighborhood PCA anisotropy + direction-frontier
        // hit counts. Also records the per-direction anchors QMRNG reuses.
        compute_quago_score_frontier(parameters.Get<size_t>("knn_size"),
                                     parameters.Get<uint32_t>("sel_dirs"),
                                     parameters.Get<float>("sel_wpos"),
                                     parameters.Get<float>("sel_wgeo"),
                                     parameters.Get<float>("sel_wdir"),
                                     parameters.Get<float>("sel_dir_tau"));
        quago_ids = select_quago_region(quago_ratio);
    } else {
        std::cout << "[Warning] Unknown QUAGO strategy '" << quago_strategy
                  << "'; skipping enhancement." << std::endl;
        return false;
    }

    const uint32_t M_nhood = parameters.Get<uint32_t>("M_nhood");
    if (enhance_method == "knn") {
        enhancement_mode_ = 0;  // add (overlay)
        enhance_subgraph_with_knn(quago_ids, parameters.Get<size_t>("knn_size"), M_nhood);
        return true;
    }
    if (enhance_method == "qmrng" || enhance_method == "QMRNG") {
        // Region-skeleton rebuild PLUS a direction-frontier MST overlay: reuses the frontier selector's
        // sampled directions (their hill-climb anchors) and the loaded knn to build a Prim MST inside
        // each anchor's gt20 set, unioned on top of the rebuilt delta (replace mode).
        enhance_subgraph_with_qmrng(quago_ids, M_nhood, parameters.Get<uint32_t>("skeleton_search_L"),
                                    parameters.Get<float>("skeleton_alpha"),
                                    static_cast<uint32_t>(parameters.Get<size_t>("knn_size")));
        return true;
    }

    std::cerr << "[Error] Unknown enhance method: " << enhance_method << std::endl;
    return false;
}

}  // namespace quago
