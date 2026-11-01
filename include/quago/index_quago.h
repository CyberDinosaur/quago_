#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "quago/core/common.h"
#include "quago/core/index.h"
#include "quago/core/neighbor.h"
#include "quago/core/util.h"
#include "quago/core/visited_list_pool.h"

namespace quago {

// IndexQuago turns a pre-built ANN base graph into an enhanced graph.
//
// QUAGO does not build base graphs. Given (1) a base graph loaded from one of
// the supported formats and (2) a k-NN graph (KNNG), it scores every point for
// how much it sits in the hard-to-navigate "QUAGO region" (the thin outer cap of
// the data distribution), selects the top-scoring fraction, and rebuilds/augments
// the edges within that region. The result is written as a delta that search-time
// code overlays onto the base graph.
//
// Pipeline (see apps/quago_build.cpp):
//   LoadBaseGraph + LoadKnng -> quago_enhancement(strategy, method) -> SaveQuagoDelta
// Search (see apps/quago_search.cpp):
//   LoadBaseGraph + LoadQuagoDelta + apply_quago_delta -> Search_normal
class IndexQuago : public Index {
 public:
    IndexQuago(size_t dimension, size_t n, Metric m);
    ~IndexQuago() override = default;

    IndexQuago(const IndexQuago&) = delete;
    IndexQuago& operator=(const IndexQuago&) = delete;

    // Allocate the per-thread visited-list pool used during search. Call once
    // before Search_normal, with the number of search threads.
    void InitVisitedListPool(uint32_t num_threads) {
        visited_list_pool_ = std::make_unique<VisitedListPool>(num_threads, static_cast<int>(nd_));
    }

    // ---- build pipeline ----
    void BuildQuagoGraph(const float* data, uint32_t total_num, uint32_t dim, const Parameters& parameters);

    // Region scoring: three query-agnostic signals fused --
    //   (a) position d_c/r_k, (b) k-neighborhood PCA anisotropy (Gram + power iteration),
    //   (c) frontier hit counts of M spherical-code directions hill-climbed on the base graph.
    // Continuous signals are rank-normalized; the sparse direction count uses a non-saturating
    // exponential activation a_hat = expm1(hits / dir_tau). Negative (-1) fusion weights select a
    // metric-dependent default (see implementation). The position sub-weights are not tunable:
    // they are fixed at d_c 0.3 / r_k 0.7 inside the implementation.
    // Fills quago_scores_ for all nd_ points and records frontier_anchors_ for QMRNG.
    void compute_quago_score_frontier(size_t knn_size, uint32_t M_dirs = 8192,
                                      float w_pos = -1.f, float w_geo = -1.f, float w_dir = -1.f,
                                      float dir_tau = 1.0f);

    // Region selection: top `ratio` fraction by score, or a uniformly random fraction (ablation).
    std::vector<uint32_t> select_quago_region(float top_ratio);
    std::vector<uint32_t> select_quago_region_random(float ratio);

    // Enhancement methods. Each fills quago_delta_ (and enhancement_mode_):
    //   knn   : overlay intra-region KNNG edges (mode 0 = add).
    void enhance_subgraph_with_knn(const std::vector<uint32_t>& node_ids, uint32_t knn_size, uint32_t max_degree);
    //   QMRNG : a region-skeleton rebuild PLUS a direction-frontier MST overlay (mode 1 = replace).
    //           max_degree is the skeleton R; search_L the build-time beam width; alpha the RobustPrune
    //           slack used on the second pass. The overlay reuses the M spherical-code directions the
    //           `frontier` selector already hill-climbed: for each direction's anchor node (its argmax
    //           on the base graph) the "gt20" set is that anchor's precomputed knn_size-NN, a Prim MST
    //           is built inside each set, and the MST edges are unioned on top of the skeleton delta.
    //           Requires quago_strategy=frontier to have run (it populates frontier_anchors_); region
    //           scoring/selection is unchanged.
    void enhance_subgraph_with_qmrng(const std::vector<uint32_t>& node_ids, uint32_t max_degree,
                                     uint32_t search_L, float alpha, uint32_t knn_size);

    // Dispatch region scoring + selection + enhancement. Returns false if disabled/skipped.
    bool quago_enhancement(const std::string& quago_strategy, const std::string& enhance_method,
                           const Parameters& parameters);

    // distance_->compare() has metric-dependent semantics; this converts it to a
    // non-negative squared distance so sqrt() is safe under IP/COSINE (see implementation).
    float squared_distance(const float* a, const float* b) const;

    // ---- search ----
    void SetEntryPoint(uint32_t ep) { entry_point_ = ep; }  // override the loader-provided entry point
    uint32_t GetEntryPoint() const { return entry_point_; }  // read the loader-provided entry point
    void LoadSearchNeededData(const char* base_file);
    // Frontier data structure used by Search_normal (and inherited by the early-stop searches).
    // false (default) = the shipped sorted array (core/neighbor.h); true = the hnswlib-style two-heap
    // frontier (core/heap_queue.h), which drops the O(L) memmove per insertion. The two agree on
    // recall to <1e-3 and on cmps to <0.01% at large L; the heap is up to 1.8x faster at L=20000 and
    // the gap widens with L, while the sorted array is marginally faster below L~1000 where it is
    // L1-resident.
    void SetHeapFrontier(bool v) { use_heap_frontier_ = v; }
    bool heap_frontier() const { return use_heap_frontier_; }
    // Best-first search; returns (hops, distance_comparisons).
    std::pair<uint32_t, uint32_t> Search_normal(float* query, uint32_t& qid, size_t k,
                                                const Parameters& parameters, unsigned* indices);

    // ---- graph IO ----
    // Base-graph loaders. LoadBaseGraph dispatches on format:
    //   "quago"  -> native adjacency format written by SaveQuagoGraph
    //   "nsg"    -> NSG .index format        ([u32 width][u32 ep][per-node deg+neighbors])
    //   "hnsw"   -> hnswlib level-0 binary   (96-byte header + per-element blocks)
    void LoadBaseGraph(const std::string& filepath, const std::string& format);
    void LoadQuagoGraph(const std::string& filepath);
    void LoadNsgGraph(const std::string& filepath);
    void LoadHnswGraph(const std::string& filepath);
    void LoadKnng(const std::string& filepath);
    void SaveQuagoGraph(const std::string& filepath);

    // Enhancement-delta IO and overlay.
    void SaveQuagoDelta(const std::string& filepath);
    void LoadQuagoDelta(const std::string& filepath);
    void apply_quago_delta();

    // ---- diagnostics ----
    void PrintGraphStats();
    std::pair<size_t, size_t> CheckConnectivity();

    bool need_normalize = false;

 protected:
    // Heap-frontier variant of Search_normal; see SetHeapFrontier.
    std::pair<uint32_t, uint32_t> Search_normal_heap(const float* query, size_t k, uint32_t L_pq,
                                                     unsigned* indices);

    bool use_heap_frontier_ = false;

    std::vector<std::vector<uint32_t>> quago_graph_;  // base graph, overlaid in place at search time
    std::vector<std::vector<uint32_t>> knng_indices_;  // loaded KNNG (ids only)
    uint32_t knng_k_ = 0;

    std::vector<float> quago_scores_;                   // per-point region score
    std::vector<std::vector<uint32_t>> quago_delta_;    // per-node enhancement edges

    // Per-direction hill-climb winners recorded by compute_quago_score_frontier (one anchor node id per
    // sampled spherical-code direction). Reused by the QMRNG enhancement so the MST overlay is built from
    // exactly the directions the frontier selector already probed -- no extra direction sampling or search.
    std::vector<uint32_t> frontier_anchors_;

    // Delta application mode: 0 = add (overlay, knn), 1 = replace (qmrng:
    // rebuild region<->region edges but keep each region node's original edges to
    // non-region nodes; see apply_quago_delta). enhancement_mode_ is set by the
    // dispatcher and written to the delta header; delta_mode_ is read back on load.
    int enhancement_mode_ = 0;
    int delta_mode_ = 0;

 private:
    uint32_t entry_point_ = 0;
    uint64_t total_edges_ = 0;
    uint32_t total_points_ = 0;

    std::unique_ptr<VisitedListPool> visited_list_pool_;
};

}  // namespace quago
