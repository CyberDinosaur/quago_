#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "quago/core/heap_queue.h"
#include "quago/core/neighbor.h"
#include "quago/core/visited_list_pool.h"
#include "quago/index_quago.h"

namespace quago {

// Outcome of one early-stopped search.
struct EarlyStopStats {
    uint32_t hops = 0;   // nodes expanded
    uint32_t cmps = 0;   // distance computations
    bool early = false;  // terminated by the stopping rule (false => exhausted the L_max budget)
};

class EarlyStopIndex : public IndexQuago {
 public:
    using IndexQuago::IndexQuago;

    // Allocate the per-thread visited-list pool used by the early-stop searches. Call once
    // before searching, with the number of search threads.
    void InitEarlyStopPool(uint32_t num_threads) {
        es_pool_ = std::make_unique<VisitedListPool>(static_cast<int>(num_threads), static_cast<int>(nd_));
    }

    // compare() semantics depend on the metric; record whether distances are negated-dot so the
    // squared-distance transform below is correct.
    void SetInnerProductMetric(bool v) { metric_ip_ = v; }

    // FAST always runs on the two-heap frontier (core/heap_queue.h); this budget is where its
    // STOPPING RULE changes. Below it a stall is not yet evidence of convergence -- the query has
    // simply not explored enough for the top-k to settle -- so the geometric test decides alone. At
    // or above it the temporal half joins, and both must agree before the query quits.
    static constexpr uint32_t kFastSwitchL = 3000;

    // The frontier rule takes its data structure from IndexQuago::SetHeapFrontier; FAST always uses
    // the heap and ignores that setting.

    // Stop when the best still-unexpanded candidate exceeds theta * (k-th result distance).
    // theta is a dimensionless ratio; theta=1 is the classic greedy kNN boundary (too aggressive
    // on graphs), useful range ~1.04-1.15.
    EarlyStopStats SearchFrontier(const float* query, size_t k, uint32_t L_max, float theta,
                                  uint32_t min_hops, unsigned* indices) {
        return SearchCore(query, k, L_max, /*fast=*/false, theta, 0, min_hops, indices);
    }

    // FAST: the same geometric test as SearchFrontier, plus a patience floor that joins it at
    // L >= kFastSwitchL. So FAST == frontier on the heap, except that a large budget must ALSO see
    // the top-k stall before the query quits.
    EarlyStopStats SearchFast(const float* query, size_t k, uint32_t L_max, float theta,
                              uint32_t patience, uint32_t min_hops, unsigned* indices) {
        return SearchCore(query, k, L_max, /*fast=*/true, theta, patience, min_hops, indices);
    }

 private:
    // squared-distance transform of a compare() value (monotone, non-negative).
    inline float to_d2(float cmp) const {
        if (metric_ip_) {
            const float s = 2.0f + 2.0f * cmp;
            return s > 0.f ? s : 0.f;
        }
        return cmp;
    }

    // Dispatch on the frontier the rule runs on. FAST is always on the heap; the frontier rule
    // follows SetHeapFrontier. `use_patience` is the only thing separating the two rules' stopping
    // conditions, and it is never true off the heap, so the sorted core does not take it.
    EarlyStopStats SearchCore(const float* query, size_t k, uint32_t L_max, bool fast, float theta,
                              uint32_t patience, uint32_t min_hops, unsigned* indices) {
        if (fast || use_heap_frontier_) {
            const bool use_patience = fast && L_max >= kFastSwitchL;
            return SearchCoreHeap(query, k, L_max, theta, patience, min_hops, use_patience, indices);
        }
        return SearchCoreSorted(query, k, L_max, theta, min_hops, indices);
    }

    // Sorted-array frontier: the shipped structure (core/neighbor.h). Only the frontier rule reaches
    // it, since FAST is always on the heap.
    EarlyStopStats SearchCoreSorted(const float* query, size_t k, uint32_t L_max, float theta,
                                    uint32_t min_hops, unsigned* indices) {
        VisitedList* vl = es_pool_->getFreeVisitedList();
        vl_type* visited = vl->mass;
        const vl_type tag = vl->curV;

        const uint32_t entry = GetEntryPoint();
        NeighborPriorityQueue q(L_max);
        prefetch_vector(reinterpret_cast<const char*>(data_ + static_cast<size_t>(entry) * dimension_),
                        dimension_);
        q.insert(Neighbor(entry,
                          distance_->compare(data_ + static_cast<size_t>(entry) * dimension_, query,
                                             static_cast<unsigned>(dimension_)),
                          false));

        EarlyStopStats st;
        st.cmps = 1;  // the entry-point distance
        bool have_k = (q.size() >= k);

        while (q.has_unexpanded_node()) {
            // Pop the best unexpanded node first: its distance is the current search frontier, which
            // the geometric rule is expressed in terms of.
            const Neighbor cur = q.closest_unexpanded();

            if (have_k && st.hops >= min_hops &&
                to_d2(cur.distance) > theta * to_d2(q[k - 1].distance)) {
                st.early = true;
                break;
            }

            // Expand `cur`.
            const std::vector<uint32_t>& nbrs = quago_graph_[cur.id];
            const uint32_t* cur_nbrs = nbrs.data();
            const size_t deg = nbrs.size();
            ++st.hops;

            for (size_t j = 0; j < deg; ++j) {
                const uint32_t nbr = cur_nbrs[j];
                if (j + 1 < deg) {
                    _mm_prefetch(reinterpret_cast<const char*>(visited + cur_nbrs[j + 1]), _MM_HINT_T0);
                    _mm_prefetch(
                        reinterpret_cast<const char*>(data_ + static_cast<size_t>(cur_nbrs[j + 1]) * dimension_),
                        _MM_HINT_T0);
                }
                if (visited[nbr] != tag) {
                    visited[nbr] = tag;
                    const float distance = distance_->compare(
                        data_ + static_cast<size_t>(nbr) * dimension_, query, static_cast<unsigned>(dimension_));
                    ++st.cmps;
                    q.insert({nbr, distance, false});
                }
            }
            if (!have_k && q.size() >= k) have_k = true;
        }
        es_pool_->releaseVisitedList(vl);

        for (size_t i = 0; i < k; ++i) indices[i] = q[i].id;
        return st;
    }

    // Heap frontier (core/heap_queue.h). Same hop/cmps accounting as the sorted-array core; the
    // candidate set is two binary heaps instead of one sorted array, so an insertion sifts O(log L)
    // 8-byte elements rather than memmove'ing an O(L) 12-byte tail. FAST always lands here; the
    // frontier rule lands here when SetHeapFrontier(true).
    EarlyStopStats SearchCoreHeap(const float* query, size_t k, uint32_t L_max, float theta,
                                  uint32_t patience, uint32_t min_hops, bool use_patience,
                                  unsigned* indices) {
        VisitedList* vl = es_pool_->getFreeVisitedList();
        vl_type* visited = vl->mass;
        const vl_type tag = vl->curV;

        const uint32_t entry = GetEntryPoint();
        HeapFrontier f(L_max, k);
        prefetch_vector(reinterpret_cast<const char*>(data_ + static_cast<size_t>(entry) * dimension_),
                        dimension_);
        f.seed(entry, distance_->compare(data_ + static_cast<size_t>(entry) * dimension_, query,
                                         static_cast<unsigned>(dimension_)));

        EarlyStopStats st;
        st.cmps = 1;  // the entry-point distance
        uint32_t last_improve_hop = 0;
        bool have_k = (f.size() >= k);

        while (!f.empty()) {
            // The closest unexpanded candidate: its distance is the current search frontier.
            const float cur_dist = f.next_dist();
            const uint32_t cur_id = f.next_id();
            if (f.budget_spent()) break;  // nothing retained is left to expand

            if (have_k && st.hops >= min_hops &&
                to_d2(cur_dist) > theta * to_d2(f.kth_dist()) &&
                (!use_patience || st.hops - last_improve_hop >= patience)) {
                st.early = true;
                break;
            }

            f.pop_candidate();
            if (!f.expandable(cur_dist)) continue;  // evicted from the top-L after being queued

            const std::vector<uint32_t>& nbrs = quago_graph_[cur_id];
            const uint32_t* cur_nbrs = nbrs.data();
            const size_t deg = nbrs.size();
            ++st.hops;

            bool topk_changed = false;
            for (size_t j = 0; j < deg; ++j) {
                const uint32_t nbr = cur_nbrs[j];
                if (j + 1 < deg) {
                    _mm_prefetch(reinterpret_cast<const char*>(visited + cur_nbrs[j + 1]), _MM_HINT_T0);
                    _mm_prefetch(
                        reinterpret_cast<const char*>(data_ + static_cast<size_t>(cur_nbrs[j + 1]) * dimension_),
                        _MM_HINT_T0);
                }
                if (visited[nbr] != tag) {
                    visited[nbr] = tag;
                    const float distance = distance_->compare(
                        data_ + static_cast<size_t>(nbr) * dimension_, query, static_cast<unsigned>(dimension_));
                    ++st.cmps;
                    const int32_t pos = f.push(nbr, distance);
                    if (pos >= 0) topk_changed = true;  // HeapFrontier::push already caps at k
                }
            }
            if (!have_k && f.size() >= k) { have_k = true; topk_changed = true; }
            if (topk_changed) last_improve_hop = st.hops;
        }
        es_pool_->releaseVisitedList(vl);

        for (size_t i = 0; i < k; ++i) indices[i] = f.topk_id(i);
        return st;
    }

    bool metric_ip_ = true;
    std::unique_ptr<VisitedListPool> es_pool_;
};

}  // namespace quago
