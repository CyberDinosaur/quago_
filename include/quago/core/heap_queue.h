// Heap-based search frontier, structurally the one hnswlib's searchBaseLayer uses.
//
// Why this exists: the shipped frontier (NeighborPriorityQueue, core/neighbor.h) is a sorted ARRAY
// of capacity L, so every accepted insertion memmove's its tail. Tail length is set by the capacity
// rather than by the work actually done, so total bytes moved scale as L^2 while distance
// computations scale as L -- at L=20000 that memmove is 54% of fixed-L search time and 89% of
// early-stopped search time. Splitting the frontier into two binary heaps removes the tail shift
// entirely: insertion becomes O(log L) sifts of 8 bytes. Measured on clip_webvid2.5M that is 1.78x
// faster at L=20000 at identical recall, moving 7.3 MB per query instead of 3255 MB.
//
// The two heaps play the two roles the sorted array played with a single structure:
//   top_  -- MAX-heap of the L best, so the worst (the eviction victim and the `lower_bound`) is
//            O(1). This is hnswlib's `top_candidates`.
//   cand_ -- MIN-heap of unexpanded candidates, so the closest is O(1). This is hnswlib's
//            `candidate_set`; eviction from top_ is handled lazily by re-testing against
//            `lower_bound` at pop time rather than by removing the stale entry.
//
// A max-heap exposes only the WORST element in O(1), but the early-stop rules need the k-th best on
// every hop (`d_k` in quago/early_stop.h) and the read-out needs the first k in order. So this also
// keeps `topk_`, a sorted array of just the first k entries: at most k elements, at most 8*k bytes
// shifted, and only when the top-k actually changes, so it stays permanently L1-resident. That
// restores the O(1) random access and the O(k) result read-out the sorted array gave for free, which
// a bare pair of heaps does not.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace quago {

// Binary heap over (distance, id). Ordering is total and stable: distance first, id as tie-break,
// matching Neighbor::operator< so results agree with the sorted-array frontier.
template <bool kMinOnTop>
class BinaryHeap {
 public:
    using Item = std::pair<float, uint32_t>;

    void reserve(size_t n) { a_.reserve(n); }
    void clear() { a_.clear(); }
    size_t size() const { return a_.size(); }
    bool empty() const { return a_.empty(); }
    const Item& top() const { return a_.front(); }

    void push(const Item& v) {
        a_.push_back(v);
        size_t i = a_.size() - 1;
        while (i > 0) {
            const size_t parent = (i - 1) >> 1;
            if (!better(a_[i], a_[parent])) break;
            std::swap(a_[i], a_[parent]);
            i = parent;
        }
    }

    void pop() {
        a_.front() = a_.back();
        a_.pop_back();
        size_t i = 0;
        const size_t n = a_.size();
        for (;;) {
            const size_t l = 2 * i + 1, r = l + 1;
            size_t best = i;
            if (l < n && better(a_[l], a_[best])) best = l;
            if (r < n && better(a_[r], a_[best])) best = r;
            if (best == i) break;
            std::swap(a_[i], a_[best]);
            i = best;
        }
    }

 private:
    static bool better(const Item& x, const Item& y) { return kMinOnTop ? (x < y) : (y < x); }
    std::vector<Item> a_;
};

class HeapFrontier {
 public:
    using Item = std::pair<float, uint32_t>;

    // `capacity` is L (the budget / number of retained results); `k` is the result count whose best
    // and k-th distance the stopping rules read every hop. Requires k <= capacity.
    HeapFrontier(size_t capacity, size_t k)
        : cap_(capacity), k_(k), lower_bound_(std::numeric_limits<float>::infinity()), topk_(k) {
        top_.reserve(capacity + 1);
        cand_.reserve(capacity + 1);
    }

    // First (entry-point) candidate: always retained, so no capacity test.
    void seed(uint32_t id, float dist) {
        top_.push({dist, id});
        cand_.push({dist, id});
        lower_bound_ = dist;
        topk_insert(id, dist);
    }

    // ---- traversal ----
    bool empty() const { return cand_.empty(); }
    float next_dist() const { return cand_.top().first; }
    uint32_t next_id() const { return cand_.top().second; }
    void pop_candidate() { cand_.pop(); }

    // Budget exhausted: the closest remaining candidate has already been evicted from the top-L, so
    // nothing retained is left to expand. This is the heap analogue of the sorted array running out
    // of unflagged entries.
    bool budget_spent() const { return top_.size() >= cap_ && cand_.top().first > lower_bound_; }

    // A popped candidate is worth expanding only if it is still inside the top-L; stale entries
    // (evicted after being pushed) are skipped, which is how the min-heap avoids deletion.
    bool expandable(float dist) const { return top_.size() < cap_ || dist < lower_bound_; }

    // ---- insertion ----
    // Retain `id` if it belongs in the top-L. Returns its position within the top-k, or -1 if it did
    // not enter the top-k (which includes being rejected from the top-L entirely). The return value
    // mirrors NeighborPriorityQueue::insert's "did the top-k change" contract.
    int32_t push(uint32_t id, float dist) {
        if (top_.size() >= cap_ && !(dist < lower_bound_)) return -1;
        top_.push({dist, id});
        cand_.push({dist, id});
        if (top_.size() > cap_) top_.pop();
        lower_bound_ = top_.top().first;
        return topk_insert(id, dist);
    }

    // ---- accessors ----
    size_t size() const { return top_.size(); }  // retained count, == NeighborPriorityQueue::size()
    float lower_bound() const { return lower_bound_; }
    float kth_dist() const { return topk_[k_ - 1].first; }
    uint32_t topk_id(size_t i) const { return topk_[i].second; }

 private:
    // Insert into the sorted top-k mini-array. Capacity k_, so at most k_-1 elements shift (8*k_
    // bytes, L1-resident) and only when the new candidate actually belongs in the top-k.
    // An entry evicted from top_ can never be in the top-k (eviction removes the (cap_+1)-th best
    // and k_ <= cap_), so this array never goes stale.
    int32_t topk_insert(uint32_t id, float dist) {
        const size_t n = ntopk_;
        if (n >= k_) {
            const Item& worst = topk_[k_ - 1];
            const bool better_than_worst =
                dist < worst.first || (dist == worst.first && id < worst.second);
            if (!better_than_worst) return -1;
        }
        size_t lo = 0, hi = n < k_ ? n : k_;
        while (lo < hi) {
            const size_t mid = (lo + hi) >> 1;
            const Item& m = topk_[mid];
            if (m.first < dist || (m.first == dist && m.second < id)) lo = mid + 1;
            else hi = mid;
        }
        const size_t end = (n < k_) ? n : k_ - 1;
        // Shift [lo, end) right by one. copy_backward handles the overlapping range for a
        // non-trivially-copyable element type (memmove would trip -Wclass-memaccess).
        if (lo < end) std::copy_backward(topk_.begin() + lo, topk_.begin() + end, topk_.begin() + end + 1);
        topk_[lo] = {dist, id};
        if (n < k_) ++ntopk_;
        return static_cast<int32_t>(lo);
    }

    size_t cap_, k_, ntopk_ = 0;
    float lower_bound_;
    BinaryHeap<false> top_;   // worst of the L best on top
    BinaryHeap<true> cand_;   // closest unexpanded on top
    std::vector<Item> topk_;  // the first k, sorted ascending
};

}  // namespace quago
