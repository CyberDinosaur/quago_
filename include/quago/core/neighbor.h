// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "quago/core/common.h"  // QUAGO_UNLIKELY

namespace quago {

// A single candidate during graph search: a node id, its distance to the query,
// and whether it has already been expanded.
struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    // Ordered by distance, tie-broken by id so the ordering is total and stable.
    bool operator<(const Neighbor& other) const {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }
    bool operator==(const Neighbor& other) const { return id == other.id; }
};

// Fixed-capacity min-ordered candidate set with an "unexpanded" cursor, used as
// the best-first search frontier. Insertions keep the array sorted by distance
// and drop duplicates or anything worse than the current capacity-th element.
class NeighborPriorityQueue {
 public:
    NeighborPriorityQueue() : size_(0), capacity_(0), cur_(0) {}
    explicit NeighborPriorityQueue(size_t capacity)
        : size_(0), capacity_(capacity), cur_(0), data_(capacity + 1) {}

    // Insert nbr in sorted order. Returns the insertion index, or -1 if nbr was
    // dropped (duplicate id, or worse than the capacity-th best when full).
    int32_t insert(const Neighbor& nbr) {
        if (size_ == capacity_ && data_[size_ - 1] < nbr) return -1;

        size_t lo = 0, hi = size_;
        while (lo < hi) {
            const size_t mid = (lo + hi) >> 1;
            if (nbr < data_[mid]) {
                hi = mid;
            } else if (QUAGO_UNLIKELY(data_[mid].id == nbr.id)) {
                return -1;  // duplicate id
            } else {
                lo = mid + 1;
            }
        }

        if (lo < capacity_) {
            std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
        }
        data_[lo] = {nbr.id, nbr.distance, false};

        if (size_ < capacity_) ++size_;
        if (lo < cur_) cur_ = lo;
        return static_cast<int32_t>(lo);
    }

    // Mark the closest unexpanded node as expanded and return it.
    Neighbor closest_unexpanded() {
        data_[cur_].flag = true;
        const size_t pre = cur_;
        while (cur_ < size_ && data_[cur_].flag) ++cur_;
        return data_[pre];
    }

    bool has_unexpanded_node() const { return cur_ < size_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    void reserve(size_t capacity) {
        if (capacity + 1 > data_.size()) data_.resize(capacity + 1);
        capacity_ = capacity;
    }

    Neighbor& operator[](size_t i) { return data_[i]; }
    const Neighbor& operator[](size_t i) const { return data_[i]; }

    void clear() {
        size_ = 0;
        cur_ = 0;
    }
    std::vector<Neighbor>& get_data() { return data_; }

 private:
    size_t size_, capacity_, cur_;
    std::vector<Neighbor> data_;
};

}  // namespace quago
