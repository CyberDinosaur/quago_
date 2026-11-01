// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
#include "quago/core/index.h"

#include <iostream>
#include <memory>

namespace quago {

Index::Index(const size_t dimension, const size_t n, Metric metric)
    : dimension_(dimension), data_(nullptr), nd_(n), has_built_(false), metric_(metric) {
    switch (metric_) {
        case COSINE:
        case INNER_PRODUCT:
            distance_ = std::make_unique<DistanceInnerProduct>();
            std::cout << "[Info] Using inner-product distance (data normalized)." << std::endl;
            break;
        case L2:
        case FAST_L2:
        default:
            distance_ = std::make_unique<DistanceL2>();
            std::cout << "[Info] Using L2 distance." << std::endl;
            break;
    }
}

}  // namespace quago
