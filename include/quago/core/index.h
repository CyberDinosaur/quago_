// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
#pragma once

#include <cstddef>
#include <memory>

#include "quago/core/distance.h"
#include "quago/core/parameters.h"

namespace quago {

// Thin concrete base holding the state every graph index needs: dataset
// dimensions, a non-owning pointer to the row-major vector data, the metric,
// and the distance functor (owned via unique_ptr).
//
// `data_` is non-owning: the calling application allocates and frees the base
// vectors. IndexQuago may internally point `data_` at an aligned copy it loads
// for search; that buffer lives for the lifetime of the process.
class Index {
 public:
  Index(size_t dimension, size_t n, Metric metric);
  virtual ~Index() = default;

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;

  bool HasBuilt() const { return has_built_; }
  size_t GetDimension() const { return dimension_; }
  size_t GetSizeOfDataset() const { return nd_; }
  const float* GetDataset() const { return data_; }
  Metric GetMetric() const { return metric_; }

 protected:
  // Must be 64-bit: the whole project locates vectors with `data_ + id * dimension_`
  // where id is uint32_t. If dimension_ were 32-bit, the multiply would wrap once
  // n*dim >= 2^32 (e.g. 10,004,480 x 512 = 5.12e9), silently reading the wrong
  // vector for ~16% of points and capping recall far below its true value.
  const size_t dimension_;
  const float* data_;
  size_t nd_;
  bool has_built_;
  Metric metric_;
  std::unique_ptr<Distance> distance_;
};

}  // namespace quago
