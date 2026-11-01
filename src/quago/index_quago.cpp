#include "quago/index_quago.h"

#include <omp.h>

#include <chrono>
#include <sstream>
#include <stdexcept>

#include "quago/core/common.h"
#include "quago/core/file_io.h"
#include "quago/core/heap_queue.h"
#include "quago/core/parameters.h"
#include "quago/core/util.h"

namespace quago {

namespace {
// Extra per-node capacity reserved on the base graph so the enhancement delta can
// be overlaid without repeated reallocation.
constexpr uint32_t kProjectionSlack = 2;
}  // namespace

IndexQuago::IndexQuago(const size_t dimension, const size_t n, Metric m) : Index(dimension, n, m) {
    if (m == COSINE || m == INNER_PRODUCT) need_normalize = true;
}

/********************************************************************************************
 * Build pipeline
 ********************************************************************************************/
void IndexQuago::BuildQuagoGraph(const float* data, uint32_t total_num, uint32_t /*dim*/,
                                 const Parameters& parameters) {
    const std::string save_dir = parameters.Get<std::string>("quago_index_save_dir");
    data_ = data;
    total_points_ = total_num;
    std::cout << "################ Start building QUAGO enhanced index ################" << std::endl;

    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();

    // Base and query vectors must be normalized for IP/COSINE so that ||a-b||^2 = 2 - 2*dot.
    if (need_normalize) {
        std::cout << "Normalizing base data" << std::endl;
        float* mutable_data = const_cast<float*>(data_);
        for (size_t i = 0; i < nd_; ++i) normalize(mutable_data + i * dimension_, dimension_);
    }

    // Load the externally-built base graph and the KNNG that region scoring consumes.
    const std::string base_index_file = parameters.Get<std::string>("base_index_file");
    const std::string base_index_format = parameters.Get<std::string>("base_index_format");
    const std::string knng_path = parameters.Get<std::string>("knng_path");
    LoadBaseGraph(base_index_file, base_index_format);
    LoadKnng(knng_path);

    // Reserve overlay headroom on the loaded adjacency lists.
    const uint32_t M_nhood = parameters.Get<uint32_t>("M_nhood");
    for (auto& adj : quago_graph_) adj.reserve(M_nhood * kProjectionSlack);

    // Score, select and enhance the QUAGO region, then persist the delta.
    const std::string strategy = parameters.Get<std::string>("quago_strategy");    // frontier/random/none
    const std::string method = parameters.Get<std::string>("enhance_method");      // knn/qmrng
    if (quago_enhancement(strategy, method, parameters)) {
        std::cout << "[Completed] QUAGO enhancement done." << std::endl;
        std::ostringstream ratio_tag;
        ratio_tag << std::fixed << std::setprecision(2) << parameters.Get<float>("quago_ratio");
        const std::string delta_path =
            join_path({save_dir, strategy + "_" + method + "_r" + ratio_tag.str() + ".delta"});
        SaveQuagoDelta(delta_path);
    } else {
        std::cout << "[Skip] QUAGO enhancement skipped." << std::endl;
    }

    has_built_ = true;
    const std::chrono::duration<double> dt = Clock::now() - t0;
    std::cout << "Total time of building QUAGO graph (s): " << dt.count() << std::endl;
}

/********************************************************************************************
 * Distance helper
 * ------------------------------------------------------------------------------------------
 * distance_->compare() has metric-dependent semantics:
 *   L2 / FAST_L2 -> squared Euclidean distance, non-negative.
 *   IP  / COSINE -> -dot, negative for similar vectors.
 * So compare() must never be sqrt()'d directly: under IP it would yield NaN, which then
 * poisons quago_scores_ through min-max normalization and makes select_quago_region's
 * quantile threshold NaN. Because `score >= NaN` is false, that would silently select the
 * EMPTY region (defeating quago_ratio); select_quago_region rejects a NaN threshold explicitly.
 *
 * Under IP/COSINE need_normalize is true, so both operands are unit vectors and
 * ||a-b||^2 = 2 - 2*dot = 2 + 2*compare(). If b is not a unit vector (e.g. a centroid) the
 * expression differs from the true squared distance by a constant only; it is still
 * monotonic in dot, so rankings are unchanged.
 ********************************************************************************************/
float IndexQuago::squared_distance(const float* a, const float* b) const {
    const float c = distance_->compare(a, b, static_cast<unsigned>(dimension_));
    if (metric_ == INNER_PRODUCT || metric_ == COSINE) {
        const float sq = 2.0f + 2.0f * c;
        return sq > 0.0f ? sq : 0.0f;
    }
    return c;
}

/********************************************************************************************
 * Search
 ********************************************************************************************/
void IndexQuago::LoadSearchNeededData(const char* base_file) {
    uint32_t base_num = 0, base_dim = 0;
    load_meta<float>(base_file, base_num, base_dim);

    float* base_data = nullptr;
    load_data_aligned<float>(base_file, base_num, base_dim, base_data);

    if (need_normalize) {
        std::cout << "Normalizing base data" << std::endl;
        for (size_t i = 0; i < base_num; ++i) {
            normalize<float>(base_data + i * static_cast<uint64_t>(base_dim), static_cast<uint64_t>(base_dim));
        }
    }
    // data_align frees base_data and returns an aligned copy; data_ points at it for the
    // lifetime of the process.
    data_ = data_align(base_data, base_num, base_dim);
}

std::pair<uint32_t, uint32_t> IndexQuago::Search_normal(float* query, uint32_t& /*qid*/, size_t k,
                                                        const Parameters& parameters, unsigned* indices) {
    const uint32_t L_pq = parameters.Get<uint32_t>("L_search");
    if (use_heap_frontier_) return Search_normal_heap(query, k, L_pq, indices);
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    const vl_type visited_array_tag = vl->curV;

    NeighborPriorityQueue search_queue(L_pq);
    prefetch_vector(reinterpret_cast<const char*>(data_ + entry_point_ * dimension_), dimension_);
    search_queue.insert(Neighbor(entry_point_,
                                 distance_->compare(data_ + entry_point_ * dimension_, query,
                                                    static_cast<unsigned>(dimension_)),
                                 false));

    uint32_t cmps = 0;
    uint32_t hops = 0;
    while (search_queue.has_unexpanded_node()) {
        const auto cur_check_node = search_queue.closest_unexpanded();
        const uint32_t cur_id = cur_check_node.id;
        const std::vector<uint32_t>& nbrs = quago_graph_[cur_id];
        const uint32_t* cur_nbrs = nbrs.data();
        const size_t deg = nbrs.size();

        ++hops;
        for (size_t j = 0; j < deg; ++j) {
            const uint32_t nbr = cur_nbrs[j];
            // Prefetch the next neighbor's visited-flag and vector while computing this one.
            if (j + 1 < deg) {
                _mm_prefetch(reinterpret_cast<const char*>(visited_array + cur_nbrs[j + 1]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(data_ + cur_nbrs[j + 1] * dimension_), _MM_HINT_T0);
            }
            if (visited_array[nbr] != visited_array_tag) {
                visited_array[nbr] = visited_array_tag;
                const float distance =
                    distance_->compare(data_ + nbr * dimension_, query, static_cast<unsigned>(dimension_));
                ++cmps;
                search_queue.insert({nbr, distance, false});
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (QUAGO_UNLIKELY(search_queue.size() < k)) {
        throw std::runtime_error("not enough results: " + std::to_string(search_queue.size()) +
                                 ", expected: " + std::to_string(k));
    }
    for (size_t i = 0; i < k; ++i) indices[i] = search_queue[i].id;
    return {hops, cmps};
}

// Heap-frontier variant of Search_normal. Same traversal and same accounting (hops = expansions,
// cmps = distance computations including the entry point); only the candidate-set representation
// differs, so the O(L) memmove per accepted insertion becomes O(log L) sifts of 8 bytes.
std::pair<uint32_t, uint32_t> IndexQuago::Search_normal_heap(const float* query, size_t k,
                                                             uint32_t L_pq, unsigned* indices) {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    const vl_type visited_array_tag = vl->curV;

    HeapFrontier frontier(L_pq, k);
    const uint32_t entry = entry_point_;
    prefetch_vector(reinterpret_cast<const char*>(data_ + static_cast<size_t>(entry) * dimension_),
                    dimension_);
    frontier.seed(entry, distance_->compare(data_ + static_cast<size_t>(entry) * dimension_, query,
                                            static_cast<unsigned>(dimension_)));
    visited_array[entry] = visited_array_tag;

    uint32_t cmps = 1, hops = 0;
    while (!frontier.empty()) {
        if (frontier.budget_spent()) break;
        const float cur_dist = frontier.next_dist();
        const uint32_t cur_id = frontier.next_id();
        frontier.pop_candidate();
        if (!frontier.expandable(cur_dist)) continue;  // evicted from the top-L after being queued

        const std::vector<uint32_t>& nbrs = quago_graph_[cur_id];
        const uint32_t* cur_nbrs = nbrs.data();
        const size_t deg = nbrs.size();
        ++hops;
        for (size_t j = 0; j < deg; ++j) {
            const uint32_t nbr = cur_nbrs[j];
            if (j + 1 < deg) {
                _mm_prefetch(reinterpret_cast<const char*>(visited_array + cur_nbrs[j + 1]), _MM_HINT_T0);
                _mm_prefetch(
                    reinterpret_cast<const char*>(data_ + static_cast<size_t>(cur_nbrs[j + 1]) * dimension_),
                    _MM_HINT_T0);
            }
            if (visited_array[nbr] != visited_array_tag) {
                visited_array[nbr] = visited_array_tag;
                const float distance = distance_->compare(
                    data_ + static_cast<size_t>(nbr) * dimension_, query, static_cast<unsigned>(dimension_));
                ++cmps;
                frontier.push(nbr, distance);
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (QUAGO_UNLIKELY(frontier.size() < k)) {
        throw std::runtime_error("not enough results: " + std::to_string(frontier.size()) +
                                 ", expected: " + std::to_string(k));
    }
    for (size_t i = 0; i < k; ++i) indices[i] = frontier.topk_id(i);
    return {hops, cmps};
}

/********************************************************************************************
 * Native QUAGO graph format
 * ------------------------------------------------------------------------------------------
 * A compact adjacency-list layout that also records the search entry point:
 *   [u64 total_edges][u32 total_points][u32 entry_point]
 *   then, for each node i in [0, total_points): [u32 degree][degree * u32 neighbor_ids]
 ********************************************************************************************/
void IndexQuago::SaveQuagoGraph(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Failed to open QUAGO index file: " + filepath);

    total_edges_ = 0;
    for (const auto& adj : quago_graph_) total_edges_ += adj.size();
    total_points_ = static_cast<uint32_t>(quago_graph_.size());

    out.write(reinterpret_cast<const char*>(&total_edges_), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&total_points_), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(uint32_t));
    for (uint32_t i = 0; i < total_points_; ++i) {
        const uint32_t deg = static_cast<uint32_t>(quago_graph_[i].size());
        out.write(reinterpret_cast<const char*>(&deg), sizeof(uint32_t));
        if (deg > 0) out.write(reinterpret_cast<const char*>(quago_graph_[i].data()), deg * sizeof(uint32_t));
    }
    if (!out) throw std::runtime_error("[Save] Write error: " + filepath);
    std::cout << "[Save] QUAGO graph -> " << filepath << " (points=" << total_points_
              << ", edges=" << total_edges_ << ", entry=" << entry_point_ << ")" << std::endl;
}

void IndexQuago::LoadQuagoGraph(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open QUAGO index file: " + filepath);

    in.read(reinterpret_cast<char*>(&total_edges_), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&total_points_), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&entry_point_), sizeof(uint32_t));
    if (!in) throw std::runtime_error("[Load] Failed to read QUAGO graph header: " + filepath);

    quago_graph_.assign(total_points_, {});
    for (uint32_t i = 0; i < total_points_; ++i) {
        uint32_t deg = 0;
        in.read(reinterpret_cast<char*>(&deg), sizeof(uint32_t));
        quago_graph_[i].resize(deg);
        if (deg > 0) in.read(reinterpret_cast<char*>(quago_graph_[i].data()), deg * sizeof(uint32_t));
        if (!in) throw std::runtime_error("[Load] Unexpected EOF at node " + std::to_string(i));
    }
    std::cout << "[Load] QUAGO graph: points=" << total_points_ << ", edges=" << total_edges_
              << ", entry=" << entry_point_ << " from " << filepath << std::endl;
}

/********************************************************************************************
 * NSG .index format: [u32 width][u32 entry_point] then per-node [u32 deg][deg * u32].
 ********************************************************************************************/
void IndexQuago::LoadNsgGraph(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open NSG index file: " + filepath);

    uint32_t width = 0, ep = 0;
    in.read(reinterpret_cast<char*>(&width), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&ep), sizeof(uint32_t));
    entry_point_ = ep;

    quago_graph_.clear();
    total_edges_ = 0;
    while (in) {
        uint32_t deg = 0;
        in.read(reinterpret_cast<char*>(&deg), sizeof(uint32_t));
        if (!in) break;
        std::vector<uint32_t> neighbors(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(neighbors.data()), deg * sizeof(uint32_t));
            if (!in) break;
        }
        total_edges_ += deg;
        quago_graph_.push_back(std::move(neighbors));
    }
    total_points_ = static_cast<uint32_t>(quago_graph_.size());
    std::cout << "[Load] NSG graph: width=" << width << ", points=" << total_points_
              << ", edges=" << total_edges_ << ", entry=" << entry_point_ << " from " << filepath << std::endl;
}

/********************************************************************************************
 * hnswlib binary format (level-0 only). Standalone parser of hnswlib's on-disk layout; does
 * not depend on the hnswlib library.
 *   Header (96B): offsetLevel0, max_elements, cur_element_count, size_data_per_element,
 *     label_offset, offsetData, maxlevel(i32), enterpoint_node(u32), maxM, maxM0, M,
 *     mult(f64), ef_construction.
 *   Level-0 block: N * size_data_per_element; each element is
 *     [linklistsizeint count][maxM0 * u32 tableint slots][vector data][labeltype label].
 * Level-0 links store hnswlib INTERNAL ids (tableint), not base-row ids. Element i is stored
 * in tableint order (tableint == i) and carries its external label at +label_offset. We read
 * every label to build a tableint -> label map, then remap all neighbor slots and the entry
 * point into base-row space. This makes the loader correct regardless of insertion order
 * (labels are assumed to be base-row ids, the standard hnswlib usage).
 ********************************************************************************************/
void IndexQuago::LoadHnswGraph(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Failed to open HNSW index file: " + filepath);

    size_t offsetLevel0 = 0, max_elements = 0, cur_element_count = 0, size_data_per_element = 0;
    size_t label_offset = 0, offsetData = 0, maxM = 0, maxM0 = 0, M = 0, ef_construction = 0;
    int maxlevel = 0;
    uint32_t enterpoint_node = 0;
    double mult = 0.0;

    in.read(reinterpret_cast<char*>(&offsetLevel0), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&max_elements), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cur_element_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&size_data_per_element), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&label_offset), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&offsetData), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&maxlevel), sizeof(int));
    in.read(reinterpret_cast<char*>(&enterpoint_node), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&maxM), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&maxM0), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&M), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&mult), sizeof(double));
    in.read(reinterpret_cast<char*>(&ef_construction), sizeof(size_t));

    const uint32_t N = static_cast<uint32_t>(cur_element_count);
    total_points_ = N;
    // linklistsizeint width = size_links_level0 - maxM0 * sizeof(tableint); 4 in current hnswlib.
    const size_t count_size = (offsetData > maxM0 * sizeof(uint32_t))
                                  ? (offsetData - maxM0 * sizeof(uint32_t)) : sizeof(uint32_t);

    // Pass 1: tableint (== element index) -> external label (base-row id).
    std::vector<uint32_t> ti2label(N);
    for (uint32_t i = 0; i < N; ++i) {
        in.seekg(static_cast<std::streamoff>(96 + static_cast<size_t>(i) * size_data_per_element + label_offset),
                 std::ios::beg);
        uint64_t lab = 0;
        in.read(reinterpret_cast<char*>(&lab), sizeof(uint64_t));
        if (!in) throw std::runtime_error("[Load] HNSW: EOF reading label of element " + std::to_string(i));
        if (lab >= N) {
            throw std::runtime_error("[Load] HNSW: label " + std::to_string(lab) +
                                     " out of range [0," + std::to_string(N) +
                                     "); labels are not base-row ids, cannot remap.");
        }
        ti2label[i] = static_cast<uint32_t>(lab);
    }
    if (enterpoint_node < N) entry_point_ = ti2label[enterpoint_node];

    // Pass 2: read level-0 links (tableint slots) and remap them into base-row space.
    quago_graph_.assign(N, {});
    total_edges_ = 0;
    std::vector<uint32_t> slots;
    for (uint32_t i = 0; i < N; ++i) {
        in.seekg(static_cast<std::streamoff>(96 + static_cast<size_t>(i) * size_data_per_element), std::ios::beg);
        uint32_t count = 0;
        if (count_size == sizeof(uint16_t)) {
            uint16_t c16 = 0; in.read(reinterpret_cast<char*>(&c16), sizeof(uint16_t)); count = c16;
        } else {
            in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        }
        if (count > maxM0) count = static_cast<uint32_t>(maxM0);  // sanity clamp
        slots.resize(count);
        if (count > 0) in.read(reinterpret_cast<char*>(slots.data()), count * sizeof(uint32_t));
        if (!in) throw std::runtime_error("[Load] HNSW: EOF reading links of element " + std::to_string(i));

        auto& adj = quago_graph_[ti2label[i]];
        adj.reserve(count);
        for (uint32_t t : slots) {
            if (t < N) { adj.push_back(ti2label[t]); ++total_edges_; }
        }
    }
    std::cout << "[Load] HNSW graph (level-0, label-remapped): points=" << total_points_
              << ", edges=" << total_edges_ << ", maxM0=" << maxM0 << ", entry=" << entry_point_
              << " from " << filepath << std::endl;
}

void IndexQuago::LoadBaseGraph(const std::string& filepath, const std::string& format) {
    if (format == "quago") {
        LoadQuagoGraph(filepath);
    } else if (format == "nsg") {
        LoadNsgGraph(filepath);
    } else if (format == "hnsw") {
        LoadHnswGraph(filepath);
    } else {
        throw std::runtime_error("Unknown base index format: '" + format +
                                 "'. Supported: quago, nsg, hnsw");
    }
}

/********************************************************************************************
 * KNNG format (NSG-style, ids only):
 *   Header:   [u32 k]
 *   Per node: [u32 k_dummy][k * u32 neighbor_ids]   (no distances)
 ********************************************************************************************/
void IndexQuago::LoadKnng(const std::string& filepath) {
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin.is_open()) throw std::runtime_error("[Load] Cannot open KNNG file: " + filepath);

    fin.seekg(0, std::ios::end);
    const std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    uint32_t K = 0;
    fin.read(reinterpret_cast<char*>(&K), sizeof(uint32_t));

    // file_size = 4 + N * (K + 1) * 4  =>  N = (file_size - 4) / ((K + 1) * 4)
    const std::streamsize per_node = static_cast<std::streamsize>(K + 1) * sizeof(uint32_t);
    const std::streamsize data_size = file_size - static_cast<std::streamsize>(sizeof(uint32_t));
    if (per_node == 0 || data_size % per_node != 0) {
        throw std::runtime_error("[Load] KNNG file size mismatch, cannot determine N.");
    }
    const size_t N = static_cast<size_t>(data_size / per_node);
    if (N != total_points_) {
        throw std::runtime_error("[Load] KNNG size mismatch: inferred N=" + std::to_string(N) +
                                 " but total_points_=" + std::to_string(total_points_));
    }

    knng_k_ = K;
    knng_indices_.assign(N, std::vector<uint32_t>(K));
    for (size_t i = 0; i < N; ++i) {
        uint32_t dummy = 0;
        fin.read(reinterpret_cast<char*>(&dummy), sizeof(uint32_t));  // skip k_dummy
        fin.read(reinterpret_cast<char*>(knng_indices_[i].data()), K * sizeof(uint32_t));
        if (!fin) throw std::runtime_error("[Load] Unexpected EOF at KNNG point " + std::to_string(i));
    }
    std::cout << "[Load] KNNG: N=" << N << ", K=" << K << " from " << filepath << std::endl;
}

/********************************************************************************************
 * Diagnostics: degree statistics, invalid-id / isolation / symmetry checks, and BFS
 * connectivity (component count + largest-component size).
 ********************************************************************************************/
void IndexQuago::PrintGraphStats() {
    const size_t num_nodes = quago_graph_.size();
    std::cout << "\n==================== [QUAGO Graph Statistics & Diagnostics] ====================\n";
    std::cout << "[Graph Stats] Total nodes: " << num_nodes << std::endl;
    if (num_nodes == 0) return;

    uint64_t total_edges = 0;
    uint32_t min_deg = UINT32_MAX, max_deg = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        const uint32_t deg = static_cast<uint32_t>(quago_graph_[i].size());
        total_edges += deg;
        min_deg = std::min(min_deg, deg);
        max_deg = std::max(max_deg, deg);
    }
    const double avg_deg = static_cast<double>(total_edges) / static_cast<double>(num_nodes);
    std::cout << "\n[Statistics] Degree distribution:" << std::endl;
    std::cout << "  - Total edges: " << total_edges << std::endl;
    std::cout << "  - Avg degree: " << avg_deg << std::endl;
    std::cout << "  - Min degree: " << min_deg << std::endl;
    std::cout << "  - Max degree: " << max_deg << std::endl;

    std::cout << "\n[Diagnostics] Checking graph structure..." << std::endl;
    size_t invalid_ids = 0;
    for (size_t i = 0; i < num_nodes; ++i)
        for (uint32_t nb : quago_graph_[i])
            if (nb >= num_nodes) ++invalid_ids;
    std::cout << "  - Invalid neighbor IDs: " << invalid_ids << (invalid_ids == 0 ? " [OK]" : " [FAIL]") << std::endl;

    size_t isolated = 0;
    for (size_t i = 0; i < num_nodes; ++i)
        if (quago_graph_[i].empty()) ++isolated;
    std::cout << "  - Isolated nodes: " << isolated << " (" << 100.0 * isolated / num_nodes << "%)" << std::endl;

    size_t asymmetric_count = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        for (uint32_t nb : quago_graph_[i]) {
            if (nb >= num_nodes) continue;
            if (std::find(quago_graph_[nb].begin(), quago_graph_[nb].end(), static_cast<uint32_t>(i)) ==
                quago_graph_[nb].end()) {
                ++asymmetric_count;
            }
        }
    }
    std::cout << "  - Asymmetric (non-mutual) edges: " << asymmetric_count
              << (asymmetric_count == 0 ? " [OK]" : " [directed or missing backlinks]") << std::endl;

    const auto [components, largest_comp] = CheckConnectivity();
    std::cout << "\n[Diagnostics] BFS connectivity analysis..." << std::endl;
    std::cout << "  - Connected components: " << components << std::endl;
    std::cout << "  - Largest component size: " << largest_comp << " ("
              << 100.0 * largest_comp / num_nodes << "%)" << std::endl;
    std::cout << "=========================================================================\n";
}

std::pair<size_t, size_t> IndexQuago::CheckConnectivity() {
    const size_t num_nodes = quago_graph_.size();
    if (num_nodes == 0) return {0, 0};

    std::vector<bool> visited(num_nodes, false);
    std::queue<uint32_t> q;
    size_t components = 0, largest_comp = 0;

    for (size_t start = 0; start < num_nodes; ++start) {
        if (visited[start]) continue;
        ++components;
        size_t comp_size = 0;
        q.push(static_cast<uint32_t>(start));
        visited[start] = true;
        while (!q.empty()) {
            const uint32_t cur = q.front();
            q.pop();
            ++comp_size;
            for (uint32_t nb : quago_graph_[cur]) {
                if (nb < num_nodes && !visited[nb]) {
                    visited[nb] = true;
                    q.push(nb);
                }
            }
        }
        largest_comp = std::max(largest_comp, comp_size);
    }
    return {components, largest_comp};
}

}  // namespace quago
