#include "quago/index_quago.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "quago/core/common.h"

namespace quago {

// ============================================================================
//  frontier selector
// ----------------------------------------------------------------------------
//  Fuses three query-agnostic signals, all from base data + KNNG + base graph:
//  (a) position : d_c (global centroid dist), r_k (k-th NN dist)
//  (b) geometry : k-neighborhood PCA anisotropy  aniso = lambda1*(k-2) / (tr(G)-lambda1).
//      Implementation: neighborhood Gram matrix B = Y Y^T (k x k); the centered Gram is
//      G = B - rowmean_a - rowmean_b + totalmean, with tr(G) = tr(B) - k*totalmean in closed
//      form. lambda1 is obtained by power iteration kept inside the sum-zero subspace (G*1 = 0,
//      so G*v = B*v - (rowmean.v)*1). After centering, k points have rank <= k-1, so
//      lambda1 / mean(lambda2..lambda_{k-1}) equals the expression above and avoids sensitivity
//      to the noisy smallest eigenvalue.
//  (c) direction: M greedy-farthest-point spherical-code directions u_j are hill-climbed on
//      quago_graph_ to maximize <x-c, u_j> (= maximize dot(x, u_j) up to a constant); a(x) is
//      the number of times x is the argmax (two starts: entry_point_ + a random point, keep the
//      better one, approximating the global argmax).
//  Normalization: continuous signals use rank normalization (robust to heavy tails); the sparse
//  count a uses the non-saturating activation below. Final score:
//      s = w_pos*(0.3*d_c + 0.7*r_k) + w_geo*aniso + w_dir*a_hat, renormalized.
//  The position sub-weights are FIXED, not tunable -- see the definition below.
// ============================================================================
void IndexQuago::compute_quago_score_frontier(size_t knn_size, uint32_t M_dirs,
                                              float w_pos, float w_geo, float w_dir,
                                              float dir_tau) {
    // Position sub-weights are fixed, not tunable: the position signal is a 0.3/0.7 blend of the
    // global centroid distance and the k-th-NN distance.
    const float w_c = 0.3f, w_r = 0.7f;
    std::cout << "[Frontier] Computing region scores (frontier selector)..." << std::endl;
    if (knng_indices_.empty())
        throw std::runtime_error("[Frontier] KNN graph not loaded. Call LoadKnng() first.");
    if (knn_size > knng_k_)
        throw std::runtime_error("[Frontier] Requested k=" + std::to_string(knn_size) +
                                 " exceeds loaded knng_k_=" + std::to_string(knng_k_));
    if (knn_size < 3)
        throw std::runtime_error("[Frontier] knn_size must be >= 3 for PCA anisotropy.");
    if (quago_graph_.empty())
        throw std::runtime_error("[Frontier] base graph not loaded (needed for direction probing).");

    const size_t N = nd_, d = dimension_, k = knn_size;
    const float eps = 1e-12f;

    // ---- 1) global centroid c ----
    std::vector<float> c(d, 0.f);
    {
        const int nt = omp_get_max_threads();
        std::vector<std::vector<float>> partial(static_cast<size_t>(nt), std::vector<float>(d, 0.f));
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                const float* x = data_ + i * d;
                for (size_t t = 0; t < d; ++t) partial[static_cast<size_t>(tid)][t] += x[t];
            }
        }
        for (const auto& pv : partial)
            for (size_t t = 0; t < d; ++t) c[t] += pv[t];
        for (size_t t = 0; t < d; ++t) c[t] /= static_cast<float>(N);
    }

    // ---- 2) per-point stats: d_c, r_k, aniso ----
    std::vector<float> s_dc(N), s_rk(N), s_aniso(N);
#pragma omp parallel
    {
        std::vector<float> ybuf(static_cast<size_t>(k) * d);
        std::vector<float> B(static_cast<size_t>(k) * k), rowmean(k);
        std::vector<float> v(k), w(k);
        std::mt19937 rng(0xF2071Eu + static_cast<unsigned>(omp_get_thread_num()) * 7919u);
        std::normal_distribution<float> gauss(0.f, 1.f);
#pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < N; ++i) {
            const float* x = data_ + i * d;
            s_dc[i] = squared_distance(x, c.data());

            const std::vector<uint32_t>& nb = knng_indices_[i];
            // Neighborhood matrix Y (k x d), consumed by the Gram product below.
            for (size_t j = 0; j < k; ++j) {
                const float* y = data_ + static_cast<size_t>(nb[j]) * d;
                std::copy(y, y + d, ybuf.begin() + j * d);
            }
            const float invk = 1.0f / static_cast<float>(k);  // also used by the Gram/power iteration
            s_rk[i] = squared_distance(x, data_ + static_cast<size_t>(nb[k - 1]) * d);

            // Gram B = Y Y^T (symmetric; compute upper triangle only).
            for (size_t a = 0; a < k; ++a) {
                const float* ya = ybuf.data() + a * d;
                for (size_t b = a; b < k; ++b) {
                    const float* yb = ybuf.data() + b * d;
                    float dot = 0.f;
                    for (size_t t = 0; t < d; ++t) dot += ya[t] * yb[t];
                    B[a * k + b] = dot;
                    B[b * k + a] = dot;
                }
            }
            float totalmean = 0.f, traceB = 0.f;
            for (size_t a = 0; a < k; ++a) {
                float rs = 0.f;
                for (size_t b = 0; b < k; ++b) rs += B[a * k + b];
                rowmean[a] = rs * invk;
                totalmean += rowmean[a];
                traceB += B[a * k + a];
            }
            totalmean *= invk;
            const float traceG = traceB - static_cast<float>(k) * totalmean;  // tr(G) = tr(B) - k*totalmean

            // Power iteration for lambda1(G); v stays in the sum-zero subspace (G*1 = 0).
            float vsum = 0.f;
            for (size_t a = 0; a < k; ++a) { v[a] = gauss(rng); vsum += v[a]; }
            float vm = vsum * invk, vn = 0.f;
            for (size_t a = 0; a < k; ++a) { v[a] -= vm; vn += v[a] * v[a]; }
            vn = 1.0f / std::sqrt(vn + eps);
            for (size_t a = 0; a < k; ++a) v[a] *= vn;
            for (int it = 0; it < 40; ++it) {
                float s = 0.f;
                for (size_t b = 0; b < k; ++b) s += rowmean[b] * v[b];  // rowmean . v
                for (size_t a = 0; a < k; ++a) {
                    float bv = 0.f;
                    const float* Ba = B.data() + a * k;
                    for (size_t b = 0; b < k; ++b) bv += Ba[b] * v[b];
                    w[a] = bv - s;  // (G v)_a  (sum-zero)
                }
                float wm = 0.f;
                for (size_t a = 0; a < k; ++a) wm += w[a];
                wm *= invk;
                vn = 0.f;
                for (size_t a = 0; a < k; ++a) { v[a] = w[a] - wm; vn += v[a] * v[a]; }
                vn = 1.0f / std::sqrt(vn + eps);
                for (size_t a = 0; a < k; ++a) v[a] *= vn;
            }
            // lambda1 = v^T G v (v normalized and sum-zero => v^T G v = v . (B v)).
            float lambda1;
            {
                float bv_dot = 0.f;
                for (size_t a = 0; a < k; ++a) {
                    float bv = 0.f;
                    const float* Ba = B.data() + a * k;
                    for (size_t b = 0; b < k; ++b) bv += Ba[b] * v[b];
                    bv_dot += v[a] * bv;
                }
                lambda1 = std::max(0.f, bv_dot);
            }
            const float rest = traceG - lambda1;
            s_aniso[i] = (rest > eps) ? lambda1 * static_cast<float>(k - 2) / rest
                                      : (traceG > eps ? 1e9f : 0.f);
        }
    }
    std::cout << "[Frontier] positional + PCA stats done." << std::endl;

    // ---- 3) direction-frontier probing: FPS spherical code + hill-climb hit counts ----
    std::vector<uint32_t> hits(N, 0);
    {
        if (M_dirs < 1) M_dirs = 1;
        std::vector<float> U(static_cast<size_t>(M_dirs) * d);
        {
            std::mt19937 rng(0x5CA1EC0Du);
            std::normal_distribution<float> gauss(0.f, 1.f);
            auto nrm = [&](float* p) {
                float ss = 0.f;
                for (size_t t = 0; t < d; ++t) ss += p[t] * p[t];
                const float inv = 1.0f / std::sqrt(ss + eps);
                for (size_t t = 0; t < d; ++t) p[t] *= inv;
            };
            for (uint32_t t = 0; t < d; ++t) U[t] = gauss(rng);
            nrm(U.data());
            const uint32_t P = 32;  // candidates per step
            std::vector<float> cand(static_cast<size_t>(P) * d), cand_mx(P, -2.f);
            for (uint32_t j = 1; j < M_dirs; ++j) {
                for (uint32_t q = 0; q < P; ++q) {
                    float* cc = cand.data() + static_cast<size_t>(q) * d;
                    for (size_t t = 0; t < d; ++t) cc[t] = gauss(rng);
                    nrm(cc);
                    cand_mx[q] = -2.f;
                }
                // Incrementally track each candidate's max inner product with the chosen set.
                for (uint32_t t = 0; t < j; ++t) {
                    const float* ut = U.data() + static_cast<size_t>(t) * d;
                    for (uint32_t q = 0; q < P; ++q) {
                        const float* cc = cand.data() + static_cast<size_t>(q) * d;
                        float dot = 0.f;
                        for (size_t x2 = 0; x2 < d; ++x2) dot += ut[x2] * cc[x2];
                        if (dot > cand_mx[q]) cand_mx[q] = dot;
                    }
                }
                uint32_t best = 0;
                for (uint32_t q = 1; q < P; ++q)
                    if (cand_mx[q] < cand_mx[best]) best = q;
                std::copy(cand.data() + static_cast<size_t>(best) * d,
                          cand.data() + static_cast<size_t>(best + 1) * d,
                          U.begin() + static_cast<size_t>(j) * d);
            }
        }
        std::cout << "[Frontier] " << M_dirs << " spherical-code directions built; probing..." << std::endl;

        // Record each direction's hill-climb winner so QMRNG can reuse the exact direction set the
        // selector probed (frontier_anchors_[j] = the base node maximizing <x, u_j>). Each j is written
        // by a single iteration, so no synchronization is needed beyond the hits counter's.
        frontier_anchors_.assign(M_dirs, 0u);

        // Hill-climb to maximize dot(x, u) (equivalently the argmax of <x-c, u>).
        std::atomic<size_t> total_hops{0};
#pragma omp parallel for schedule(dynamic, 8)
        for (uint32_t j = 0; j < M_dirs; ++j) {
            const float* u = U.data() + static_cast<size_t>(j) * d;
            auto dot_u = [&](uint32_t id) {
                const float* xx = data_ + static_cast<size_t>(id) * d;
                float s = 0.f;
                for (size_t t = 0; t < d; ++t) s += xx[t] * u[t];
                return s;
            };
            auto climb = [&](uint32_t start) {
                uint32_t x = start;
                float fx = dot_u(x);
                for (;;) {
                    uint32_t best = x;
                    float fb = fx;
                    for (uint32_t v : quago_graph_[x]) {
                        const float fv = dot_u(v);
                        if (fv > fb) { fb = fv; best = v; }
                    }
                    ++total_hops;
                    if (best == x) break;
                    x = best;
                    fx = fb;
                }
                return std::make_pair(x, fx);
            };
            const auto r1 = climb(entry_point_);
            uint32_t rnd_start;
            {  // deterministic random second start
                std::mt19937 rng(0xABCDu + j * 2654435761u);
                rnd_start = static_cast<uint32_t>(rng() % static_cast<unsigned>(N));
            }
            const auto r2 = climb(rnd_start);
            const uint32_t winner = (r2.second > r1.second) ? r2.first : r1.first;
            frontier_anchors_[j] = winner;  // each j written by one iteration; no sync needed
#pragma omp atomic
            hits[winner]++;
        }
        std::cout << "[Frontier] probing done, total hops=" << total_hops.load() << std::endl;
    }

    // ---- 4) normalization + fusion ----
    auto rank_norm = [N](const std::vector<float>& src, std::vector<float>& dst) {
        std::vector<uint32_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return src[a] < src[b]; });
        const float inv = 1.0f / static_cast<float>(N - 1);
        size_t t = 0;
        while (t < N) {  // ties take the average rank
            size_t t2 = t + 1;
            while (t2 < N && src[idx[t2]] == src[idx[t]]) ++t2;
            const float r = 0.5f * static_cast<float>(t + t2 - 1) * inv;
            for (size_t z = t; z < t2; ++z) dst[idx[z]] = r;
            t = t2;
        }
    };
    std::vector<float> h_dc(N), h_rk(N), h_aniso(N);
    rank_norm(s_dc, h_dc);
    rank_norm(s_rk, h_rk);
    rank_norm(s_aniso, h_aniso);
    uint32_t max_hits = 0;
    for (size_t i = 0; i < N; ++i)
        if (hits[i] > max_hits) max_hits = hits[i];
    std::cout << "[Frontier] max direction hits=" << max_hits << std::endl;

    // ---- automatic fusion weights: when all three are the negative sentinel (-1) ----
    // IP/COSINE: geo is disabled (near-duplicate / degenerate neighborhoods produce aniso
    //            spikes that hijack selection after rank normalization); pos 0.8 + dir 0.2.
    // L2:        geo enabled but small (0.1); pos 0.7 + geo 0.1 + dir 0.2.
    // Explicit (>= 0) values override the automatic scheme; a lone -1 is treated as 0.
    if (w_pos < 0.f && w_geo < 0.f && w_dir < 0.f) {
        if (metric_ == INNER_PRODUCT || metric_ == COSINE) {
            w_pos = 0.8f; w_geo = 0.0f; w_dir = 0.2f;
        } else {
            w_pos = 0.7f; w_geo = 0.1f; w_dir = 0.2f;
        }
    } else {
        if (w_pos < 0.f) w_pos = 0.f;
        if (w_geo < 0.f) w_geo = 0.f;
        if (w_dir < 0.f) w_dir = 0.f;
    }
    std::cout << "[Frontier] fusion weights: pos=" << w_pos << " geo=" << w_geo
              << " dir=" << w_dir << " (dir_tau=" << dir_tau << ")" << std::endl;

    float sf = w_pos + w_geo + w_dir;
    if (!(sf > 0.f)) sf = 1.f;
    if (!(dir_tau > 0.f)) dir_tau = 1.0f;
    const float inv_tau = 1.0f / dir_tau;
    quago_scores_.assign(N, 0.f);
    std::vector<float> s_pos_v(N), a_hat_v(N);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const float s_pos = w_c * h_dc[i] + w_r * h_rk[i];
        // Non-saturating exponential activation for the (near-binary) direction count: a(x)
        // has p50 = 1 hit, so min-max normalization would be crushed by the spikes and go inert.
        // a_hat = expm1(hits / tau) is unbounded: hits=1 -> 1.72 (mild boost, pos still leads),
        // hits=2 -> ~6.4*w_dir already exceeds the full pos range, hits>=3 -> near-certain select.
        // The exponent is clamped at 30 to avoid float overflow (e^30 ~ 1e13 already dominates).
        float a_hat = 0.f;
        if (hits[i] > 0) {
            float ex = static_cast<float>(hits[i]) * inv_tau;
            if (ex > 30.f) ex = 30.f;
            a_hat = std::expm1(ex);
        }
        s_pos_v[i] = s_pos;
        a_hat_v[i] = a_hat;
        quago_scores_[i] = (w_pos * s_pos + w_geo * h_aniso[i] + w_dir * a_hat) / sf;
    }
    std::cout << "[Frontier] Region scores computed for " << N << " points." << std::endl;
}

// ============================================================================
//  Region selection
// ============================================================================
std::vector<uint32_t> IndexQuago::select_quago_region(float top_ratio) {
    if (quago_scores_.empty())
        throw std::runtime_error("[select_quago_region] quago_scores_ empty. Call a compute_quago_score_* first.");
    if (top_ratio <= 0.f || top_ratio > 1.f)
        throw std::runtime_error("[select_quago_region] top_ratio must be in (0, 1], got " + std::to_string(top_ratio));

    const size_t N = nd_;
    std::vector<float> tmp(quago_scores_);
    size_t q_idx = static_cast<size_t>(std::floor((1.0f - top_ratio) * static_cast<float>(N)));
    q_idx = std::clamp(q_idx, static_cast<size_t>(0), N - 1);

    std::nth_element(tmp.begin(), tmp.begin() + q_idx, tmp.end());
    const float threshold = tmp[q_idx];

    // A NaN threshold would make the `>=` comparison below false for every point (IEEE-754:
    // x >= NaN is always false), silently selecting the EMPTY set and defeating top_ratio.
    // Fail loudly instead of emitting a delta that merely looks normal.
    if (std::isnan(threshold))
        throw std::runtime_error("[select_quago_region] threshold is NaN: quago_scores_ contains NaN, "
                                 "usually a metric/sqrt mismatch (see IndexQuago::squared_distance).");

    std::vector<uint32_t> quago_ids;
    quago_ids.reserve(static_cast<size_t>(top_ratio * static_cast<float>(N) * 1.1f));
    for (size_t i = 0; i < N; ++i)
        if (quago_scores_[i] >= threshold) quago_ids.push_back(static_cast<uint32_t>(i));

    std::cout << "[select_quago_region] threshold=" << threshold << ", selected=" << quago_ids.size()
              << "/" << N << " (ratio=" << top_ratio << ")" << std::endl;
    return quago_ids;
}

std::vector<uint32_t> IndexQuago::select_quago_region_random(float ratio) {
    if (ratio <= 0.f || ratio > 1.f)
        throw std::runtime_error("[select_quago_region_random] ratio must be in (0, 1], got " + std::to_string(ratio));

    const size_t N = nd_;
    const size_t select_count = static_cast<size_t>(std::round(ratio * static_cast<float>(N)));

    std::vector<uint32_t> all_ids(N);
    std::iota(all_ids.begin(), all_ids.end(), 0u);
    std::mt19937 rng(42u);
    std::shuffle(all_ids.begin(), all_ids.end(), rng);
    all_ids.resize(select_count);

    std::cout << "[select_quago_region_random] selected=" << all_ids.size() << "/" << N
              << " (ratio=" << ratio << ")" << std::endl;
    return all_ids;
}

// ============================================================================
//  Enhancement method: KNN overlay (mode 0 = add).
//  For each region node u, add intra-region KNNG edges (and their reverses) that are
//  not already present, respecting the max_degree budget on both endpoints.
// ============================================================================
void IndexQuago::enhance_subgraph_with_knn(const std::vector<uint32_t>& node_ids, uint32_t knn_size,
                                           uint32_t max_degree) {
    if (node_ids.empty()) {
        std::cout << "[KNN Enhance] No nodes provided, skip." << std::endl;
        return;
    }
    if (knng_indices_.empty()) throw std::runtime_error("[KNN Enhance] KNN graph not loaded.");
    if (knn_size > knng_k_) throw std::runtime_error("[KNN Enhance] knn_size exceeds knng_k_.");

    quago_delta_.assign(total_points_, {});
    const std::unordered_set<uint32_t> node_set(node_ids.begin(), node_ids.end());
    size_t added_edges = 0;
    for (uint32_t u : node_ids) {
        std::unordered_set<uint32_t> existing(quago_graph_[u].begin(), quago_graph_[u].end());
        existing.insert(u);

        for (size_t j = 0; j < knn_size; ++j) {
            const uint32_t v = knng_indices_[u][j];
            if (!node_set.count(v)) continue;
            if (existing.count(v)) continue;

            const size_t cur_deg = quago_graph_[u].size() + quago_delta_[u].size();
            if (cur_deg >= static_cast<size_t>(max_degree)) break;

            quago_delta_[u].push_back(v);  // u -> v
            existing.insert(v);
            ++added_edges;

            // reverse edge v -> u
            const std::unordered_set<uint32_t> existing_v(quago_graph_[v].begin(), quago_graph_[v].end());
            const size_t cur_deg_v = quago_graph_[v].size() + quago_delta_[v].size();
            if (cur_deg_v < static_cast<size_t>(max_degree) && !existing_v.count(u)) {
                quago_delta_[v].push_back(u);
                ++added_edges;
            }
        }
    }
    std::cout << "[KNN Enhance] Added " << added_edges << " new edges into quago_delta_." << std::endl;
}

// ============================================================================
//  Enhancement method: QMRNG (mode 1 = replace).
// ----------------------------------------------------------------------------
//  Two stages over the selected region, in one function:
//
//  1) Region skeleton rebuild. Builds a skeleton graph from scratch over the region subset and
//     writes each region node's new adjacency into quago_delta_ (enhancement_mode_ = 1).
//     apply_quago_delta then replaces the region<->region base edges with these edges but keeps
//     each region node's original edges to non-region nodes (cross-boundary edges) so the graph
//     stays connected. Construction: random init + medoid entry + two
//     passes (alpha=1 -> alpha) of greedy beam search + RobustPrune. All distances use
//     squared_distance(), which is non-negative and smaller-is-closer for L2/IP/COSINE, so alpha
//     scaling is meaningful.
//
//  2) Direction-frontier MST overlay. Reuses the M spherical-code directions the `frontier`
//     selector already hill-climbed (frontier_anchors_): for each direction's anchor a_j -- the
//     base point maximizing <x, u_j>, i.e. a boundary extreme -- the "gt20" set is a_j together
//     with its precomputed knn_size nearest neighbors from the KNNG. A Prim MST is built inside
//     each gt20 set (edge weight = squared_distance, rooted at the anchor), and the MST edges
//     are unioned on top of the rebuilt delta.
//
//  Rationale: the anchor's knn are exactly the boundary points an OOD query arriving along u_j must
//  resolve among, but the base graph reaches them only through the dense core (the boundary is thinly
//  connected within itself). The MST adds the intra-frontier edges, and because a spanning tree keeps
//  the longer edges an RNG/degree prune would drop, it restores exactly the tangential connectivity
//  that makes each directional extreme internally navigable. It costs no extra neighbor search: the
//  directions were already probed for scoring and the knn is already loaded.
//
//  Region scoring/selection is unchanged. Requires quago_strategy=frontier (which populates
//  frontier_anchors_).
// ============================================================================
void IndexQuago::enhance_subgraph_with_qmrng(const std::vector<uint32_t>& node_ids, uint32_t max_degree,
                                             uint32_t search_L, float alpha, uint32_t knn_size) {
    // ---- 1) skeleton rebuild of the region (fills quago_delta_, sets enhancement_mode_ = 1) ----
    const size_t S = node_ids.size();
    if (S < 2) {
        std::cout << "[skeleton build] Fewer than 2 region nodes, skip." << std::endl;
        return;
    }
    const uint32_t R = max_degree;  // the skeleton's degree budget R reuses M_nhood
    if (R < 1) throw std::runtime_error("[skeleton build] max_degree(R) must be >= 1");
    if (search_L < R) search_L = R;  // beam width must be >= target degree to fill R candidates

    quago_delta_.assign(total_points_, {});
    enhancement_mode_ = 1;  // replace semantics

    const size_t dim = dimension_;
    auto vec_of = [&](uint32_t local) -> const float* { return data_ + static_cast<size_t>(node_ids[local]) * dim; };
    auto dist_ll = [&](uint32_t a, uint32_t b) -> float { return squared_distance(vec_of(a), vec_of(b)); };

    std::cout << "[skeleton build] Rebuilding region subset: S=" << S << ", R=" << R << ", L=" << search_L
              << ", alpha=" << alpha << std::endl;

    // ---- random init of local adjacency (guarantees a greedy-search start) ----
    std::vector<std::vector<uint32_t>> adj(S);
    {
        std::mt19937 rng(0x5EEDu);
        const uint32_t init_deg = std::min<uint32_t>(R, static_cast<uint32_t>(S - 1));  // GenRandom needs size < N=S
        std::vector<unsigned> picks(init_deg);
        for (size_t i = 0; i < S; ++i) {
            GenRandom(rng, picks.data(), init_deg, static_cast<unsigned>(S));
            adj[i].reserve(R + 1);
            for (unsigned p : picks) {
                const uint32_t v = static_cast<uint32_t>(p);
                if (v != static_cast<uint32_t>(i)) adj[i].push_back(v);
            }
        }
    }

    uint32_t medoid = 0;
    {
        const size_t n_samples = std::min<size_t>(256, S - 1);  // also ensures size < N=S
        std::vector<unsigned> sp(n_samples);
        std::mt19937 rng(0xC0FFEEu);
        GenRandom(rng, sp.data(), static_cast<unsigned>(n_samples), static_cast<unsigned>(S));

        float best = std::numeric_limits<float>::max();
        for (size_t i = 0; i < S; ++i) {
            float sum = 0.f;
            for (unsigned s : sp) sum += dist_ll(static_cast<uint32_t>(i), static_cast<uint32_t>(s));
            if (sum < best) { best = sum; medoid = static_cast<uint32_t>(i); }
        }
    }
    std::cout << "[skeleton build] medoid local=" << medoid << " global=" << node_ids[medoid] << std::endl;

    auto robust_prune = [&](uint32_t p, std::vector<uint32_t> cand, float a) {
        std::vector<std::pair<float, uint32_t>> cd;
        cd.reserve(cand.size());
        for (uint32_t c : cand)
            if (c != p) cd.push_back({dist_ll(p, c), c});
        std::sort(cd.begin(), cd.end());

        std::vector<uint32_t> res;
        res.reserve(R);
        std::vector<char> killed(cd.size(), 0);
        for (size_t i = 0; i < cd.size(); ++i) {
            if (killed[i]) continue;
            const uint32_t pstar = cd[i].second;
            res.push_back(pstar);
            if (res.size() >= R) break;
            for (size_t j = i + 1; j < cd.size(); ++j) {
                if (killed[j]) continue;
                if (a * dist_ll(pstar, cd[j].second) <= cd[j].first) killed[j] = 1;
            }
        }
        return res;
    };

    std::vector<std::mutex> local_locks(S);
    std::vector<uint32_t> order(S);
    std::iota(order.begin(), order.end(), 0u);

    for (int pass = 0; pass < 2; ++pass) {
        const float a = (pass == 0) ? 1.0f : alpha;
        {
            std::mt19937 rng(1234u + static_cast<unsigned>(pass));
            std::shuffle(order.begin(), order.end(), rng);
        }

#pragma omp parallel
        {
            std::vector<uint32_t> visited(S, 0);  // per-thread timestamped visited table
            uint32_t tag = 0;
#pragma omp for schedule(dynamic, 64)
            for (size_t idx = 0; idx < S; ++idx) {
                const uint32_t p = order[idx];
                if (++tag == 0) { std::fill(visited.begin(), visited.end(), 0u); tag = 1; }

                // greedy beam search: query = p's own vector, find the nearest search_L on local adj
                NeighborPriorityQueue q(search_L);
                visited[medoid] = tag;
                q.insert(Neighbor(medoid, dist_ll(medoid, p), false));
                while (q.has_unexpanded_node()) {
                    const uint32_t cid = q.closest_unexpanded().id;
                    std::vector<uint32_t> nb;
                    { std::lock_guard<std::mutex> g(local_locks[cid]); nb = adj[cid]; }
                    for (uint32_t v : nb) {
                        if (visited[v] == tag) continue;
                        visited[v] = tag;
                        q.insert(Neighbor(v, dist_ll(v, p), false));
                    }
                }

                // candidates = search results U existing adjacency
                std::vector<uint32_t> cand;
                cand.reserve(q.size() + R);
                for (size_t i = 0; i < q.size(); ++i) cand.push_back(q[i].id);
                { std::lock_guard<std::mutex> g(local_locks[p]); cand.insert(cand.end(), adj[p].begin(), adj[p].end()); }

                const std::vector<uint32_t> newnbrs = robust_prune(p, std::move(cand), a);
                { std::lock_guard<std::mutex> g(local_locks[p]); adj[p] = newnbrs; }

                // reverse edges: add p to each n; re-prune n if it exceeds R. Only one lock is held
                // at a time (no nesting), so there is no deadlock. Iterating the local copy newnbrs
                // (not adj[p]) avoids racing with other threads writing adj[p].
                for (uint32_t n : newnbrs) {
                    std::lock_guard<std::mutex> g(local_locks[n]);
                    if (std::find(adj[n].begin(), adj[n].end(), p) == adj[n].end()) {
                        adj[n].push_back(p);
                        if (adj[n].size() > R) {
                            std::vector<uint32_t> pr = robust_prune(n, adj[n], a);
                            adj[n].swap(pr);
                        }
                    }
                }
            }
        }
        std::cout << "[skeleton build] pass " << pass << " (alpha=" << a << ") done." << std::endl;
    }

    // ---- extract delta (local -> global); under replace semantics these are the new adjacencies ----
    size_t total_edges = 0;
    for (size_t i = 0; i < S; ++i) {
        const uint32_t u = node_ids[i];
        auto& dd = quago_delta_[u];
        dd.reserve(adj[i].size());
        for (uint32_t v : adj[i]) dd.push_back(node_ids[v]);
        total_edges += dd.size();
    }
    std::cout << "[skeleton build] Rebuilt region: nodes=" << S << ", edges=" << total_edges
              << ", avg_deg=" << (S ? static_cast<double>(total_edges) / static_cast<double>(S) : 0.0)
              << " -> quago_delta_ (replace mode)." << std::endl;

    // ---- 2) direction-frontier gt20 MST overlay ----
    if (frontier_anchors_.empty())
        throw std::runtime_error("[QMRNG] no direction anchors recorded; QMRNG must run with "
                                 "quago_strategy=frontier so it can reuse the sampled directions.");
    if (knng_indices_.empty()) throw std::runtime_error("[QMRNG] KNN graph not loaded.");
    if (knn_size > knng_k_) knn_size = knng_k_;
    if (knn_size < 2) {
        std::cout << "[QMRNG] knn_size=" << knn_size << " < 2; no MST edges to add." << std::endl;
        return;
    }
    enhancement_mode_ = 1;  // replace (already set by stage 1; asserted for clarity)

    // Dedup anchors: many directions hill-climb to the same extreme, and re-running its MST would only
    // regenerate the same edges.
    std::vector<uint32_t> anchors = frontier_anchors_;
    std::sort(anchors.begin(), anchors.end());
    anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
    const size_t A = anchors.size();
    std::cout << "[QMRNG] direction anchors: " << frontier_anchors_.size() << " -> " << A
              << " unique; gt20 size=" << knn_size << std::endl;

    // Per-anchor MST edges (undirected pairs), computed in parallel; merged into the delta serially.
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> mst_edges(A);
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t ai = 0; ai < A; ++ai) {
        const uint32_t a = anchors[ai];
        // gt20 set = {a} U its knn_size nearest neighbors (dedup; drop self / out-of-range ids).
        std::vector<uint32_t> gt20;
        gt20.reserve(static_cast<size_t>(knn_size) + 1);
        gt20.push_back(a);
        {
            std::unordered_set<uint32_t> seen;
            seen.insert(a);
            const std::vector<uint32_t>& nb = knng_indices_[a];
            for (size_t j = 0; j < static_cast<size_t>(knn_size) && j < nb.size(); ++j) {
                const uint32_t v = nb[j];
                if (v >= total_points_ || !seen.insert(v).second) continue;
                gt20.push_back(v);
            }
        }
        const size_t n = gt20.size();
        if (n < 2) continue;

        auto vec = [&](uint32_t id) -> const float* { return data_ + static_cast<size_t>(id) * dim; };

        // Dense Prim rooted at gt20[0] (the anchor): O(n^2) distance evals, n <= knn_size + 1.
        const float INF = std::numeric_limits<float>::max();
        std::vector<float> key(n, INF);
        std::vector<int32_t> parent(n, -1);
        std::vector<char> in_tree(n, 0);
        key[0] = 0.f;
        auto& out = mst_edges[ai];
        out.reserve(n - 1);
        for (size_t it = 0; it < n; ++it) {
            size_t u = 0; float best = INF; bool found = false;
            for (size_t i2 = 0; i2 < n; ++i2)
                if (!in_tree[i2] && key[i2] < best) { best = key[i2]; u = i2; found = true; }
            if (!found) break;  // the set spans a complete graph, so this only guards a degenerate n
            in_tree[u] = 1;
            if (parent[u] >= 0) out.emplace_back(gt20[u], gt20[static_cast<size_t>(parent[u])]);
            const float* xu = vec(gt20[u]);
            for (size_t v2 = 0; v2 < n; ++v2) {
                if (in_tree[v2]) continue;
                const float w = squared_distance(xu, vec(gt20[v2]));
                if (w < key[v2]) { key[v2] = w; parent[v2] = static_cast<int32_t>(u); }
            }
        }
    }

    // ---- merge: add every MST edge in BOTH orientations, deduped against the skeleton edges already in
    //      the delta and against each other. Flatten + sort groups all edges by source node so each
    //      node's dedup set is built once. ----
    std::vector<std::pair<uint32_t, uint32_t>> directed;
    size_t undirected = 0;
    for (const auto& e : mst_edges) {
        undirected += e.size();
        for (const auto& uv : e) {
            directed.emplace_back(uv.first, uv.second);
            directed.emplace_back(uv.second, uv.first);
        }
    }
    std::sort(directed.begin(), directed.end());

    size_t added = 0, touched = 0;
    for (size_t i = 0; i < directed.size();) {
        const uint32_t u = directed[i].first;
        auto& dd = quago_delta_[u];
        std::unordered_set<uint32_t> have(dd.begin(), dd.end());
        have.insert(u);
        size_t j = i;
        for (; j < directed.size() && directed[j].first == u; ++j) {
            const uint32_t v = directed[j].second;
            if (have.insert(v).second) { dd.push_back(v); ++added; }
        }
        ++touched;
        i = j;
    }
    std::cout << "[QMRNG] MST overlay: unique_anchors=" << A << ", undirected_mst_edges=" << undirected
              << ", directed_edges_added=" << added << ", nodes_touched=" << touched
              << " (unioned on top of the skeleton rebuild, replace mode)." << std::endl;
}

// ============================================================================
//  Enhancement-delta IO
// ----------------------------------------------------------------------------
//  Binary format (magic + mode):
//    [u32 MAGIC = 0x51474F44 ("QGOD")][u32 mode][u32 N]
//    then for each node i in [0, N): [u32 deg][deg * u32 neighbor_ids]
//  mode: 0 = add (overlay), 1 = replace (rebuild region<->region, keep cross-boundary edges).
//  A legacy file (no magic; first field is N) is still accepted as mode = add.
// ============================================================================
void IndexQuago::SaveQuagoDelta(const std::string& filepath) {
    if (quago_delta_.empty()) throw std::runtime_error("[Save] quago_delta_ is empty.");

    std::ofstream fout(filepath, std::ios::binary);
    if (!fout.is_open()) throw std::runtime_error("[Save] Cannot open file: " + filepath);

    const uint32_t MAGIC = 0x51474F44u;  // "QGOD"; larger than any real N, so it cannot collide with a legacy header
    const uint32_t mode = static_cast<uint32_t>(enhancement_mode_);
    const uint32_t N = static_cast<uint32_t>(quago_delta_.size());
    fout.write(reinterpret_cast<const char*>(&MAGIC), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&mode), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&N), sizeof(uint32_t));

    size_t total_edges = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const auto& nbrs = quago_delta_[i];
        const uint32_t deg = static_cast<uint32_t>(nbrs.size());
        fout.write(reinterpret_cast<const char*>(&deg), sizeof(uint32_t));
        if (deg > 0) fout.write(reinterpret_cast<const char*>(nbrs.data()), deg * sizeof(uint32_t));
        total_edges += deg;
    }
    if (!fout) throw std::runtime_error("[Save] Write error.");
    std::cout << "[SaveQuagoDelta] Saved N=" << N << ", mode=" << mode << ", total_edges=" << total_edges
              << " -> " << filepath << std::endl;
}

void IndexQuago::LoadQuagoDelta(const std::string& filepath) {
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin.is_open()) throw std::runtime_error("[Load] Cannot open file: " + filepath);

    // Peek the first field: MAGIC -> v2 header; otherwise legacy (first field is N, mode = add).
    uint32_t first = 0;
    fin.read(reinterpret_cast<char*>(&first), sizeof(uint32_t));
    if (!fin) throw std::runtime_error("[Load] Failed to read header.");

    uint32_t N = 0;
    if (first == 0x51474F44u) {
        uint32_t mode = 0;
        fin.read(reinterpret_cast<char*>(&mode), sizeof(uint32_t));
        fin.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
        delta_mode_ = static_cast<int>(mode);
    } else {
        N = first;
        delta_mode_ = 0;
    }
    if (!fin) throw std::runtime_error("[Load] Failed to read header.");
    if (N != static_cast<uint32_t>(total_points_))
        throw std::runtime_error("[Load] Size mismatch: file N=" + std::to_string(N) +
                                 " but total_points_=" + std::to_string(total_points_));

    quago_delta_.assign(N, {});
    size_t total_edges = 0;
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t deg = 0;
        fin.read(reinterpret_cast<char*>(&deg), sizeof(uint32_t));
        if (!fin) throw std::runtime_error("[Load] Unexpected EOF at node " + std::to_string(i));
        if (deg > 0) {
            quago_delta_[i].resize(deg);
            fin.read(reinterpret_cast<char*>(quago_delta_[i].data()), deg * sizeof(uint32_t));
            if (!fin) throw std::runtime_error("[Load] Unexpected EOF reading neighbors of node " + std::to_string(i));
        }
        total_edges += deg;
    }
    std::cout << "[Load] Loaded N=" << N << ", mode=" << delta_mode_ << ", total_edges=" << total_edges
              << " from " << filepath << std::endl;
}

void IndexQuago::apply_quago_delta() {
    if (quago_delta_.empty()) throw std::runtime_error("[Error] quago_delta_ is empty.");
    if (quago_graph_.empty()) throw std::runtime_error("[Error] quago_graph_ is empty.");
    if (quago_delta_.size() != quago_graph_.size())
        throw std::runtime_error("[Error] Size mismatch: delta=" + std::to_string(quago_delta_.size()) +
                                 " quago_graph_=" + std::to_string(quago_graph_.size()));

    // Replace mode (qmrng): rebuild only region<->region edges, but keep each region
    // node's original base edges to NON-region nodes (cross-boundary edges):
    //   new_adj(u) = delta_edges(u) U { v in base_adj(u) : v is not a region node }
    // Original edges pointing at region nodes are dropped (replaced by the rebuilt edges). This
    // rebuilds the region backbone while preserving region -> non-region reachability, so the
    // graph is not cut apart. The region set is exactly the nodes with a non-empty delta:
    // the QMRNG builder guarantees every region node has >= 1 edge, and non-region
    // nodes always have an empty delta.
    if (delta_mode_ == 1) {
        std::vector<char> is_quago(quago_graph_.size(), 0);
        for (size_t i = 0; i < quago_delta_.size(); ++i)
            if (!quago_delta_[i].empty()) is_quago[i] = 1;

        size_t total_replaced = 0, total_skeleton = 0, total_kept = 0;
        for (size_t i = 0; i < quago_graph_.size(); ++i) {
            if (!is_quago[i]) continue;
            const auto& delta = quago_delta_[i];

            std::vector<uint32_t> nbrs;
            nbrs.reserve(delta.size() + quago_graph_[i].size());
            nbrs.insert(nbrs.end(), delta.begin(), delta.end());  // 1) rebuilt region edges
            total_skeleton += delta.size();
            for (uint32_t v : quago_graph_[i]) {  // 2) keep base edges to non-region nodes
                if (v == i || v >= is_quago.size()) continue;
                if (!is_quago[v]) { nbrs.push_back(v); ++total_kept; }
            }
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
            quago_graph_[i].swap(nbrs);
            ++total_replaced;
        }
        std::cout << "[Done] Region rebuild (replace, keep cross-boundary): nodes=" << total_replaced
                  << ", skeleton_edges=" << total_skeleton << ", kept_cross_edges=" << total_kept << std::endl;
        quago_delta_.clear();
        quago_delta_.shrink_to_fit();
        return;
    }

    // Add mode (knn): overlay delta edges onto the base graph, skipping duplicates.
    size_t total_added = 0, total_skipped = 0;
    for (size_t i = 0; i < quago_graph_.size(); ++i) {
        const auto& delta = quago_delta_[i];
        if (delta.empty()) continue;
        std::unordered_set<uint32_t> existing(quago_graph_[i].begin(), quago_graph_[i].end());
        for (uint32_t v : delta) {
            if (existing.count(v)) { ++total_skipped; continue; }
            quago_graph_[i].push_back(v);
            existing.insert(v);
            ++total_added;
        }
    }
    std::cout << "[Done] Enhancement delta edges added=" << total_added << ", skipped(dup)=" << total_skipped << std::endl;
    quago_delta_.clear();
    quago_delta_.shrink_to_fit();
}

}  // namespace quago
