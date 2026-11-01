# QUAGO

Query-agnostic graph enhancement for approximate nearest neighbor search.

QUAGO does not build base graphs. Given an existing ANN graph (HNSW / NSG / native) and a k-NN graph,
it scores every base point for how far it sits in the hard-to-navigate outer *frontier* of the data
distribution, selects the top-scoring fraction, and rebuilds the edges inside that region. The output
is a small **delta** file that search overlays onto the base graph. The base graph on disk is never
modified, and no queries are used at build time — the enhancement is query-agnostic, which is what
makes it transfer to out-of-distribution (e.g. cross-modal) queries.

Region scoring (`frontier`) fuses three signals: position (centroid distance and k-th NN distance),
k-neighborhood PCA anisotropy, and hit counts from `M` spherical-code directions hill-climbed on the
base graph. `random` selects a uniform fraction instead, as an ablation.

Two enhancement methods:

| `--enhance_method` | what it does | delta mode |
| --- | --- | --- |
| `knn` | overlays intra-region k-NN edges onto the base graph | 0 = add |
| `qmrng` (default) | rebuilds the region skeleton (two-pass greedy beam search + RobustPrune), then unions a direction-frontier MST overlay on top | 1 = replace |

In replace mode only region↔region edges are rebuilt; each region node keeps its original edges to
non-region nodes, so the graph stays connected.

## Dependencies

| requirement | version | notes |
| --- | --- | --- |
| CMake | >= 3.17 (tested 4.3.2) | |
| C++ compiler | C++17 with OpenMP (tested GCC 9.4) | |
| Boost | tested 1.71 | `program_options` only |
| Intel oneAPI MKL | ILP64 interface + `iomp5` | required; BLAS for the k-means / rotation utilities in `quago_core` |
| Python | >= 3.8 (tested 3.8.10) | `numpy` for the KNNG tool, `pandas` for the measurement harness |
| FAISS | optional (tested 1.8.0) | only for `apps/tools/build_knng.py` |

MKL is looked up in `MKL_ROOT` (default `/opt/intel/oneapi/mkl/latest`), then `$MKLROOT`, then the
usual distro paths. Point it at your installation if it lives elsewhere:

```bash
cmake -B build -DMKL_ROOT=/path/to/mkl -DCMAKE_BUILD_TYPE=Release
```

## Build

```bash
bash scripts/build.sh          # Release into build/, binaries in build/apps/
```

or directly:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`-DQUAGO_NATIVE_BUILD=OFF` drops `-march=native` for portable binaries. Use a Release build for any
timing measurement: a Debug build is ~2.7x slower at identical distance-computation counts.

## Usage

Script templates in `scripts/`, one per stage. They carry no dataset paths: every input is supplied by
environment variable (or edit `scripts/common.sh`).

| script | stage |
| --- | --- |
| `scripts/build.sh` | build the project, Release into `build/` |
| `scripts/build_index.sh` | score/select the region, enhance it, write the delta |
| `scripts/search.sh` | fixed-L search, base graph only then base graph + delta |
| `scripts/search_fast.sh` | early-stop search on the enhanced graph |

```bash
bash scripts/build.sh

export BASE_DATA=/data/base.fbin
export BASE_INDEX=/data/hnsw.index BASE_FORMAT=hnsw
export KNNG=/data/knng.graph
export QUERY=/data/query.fbin GT=/data/gt.ibin
export DIST=ip M_NHOOD=96 RATIO=0.20 KNN_SIZE=20

bash scripts/build_index.sh
bash scripts/search.sh
bash scripts/search_fast.sh
```

A missing input fails immediately and names the variable to set. Deltas and CSVs go to `results/`
(git-ignored) unless `DELTA_DIR` / `OUT_DIR` say otherwise. `search.sh` runs the base graph first and
the enhanced graph second, so the two curves are directly comparable; `FORCE=1` rebuilds a delta that
already exists.

To produce a KNNG and ground truth with FAISS:

```bash
python3 apps/tools/build_knng.py --base base.fbin --k 200 --dist ip --out knng.graph
python3 apps/tools/build_knng.py --base base.fbin --query query.fbin --k 100 --dist ip --out gt.ibin
```

Or call the binaries directly — every one documents its arguments with `--help`:

```bash
build/apps/quago_build \
    --data_type float --dist ip \
    --base_data_path base.fbin \
    --base_index_path hnsw.index --base_index_format hnsw \
    --knng_path knng.graph \
    --M_nhood 96 --quago_strategy frontier --enhance_method qmrng \
    --knn_size 20 --quago_ratio 0.20 --sel_dirs 8192 \
    --skeleton_search_L 100 --skeleton_alpha 1.2 \
    --quago_index_save_dir ./out -T 32

build/apps/quago_search \
    --data_type float --dist ip \
    --base_data_path base.fbin --query_path query.fbin --gt_path gt.ibin \
    --base_index_path hnsw.index --base_index_format hnsw \
    --enhancement_delta_path ./out/frontier_qmrng_r0.20.delta \
    --L_pq 100 200 400 800 --k 10 100 \
    --method_name qmrng --eval_path ./results -T 1
```

Omit `--enhancement_delta_path` to measure the unenhanced base graph.

### Template variables

| variable | default | meaning |
| --- | --- | --- |
| `BASE_DATA` / `QUERY` / `GT` | unset, required | base vectors, queries, ground truth |
| `BASE_INDEX` / `BASE_FORMAT` | unset / `hnsw` | base graph and its format (`hnsw`, `nsg`, `quago`) |
| `KNNG` | unset, required by `build_index.sh` | k-NN graph used for scoring and enhancement |
| `DELTA_DIR` / `DELTA` | `results/` / derived | where the delta is written / read |
| `OUT_DIR` | `results/` | search CSV output |
| `QUAGO_BIN_DIR` | `build/apps` | directory holding the binaries |
| `DIST` / `DATA_TYPE` | `l2` / `float` | metric (`l2`, `ip`, `cosine`) and vector type |
| `STRATEGY` | `frontier` | region selector: `frontier`, `random`, or `none` to skip |
| `METHOD` | `qmrng` | enhancement: `qmrng` or `knn` |
| `RATIO` | `0.20` | fraction of points selected as the region |
| `M_NHOOD` | `48` | region degree budget (the skeleton's R) |
| `KNN_SIZE` | `20` | neighborhood size for scoring and the MST overlay; must be <= the KNNG's k |
| `SEL_DIRS` | `8192` | number of spherical-code direction probes |
| `SKELETON_L` / `SKELETON_ALPHA` | `100` / `1.2` | skeleton beam width and RobustPrune slack |
| `BUILD_THREADS` / `SEARCH_THREADS` | `32` / `1` | thread counts |
| `L_LIST` / `K_LIST` | `100 200 400` / `10` | search budget grid and k values |
| `THETA` / `PATIENCE` / `MIN_HOPS` | `1.20` / `50` / `20` | early-stop knobs |
| `FRONTIER` | `sorted` | `quago_search` frontier structure: `sorted` or `heap` |
| `FORCE` | `0` | `1` rebuilds even if the delta exists |

### Choosing parameters

- `M_NHOOD` must exceed the base graph's average degree, otherwise the rebuilt region has no room and
  the delta comes out near-empty.
- `RATIO` trades coverage against build cost and graph size; 0.1–0.4 is the useful range.
- `THETA` and `PATIENCE` are **not portable between datasets**: theta tracks distance concentration and
  patience tracks top-k stall behavior. The defaults are a starting point — fit them on your own data
  at your target recall. `PATIENCE` only takes effect at L >= 3000.
- `--enhance_method knn` adds nothing when the base graph already contains the region's k-NN edges.

## Binaries

| binary | purpose |
| --- | --- |
| `quago_build` | load base graph + KNNG, score/select the region, enhance, write the delta |
| `quago_search` | fixed-L search; `--frontier sorted\|heap` selects the frontier structure |
| `quago_search_fast` | early stop: frontier ratio test (`--theta`) plus a stalled-top-k floor (`--patience`) that joins it at L >= 3000 |
| `quago_search_frontier` | early stop with the ratio test only (`--theta`, no patience floor) |
| `file_io_check` | smoke-test the fbin / ibin loaders |
| `kmeans_run` | standalone k-means++ utility (MKL-backed); not part of the enhancement pipeline |

Every binary documents its own arguments with `--help`.

## Input formats

| file | layout |
| --- | --- |
| base / query vectors (`.fbin`) | `[n:u32][dim:u32][n*dim f32]` |
| ground truth (`.ibin`) | `[n:u32][k:u32][n*k u32 ids][n*k f32 dists]` |
| KNNG (ids-only) | `[k:u32]`, then per point `[self_id:u32][k u32 neighbors]` |
| base graph `hnsw` | hnswlib level-0 binary; internal ids are remapped through the stored labels |
| base graph `nsg` | `[u32 width][u32 ep]`, then per node `[u32 deg][deg u32]` |
| base graph `quago` | `[u64 edges][u32 N][u32 ep]`, then per node `[u32 deg][deg u32]` |
| enhancement delta | `[u32 magic 0x51474F44][u32 mode][u32 N]`, then per node `[u32 deg][deg u32]` |

`quago_build` names the delta itself: `<strategy>_<method>_r<ratio with 2 decimals>.delta`, written
into `--quago_index_save_dir`. A delta is only checked against the base graph's node count, so pair it
with the base graph it was built from.

For IP / cosine, base and query vectors are L2-normalized on load.

## Measurement harness

`scripts/env.sh` provides a single-thread QPS protocol for reproducible curves: warmup passes plus
`QUAGO_REPEATS` passes over the L grid in one app invocation, a host-wide measurement lock, and a
load-average gate. `scripts/lib/aggregate_repeats.py` keeps the fastest pass per L and reports whether
any pass was clean. Source `env.sh` from your own runner and call `measure`; see the variables at the
top of the file for the knobs.

## Layout

```
include/quago/      public headers (core utilities + IndexQuago + early stopping)
src/core/           distance functors, file IO, neighbor queue, math utilities
src/quago/          IndexQuago: loaders, scoring, enhancement, search
apps/               CLI drivers and standalone tools
scripts/            build / build_index / search / search_fast templates, env.sh measurement harness
```