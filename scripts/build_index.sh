#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var BASE_DATA "base vectors, fbin"
require_var BASE_INDEX "base graph file"
require_var KNNG "k-NN graph, ids-only"
check_bin quago_build
check_file "$BASE_DATA"
check_file "$BASE_INDEX"
check_file "$KNNG"

if [[ -s "$DELTA" && "$FORCE" -eq 0 ]]; then
    echo "[build_index] skip (exists): $DELTA"
    exit 0
fi

mkdir -p "$DELTA_DIR"
echo "[build_index] strategy=$STRATEGY method=$METHOD ratio=$RATIO M_nhood=$M_NHOOD knn_size=$KNN_SIZE"
echo "[build_index] skeleton_L=$SKELETON_L alpha=$SKELETON_ALPHA sel_dirs=$SEL_DIRS threads=$BUILD_THREADS"
echo "[build_index] out=$DELTA"

"$BIN/quago_build" \
    --data_type "$DATA_TYPE" --dist "$DIST" \
    --base_data_path "$BASE_DATA" \
    --base_index_path "$BASE_INDEX" --base_index_format "$BASE_FORMAT" \
    --knng_path "$KNNG" \
    --M_nhood "$M_NHOOD" \
    --quago_strategy "$STRATEGY" --enhance_method "$METHOD" \
    --knn_size "$KNN_SIZE" --quago_ratio "$RATIO" \
    --sel_dirs "$SEL_DIRS" \
    --skeleton_search_L "$SKELETON_L" --skeleton_alpha "$SKELETON_ALPHA" \
    --quago_index_save_dir "$DELTA_DIR" -T "$BUILD_THREADS"

check_file "$DELTA"
echo "[build_index] wrote $DELTA ($(du -h "$DELTA" | cut -f1))"