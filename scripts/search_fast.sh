#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var BASE_DATA "base vectors, fbin"
require_var QUERY "query vectors, fbin"
require_var GT "ground truth, ibin"
require_var BASE_INDEX "base graph file"
check_bin quago_search_fast
check_file "$BASE_DATA"
check_file "$QUERY"
check_file "$GT"
check_file "$BASE_INDEX"
check_file "$DELTA"

mkdir -p "$OUT_DIR"
for k in $K_LIST; do
    rm -f "$OUT_DIR/${METHOD}_fast_k${k}.csv"
done

echo "[search_fast] theta=$THETA patience=$PATIENCE min_hops=$MIN_HOPS  (L=$L_LIST, k=$K_LIST)"
"$BIN/quago_search_fast" \
    --data_type "$DATA_TYPE" --dist "$DIST" \
    --base_data_path "$BASE_DATA" --query_path "$QUERY" --gt_path "$GT" \
    --base_index_path "$BASE_INDEX" --base_index_format "$BASE_FORMAT" \
    --enhancement_delta_path "$DELTA" \
    --theta "$THETA" --patience "$PATIENCE" --min_hops "$MIN_HOPS" \
    --L_pq $L_LIST --k $K_LIST \
    --method_name "${METHOD}_fast" --eval_path "$OUT_DIR" -T "$SEARCH_THREADS"

echo "[search_fast] CSVs in $OUT_DIR"