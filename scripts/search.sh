#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_var BASE_DATA "base vectors, fbin"
require_var QUERY "query vectors, fbin"
require_var GT "ground truth, ibin"
require_var BASE_INDEX "base graph file"
check_bin quago_search
check_file "$BASE_DATA"
check_file "$QUERY"
check_file "$GT"
check_file "$BASE_INDEX"
check_file "$DELTA"

mkdir -p "$OUT_DIR"
for k in $K_LIST; do
    rm -f "$OUT_DIR/base_k${k}.csv" "$OUT_DIR/${METHOD}_k${k}.csv"
done

echo "[search] base graph only  (L=$L_LIST, k=$K_LIST, T=$SEARCH_THREADS)"
"$BIN/quago_search" \
    --data_type "$DATA_TYPE" --dist "$DIST" \
    --base_data_path "$BASE_DATA" --query_path "$QUERY" --gt_path "$GT" \
    --base_index_path "$BASE_INDEX" --base_index_format "$BASE_FORMAT" \
    --frontier "${FRONTIER:-sorted}" \
    --L_pq $L_LIST --k $K_LIST \
    --method_name base --eval_path "$OUT_DIR" -T "$SEARCH_THREADS"

echo "[search] base graph + $DELTA"
"$BIN/quago_search" \
    --data_type "$DATA_TYPE" --dist "$DIST" \
    --base_data_path "$BASE_DATA" --query_path "$QUERY" --gt_path "$GT" \
    --base_index_path "$BASE_INDEX" --base_index_format "$BASE_FORMAT" \
    --enhancement_delta_path "$DELTA" \
    --frontier "${FRONTIER:-sorted}" \
    --L_pq $L_LIST --k $K_LIST \
    --method_name "$METHOD" --eval_path "$OUT_DIR" -T "$SEARCH_THREADS"

echo "[search] CSVs in $OUT_DIR"