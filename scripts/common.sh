#!/bin/bash
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${QUAGO_BIN_DIR:-$PROJ_ROOT/build/apps}"

BASE_DATA="${BASE_DATA:-}"
BASE_INDEX="${BASE_INDEX:-}"
BASE_FORMAT="${BASE_FORMAT:-hnsw}"
KNNG="${KNNG:-}"
QUERY="${QUERY:-}"
GT="${GT:-}"
DELTA_DIR="${DELTA_DIR:-$PROJ_ROOT/results}"
OUT_DIR="${OUT_DIR:-$PROJ_ROOT/results}"

DIST="${DIST:-l2}"
DATA_TYPE="${DATA_TYPE:-float}"
STRATEGY="${STRATEGY:-frontier}"
METHOD="${METHOD:-qmrng}"
RATIO="${RATIO:-0.20}"
M_NHOOD="${M_NHOOD:-48}"
KNN_SIZE="${KNN_SIZE:-20}"
SEL_DIRS="${SEL_DIRS:-8192}"
SKELETON_L="${SKELETON_L:-100}"
SKELETON_ALPHA="${SKELETON_ALPHA:-1.2}"

DELTA="${DELTA:-$DELTA_DIR/${STRATEGY}_${METHOD}_r$(printf '%.2f' "$RATIO").delta}"

BUILD_THREADS="${BUILD_THREADS:-32}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
K_LIST="${K_LIST:-10}"
L_LIST="${L_LIST:-100 200 400}"
THETA="${THETA:-1.20}"
PATIENCE="${PATIENCE:-50}"
MIN_HOPS="${MIN_HOPS:-20}"
FORCE="${FORCE:-0}"

require_var() {
    if [[ -z "${!1}" ]]; then
        echo "[FATAL] $1 is not set ($2)" >&2
        echo "        export it, or edit scripts/common.sh" >&2
        exit 1
    fi
}

check_bin() {
    if [[ ! -x "$BIN/$1" ]]; then
        echo "[FATAL] $BIN/$1 not found. Build first: bash scripts/build.sh" >&2
        exit 1
    fi
}

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "[FATAL] file not found: $1" >&2
        exit 1
    fi
}