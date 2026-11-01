#!/bin/bash
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${QUAGO_RESULTS_ROOT:=$PROJ_ROOT/results}"
RESULTS_ROOT="$QUAGO_RESULTS_ROOT"

: "${QUAGO_REPEATS:=3}"
: "${QUAGO_WARMUP_N:=3}"
: "${QUAGO_WARMUP_L:=200}"
: "${QUAGO_MAX_LOAD:=8}"
: "${QUAGO_LOAD_TIMEOUT:=21600}"
: "${QUAGO_MEASURE_LOCK:=/tmp/quago_measure.lock}"
: "${QUAGO_BIN_DIR:=}"

FORCE="${FORCE:-0}"

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "[FATAL] file not found: $1" >&2
        exit 1
    fi
}

build_type_of() {
    local cache; cache="$(dirname "$1")/CMakeCache.txt"
    [[ -f "$cache" ]] || { echo "unknown"; return 0; }
    sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$cache" | head -1
}

is_release_build() {
    case "$(build_type_of "$1")" in
        Release|RelWithDebInfo|MinSizeRel) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_bin() {
    if [[ -n "$QUAGO_BIN_DIR" ]]; then
        if [[ -x "$QUAGO_BIN_DIR/$1" ]]; then echo "$QUAGO_BIN_DIR/$1"; return 0; fi
        echo "[FATAL] QUAGO_BIN_DIR=$QUAGO_BIN_DIR has no '$1'" >&2
        return 1
    fi
    local d="$PROJ_ROOT/build/apps"
    if [[ -x "$d/$1" ]]; then
        if is_release_build "$d"; then echo "$d/$1"; return 0; fi
        echo "[warn] skipping $d/$1: CMAKE_BUILD_TYPE=$(build_type_of "$d") is not Release" >&2
    fi
    echo "[FATAL] no Release build provides '$1'. Build one with:" >&2
    echo "        bash scripts/build.sh" >&2
    return 1
}

parse_force() {
    local a
    for a in "$@"; do
        case "$a" in
            --force) FORCE=1 ;;
            *) echo "usage: $0 [--force]" >&2; exit 2 ;;
        esac
    done
}

require_delta() {
    if [[ ! -s "$DELTA" ]]; then
        echo "[FATAL] enhancement delta missing: $DELTA" >&2
        echo "        build it first with quago_build (see scripts/build_index.sh)" >&2
        exit 1
    fi
}

base_stem() { local b; b="$(basename "$1")"; echo "${b%.index}"; }
scheme_of() { local b; b="$(basename "$1")"; echo "${b%.delta}"; }

acquire_measure_lock() {
    while ! mkdir "$QUAGO_MEASURE_LOCK" 2>/dev/null; do
        local owner; owner="$(cat "$QUAGO_MEASURE_LOCK/pid" 2>/dev/null || echo '')"
        if [[ -z "$owner" ]] || ! kill -0 "$owner" 2>/dev/null; then
            echo "[lock] reclaiming stale lock (owner '${owner:-none}' is dead)"
            rm -rf "$QUAGO_MEASURE_LOCK"; continue
        fi
        if [[ "$owner" == "$$" || "$owner" == "$PPID" ]]; then return 0; fi
        echo "[lock] pid $owner holds the measurement lock, waiting 60s"
        sleep 60
    done
    echo $$ > "$QUAGO_MEASURE_LOCK/pid"
    echo "[lock] acquired ($QUAGO_MEASURE_LOCK, pid $$)"
    trap 'rm -rf "$QUAGO_MEASURE_LOCK"' EXIT
}

wait_for_quiet() {
    local waited=0 la
    while :; do
        la="$(cut -d' ' -f1 /proc/loadavg)"
        if awk -v l="$la" -v m="$QUAGO_MAX_LOAD" 'BEGIN{exit !(l+0 < m+0)}'; then
            if [[ "$waited" -gt 0 ]]; then
                echo "[load] quiet now (1-min avg $la < $QUAGO_MAX_LOAD) after ${waited}s"
            fi
            return 0
        fi
        if [[ "$waited" -ge "$QUAGO_LOAD_TIMEOUT" ]]; then
            echo "[FATAL] 1-min load average $la still >= $QUAGO_MAX_LOAD after ${waited}s;" \
                 "refusing to record single-thread QPS on a contended host" >&2
            exit 1
        fi
        if [[ "$waited" -eq 0 ]]; then
            echo "[load] 1-min avg $la >= $QUAGO_MAX_LOAD -- waiting for a quiet window" \
                 "(timeout ${QUAGO_LOAD_TIMEOUT}s)"
        fi
        sleep 60
        waited=$((waited + 60))
    done
}

measure() {
    local app="$1" out_dir="$2" mn="$3" logf="$4" grid="$5"; shift 5
    if [[ "${1:-}" == "--" ]]; then shift; fi

    local bin; bin="$(resolve_bin "$app")" || return 1

    local kmax=0 k L
    for k in $K_LIST; do if [[ "$k" -gt "$kmax" ]]; then kmax="$k"; fi; done
    for L in $grid; do
        if [[ "$L" -lt "$kmax" ]]; then
            echo "[FATAL] $mn: L=$L < max(k)=$kmax -- raise the L grid or lower QUAGO_K_LIST" >&2
            return 1
        fi
    done
    local warmup_l="$QUAGO_WARMUP_L"
    if [[ "$warmup_l" -lt "$kmax" ]]; then warmup_l="$kmax"; fi

    local csvs=() f
    for k in $K_LIST; do csvs+=("$out_dir/${mn}_k${k}.csv"); done
    if [[ "$FORCE" -eq 0 ]]; then
        local complete=1
        for f in "${csvs[@]}"; do [[ -s "$f" ]] || complete=0; done
        if [[ "$complete" -eq 1 ]]; then
            echo "[skip] $(basename "$out_dir")/${mn}_k{$(echo $K_LIST | tr ' ' ',')}.csv exist" \
                 "(--force to re-measure)"
            return 0
        fi
    fi

    acquire_measure_lock
    wait_for_quiet

    mkdir -p "$out_dir/raw" "$LOG_DIR"
    for f in "${csvs[@]}"; do rm -f "$f"; done
    for k in $K_LIST; do rm -f "$out_dir/raw/${mn}_raw_k${k}.csv"; done

    local L_seq="" i
    for ((i = 0; i < QUAGO_WARMUP_N; i++)); do L_seq="$L_seq $warmup_l"; done
    for ((i = 0; i < QUAGO_REPEATS; i++)); do L_seq="$L_seq $grid"; done

    echo "[measure] $(date '+%F %T') $mn"
    echo "[measure]   bin=$bin  T=$SEARCH_THREADS  k=[$K_LIST]"
    echo "[measure]   warmup=${QUAGO_WARMUP_N}x${warmup_l}  repeats=${QUAGO_REPEATS}"
    echo "[measure]   load:$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "[measure]   out=$out_dir"
    "$bin" "$@" --L_pq $L_seq --k $K_LIST \
        --method_name "${mn}_raw" --eval_path "$out_dir/raw" -T "$SEARCH_THREADS" \
        2>&1 | tee "$logf"

    local raw
    for k in $K_LIST; do
        raw="$out_dir/raw/${mn}_raw_k${k}.csv"
        if [[ ! -s "$raw" ]]; then echo "[FATAL] no raw CSV: $raw" >&2; return 1; fi
        python3 "$PROJ_ROOT/scripts/lib/aggregate_repeats.py" "$raw" "$out_dir/${mn}_k${k}.csv" \
            --drop-warmup "$QUAGO_WARMUP_N" 2>&1 | tee -a "$logf"
    done
    echo "[measure] done: $(printf '%s ' "${csvs[@]}")"
}
