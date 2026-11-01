#!/bin/bash

set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-$(nproc)}"

echo "[build] $(date '+%F %T') configuring $PROJ_ROOT/build as Release (-j $JOBS)"
cmake -B "$PROJ_ROOT/build" -S "$PROJ_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJ_ROOT/build" -j "$JOBS"
echo "[build] $(date '+%F %T') done: CMAKE_BUILD_TYPE=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' \
    "$PROJ_ROOT/build/CMakeCache.txt" | head -1)"
ls -l "$PROJ_ROOT/build/apps" | grep -E 'quago_(build|search)' || true
