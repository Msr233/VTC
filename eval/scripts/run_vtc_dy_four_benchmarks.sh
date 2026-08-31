#!/bin/bash
# VTC_Dy: DyRSO + DyTA + SC-TopK.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VTC_REPO="${VTC_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
cd "$VTC_REPO"

BENCHMARKS="${BENCHMARKS:-mlvu videomme longvideobench lvbench}"
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-./outputs_vtc_dy}"
RUNS="${RUNS:-64:0.5 64:0.25}"
DYRSO_I_PAIR_STRICT="${DYRSO_I_PAIR_STRICT:-0}"

run_suite() {
    local benchmark="$1"
    local script=""
    local data_prefix=""
    case "$benchmark" in
        videomme) script="eval/scripts/run_videomme.sh"; data_prefix="VIDEOMME" ;;
        longvideobench|lvb-long) script="eval/scripts/run_longvideobench.sh"; data_prefix="LONGVIDEOBENCH" ;;
        mlvu) script="eval/scripts/run_mlvu.sh"; data_prefix="MLVU" ;;
        lvbench) script="eval/scripts/run_lvbench.sh"; data_prefix="LVBENCH" ;;
        *)
            echo "ERROR: unknown benchmark '${benchmark}'"
            return 1
            ;;
    esac
    local video_var="${data_prefix}_VIDEO_DIR"
    local gt_var="${data_prefix}_GT_FILE"
    local video_dir="${!video_var:-${VIDEO_DIR:-}}"
    local gt_file="${!gt_var:-${GT_FILE:-}}"
    if [ -z "$video_dir" ] || [ -z "$gt_file" ]; then
        echo "ERROR: ${benchmark} requires ${video_var} and ${gt_var}."
        return 1
    fi

    echo "=== VTC_Dy ${benchmark}: DyRSO + DyTA + SC-TopK ==="
    VERSION="v5.0.0" \
    FRAME_EXTRACT_VERSION="v1" \
    DYRSO_I_PAIR_MODE="fixed" \
    DYRSO_I_PAIR_STRICT="$DYRSO_I_PAIR_STRICT" \
    MOSAIC_REQUIRE_HOMOGENEOUS_PAIR="1" \
    MOSAIC_STRICT_PAIRING="1" \
    VTC_TOKEN_ALLOCATION_MODE="dyta" \
    SEMANTIC_CODEC_BITCOST_ENABLED="1" \
    VTC_I_FRAME_COMPRESSION="${VTC_I_FRAME_COMPRESSION:-lightweight}" \
    VTC_TOKEN_SELECTION_MODE="${VTC_TOKEN_SELECTION_MODE:-sc_topk}" \
    VIDEO_DIR="$video_dir" \
    GT_FILE="$gt_file" \
    OUTPUT_BASE="${OUTPUT_BASE_ROOT}/${benchmark}" \
    RUNS="$RUNS" \
    bash "$script"
}

failed=0
for benchmark in $BENCHMARKS; do
    run_suite "$benchmark" || failed=1
done

echo "Results directory: ${OUTPUT_BASE_ROOT}/"
exit "$failed"
