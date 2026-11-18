#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run_benchmark.sh videomme vtcdy 0.5 [extra evaluation options]
benchmark="${1:?Specify videomme, mlvu, lvbench, or longvideobench}"
method="${2:-vtcdy}"
ratio="${3:-0.5}"
shift "$(( $# >= 3 ? 3 : $# ))"
case "$benchmark" in
    videomme|mlvu|lvbench|longvideobench) ;;
    *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
: "${VIDEO_DIR:?Set VIDEO_DIR to this benchmark's video directory}"
: "${GT_FILE:?Set GT_FILE to this benchmark's JSON annotation file}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
out="${OUTPUT_DIR:-outputs/${benchmark}/${method}_r${ratio}}"
mkdir -p "$out"
"${PYTHON:-python}" -m "evaluation.${benchmark}" \
    --model-path "${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}" \
    --video-dir "$VIDEO_DIR" --gt-file "$GT_FILE" --output-dir "$out" \
    --method "$method" --retention-ratio "$ratio" \
    --frames "${FRAMES:-64}" --alpha "${ALPHA:-0.5}" \
    --vit-chunk-frames "${VIT_CHUNK_FRAMES:-16}" \
    --codec-cache-dir "${CODEC_CACHE_DIR:-outputs/codec_cache}" \
    "$@" 2>&1 | tee -a "$out/run.log"
