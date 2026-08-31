#!/bin/bash
# LVBench semantic-pruning/baseline streaming chunked-ViT tests.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VTC_REPO="${VTC_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
cd "$VTC_REPO"
export PYTHONPATH="${VTC_REPO}:${VTC_REPO}/eval:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
VIDEO_DIR="${VIDEO_DIR:-}"
GT_FILE="${GT_FILE:-}"
OUTPUT_BASE="${OUTPUT_BASE:-./outputs_vtc_dy/lvbench}"

VERSION="${VERSION:-v5.0.0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
if [ "$VERSION" != "v5.0.0" ]; then
    echo "ERROR: this submission package only supports VERSION=v5.0.0"
    exit 1
fi
if [ -z "$VIDEO_DIR" ] || [ -z "$GT_FILE" ]; then
    echo "ERROR: set VIDEO_DIR and GT_FILE for LVBench"
    exit 1
fi
VIT_CHUNK_FRAMES="${VIT_CHUNK_FRAMES:-16}"
VIT_CHUNK_STAGING_DEVICE="${VIT_CHUNK_STAGING_DEVICE:-cuda}"
VIT_CHUNK_COLLECT_STATS="${VIT_CHUNK_COLLECT_STATS:-1}"
MAX_PIXELS="${MAX_PIXELS:-0}"
FRAME_EXTRACT_VERSION="${FRAME_EXTRACT_VERSION:-v1}"
MOSAIC_SOURCE_FRAMES="${MOSAIC_SOURCE_FRAMES:-8}"
MOSAIC_READ_BATCH_SIZE="${MOSAIC_READ_BATCH_SIZE:-256}"
MOSAIC_PACKING_ORDER="${MOSAIC_PACKING_ORDER:-}"
MOSAIC_PAIR_PACK_ORDER="${MOSAIC_PAIR_PACK_ORDER:-sequential}"
MOSAIC_REQUIRE_HOMOGENEOUS_PAIR="${MOSAIC_REQUIRE_HOMOGENEOUS_PAIR:-1}"
MOSAIC_STRICT_PAIRING="${MOSAIC_STRICT_PAIRING:-1}"
MOSAIC_DUMP_PAIR_MAPPING="${MOSAIC_DUMP_PAIR_MAPPING:-0}"
DYRSO_I_PAIR_MODE="${DYRSO_I_PAIR_MODE:-fixed}"
DYRSO_HEVC_PAIR_MAX_WEIGHT="${DYRSO_HEVC_PAIR_MAX_WEIGHT:-0.3}"
DYRSO_I_PAIR_CANDIDATES="${DYRSO_I_PAIR_CANDIDATES:-}"
DYRSO_I_PAIR_STRICT="${DYRSO_I_PAIR_STRICT:-1}"
DYRSO_DUMP_I_PAIR_SELECTION="${DYRSO_DUMP_I_PAIR_SELECTION:-0}"

VTC_P_FRAME_MIN_TOKENS="${VTC_P_FRAME_MIN_TOKENS:-8}"
VTC_SEMANTIC_I_THRESHOLD="${VTC_SEMANTIC_I_THRESHOLD:-2.0}"
SEMANTIC_WINDOW_SIZE="${SEMANTIC_WINDOW_SIZE:-8}"
SEMANTIC_MAX_IFRAMES="${SEMANTIC_MAX_IFRAMES:--1}"
SEMANTIC_MIN_IFRAME_GAP="${SEMANTIC_MIN_IFRAME_GAP:--1}"
SEMANTIC_BUDGET_INFEASIBLE_POLICY="${SEMANTIC_BUDGET_INFEASIBLE_POLICY:-adjust_p_min}"
SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA="${SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA:-1}"
VTC_TOKEN_ALLOCATION_MODE="${VTC_TOKEN_ALLOCATION_MODE:-dyta}"
SEMANTIC_CODEC_BITCOST_ENABLED="${SEMANTIC_CODEC_BITCOST_ENABLED:-1}"
SEMANTIC_CODEC_CACHE_DIR="${SEMANTIC_CODEC_CACHE_DIR:-${VTC_REPO}/outputs_codec_cache_v2}"
SEMANTIC_CODEC_FFPROBE_PATH="${SEMANTIC_CODEC_FFPROBE_PATH:-ffprobe}"
SEMANTIC_CODEC_STRICT="${SEMANTIC_CODEC_STRICT:-0}"
SEMANTIC_CODEC_AGGREGATION="${SEMANTIC_CODEC_AGGREGATION:-mean_p90}"
SEMANTIC_CODEC_MEAN_WEIGHT="${SEMANTIC_CODEC_MEAN_WEIGHT:-0.5}"
SEMANTIC_CODEC_P90_WEIGHT="${SEMANTIC_CODEC_P90_WEIGHT:-0.5}"
SEMANTIC_CODEC_NORMALIZATION="${SEMANTIC_CODEC_NORMALIZATION:-robust_pb}"
SEMANTIC_CODEC_I_POLICY="${SEMANTIC_CODEC_I_POLICY:-exclude}"
SEMANTIC_CODEC_Z_CLIP="${SEMANTIC_CODEC_Z_CLIP:-5.0}"
SEMANTIC_CODEC_BUDGET_POWER="${SEMANTIC_CODEC_BUDGET_POWER:-0.5}"
SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER="${SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER:-2.0}"
VTC_I_FRAME_COMPRESSION="${VTC_I_FRAME_COMPRESSION:-lightweight}"
VTC_I_FRAME_RETENTION_RATIO="${VTC_I_FRAME_RETENTION_RATIO:-0.9}"
SEMANTIC_FIRST_IFRAME_KEEP_RATIO="${SEMANTIC_FIRST_IFRAME_KEEP_RATIO:-}"
SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO="${SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO:-}"
SEMANTIC_IFRAME_MIN_KEEP_RATIO="${SEMANTIC_IFRAME_MIN_KEEP_RATIO:-0.8}"
SEMANTIC_IFRAME_ANCHOR_RATIO="${SEMANTIC_IFRAME_ANCHOR_RATIO:-0.2}"
SEMANTIC_IFRAME_SELECTION_MODE="${SEMANTIC_IFRAME_SELECTION_MODE:-spatial_anchor_salient}"
SEMANTIC_IFRAME_ANCHOR_GRID="${SEMANTIC_IFRAME_ANCHOR_GRID:-4x6}"
VTC_I_FRAME_RESIDUAL_WEIGHT="${VTC_I_FRAME_RESIDUAL_WEIGHT:-0.5}"
SEMANTIC_IFRAME_COMPRESS_WHEN_RATIO_ONE="${SEMANTIC_IFRAME_COMPRESS_WHEN_RATIO_ONE:-0}"
VTC_TOKEN_SELECTION_MODE="${VTC_TOKEN_SELECTION_MODE:-sc_topk}"
VTC_REGION_GRID_H="${VTC_REGION_GRID_H:-4}"
VTC_REGION_GRID_W="${VTC_REGION_GRID_W:-4}"
VTC_COVERAGE_RATIO="${VTC_COVERAGE_RATIO:-0.20}"
VTC_REGION_MIN_TOKENS="${VTC_REGION_MIN_TOKENS:-1}"

NUM_CHUNKS="${NUM_CHUNKS:-2}"
GPU_GROUP="${GPU_GROUP:-2,3}"
GPUS_PER_CHUNK="${GPUS_PER_CHUNK:-1}"
LINK_EXTERNAL_CHUNK_MODE="${LINK_EXTERNAL_CHUNK_MODE:-0}"
LINK_EXTERNAL_CHUNK_IDX="${LINK_EXTERNAL_CHUNK_IDX:-${CHUNK_IDX:-${CHUNK_ID:-0}}}"
DEBUG="${DEBUG:-0}"

RUNS="${RUNS:-256:0.12}"

case "$FRAME_EXTRACT_VERSION" in
    v0)
        VIDEO_LOAD_BRANCH="default"
        ;;
    v1|v1.1)
        VIDEO_LOAD_BRANCH="mosaic"
        ;;
    *)
        echo "ERROR: FRAME_EXTRACT_VERSION must be 'v0', 'v1', or 'v1.1', got ${FRAME_EXTRACT_VERSION}"
        exit 1
        ;;
esac

if [ "$NUM_CHUNKS" -le 0 ]; then
    echo "ERROR: NUM_CHUNKS must be > 0, got ${NUM_CHUNKS}"
    exit 1
fi

if [ "$GPUS_PER_CHUNK" -le 0 ]; then
    echo "ERROR: GPUS_PER_CHUNK must be > 0, got ${GPUS_PER_CHUNK}"
    exit 1
fi

if [ "$LINK_EXTERNAL_CHUNK_MODE" = "1" ]; then
    if [ "$LINK_EXTERNAL_CHUNK_IDX" -lt 0 ] || [ "$LINK_EXTERNAL_CHUNK_IDX" -ge "$NUM_CHUNKS" ]; then
        echo "ERROR: LINK_EXTERNAL_CHUNK_IDX must be in [0, $((NUM_CHUNKS - 1))], got ${LINK_EXTERNAL_CHUNK_IDX}"
        exit 1
    fi
fi

GPU_CHUNK_ARRAY=()
if [ "$LINK_EXTERNAL_CHUNK_MODE" = "1" ]; then
    for ((chunk=0; chunk<NUM_CHUNKS; chunk++)); do
        GPU_CHUNK_ARRAY+=("$GPU_GROUP")
    done
elif [[ "$GPU_GROUP" == *";"* ]]; then
    IFS=';' read -ra GPU_CHUNK_ARRAY <<< "$GPU_GROUP"
    for ((idx=0; idx<${#GPU_CHUNK_ARRAY[@]}; idx++)); do
        GPU_CHUNK_ARRAY[$idx]="${GPU_CHUNK_ARRAY[$idx]//[[:space:]]/}"
        if [ -z "${GPU_CHUNK_ARRAY[$idx]}" ]; then
            echo "ERROR: GPU_GROUP contains an empty chunk group: '${GPU_GROUP}'"
            exit 1
        fi
    done
    if [ "$NUM_CHUNKS" -gt "${#GPU_CHUNK_ARRAY[@]}" ]; then
        echo "ERROR: NUM_CHUNKS=${NUM_CHUNKS} requires at least ${NUM_CHUNKS} GPU groups, but GPU_GROUP='${GPU_GROUP}' has ${#GPU_CHUNK_ARRAY[@]}"
        exit 1
    fi
else
    IFS=',' read -ra FLAT_GPU_ARRAY <<< "$GPU_GROUP"
    NUM_GPUS=${#FLAT_GPU_ARRAY[@]}
    REQUIRED_GPUS=$((NUM_CHUNKS * GPUS_PER_CHUNK))
    if [ "$REQUIRED_GPUS" -gt "$NUM_GPUS" ]; then
        echo "ERROR: NUM_CHUNKS=${NUM_CHUNKS} with GPUS_PER_CHUNK=${GPUS_PER_CHUNK} requires ${REQUIRED_GPUS} GPUs, but GPU_GROUP='${GPU_GROUP}' has ${NUM_GPUS}"
        exit 1
    fi
    for ((chunk=0; chunk<NUM_CHUNKS; chunk++)); do
        start=$((chunk * GPUS_PER_CHUNK))
        group="${FLAT_GPU_ARRAY[$start]}"
        for ((offset=1; offset<GPUS_PER_CHUNK; offset++)); do
            group="${group},${FLAT_GPU_ARRAY[$((start + offset))]}"
        done
        GPU_CHUNK_ARRAY+=("$group")
    done
fi

require_codec_bitcost_tool() {
    if [ "$VERSION" = "v4.7.3" ]; then
        :
    elif { [ "$VERSION" != "v4.5" ] && [ "$VERSION" != "v4.5.1" ] && [ "$VERSION" != "v4.7.2" ] && [ "$VERSION" != "v5.0.0" ]; } || [ "$VTC_TOKEN_ALLOCATION_MODE" != "dyta" ] || [ "$SEMANTIC_CODEC_BITCOST_ENABLED" = "0" ]; then
        return 0
    fi

    if [[ "$SEMANTIC_CODEC_FFPROBE_PATH" == */* ]]; then
        if [ -x "$SEMANTIC_CODEC_FFPROBE_PATH" ]; then
            return 0
        fi
    elif command -v "$SEMANTIC_CODEC_FFPROBE_PATH" >/dev/null 2>&1; then
        return 0
    fi

    echo "ERROR: VERSION=${VERSION} requires executable ffprobe for codec metadata."
    echo "       Current SEMANTIC_CODEC_FFPROBE_PATH='${SEMANTIC_CODEC_FFPROBE_PATH}' is not executable or not in PATH."
    echo "       Install ffmpeg/ffprobe or set SEMANTIC_CODEC_FFPROBE_PATH=/abs/path/to/ffprobe."
    echo "       Use VTC_TOKEN_ALLOCATION_MODE=residual_mean for residual-mean allocation without ffprobe, or uniform for equal allocation."
    return 1
}

write_experiment_config() {
    local result_log="$1"
    local benchmark="$2"
    local frames="$3"
    local ratio="$4"
    local mode_dir="$5"
    local run_dir="$6"
    local res_file="$7"
    local vtc_enabled="1"
    if [ "$ratio" = "baseline" ]; then
        vtc_enabled="0"
    fi

    {
        echo "============================================================"
        echo "Experiment Configuration"
        echo "============================================================"
        echo "[Experiment]"
        echo "benchmark=${benchmark}"
        echo "version=${VERSION}"
        echo "frames=${frames}"
        echo "ratio=${ratio}"
        echo "mode_dir=${mode_dir}"
        echo "run_dir=${run_dir}"
        echo "vtc_enabled=${vtc_enabled}"
        echo "model_path=${MODEL_PATH}"
        echo "attn_implementation=${ATTN_IMPLEMENTATION:-n/a}"
        echo
        echo "[Dataset]"
        echo "video_dir=${VIDEO_DIR}"
        echo "gt_file=${GT_FILE}"
        echo "res_file=${res_file}"
        echo
        echo "[Runtime / Chunking]"
        echo "num_chunks=${NUM_CHUNKS}"
        echo "gpu_group=${GPU_GROUP}"
        echo "gpu_chunk_assignment=${GPU_CHUNK_ARRAY[*]}"
        echo "gpus_per_chunk=${GPUS_PER_CHUNK}"
        echo "vit_chunk_frames=${VIT_CHUNK_FRAMES}"
        echo "vit_chunk_staging_device=${VIT_CHUNK_STAGING_DEVICE}"
        echo "vit_chunk_collect_stats=${VIT_CHUNK_COLLECT_STATS}"
        echo "max_pixels=${MAX_PIXELS}"
        echo
        echo "[Frame Extract / DyRSO]"
        echo "video_load_branch=${VIDEO_LOAD_BRANCH}"
        echo "frame_extract_version=${FRAME_EXTRACT_VERSION}"
        echo "mosaic_source_frames=${MOSAIC_SOURCE_FRAMES}"
        echo "mosaic_read_batch_size=${MOSAIC_READ_BATCH_SIZE}"
        echo "mosaic_packing_order=${MOSAIC_PACKING_ORDER:-<version default>}"
        echo "mosaic_pair_pack_order=${MOSAIC_PAIR_PACK_ORDER}"
        echo "mosaic_require_homogeneous_pair=${MOSAIC_REQUIRE_HOMOGENEOUS_PAIR}"
        echo "mosaic_strict_pairing=${MOSAIC_STRICT_PAIRING}"
        echo "dyrso_i_pair_mode=${DYRSO_I_PAIR_MODE}"
        echo "dyrso_hevc_pair_max_weight=${DYRSO_HEVC_PAIR_MAX_WEIGHT}"
        echo "dyrso_i_pair_candidates=${DYRSO_I_PAIR_CANDIDATES:-<auto>}"
        echo "dyrso_i_pair_strict=${DYRSO_I_PAIR_STRICT}"
        echo "dyrso_dump_i_pair_selection=${DYRSO_DUMP_I_PAIR_SELECTION}"
        echo
        echo "[Semantic Core]"
        echo "vtc_semantic_i_threshold=${VTC_SEMANTIC_I_THRESHOLD}"
        echo "semantic_window_size=${SEMANTIC_WINDOW_SIZE}"
        echo "semantic_p_min_keep=${VTC_P_FRAME_MIN_TOKENS}"
        echo "semantic_max_iframes=${SEMANTIC_MAX_IFRAMES}"
        echo "semantic_min_iframe_gap=${SEMANTIC_MIN_IFRAME_GAP}"
        echo "semantic_budget_infeasible_policy=${SEMANTIC_BUDGET_INFEASIBLE_POLICY}"
        echo "semantic_enable_iframe_prefix_quota=${SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA}"
        echo
        echo "[DyTA / P-frame Budget]"
        echo "vtc_token_allocation_mode=${VTC_TOKEN_ALLOCATION_MODE}"
        echo "semantic_codec_bitcost_enabled=${SEMANTIC_CODEC_BITCOST_ENABLED}"
        echo "semantic_codec_cache_dir=${SEMANTIC_CODEC_CACHE_DIR}"
        echo "semantic_codec_ffprobe_path=${SEMANTIC_CODEC_FFPROBE_PATH}"
        echo "semantic_codec_strict=${SEMANTIC_CODEC_STRICT}"
        echo "semantic_codec_aggregation=${SEMANTIC_CODEC_AGGREGATION}"
        echo "semantic_codec_mean_weight=${SEMANTIC_CODEC_MEAN_WEIGHT}"
        echo "semantic_codec_p90_weight=${SEMANTIC_CODEC_P90_WEIGHT}"
        echo "semantic_codec_normalization=${SEMANTIC_CODEC_NORMALIZATION}"
        echo "semantic_codec_i_policy=${SEMANTIC_CODEC_I_POLICY}"
        echo "semantic_codec_z_clip=${SEMANTIC_CODEC_Z_CLIP}"
        echo "semantic_codec_budget_power=${SEMANTIC_CODEC_BUDGET_POWER}"
        echo "semantic_codec_budget_max_multiplier=${SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER}"
        echo
        echo "[dyI / I-frame Compression]"
        echo "vtc_i_frame_compression=${VTC_I_FRAME_COMPRESSION}"
        echo "vtc_i_frame_retention_ratio=${VTC_I_FRAME_RETENTION_RATIO}"
        echo "semantic_first_iframe_keep_ratio=${SEMANTIC_FIRST_IFRAME_KEEP_RATIO:-<default>}"
        echo "semantic_dynamic_iframe_keep_ratio=${SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO:-<default>}"
        echo "semantic_iframe_min_keep_ratio=${SEMANTIC_IFRAME_MIN_KEEP_RATIO}"
        echo "semantic_iframe_anchor_ratio=${SEMANTIC_IFRAME_ANCHOR_RATIO}"
        echo "semantic_iframe_selection_mode=${SEMANTIC_IFRAME_SELECTION_MODE}"
        echo "semantic_iframe_anchor_grid=${SEMANTIC_IFRAME_ANCHOR_GRID}"
        echo "vtc_i_frame_residual_weight=${VTC_I_FRAME_RESIDUAL_WEIGHT}"
        echo "semantic_iframe_compress_when_ratio_one=${SEMANTIC_IFRAME_COMPRESS_WHEN_RATIO_ONE}"
        echo
        echo "[token_spa / Token Selection]"
        echo "vtc_token_selection_mode=${VTC_TOKEN_SELECTION_MODE}"
        echo "vtc_region_grid_h=${VTC_REGION_GRID_H}"
        echo "vtc_region_grid_w=${VTC_REGION_GRID_W}"
        echo "vtc_coverage_ratio=${VTC_COVERAGE_RATIO}"
        echo "vtc_region_min_tokens=${VTC_REGION_MIN_TOKENS}"
        echo
        echo "[Output]"
        echo "output_dir=${out_dir}"
        echo "output_base=${OUTPUT_BASE}"
        echo "result_log=${result_log}"
        echo "============================================================"
        echo "Evaluation Result"
        echo "============================================================"
    } > "$result_log"
}

run_one() {
    local frames="$1"
    local ratio="$2"
    local mp_dir="mp${MAX_PIXELS}"
    if [ "$MAX_PIXELS" = "0" ]; then
        mp_dir="mpdefault"
    fi
    local exp_name="pred"
    local mode_dir="pruned"

    local run_dir="f${frames}_r${ratio}"
    local vtc_dir="/ver${VERSION}_f${frames}_r${ratio}/fev${FRAME_EXTRACT_VERSION}_pair${MOSAIC_PAIR_PACK_ORDER}_ipair${DYRSO_I_PAIR_MODE}_hp${MOSAIC_REQUIRE_HOMOGENEOUS_PAIR}_strict${MOSAIC_STRICT_PAIRING}"
    vtc_dir="${vtc_dir}/pb${VTC_TOKEN_ALLOCATION_MODE}_agg${SEMANTIC_CODEC_AGGREGATION}_norm${SEMANTIC_CODEC_NORMALIZATION}_ip${SEMANTIC_CODEC_I_POLICY}_pow${SEMANTIC_CODEC_BUDGET_POWER}_maxmul${SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER}"
    vtc_dir="${vtc_dir}/iframe_${VTC_I_FRAME_COMPRESSION}_keep${VTC_I_FRAME_RETENTION_RATIO}_min${SEMANTIC_IFRAME_MIN_KEEP_RATIO}_sel${SEMANTIC_IFRAME_SELECTION_MODE}_anchor${SEMANTIC_IFRAME_ANCHOR_RATIO}_${SEMANTIC_IFRAME_ANCHOR_GRID}"
    vtc_dir="${vtc_dir}/toksel_${VTC_TOKEN_SELECTION_MODE}_sg${VTC_REGION_GRID_H}x${VTC_REGION_GRID_W}_cov${VTC_COVERAGE_RATIO}_min${VTC_REGION_MIN_TOKENS}"
    local pruning_args=(
        --enable_vtc
        --semantic_version "$VERSION"
        --vtc_retention_ratio "$ratio"
        --vtc_semantic_i_threshold "$VTC_SEMANTIC_I_THRESHOLD"
        --semantic_window_size "$SEMANTIC_WINDOW_SIZE"
        --semantic_warmup_frames 2
    )

    if [ "$ratio" = "baseline" ]; then
        mode_dir="baseline"
        run_dir="f${frames}"
        pruning_args=(--semantic_version "$VERSION")
    fi

    local out_dir
    if [ "$ratio" = "baseline" ]; then
        out_dir="${OUTPUT_BASE}/${VERSION}/${mode_dir}/${run_dir}/${mp_dir}/chunks${NUM_CHUNKS}"
    else
        out_dir="${OUTPUT_BASE}/${VERSION}/${mode_dir}/${run_dir}/vitc${VIT_CHUNK_FRAMES}_${VIT_CHUNK_STAGING_DEVICE}/pmin${VTC_P_FRAME_MIN_TOKENS}_maxi${SEMANTIC_MAX_IFRAMES}_gap${SEMANTIC_MIN_IFRAME_GAP}_prefix${SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA}${vtc_dir}/${mp_dir}/chunks${NUM_CHUNKS}"
    fi
    mkdir -p "$out_dir"

    local collect_stats_arg=()
    if [ "$VIT_CHUNK_COLLECT_STATS" = "1" ]; then
        collect_stats_arg=(--vit_chunk_collect_stats)
    fi

    local max_pixels_arg=()
    if [ "$MAX_PIXELS" != "0" ]; then
        max_pixels_arg=(--max_pixels "$MAX_PIXELS")
    fi
    local video_load_args=(--frame_extract_version "$FRAME_EXTRACT_VERSION")
    if [ "$FRAME_EXTRACT_VERSION" != "v0" ]; then
        local homogeneous_pair_arg=(--no-mosaic_require_homogeneous_pair)
        if [ "$MOSAIC_REQUIRE_HOMOGENEOUS_PAIR" = "1" ]; then
            homogeneous_pair_arg=(--mosaic_require_homogeneous_pair)
        fi
        local strict_pairing_arg=(--mosaic_strict_pairing)
        if [ "$MOSAIC_STRICT_PAIRING" = "0" ]; then
            strict_pairing_arg=(--no-mosaic_strict_pairing)
        fi
        local dump_pair_mapping_arg=()
        if [ "$MOSAIC_DUMP_PAIR_MAPPING" = "1" ]; then
            dump_pair_mapping_arg=(--mosaic_dump_pair_mapping)
        fi
        local dyrso_i_pair_strict_arg=(--dyrso_i_pair_strict)
        if [ "$DYRSO_I_PAIR_STRICT" = "0" ]; then
            dyrso_i_pair_strict_arg=(--no-dyrso_i_pair_strict)
        fi
        local dyrso_dump_arg=()
        if [ "$DYRSO_DUMP_I_PAIR_SELECTION" = "1" ]; then
            dyrso_dump_arg=(--dyrso_dump_i_pair_selection)
        fi
        local dyrso_candidate_arg=()
        if [ -n "$DYRSO_I_PAIR_CANDIDATES" ]; then
            dyrso_candidate_arg=(--dyrso_i_pair_candidates "$DYRSO_I_PAIR_CANDIDATES")
        fi
        video_load_args+=(
            --mosaic_source_frames "$MOSAIC_SOURCE_FRAMES"
            --mosaic_read_batch_size "$MOSAIC_READ_BATCH_SIZE"
            --mosaic_pair_pack_order "$MOSAIC_PAIR_PACK_ORDER"
            "${homogeneous_pair_arg[@]}"
            "${strict_pairing_arg[@]}"
            "${dump_pair_mapping_arg[@]}"
            --dyrso_i_pair_mode "$DYRSO_I_PAIR_MODE"
            --dyrso_hevc_pair_max_weight "$DYRSO_HEVC_PAIR_MAX_WEIGHT"
            "${dyrso_candidate_arg[@]}"
            "${dyrso_i_pair_strict_arg[@]}"
            "${dyrso_dump_arg[@]}"
        )
        if [ -n "$MOSAIC_PACKING_ORDER" ]; then
            video_load_args+=(--mosaic_packing_order "$MOSAIC_PACKING_ORDER")
        fi
    fi

    local prefix_quota_arg=(--semantic_enable_iframe_prefix_quota)
    if [ "$SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA" = "0" ]; then
        prefix_quota_arg=(--no-semantic_enable_iframe_prefix_quota)
    fi

    local budget_args=(
        --vtc_p_frame_min_tokens "$VTC_P_FRAME_MIN_TOKENS"
        --semantic_max_iframes "$SEMANTIC_MAX_IFRAMES"
        --semantic_min_iframe_gap "$SEMANTIC_MIN_IFRAME_GAP"
        --semantic_budget_infeasible_policy "$SEMANTIC_BUDGET_INFEASIBLE_POLICY"
        "${prefix_quota_arg[@]}"
    )

    local v44_args=()

    local codec_enabled_arg=(--semantic_codec_bitcost_enabled)
    if [ "$SEMANTIC_CODEC_BITCOST_ENABLED" = "0" ]; then
        codec_enabled_arg=(--no-semantic_codec_bitcost_enabled)
    fi
    local codec_strict_arg=()
    if [ "$SEMANTIC_CODEC_STRICT" = "1" ]; then
        codec_strict_arg=(--semantic_codec_strict)
    fi
    local v45_args=()
    if [ "$VERSION" = "v4.5" ] || [ "$VERSION" = "v4.5.1" ] || [ "$VERSION" = "v4.6" ] || [ "$VERSION" = "v4.6.1" ] || [ "$VERSION" = "v4.6.2" ] || [ "$VERSION" = "v4.7.1" ] || [ "$VERSION" = "v4.7.2" ] || [ "$VERSION" = "v4.7.3" ] || [ "$VERSION" = "v5.0.0" ]; then
        v45_args=(
            --vtc_token_allocation_mode "$VTC_TOKEN_ALLOCATION_MODE"
            "${codec_enabled_arg[@]}"
            --semantic_codec_cache_dir "$SEMANTIC_CODEC_CACHE_DIR"
            --semantic_codec_ffprobe_path "$SEMANTIC_CODEC_FFPROBE_PATH"
            "${codec_strict_arg[@]}"
            --semantic_codec_aggregation "$SEMANTIC_CODEC_AGGREGATION"
            --semantic_codec_mean_weight "$SEMANTIC_CODEC_MEAN_WEIGHT"
            --semantic_codec_p90_weight "$SEMANTIC_CODEC_P90_WEIGHT"
            --semantic_codec_normalization "$SEMANTIC_CODEC_NORMALIZATION"
            --semantic_codec_i_policy "$SEMANTIC_CODEC_I_POLICY"
            --semantic_codec_z_clip "$SEMANTIC_CODEC_Z_CLIP"
            --semantic_codec_budget_power "$SEMANTIC_CODEC_BUDGET_POWER"
            --semantic_codec_budget_max_multiplier "$SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER"
        )
    fi

    local first_iframe_ratio_arg=()
    if [ -n "$SEMANTIC_FIRST_IFRAME_KEEP_RATIO" ]; then
        first_iframe_ratio_arg=(--semantic_first_iframe_keep_ratio "$SEMANTIC_FIRST_IFRAME_KEEP_RATIO")
    fi
    local dynamic_iframe_ratio_arg=()
    if [ -n "$SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO" ]; then
        dynamic_iframe_ratio_arg=(--semantic_dynamic_iframe_keep_ratio "$SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO")
    fi
    local iframe_ratio_one_arg=()
    if [ "$SEMANTIC_IFRAME_COMPRESS_WHEN_RATIO_ONE" = "1" ]; then
        iframe_ratio_one_arg=(--semantic_iframe_compress_when_ratio_one)
    fi
    local v451_args=()
    if [ "$VERSION" = "v4.5.1" ] || [ "$VERSION" = "v4.6" ] || [ "$VERSION" = "v4.6.1" ] || [ "$VERSION" = "v4.6.2" ] || [ "$VERSION" = "v4.7.1" ] || [ "$VERSION" = "v4.7.2" ] || [ "$VERSION" = "v4.7.3" ] || [ "$VERSION" = "v5.0.0" ]; then
        v451_args=(
            --vtc_i_frame_compression "$VTC_I_FRAME_COMPRESSION"
            --vtc_i_frame_retention_ratio "$VTC_I_FRAME_RETENTION_RATIO"
            "${first_iframe_ratio_arg[@]}"
            "${dynamic_iframe_ratio_arg[@]}"
            --semantic_iframe_min_keep_ratio "$SEMANTIC_IFRAME_MIN_KEEP_RATIO"
            --semantic_iframe_anchor_ratio "$SEMANTIC_IFRAME_ANCHOR_RATIO"
            --semantic_iframe_selection_mode "$SEMANTIC_IFRAME_SELECTION_MODE"
            --semantic_iframe_anchor_grid "$SEMANTIC_IFRAME_ANCHOR_GRID"
            --vtc_i_frame_residual_weight "$VTC_I_FRAME_RESIDUAL_WEIGHT"
            "${iframe_ratio_one_arg[@]}"
        )
    fi
    local v471_args=()
    if [ "$VERSION" = "v4.7.2" ] || [ "$VERSION" = "v4.7.3" ] || [ "$VERSION" = "v5.0.0" ]; then
        v471_args=(
            --vtc_token_selection_mode "$VTC_TOKEN_SELECTION_MODE"
            --vtc_region_grid_h "$VTC_REGION_GRID_H"
            --vtc_region_grid_w "$VTC_REGION_GRID_W"
            --vtc_coverage_ratio "$VTC_COVERAGE_RATIO"
            --vtc_region_min_tokens "$VTC_REGION_MIN_TOKENS"
        )
    fi

    local vit_chunk_args=(
        --enable_vit_chunking
        --vit_chunk_frames "$VIT_CHUNK_FRAMES"
        --vit_chunk_staging_device "$VIT_CHUNK_STAGING_DEVICE"
        "${collect_stats_arg[@]}"
    )

    if [ "$ratio" = "baseline" ]; then
        budget_args=()
        vit_chunk_args=()
        v44_args=()
        v45_args=()
        v451_args=()
        v471_args=()
    fi

    echo "  [lvbench ${VERSION} ${mode_dir} ${run_dir}] launching ${NUM_CHUNKS} model process(es)"
    echo "    output -> ${out_dir}"

    local chunk_start=0
    local chunk_end="$NUM_CHUNKS"
    if [ "$LINK_EXTERNAL_CHUNK_MODE" = "1" ]; then
        chunk_start="$LINK_EXTERNAL_CHUNK_IDX"
        chunk_end=$((LINK_EXTERNAL_CHUNK_IDX + 1))
    fi

    local pids=()
    local launched_chunks=()
    for ((chunk=chunk_start; chunk<chunk_end; chunk++)); do
        local gpu="${GPU_CHUNK_ARRAY[$chunk]}"
        local multi_gpu_arg=()
        if [[ "$gpu" == *,* ]]; then
            multi_gpu_arg=(--multi_gpu)
        fi
        echo "    chunk ${chunk}/${NUM_CHUNKS} -> CUDA_VISIBLE_DEVICES=${gpu}"
        local debug_arg=()
        if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
            debug_arg=(--debug)
        fi
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" eval/qwen3_lvbench.py \
            --model_path "$MODEL_PATH" \
            --video_dir "$VIDEO_DIR" \
            --gt_file "$GT_FILE" \
            --output_dir "$out_dir" \
            --output_name "$exp_name" \
            "${multi_gpu_arg[@]}" \
            --num_chunks "$NUM_CHUNKS" --chunk_idx "$chunk" \
            --topk "$frames" \
            "${max_pixels_arg[@]}" \
            --attn_implementation "$ATTN_IMPLEMENTATION" \
            "${video_load_args[@]}" \
            "${pruning_args[@]}" \
            "${budget_args[@]}" \
            "${v44_args[@]}" \
            "${v45_args[@]}" \
            "${v451_args[@]}" \
            "${v471_args[@]}" \
            "${vit_chunk_args[@]}" \
            "${debug_arg[@]}" \
            > "${out_dir}/log_c${chunk}.txt" 2>&1 &
        pids+=($!)
        launched_chunks+=("$chunk")
    done

    local all_ok=true
    for ((idx=0; idx<${#pids[@]}; idx++)); do
        local chunk="${launched_chunks[$idx]}"
        wait "${pids[$idx]}"
        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "  [lvbench ${VERSION} ${mode_dir} ${run_dir}] chunk ${chunk} FAILED rc=${rc}; see ${out_dir}/log_c${chunk}.txt"
            all_ok=false
        fi
    done

    if ! $all_ok; then
        echo "  [lvbench ${VERSION} ${mode_dir} ${run_dir}] FAILED"
        return 1
    fi

    if [ "$LINK_EXTERNAL_CHUNK_MODE" = "1" ]; then
        local chunk_file="${out_dir}/${NUM_CHUNKS}_${LINK_EXTERNAL_CHUNK_IDX}.json"
        local done_file="${chunk_file}.done"
        if [ ! -s "$chunk_file" ]; then
            echo "  [${exp_name}] missing or empty external chunk output: ${chunk_file}"
            return 1
        fi
        {
            echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
            echo "chunk_idx=${LINK_EXTERNAL_CHUNK_IDX}"
            echo "num_chunks=${NUM_CHUNKS}"
            echo "chunk_file=${chunk_file}"
        } > "${done_file}.tmp"
        mv "${done_file}.tmp" "$done_file"
        echo "  [${exp_name}] external chunk ${LINK_EXTERNAL_CHUNK_IDX}/${NUM_CHUNKS} complete; wrote ${done_file}; merge deferred to link orchestrator"
        return 0
    fi

    local res_file
    if [ "$NUM_CHUNKS" -eq 1 ]; then
        res_file="${out_dir}/${exp_name}.json"
    else
        local chunk_files=()
        for ((chunk=0; chunk<NUM_CHUNKS; chunk++)); do
            local cf="${out_dir}/${NUM_CHUNKS}_${chunk}.json"
            if [ -f "$cf" ]; then
                chunk_files+=("$cf")
            fi
        done

        if [ "${#chunk_files[@]}" -ne "$NUM_CHUNKS" ]; then
            echo "  [lvbench ${VERSION} ${mode_dir} ${run_dir}] missing chunk outputs: found ${#chunk_files[@]}/${NUM_CHUNKS}"
            return 1
        fi

        res_file="${out_dir}/merged.json"
        cat "${chunk_files[@]}" > "$res_file"
    fi

    local result_log="${out_dir}/result.log"
    write_experiment_config "$result_log" "lvbench" "$frames" "$ratio" "$mode_dir" "$run_dir" "$res_file"
    "$PYTHON" eval/ab_lvbench.py --res_file "$res_file" >> "$result_log" 2>&1
    echo "  [lvbench ${VERSION} ${mode_dir} ${run_dir}] done -> ${out_dir}/result.log"
}

echo "=== LVBench semantic-pruning/baseline streaming chunked-ViT suite ==="
echo "    NUM_CHUNKS=${NUM_CHUNKS}  GPU_GROUP=${GPU_GROUP}"
echo "    GPU chunk assignment=${GPU_CHUNK_ARRAY[*]}"
echo "    RUNS=${RUNS}"
echo "    VIDEO_DIR=${VIDEO_DIR}"
echo "    GT_FILE=${GT_FILE}"
echo "    VIT_CHUNK_FRAMES=${VIT_CHUNK_FRAMES}  VIT_CHUNK_STAGING_DEVICE=${VIT_CHUNK_STAGING_DEVICE}"
echo "    MAX_PIXELS=${MAX_PIXELS}"
echo "    P_MIN=${VTC_P_FRAME_MIN_TOKENS}  MAX_IFRAMES=${SEMANTIC_MAX_IFRAMES}  MIN_GAP=${SEMANTIC_MIN_IFRAME_GAP}"
echo "    PREFIX_QUOTA=${SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA}  INFEASIBLE_POLICY=${SEMANTIC_BUDGET_INFEASIBLE_POLICY}"
echo "    V4.7.2_DYP=${VTC_TOKEN_ALLOCATION_MODE} codec=${SEMANTIC_CODEC_BITCOST_ENABLED} agg=${SEMANTIC_CODEC_AGGREGATION} norm=${SEMANTIC_CODEC_NORMALIZATION} i_policy=${SEMANTIC_CODEC_I_POLICY}"
echo "    V4.7.2_DYI=${VTC_I_FRAME_COMPRESSION} keep=${VTC_I_FRAME_RETENTION_RATIO} first=${SEMANTIC_FIRST_IFRAME_KEEP_RATIO:-default} dynamic=${SEMANTIC_DYNAMIC_IFRAME_KEEP_RATIO:-default} min=${SEMANTIC_IFRAME_MIN_KEEP_RATIO} selection=${SEMANTIC_IFRAME_SELECTION_MODE} anchor=${SEMANTIC_IFRAME_ANCHOR_RATIO}/${SEMANTIC_IFRAME_ANCHOR_GRID}"
echo "    V4.7.2_TOKEN=${VTC_TOKEN_SELECTION_MODE} spatial=${VTC_REGION_GRID_H}x${VTC_REGION_GRID_W}/cov${VTC_COVERAGE_RATIO}/min${VTC_REGION_MIN_TOKENS}"

for spec in $RUNS; do
    ratio="${spec##*:}"
    if [ "$ratio" != "baseline" ]; then
        require_codec_bitcost_tool || exit 1
        break
    fi
done

failed=0
for spec in $RUNS; do
    frames="${spec%%:*}"
    ratio="${spec##*:}"
    run_one "$frames" "$ratio" || failed=1
done

echo "Results directory: ${OUTPUT_BASE}/"
exit "$failed"
