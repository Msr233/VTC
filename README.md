# Visual Token Coding (VTC)

This repository provides a minimal reproduction of the Visual Token Coding
framework on Qwen3-VL, including both the static variant (**VTC**) and the
dynamic variant (**VTC_Dy**).  VTC_Dy combines three components:

- **DyRSO** — Dynamic Resolution input via online Mosaic frame extraction
- **DyTA** — Dynamic Token Allocation guided by codec bit-cost priors
- **SC-TopK** — Spatial Coverage Top-K token selection

Evaluation entry-points are provided for **Video-MME**, **LongVideoBench**,
**MLVU**, and **LVBench**.

---

## 1. Project Structure

```
hevc/
├── eval/
│   ├── scripts/
│   │   ├── run_vtc_dy_four_benchmarks.sh   # high-level launcher (four-benchmark sweep)
│   │   ├── run_videomme.sh                  # per-benchmark launcher: Video-MME
│   │   ├── run_longvideobench.sh            # per-benchmark launcher: LongVideoBench
│   │   ├── run_mlvu.sh                      # per-benchmark launcher: MLVU
│   │   └── run_lvbench.sh                   # per-benchmark launcher: LVBench
│   ├── qwen3_videomme.py                    # inference script: Video-MME
│   ├── qwen3_longvideobench.py              # inference script: LongVideoBench
│   ├── qwen3_mlvu.py                        # inference script: MLVU
│   ├── qwen3_lvbench.py                     # inference script: LVBench
│   ├── ab_videomme.py                       # scoring script: Video-MME
│   ├── ab_lvb.py                            # scoring script: LongVideoBench
│   ├── ab_mlvu.py                           # scoring script: MLVU
│   ├── ab_lvbench.py                        # scoring script: LVBench
│   ├── qwen_vl_utils_new.py                 # video vision-info processing (default branch)
│   ├── qwen_vl_utils_new_2.py               # video vision-info processing (Mosaic branch)
│   ├── vtc_context.py                       # passes video metadata into the VTC pruner
│   ├── vtc_token_stats.py                   # collects per-sample token statistics
│   └── attention_backend.py                 # Flash-Attention-2 / SDPA auto-fallback
├── models/
│   ├── vtc_dy.py                            # VTC / VTC_Dy component wiring (apply_vtc_dy)
│   ├── visual_token_coding.py               # core semantic pruning / I-frame / P-frame logic
│   ├── iframe_budget_allocator.py           # I-frame retention-ratio budget allocation
│   ├── iframe_token_selector.py             # spatial-anchor + salient I-frame token selection
│   ├── codec_bitcost_allocator.py           # codec-guided per-frame budget (DyTA)
│   └── chunked_video_encoder.py             # ViT chunked encoding for long videos
├── utils/
│   ├── hevc_bitcost.py                      # ffprobe-based HEVC metadata extractor & cache
│   └── migrate_codec_cache.py               # atomic cache-format migration tool
├── tests/
│   ├── test_vtc_dy.py                       # core pruning + codec-prior unit tests
│   ├── test_hevc_bitcost_cache.py           # codec metadata cache tests
│   └── test_attention_backend.py            # attention backend resolution tests
├── environment.yml                          # conda environment specification
└── README.md
```

### Script call-chain

```
run_vtc_dy_four_benchmarks.sh
  └─ per benchmark: run_<benchmark>.sh
       └─ qwen3_<benchmark>.py  (inference, per-chunk parallel launch)
            ├─ qwen_vl_utils_new[_2].py  (video loading & Mosaic packing)
            ├─ models/vtc_dy.py          (VTC pruner injection)
            └─ ab_<benchmark>.py        (scoring after merge)
```

---

## 2. Environment

**Requirements:** Linux, NVIDIA GPU with CUDA drivers, Conda, and `ffprobe`
(part of FFmpeg, version 6+ recommended).  At least one GPU capable of loading
Qwen3-VL-8B-Instruct is needed; each data chunk defaults to one visible GPU.

```bash
conda env create -f environment.yml
conda activate vtc
ffprobe -version
python -m pytest -q tests/test_vtc_dy.py
```

The shipped `environment.yml` installs PyTorch for CUDA 11.8 with SDPA
attention (no extra compilation required).  For other CUDA versions, install a
matching PyTorch build first, then the remaining dependencies from
`environment.yml`.

---

## 3. Data Preparation

`MODEL_PATH` may be a Hugging Face model ID or a local directory (default:
`Qwen/Qwen3-VL-8B-Instruct`).  The first use of a model ID requires network
access and sufficient cache space.

The four `*_GT_FILE` variables must point to JSON arrays whose fields match the
formats expected by the corresponding inference scripts:

| Benchmark | Video dir variable | Annotation variable | Key fields in annotation |
|---|---|---|---|
| Video-MME | `VIDEOMME_VIDEO_DIR` | `VIDEOMME_GT_FILE` | `question_id`, `question`, `videoID`, `duration`, `options`, `answer` |
| LongVideoBench | `LONGVIDEOBENCH_VIDEO_DIR` | `LONGVIDEOBENCH_GT_FILE` | `id`, `question`, `candidates`, `correct_choice`, `video_path`, `duration_group` |
| MLVU | `MLVU_VIDEO_DIR` | `MLVU_GT_FILE` | `qid`, `question`, `candidates`, `answer`, `category`, `video` |
| LVBench | `LVBENCH_VIDEO_DIR` | `LVBENCH_GT_FILE` | `key`, `uid`, `question`, `answer`, optional `video_path` |

Video directories should preserve the original dataset-relative paths. MLVU
expects `${MLVU_VIDEO_DIR}/${category}/${video}`; the remaining path-matching
rules are in the `build_questions` / data-loading sections of `eval/qwen3_*.py`.

---

## 4. Quick Start — Single Benchmark

Run Video-MME with 64 frames at 50 % token retention on a single GPU:

```bash
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export VIDEOMME_VIDEO_DIR=/path/to/Video-MME/data
export VIDEOMME_GT_FILE=/path/to/videomme.json

BENCHMARKS=videomme \
RUNS=64:0.5 \
NUM_CHUNKS=1 \
GPU_GROUP=0 \
bash eval/scripts/run_vtc_dy_four_benchmarks.sh
```

Results appear under `outputs_vtc_dy/` by default (per-chunk predictions,
merged JSON, per-chunk logs, experiment config, and scoring output).

---

## 5. Four-Benchmark Sweep

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export VIDEOMME_VIDEO_DIR=/path/to/Video-MME/data
export VIDEOMME_GT_FILE=/path/to/videomme.json
export LONGVIDEOBENCH_VIDEO_DIR=/path/to/LongVideoBench/videos
export LONGVIDEOBENCH_GT_FILE=/path/to/longvideobench.json
export MLVU_VIDEO_DIR=/path/to/MLVU/video
export MLVU_GT_FILE=/path/to/mlvu.json
export LVBENCH_VIDEO_DIR=/path/to/LVBench/all_videos
export LVBENCH_GT_FILE=/path/to/lvbench.json

RUNS="64:0.5 64:0.25" \
NUM_CHUNKS=2 \
GPU_GROUP=0,1 \
bash eval/scripts/run_vtc_dy_four_benchmarks.sh
```

---

## 6. Configuration Reference

### 6.1 High-level launcher (`run_vtc_dy_four_benchmarks.sh`)

| Variable | Default | Description |
|---|---|---|
| `BENCHMARKS` | `mlvu videomme longvideobench lvbench` | Space-delimited benchmark names |
| `RUNS` | `64:0.5 64:0.25` | Space-delimited `frames:retention_ratio` specs |
| `OUTPUT_BASE_ROOT` | `./outputs_vtc_dy` | Output root; a `<benchmark>/` sub-directory is created per benchmark |
| `DYRSO_I_PAIR_STRICT` | `0` | Set to `1` to fail on missing codec metadata instead of falling back |
| `VTC_I_FRAME_COMPRESSION` | `lightweight` | I-frame compression mode |
| `VTC_TOKEN_SELECTION_MODE` | `sc_topk` | P-frame token selection mode |

### 6.2 Per-benchmark launcher (`run_<benchmark>.sh`)

These scripts also accept direct invocation (they use generic `VIDEO_DIR` and
`GT_FILE` variables) and expose every tunable parameter.

**Core runtime**

| Variable | Default | Description |
|---|---|---|
| `PYTHON` | `python3` | Python interpreter path |
| `MODEL_PATH` | `Qwen/Qwen3-VL-8B-Instruct` | Model ID or local path |
| `ATTN_IMPLEMENTATION` | `flash_attention_2` | Attention backend; auto-falls back to `sdpa` |
| `MAX_PIXELS` | `0` | Per-frame pixel cap; `0` keeps the default |

**Chunking & GPU assignment**

| Variable | Default | Description |
|---|---|---|
| `NUM_CHUNKS` | `2` | Number of data chunks to parallelize across |
| `GPU_GROUP` | `2,3` | Comma-separated GPU IDs, or semicolon-delimited per-chunk groups |
| `GPUS_PER_CHUNK` | `1` | GPUs assigned to each chunk |
| `VIT_CHUNK_FRAMES` | `16` | Raw video frames per ViT temporal chunk |
| `VIT_CHUNK_STAGING_DEVICE` | `cuda` | Device for staging per-chunk ViT features (`cuda` or `cpu`) |
| `VIT_CHUNK_COLLECT_STATS` | `1` | Collect per-chunk timing and CUDA peak-memory stats |

**Frame extraction / DyRSO**

| Variable | Default | Description |
|---|---|---|
| `FRAME_EXTRACT_VERSION` | `v1` | `v0`: uniform sampling; `v1`/`v1.1`: online DyRSO Mosaic extraction |
| `MOSAIC_SOURCE_FRAMES` | `8` | Source frames sampled per Mosaic group |
| `MOSAIC_READ_BATCH_SIZE` | `256` | Distinct frame indices per decord `get_batch` call |
| `MOSAIC_PACKING_ORDER` | *empty* | Final input order override — `qwen3_2i2p` or `miniset_1i1p` |
| `MOSAIC_PAIR_PACK_ORDER` | `sequential` | Mosaic pair packing: `sequential` or `pair_aligned` |
| `MOSAIC_REQUIRE_HOMOGENEOUS_PAIR` | `1` | Require every temporal pair to be full+full or mosaic+mosaic |
| `MOSAIC_STRICT_PAIRING` | `1` | Raise on mismatched mosaic tail or tile-pair misalignment |
| `DYRSO_I_PAIR_MODE` | `fixed` | Full-resolution I-I pair selection: `fixed`, `hevc_bitcost`, or `hevc_tile_sort` |
| `DYRSO_HEVC_PAIR_MAX_WEIGHT` | `0.3` | Weight for max bit-cost in HEVC pair scoring |
| `DYRSO_I_PAIR_STRICT` | `1` | Fail on HEVC pair-selection errors (set to `0` for fallback) |

**Semantic pruning core**

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_WINDOW_SIZE` | `8` | Rolling window for keyframe detection |
| `SEMANTIC_WARMUP_FRAMES` | `2` | Frames before keyframe detection activates |
| `VTC_SEMANTIC_I_THRESHOLD` | `2.0` | Similarity threshold for semantic I-frame insertion |
| `VTC_P_FRAME_MIN_TOKENS` | `8` | Minimum tokens retained per P-frame |
| `SEMANTIC_MAX_IFRAMES` | `-1` | Max I-frames (`-1` = no cap) |
| `SEMANTIC_MIN_IFRAME_GAP` | `-1` | Minimum frame gap between I-frames (`-1` = auto) |
| `SEMANTIC_BUDGET_INFEASIBLE_POLICY` | `adjust_p_min` | Budget infeasibility handling: `adjust_p_min` or `error` |
| `SEMANTIC_ENABLE_IFRAME_PREFIX_QUOTA` | `1` | Reserve token budget for I-frame prefix |

**DyTA / P-frame token allocation**

| Variable | Default | Description |
|---|---|---|
| `VTC_TOKEN_ALLOCATION_MODE` | `dyta` | Allocation mode: `uniform`, `residual_mean`, or `dyta` |
| `SEMANTIC_CODEC_BITCOST_ENABLED` | `1` | Enable codec bit-cost prior in DyTA |
| `SEMANTIC_CODEC_CACHE_DIR` | `${VTC_REPO}/outputs_codec_cache_v2` | Codec metadata cache directory |
| `SEMANTIC_CODEC_FFPROBE_PATH` | `ffprobe` | Path or name of ffprobe executable |
| `SEMANTIC_CODEC_STRICT` | `0` | Set to `1` to fail on codec extraction errors |
| `SEMANTIC_CODEC_AGGREGATION` | `mean_p90` | Frame bit-cost aggregation: `mean`, `p90`, `mean_p90`, `max`, `sum` |
| `SEMANTIC_CODEC_NORMALIZATION` | `robust_pb` | Per-byte normalization: `raw_bits`, `bpp`, `log_bpp`, `robust_pb` |
| `SEMANTIC_CODEC_I_POLICY` | `exclude` | I-frame bit-cost handling: `exclude`, `nearest_pb`, `clip_as_pb` |
| `SEMANTIC_CODEC_Z_CLIP` | `5.0` | Z-score clipping for robust percentile normalization |
| `SEMANTIC_CODEC_BUDGET_POWER` | `0.5` | Power-law exponent applied to normalized cost |
| `SEMANTIC_CODEC_BUDGET_MAX_MULTIPLIER` | `2.0` | Ceiling multiplier for per-frame token allocation |

**I-frame compression (dyI)**

| Variable | Default | Description |
|---|---|---|
| `VTC_I_FRAME_COMPRESSION` | `lightweight` | I-frame compression: `none` or `lightweight` |
| `VTC_I_FRAME_RETENTION_RATIO` | `0.9` | Token retention ratio applied to I-frames |
| `SEMANTIC_IFRAME_MIN_KEEP_RATIO` | `0.8` | Floor for I-frame retention under budget pressure |
| `SEMANTIC_IFRAME_ANCHOR_RATIO` | `0.2` | Fraction of I-frame budget reserved for spatial anchors |
| `SEMANTIC_IFRAME_SELECTION_MODE` | `spatial_anchor_salient` | I-frame token selection: `spatial_anchor_salient`, `salient_only`, `random` |
| `SEMANTIC_IFRAME_ANCHOR_GRID` | `4x6` | Spatial grid dimensions for anchor selection |
| `VTC_I_FRAME_RESIDUAL_WEIGHT` | `0.5` | Weight given to residual change vs. absolute saliency |

**Token selection (SC-TopK / global TopK)**

| Variable | Default | Description |
|---|---|---|
| `VTC_TOKEN_SELECTION_MODE` | `sc_topk` | P-frame token selection: `sc_topk` or `global_topk` |
| `VTC_REGION_GRID_H` | `4` | Spatial coverage grid rows |
| `VTC_REGION_GRID_W` | `4` | Spatial coverage grid columns |
| `VTC_COVERAGE_RATIO` | `0.20` | Fraction of P-frame budget reserved for spatial coverage |
| `VTC_REGION_MIN_TOKENS` | `1` | Minimum per-region token target |

**Debug**

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `0` | Set to `1` or `true` to emit per-video metadata logs alongside the tqdm progress bar |

---

## 7. Running Individual Benchmark Scripts Directly

Each `eval/scripts/run_<benchmark>.sh` can be invoked stand-alone with the
generic variables `VIDEO_DIR` and `GT_FILE`:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export VIDEO_DIR=/path/to/videos
export GT_FILE=/path/to/annotations.json
export RUNS="128:0.25"
export GPU_GROUP=0,1
export NUM_CHUNKS=2

bash eval/scripts/run_videomme.sh
```

All variables listed in §6.2 are accepted.

---

## 8. VTC vs. VTC_Dy — Method Presets

The two variants differ only in their component settings:

| Component | VTC | VTC_Dy |
|---|---|---|
| Frame extraction | Uniform sampling | DyRSO Mosaic (IIPP layout) |
| Token allocation | Uniform (all P-frames equal) | DyTA (codec-guided dynamic) |
| Codec prior | Disabled | Enabled |
| I-frame compression | None | Lightweight spatial-anchor pruning |
| P-frame token selection | Global TopK | SC-TopK |

When invoked through `run_vtc_dy_four_benchmarks.sh`, the presets are selected
automatically.  When invoking the per-benchmark scripts directly, set the
individual component variables listed in §6.2 to produce the desired
configuration.

---

## 9. Sanity Checks

Before a full-scale run, verify the environment and scripts are valid:

```bash
python -m pytest -q tests/test_vtc_dy.py
bash -n eval/scripts/run_vtc_dy_four_benchmarks.sh
bash -n eval/scripts/run_videomme.sh
bash -n eval/scripts/run_longvideobench.sh
bash -n eval/scripts/run_mlvu.sh
bash -n eval/scripts/run_lvbench.sh
```

The unit tests do not download models or data; they cover codec-prior cache
I/O, pair-aligned Mosaic packing, homogeneous temporal pair constraints, and
the alignment of pre-computed codec step statistics into the pruner.

---

## 10. Further Details

See [`docs/visual_token_coding_zh.md`](docs/visual_token_coding_zh.md) for a
technical walk-through of the DyRSO input layout, DyTA budget derivation, I-frame
compression, and SC-TopK spatial coverage algorithm.
