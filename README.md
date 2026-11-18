# Visual Token Coding for Video Multimodal Large Language Models

Implementation of **VTC** and **VTCdy** for
**Qwen3-VL-8B-Instruct**. Evaluation entry points cover VideoMME, MLVU,
LVBench and LongVideoBench. Model weights and benchmark data are obtained
separately.

## Method and code

| Paper component | Implementation |
| --- | --- |
| Semantic I/P construction and residual coding | `vtc/coding.py` |
| Codec extraction, normalization and temporal alignment | `vtc/codec_priors.py` |
| Codec-guided integer budget allocation | `vtc/dynamic_budget.py` |
| Budgeted I-frame pruning | `vtc/reference_budget.py` |
| P-frame FRS and I-frame reference selection | `vtc/frame_redundancy.py` |
| GoP input packing | `vtc/gop_packing.py` |
| Qwen3 mRoPE, DeepStack and generation integration | `vtc/qwen3.py` |
| Streaming visual encoding | `vtc/chunked_encoder.py` |

`--method vtc` uses uniform input sampling, complete I frames, uniform P
budgets and residual Top-K. `--method vtcdy` uses GoP inputs, codec-guided P
budgets, budgeted I-frame pruning and frame redundancy suppression (FRS).
`--method none` runs the uncompressed model with uniform input sampling.

The implementation preserves the experimental prediction, budget and
selection rules. Read [implementation details](docs/implementation_notes.md)
for the precise correspondence to the paper's compact equations.

## Environment

Use Python 3.10, CUDA PyTorch and FFmpeg/ffprobe on `PATH`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Optional; requires a compatible GPU and CUDA build environment.
pip install flash-attn==2.7.3 --no-build-isolation
ffprobe -version
```

The default backend is FlashAttention 2, with automatic SDPA fallback when
FlashAttention is unavailable. `--attn-implementation sdpa` selects SDPA
explicitly. Keep the actual backend fixed when comparing latency or accuracy.

## Evaluation

All commands run from the package root. The annotation schemas are described
in [data format](docs/data_format.md).

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export VIDEO_DIR=/path/to/VideoMME/videos
export GT_FILE=/path/to/annotations/videomme.json

bash scripts/run_benchmark.sh videomme vtcdy 0.5
bash scripts/run_benchmark.sh videomme vtcdy 0.25
bash scripts/run_benchmark.sh videomme vtc 0.5
bash scripts/run_benchmark.sh videomme none 1.0
```

Replace `videomme` with `mlvu`, `lvbench` or `longvideobench`, and update the
video/annotation paths. Outputs go to
`outputs/<benchmark>/<method>_r<ratio>` by default. Use a fresh `OUTPUT_DIR`
when changing settings: existing predictions resume by question ID.

The common defaults are 64 input images, dynamic default resolution,
`tau=2`, `alpha=0.5`, I retention `rho=0.9`, and 16 input images per visual
chunk. Qwen3 fuses every two images into one temporal unit; 64 input images
normally form 32 semantic units. Chunk size affects the allocation scope.
Use `--vit-chunk-frames 0` for full-video allocation in one pass.

```bash
python -m evaluation.longvideobench \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --video-dir /path/to/LongVideoBench/videos \
  --gt-file /path/to/annotations/longvideobench.json \
  --output-dir outputs/longvideobench/vtcdy_r0.5 \
  --method vtcdy --retention-ratio 0.5 --alpha 0.5
```

`--alpha 1` selects mean-only codec aggregation. `--log-frame-tokens` adds
per-frame semantic types and retained token counts to predictions;
`--debug` prints additional runtime information. These are disabled by default.
No heatmaps, feature quantiles or spatial-entropy diagnostics run during normal
inference.

## Batch execution and scoring

Set `VIDEOMME_VIDEO_DIR`, `VIDEOMME_GT_FILE`, and the analogous `MLVU_*`,
`LVBENCH_*`, `LONGVIDEOBENCH_*` resource variables for the selected datasets.

```bash
# One question shard per listed GPU; datasets/methods/ratios run in sequence.
GPU_GROUP=0,1 BENCHMARKS="videomme mlvu" METHODS="vtc vtcdy" RATIOS="0.5 0.25" \
  bash scripts/run_suite.sh
```

The suite keeps separate shard logs/configurations and writes `metrics.json`
after all shards finish. For one benchmark, `--multi-gpu` instead distributes
a single model over its visible GPUs; it is separate from question sharding.

```bash
python -m evaluation.summarize --benchmark videomme \
  --gt-file /path/to/annotations/videomme.json \
  --predictions outputs/videomme/vtcdy_r0.5/predictions.json \
  --output outputs/videomme/vtcdy_r0.5/metrics.json
```

Prediction files are JSON Lines with a `.json` extension. The scorer reports
expected/evaluated question counts, missing IDs, total accuracy and group
scores. Main accuracy uses the full supplied annotation set; unanswered
questions count as incorrect. Check `complete=true` before reporting a full
benchmark. MLVU uses total accuracy, not category macro-average.

## Efficiency options

The implementation uses batched region Top-K, cached I-frame anchor layouts,
one residual-mean CPU transfer per chunk, fixed-length gathers shared with
DeepStack, and compressed prompt attention masks cached on device.

`--profile` explicitly enables CUDA-event chunk timings and a completion
synchronization. Normal inference has no per-chunk profiling synchronization.
`--preload-codec` optionally performs extraction/alignment before `generate`;
predictions record `codec_preload_ms` separately. It changes the timing
boundary, not the total preprocessing work. Normal evaluation includes cold
codec work unless the disk/runtime cache is already populated.

The package writes schema-3 codec caches and reads schemas 1/2/3. Use
separate cache directories when comparing releases. By default, known codec
IO/metadata failures produce a warning and uniform P budgets;
`--strict-codec` raises instead. Programming errors are not swallowed.

## Scope

This package includes no model weights, benchmark videos, experiment outputs,
private machine paths or Git history. Unused packing/selection ablations,
version aliases, adaptive-I dead code, regression tests and debug-plotting
paths are omitted. [Third-party notes](THIRD_PARTY.md) list external dependencies.
This packaged revision has been checked through source inspection and static
analysis only; model inference and GPU timing have not been validated. No
measured speedup or numerical reproduction of the paper tables is claimed.
