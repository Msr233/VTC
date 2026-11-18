"""Shared configuration for the four Qwen3-VL benchmark runners."""

import argparse


def parse_common_args(benchmark):
    parser = argparse.ArgumentParser(description=f"Qwen3-VL: {benchmark}")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="predictions")
    parser.add_argument("--method", choices=("none", "vtc", "vtcdy"), default="vtcdy")
    parser.add_argument(
        "--retention-ratio",
        type=float,
        choices=(0.25, 0.5, 1.0),
        default=0.5,
        help="Fraction of visual tokens retained (not the drop fraction)",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="0 preserves the original dynamic video resolution",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mean weight; P90 weight is 1-alpha. alpha=1 skips P90",
    )
    parser.add_argument("--keyframe-threshold", type=float, default=2.0)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--min-p-tokens", type=int, default=8)
    parser.add_argument("--iframe-keep-ratio", type=float, default=0.9)
    parser.add_argument("--codec-cache-dir", default="outputs/codec_cache")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--strict-codec",
        action="store_true",
        help="Raise on failed codec extraction instead of uniform-budget fallback",
    )
    parser.add_argument(
        "--vit-chunk-frames",
        type=int,
        default=16,
        help="Input images per ViT pass; 0 disables chunking",
    )
    parser.add_argument("--vit-staging-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Shard one model over visible GPUs"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=1, help="Number of question shards"
    )
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--read-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512 if benchmark in ("videomme", "longvideobench") else 16,
    )
    parser.add_argument("--log-frame-tokens", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Explicit CUDA-event profiling; disabled for normal evaluation",
    )
    parser.add_argument(
        "--preload-codec",
        action="store_true",
        help="Prepare codec priors before generate; report preload wall time separately",
    )
    args = parser.parse_args()
    args.input_mode = "gop" if args.method == "vtcdy" else "uniform"
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be between 0 and 1")
    if args.frames <= 0 or args.frames % (4 if args.input_mode == "gop" else 2):
        parser.error(
            "frames must be positive and divisible by 4 for GoP input, or 2 for uniform input"
        )
    if args.vit_chunk_frames < 0 or args.vit_chunk_frames % 2:
        parser.error("vit-chunk-frames must be zero or a positive multiple of 2")
    if not 0 <= args.chunk_idx < args.num_chunks:
        parser.error("chunk-idx must be within num-chunks")
    return args


def configure_model(model, args):
    if args.method == "none":
        return None
    from vtc import apply_qwen3_vtc

    return apply_qwen3_vtc(
        model,
        method=args.method,
        retention_ratio=args.retention_ratio,
        keyframe_threshold=args.keyframe_threshold,
        window_size=args.history_window,
        min_p_tokens=args.min_p_tokens,
        iframe_keep_ratio=args.iframe_keep_ratio,
        codec_cache_dir=args.codec_cache_dir,
        codec_ffprobe_path=args.ffprobe,
        codec_strict=args.strict_codec,
        codec_mean_weight=args.alpha,
        enable_vit_chunking=args.vit_chunk_frames > 0,
        vit_chunk_frames=args.vit_chunk_frames or args.frames,
        vit_chunk_staging_device=args.vit_staging_device,
        vit_chunk_collect_stats=args.profile,
        log_stats=args.debug,
    )


def prepare_video_context(model, args, video_paths, metadata, inputs):
    import time
    from .sampling_context import set_vtc_video_contexts

    pruner = set_vtc_video_contexts(model, video_paths, metadata, inputs)
    model._vtc_codec_preload_ms = 0.0
    if pruner is not None and args.preload_codec and args.method == "vtcdy":
        begin = time.perf_counter()
        pruner.preload_codec()
        model._vtc_codec_preload_ms = (time.perf_counter() - begin) * 1000
