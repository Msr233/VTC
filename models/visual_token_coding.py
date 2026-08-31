from __future__ import annotations

import collections
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .codec_bitcost_allocator import CodecBitcostAllocator
from .iframe_budget_allocator import IFrameBudgetAllocator
from .iframe_token_selector import IFrameTokenSelector
from utils.hevc_bitcost import (
    HEVCBitcostAligner,
    HEVCBitcostExtractor,
    TemporalStepCodecStats,
    UnsupportedCodecError,
    VideoSamplingContext,
    summarize_metadata,
)


TOKEN_SELECTION_MODES = ("global_topk", "sc_topk")
IFRAME_COMPRESSION_MODES = ("none", "lightweight")
SEMANTIC_FRAME_TYPE_SOURCES = ("residual", "codec_ip")


class VisualTokenCoding:
    """Visual Token Coding with dynamic budgeting (VTC_Dy).

    The implementation combines semantic I/P-frame coding with Dynamic
    Resolution Input (DyRSO), Dynamic Token Allocation (DyTA), and Spatial
    Coverage Top-K (SC-TopK).
    """

    def __init__(
        self,
        retention_ratio: float = 0.5,
        semantic_i_threshold: float = 2.0,
        p_frame_min_tokens: int = 8,
        fill_mode: str = "zero",
        window_size: int = 8,
        warmup_frames: int = 2,
        spatial_merge_size: int = 2,
        codec_bitcost_enabled: bool = True,
        token_allocation_mode: str = "dyta",
        codec_cache_dir: str | None = None,
        codec_ffprobe_path: str = "ffprobe",
        codec_strict: bool = False,
        codec_aggregation: str = "mean_p90",
        codec_mean_weight: float = 0.5,
        codec_p90_weight: float = 0.5,
        codec_normalization: str = "robust_pb",
        codec_i_policy: str = "exclude",
        codec_z_clip: float = 5.0,
        codec_budget_power: float = 0.5,
        codec_budget_max_multiplier: float = 2.0,
        iframe_compression_mode: str = "lightweight",
        i_frame_retention_ratio: float = 0.9,
        first_iframe_keep_ratio: float | None = None,
        dynamic_iframe_keep_ratio: float | None = None,
        iframe_min_keep_ratio: float = 0.8,
        iframe_anchor_ratio: float = 0.2,
        iframe_selection_mode: str = "spatial_anchor_salient",
        iframe_anchor_grid: str = "4x6",
        i_frame_residual_weight: float = 0.5,
        iframe_compress_when_ratio_one: bool = False,
        token_selection_mode: str = "sc_topk",
        region_grid_h: int = 4,
        region_grid_w: int = 4,
        coverage_ratio: float = 0.20,
        region_min_tokens: int = 1,
        enable_codec_runtime_cache: bool = True,
        codec_runtime_cache_size: int = 1,
        log_stats: bool = True,
        log_frame_allocations: bool | None = None,
        semantic_version: str = "v5.0.0",
        semantic_frame_type_source: str = "residual",
    ) -> None:
        self.semantic_version = str(semantic_version)
        self.semantic_frame_type_source = str(semantic_frame_type_source)
        if self.semantic_frame_type_source not in SEMANTIC_FRAME_TYPE_SOURCES:
            raise ValueError(
                "semantic_frame_type_source must be residual/codec_ip, "
                f"got {self.semantic_frame_type_source!r}"
            )
        self.retention_ratio = float(retention_ratio)
        self.semantic_i_threshold = float(semantic_i_threshold)
        self.p_frame_min_tokens = int(p_frame_min_tokens)
        self.fill_mode = str(fill_mode)
        self.window_size = int(window_size)
        self.warmup_frames = int(warmup_frames)
        self.spatial_merge_size = int(spatial_merge_size)
        self.log_stats = bool(log_stats)
        self.log_frame_allocations = (
            self.log_stats
            if log_frame_allocations is None
            else bool(log_frame_allocations)
        )

        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")

        self.codec_bitcost_enabled = bool(codec_bitcost_enabled)
        self.token_allocation_mode = str(token_allocation_mode)
        if self.token_allocation_mode not in ("uniform", "residual_mean", "dyta"):
            raise ValueError(
                "token_allocation_mode must be uniform/residual_mean/dyta, "
                f"got {self.token_allocation_mode!r}"
            )
        self.codec_cache_dir = codec_cache_dir
        self.codec_ffprobe_path = str(codec_ffprobe_path)
        self.codec_strict = bool(codec_strict)
        self.codec_aggregation = str(codec_aggregation)
        self.codec_mean_weight = float(codec_mean_weight)
        self.codec_p90_weight = float(codec_p90_weight)
        self.codec_normalization = str(codec_normalization)
        self.codec_i_policy = str(codec_i_policy)
        self.codec_z_clip = float(codec_z_clip)
        self.codec_budget_power = float(codec_budget_power)
        self.codec_budget_max_multiplier = float(codec_budget_max_multiplier)
        self.enable_codec_runtime_cache = bool(enable_codec_runtime_cache)
        self.codec_runtime_cache_size = max(0, int(codec_runtime_cache_size))
        self._codec_runtime_cache: collections.OrderedDict[
            tuple[Any, ...],
            tuple[Any, list[TemporalStepCodecStats], dict[str, Any]],
        ] = collections.OrderedDict()
        self.temporal_patch_size = 2
        self._sampling_contexts: List[VideoSamplingContext] = []
        self._warned_fallback_reasons: set[str] = set()

        self.iframe_compression_mode = str(iframe_compression_mode)
        if self.iframe_compression_mode not in IFRAME_COMPRESSION_MODES:
            raise ValueError(
                f"iframe_compression_mode must be one of {IFRAME_COMPRESSION_MODES}, "
                f"got {self.iframe_compression_mode!r}"
            )
        self.i_frame_retention_ratio = float(i_frame_retention_ratio)
        self.first_iframe_keep_ratio = (
            None if first_iframe_keep_ratio is None else float(first_iframe_keep_ratio)
        )
        self.dynamic_iframe_keep_ratio = (
            None if dynamic_iframe_keep_ratio is None else float(dynamic_iframe_keep_ratio)
        )
        self.iframe_min_keep_ratio = float(iframe_min_keep_ratio)
        self.iframe_anchor_ratio = float(iframe_anchor_ratio)
        self.iframe_selection_mode = str(iframe_selection_mode)
        self.iframe_anchor_grid = str(iframe_anchor_grid)
        self.i_frame_residual_weight = float(i_frame_residual_weight)
        self.iframe_compress_when_ratio_one = bool(iframe_compress_when_ratio_one)
        self.iframe_budget_allocator = IFrameBudgetAllocator()
        self.iframe_token_selector = IFrameTokenSelector()

        self.token_selection_mode = str(token_selection_mode)
        if self.token_selection_mode not in TOKEN_SELECTION_MODES:
            raise ValueError(f"token_selection_mode must be one of {TOKEN_SELECTION_MODES}, got {self.token_selection_mode!r}")
        self.region_grid_h = int(region_grid_h)
        self.region_grid_w = int(region_grid_w)
        self.coverage_ratio = float(coverage_ratio)
        self.region_min_tokens = int(region_min_tokens)
        if self.region_grid_h <= 0 or self.region_grid_w <= 0:
            raise ValueError("spatial grid dimensions must be positive")
        if self.region_min_tokens < 0:
            raise ValueError("region_min_tokens must be non-negative")

        self._call_count = 0
        self._config_logged = False
        self._spatial_region_cache: Dict[
            tuple[int, int, int, int, str],
            List[torch.Tensor],
        ] = {}

    def set_temporal_patch_size(self, temporal_patch_size: int) -> None:
        self.temporal_patch_size = max(1, int(temporal_patch_size))

    def set_video_sampling_contexts(self, contexts: List[VideoSamplingContext | dict]) -> None:
        self._sampling_contexts = [self._coerce_context(ctx) for ctx in contexts]

    def clear_video_sampling_contexts(self) -> None:
        self._sampling_contexts = []

    def _coerce_context(self, ctx: VideoSamplingContext | dict) -> VideoSamplingContext:
        if isinstance(ctx, VideoSamplingContext):
            return ctx
        return VideoSamplingContext(
            video_path=str(ctx.get("video_path") or ctx.get("video") or ""),
            source_fps=float(ctx.get("source_fps") or ctx.get("fps") or 0.0),
            source_total_frames=int(ctx.get("source_total_frames") or ctx.get("total_num_frames") or 0),
            sampled_frame_indices=[int(x) for x in ctx.get("sampled_frame_indices", ctx.get("frames_indices", []))],
            sampled_timestamps=[float(x) for x in ctx.get("sampled_timestamps", [])],
            temporal_patch_size=int(ctx.get("temporal_patch_size") or self.temporal_patch_size),
            grid_t=int(ctx.get("grid_t") or 0),
            sampled_source_frame_indices=[
                [int(index) for index in group]
                for group in ctx.get("sampled_source_frame_indices", [])
            ],
            input_types=[str(value) for value in ctx.get("input_types", [])],
        )

    def _pop_sampling_context(self, full_grid_thw: Optional[torch.Tensor]) -> VideoSamplingContext | None:
        if not self._sampling_contexts:
            return None
        ctx = self._sampling_contexts.pop(0)
        if full_grid_thw is not None and ctx.grid_t <= 0:
            ctx.grid_t = int(full_grid_thw[0].item())
        if ctx.temporal_patch_size <= 0:
            ctx.temporal_patch_size = self.temporal_patch_size
        return ctx

    def _codec_runtime_cache_key(
        self,
        ctx: VideoSamplingContext,
    ) -> tuple[Any, ...] | None:
        if not self.enable_codec_runtime_cache or self.codec_runtime_cache_size <= 0:
            return None
        try:
            path = Path(ctx.video_path).resolve()
            stat = path.stat()
        except OSError:
            return None
        return (
            str(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(ctx.grid_t),
            int(ctx.temporal_patch_size),
            tuple(int(index) for index in ctx.sampled_frame_indices),
            tuple(float(timestamp) for timestamp in ctx.sampled_timestamps),
            float(ctx.source_fps),
            self.codec_aggregation,
            self.codec_normalization,
            self.codec_i_policy,
            self.codec_mean_weight,
            self.codec_p90_weight,
            self.codec_z_clip,
        )

    def _get_cached_codec_data(
        self,
        key: tuple[Any, ...] | None,
    ) -> tuple[Any, list[TemporalStepCodecStats], dict[str, Any]] | None:
        if key is None:
            return None
        cached = self._codec_runtime_cache.get(key)
        if cached is not None:
            self._codec_runtime_cache.move_to_end(key)
        return cached

    def _cache_codec_data(
        self,
        key: tuple[Any, ...] | None,
        value: tuple[Any, list[TemporalStepCodecStats], dict[str, Any]],
    ) -> None:
        if key is None:
            return
        self._codec_runtime_cache[key] = value
        self._codec_runtime_cache.move_to_end(key)
        while len(self._codec_runtime_cache) > self.codec_runtime_cache_size:
            self._codec_runtime_cache.popitem(last=False)

    def init_stream_state(self, full_grid_thw: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "energy_window": collections.deque(maxlen=self.window_size),
            "prev_full_frame": None,
            "frame_info": [],
            "keep_masks": [],
            "chunk_summaries": [],
            "original_tokens": 0,
            "kept_tokens": 0,
            "target_kept_tokens": 0,
            "target_p_frame_kept_tokens": 0,
            "semantic_iframe_count": 0,
            "total_iframe_original_tokens": 0,
            "total_iframe_kept_tokens": 0,
            "total_iframe_released_tokens": 0,
            "p_budget_total_increase_from_iframe": 0,
            "iframe_compression_warnings": [],
            "token_allocation_mode": self.token_allocation_mode,
            "codec_budget_mode": self.token_allocation_mode,
            "codec_alignment_valid": False,
            "codec_fallback_reason": None,
            "codec_ip_alignment_valid": False,
            "codec_ip_fallback_reason": None,
            "codec_step_stats": None,
            "codec_metadata_summary": None,
            "codec_alignment_time_ms": 0.0,
            "codec_extraction_time_ms": 0.0,
            "codec_summary_time_ms": 0.0,
            "codec_allocation_time_ms": 0.0,
            "codec_runtime_cache_hit": False,
            "codec_frame_count": 0,
            "total_codec_fallback_count": 0,
            "total_residual_mean_fallback_count": 0,
            "token_selection_mode": self.token_selection_mode,
        }
        if full_grid_thw is not None:
            T, H_grid, W_grid = [int(x) for x in full_grid_thw.tolist()]
            N = H_grid * W_grid // (self.spatial_merge_size ** 2)
            state.update({
                "full_T": T,
                "tokens_per_frame": N,
                "full_original_tokens": T * N,
                "full_target_kept_tokens": min(T * N, max(0, int(round(T * N * self.retention_ratio)))),
            })

        needs_codec_ip = self.semantic_frame_type_source == "codec_ip"
        needs_codec_budget = (
            self.token_allocation_mode == "dyta"
            and self.codec_bitcost_enabled
        )
        if (
            not needs_codec_ip
            and self.token_allocation_mode == "dyta"
            and not self.codec_bitcost_enabled
        ):
            state["codec_fallback_reason"] = "codec_bitcost_disabled"
            state["codec_ip_fallback_reason"] = "not_codec_ip"
            return state
        if not needs_codec_ip and not needs_codec_budget:
            state["codec_fallback_reason"] = "token_allocation_not_dyta"
            state["codec_ip_fallback_reason"] = "not_codec_ip"
            return state

        ctx = self._pop_sampling_context(full_grid_thw)
        if ctx is None or not ctx.video_path:
            self._handle_codec_failure(state, "sampling_context_missing", None)
            return state

        if full_grid_thw is not None:
            ctx.grid_t = int(full_grid_thw[0].item())
        ctx.temporal_patch_size = self.temporal_patch_size
        if len(ctx.sampled_frame_indices) < ctx.grid_t * ctx.temporal_patch_size:
            self._handle_codec_failure(state, "sampled_frame_count_mismatch", None)
            return state

        try:
            cache_key = self._codec_runtime_cache_key(ctx)
            cached = self._get_cached_codec_data(cache_key)
            if cached is None:
                t0 = time.perf_counter()
                extractor = HEVCBitcostExtractor(
                    ffprobe_path=self.codec_ffprobe_path,
                    cache_dir=self.codec_cache_dir,
                    strict=self.codec_strict,
                    z_clip=self.codec_z_clip,
                )
                metadata = extractor.extract(ctx.video_path)
                state["codec_extraction_time_ms"] = round(
                    (time.perf_counter() - t0) * 1000.0,
                    6,
                )
                align_start = time.perf_counter()
                aligner = HEVCBitcostAligner(
                    aggregation=self.codec_aggregation,
                    normalization=self.codec_normalization,
                    i_policy=self.codec_i_policy,
                    mean_weight=self.codec_mean_weight,
                    p90_weight=self.codec_p90_weight,
                    z_clip=self.codec_z_clip,
                )
                step_stats = aligner.align_to_temporal_steps(metadata, ctx)
                state["codec_alignment_time_ms"] = round(
                    (time.perf_counter() - align_start) * 1000.0,
                    6,
                )
                summary_start = time.perf_counter()
                summary = summarize_metadata(metadata)
                state["codec_summary_time_ms"] = round(
                    (time.perf_counter() - summary_start) * 1000.0,
                    6,
                )
            else:
                metadata, step_stats, summary = cached
                state["codec_runtime_cache_hit"] = True
            if len(step_stats) != ctx.grid_t:
                raise ValueError(f"codec step count {len(step_stats)} != grid_t {ctx.grid_t}")
            if not any(math.isfinite(s.codec_complexity) and s.codec_complexity > 0 for s in step_stats):
                raise ValueError("all codec complexities are invalid")
            if cached is None:
                self._cache_codec_data(
                    cache_key,
                    (metadata, step_stats, summary),
                )
            state.update({
                "codec_alignment_valid": True,
                "codec_ip_alignment_valid": bool(
                    step_stats
                    and all(step.codec_frame_type_valid for step in step_stats)
                ),
                "codec_step_stats": step_stats,
                "codec_metadata_summary": summary,
                "codec_name": metadata.codec_name,
                "codec_frame_count": metadata.total_frames,
                "codec_i_count": summary["frame_counts"].get("I", 0),
                "codec_p_count": summary["frame_counts"].get("P", 0),
                "codec_b_count": summary["frame_counts"].get("B", 0),
                "video_sampling_context": asdict(ctx),
            })
            if not state["codec_ip_alignment_valid"]:
                reasons = [
                    step.codec_frame_type_fallback_reason
                    for step in step_stats
                    if not step.codec_frame_type_valid
                ]
                state["codec_ip_fallback_reason"] = (
                    "invalid_codec_frame_type_alignment"
                    + (f": {reasons[0]}" if reasons else "")
                )
            if self.log_stats:
                print(
                    f"[CodecBitcost] video={Path(ctx.video_path).name} codec={metadata.codec_name} "
                    f"frames={metadata.total_frames} I/P/B={state['codec_i_count']}/"
                    f"{state['codec_p_count']}/{state['codec_b_count']}"
                )
        except UnsupportedCodecError as exc:
            self._handle_codec_failure(state, "unsupported_codec", exc)
        except Exception as exc:
            self._handle_codec_failure(state, "metadata_or_alignment_failed", exc)
        return state

    def _handle_codec_failure(self, state: Dict[str, Any], reason: str, exc: Exception | None) -> None:
        state["codec_alignment_valid"] = False
        state["codec_fallback_reason"] = reason if exc is None else f"{reason}: {exc}"
        state["codec_ip_alignment_valid"] = False
        state["codec_ip_fallback_reason"] = state["codec_fallback_reason"]
        if (
            self.codec_strict
            and self.semantic_frame_type_source != "codec_ip"
            and not isinstance(exc, UnsupportedCodecError)
        ):
            raise RuntimeError(f"[CodecBitcost] strict mode failed: {state['codec_fallback_reason']}") from exc
        key = str(state["codec_fallback_reason"])
        if key not in self._warned_fallback_reasons:
            print(
                f"[CodecBitcost] WARNING: {key}; falling back to uniform "
                "P-frame allocation and residual semantic I/P classification."
            )
            self._warned_fallback_reasons.add(key)

    def _allocate_streaming_chunk_budget(
        self,
        state: Dict[str, Any],
        *,
        global_end_t: int,
        num_i: int,
        num_p: int,
        tokens_per_frame: int,
        chunk_total_tokens: int,
    ) -> Tuple[List[int], int, int]:
        if num_p == 0:
            target = num_i * tokens_per_frame
            return [], target, 0

        full_T = state.get("full_T")
        full_target = state.get("full_target_kept_tokens")
        if full_T is None or full_target is None:
            target_kept = min(chunk_total_tokens, max(0, int(round(chunk_total_tokens * self.retention_ratio))))
            i_tokens = num_i * tokens_per_frame
            p_budget = min(num_p * tokens_per_frame, max(0, target_kept - i_tokens))
        else:
            already_kept = int(state.get("kept_tokens", 0))
            current_i_tokens = num_i * tokens_per_frame
            future_frames = max(0, int(full_T) - int(global_end_t))
            remaining_p_slots = num_p + future_frames
            remaining_target = max(0, int(full_target) - already_kept - current_i_tokens)
            p_budget = min(
                num_p * tokens_per_frame,
                int(round(remaining_target * (num_p / max(1, remaining_p_slots)))),
            )
            target_kept = current_i_tokens + p_budget

        base = p_budget // num_p
        remainder = p_budget % num_p
        allocations = [base + (1 if p_idx < remainder else 0) for p_idx in range(num_p)]
        return allocations, target_kept, p_budget

    def is_keyframe(self, energy_mean: float, window: collections.deque) -> bool:
        if len(window) < self.warmup_frames:
            return False
        rolling_mean = sum(window) / len(window)
        if rolling_mean < 1e-8:
            return energy_mean > 1e-6
        return energy_mean > self.semantic_i_threshold * rolling_mean

    def _select_tokens_with_k(
        self,
        Energy: torch.Tensor,
        N: int,
        K: int,
        frame_idx: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        K = min(N, max(0, int(K)))
        mask = torch.zeros(N, dtype=torch.bool, device=Energy.device)
        if K == 0:
            return mask, K
        clean = torch.nan_to_num(
            Energy.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if clean.max().item() < 1e-8:
            mask[:K] = True
            return mask, K
        _, indices = torch.topk(clean, K, largest=True)
        mask[indices] = True
        return mask, K

    def _step_stats_for(self, state: Dict[str, Any], global_t: int) -> TemporalStepCodecStats | None:
        stats = state.get("codec_step_stats")
        if stats is None:
            return None
        if 0 <= int(global_t) < len(stats):
            return stats[int(global_t)]
        return None

    def _codec_fields(self, stats: TemporalStepCodecStats | None) -> dict:
        if stats is None:
            return {
                "temporal_step_id": None,
                "sampled_frame_indices": [],
                "sampled_timestamps": [],
                "sampled_frame_pict_types": [],
                "sampled_frame_type_sources": [],
                "codec_semantic_frame_type": None,
                "codec_frame_type_valid": False,
                "codec_frame_type_fallback_reason": "no_codec_step_stats",
                "codec_window_left": None,
                "codec_window_right": None,
                "codec_pb_frame_count": 0,
                "codec_p_frame_count": 0,
                "codec_b_frame_count": 0,
                "codec_i_frame_count": 0,
                "codec_pb_mean": None,
                "codec_pb_p90": None,
                "codec_pb_max": None,
                "codec_complexity": None,
                "codec_fallback_used": True,
                "codec_fallback_reason": "no_codec_step_stats",
            }
        return {
            "temporal_step_id": int(stats.temporal_step_id),
            "sampled_frame_indices": list(stats.sampled_frame_indices),
            "sampled_timestamps": [round(float(x), 6) for x in stats.sampled_timestamps],
            "sampled_frame_pict_types": list(stats.sampled_frame_pict_types),
            "sampled_frame_type_sources": list(stats.sampled_frame_type_sources),
            "codec_semantic_frame_type": stats.codec_semantic_frame_type,
            "codec_frame_type_valid": bool(stats.codec_frame_type_valid),
            "codec_frame_type_fallback_reason": stats.codec_frame_type_fallback_reason,
            "codec_window_left": round(float(stats.left_time), 6),
            "codec_window_right": round(float(stats.right_time), 6),
            "codec_pb_frame_count": int(stats.codec_pb_frame_count),
            "codec_p_frame_count": int(stats.codec_p_frame_count),
            "codec_b_frame_count": int(stats.codec_b_frame_count),
            "codec_i_frame_count": int(stats.codec_i_frame_count),
            "codec_pb_mean": round(float(stats.pb_mean), 6),
            "codec_pb_p90": round(float(stats.pb_p90), 6),
            "codec_pb_max": round(float(stats.pb_max), 6),
            "codec_complexity": round(float(stats.codec_complexity), 6),
            "codec_fallback_used": bool(stats.used_fallback),
            "codec_fallback_reason": stats.fallback_reason,
        }

    def _uniform_p_allocations(self, num_p: int, total_budget: int, max_keep: int) -> list[int]:
        if num_p == 0:
            return []
        if total_budget < 0 or total_budget > num_p * max_keep:
            raise ValueError(f"Uniform P budget out of range: total_budget={total_budget}, capacity={num_p * max_keep}")
        base = total_budget // num_p
        remainder = total_budget % num_p
        return [base + (1 if p_idx < remainder else 0) for p_idx in range(num_p)]

    def _log_config_once(self) -> None:
        if self._config_logged or not self.log_stats:
            return
        print(
            f"[VTC_Dy {self.semantic_version}]\n"
            f"semantic_frame_type_source={self.semantic_frame_type_source}\n"
            f"token_allocation_mode={self.token_allocation_mode} codec_enabled={self.codec_bitcost_enabled}\n"
            f"iframe_mode={self.iframe_compression_mode} iframe_keep={self.i_frame_retention_ratio}\n"
            f"token_selection_mode={self.token_selection_mode} grid={self.region_grid_h}x{self.region_grid_w} "
            f"coverage={self.coverage_ratio}"
        )
        self._config_logged = True

    def _method_stack(self) -> str:
        if self.token_allocation_mode == "residual_mean":
            stack = "sc_topk+dyrso+residual_mean"
        elif self.token_allocation_mode == "uniform":
            stack = "sc_topk+dyrso+uniform"
        else:
            stack = "sc_topk+dyrso+dyta"
        if self.semantic_frame_type_source == "codec_ip":
            return f"{stack}+codec_ip"
        return stack

    def _format_frame_allocation_line(self, fi: Dict[str, Any]) -> str:
        frame = int(fi.get("frame", -1))
        frame_type = str(fi.get("type", "?"))
        kept = int(fi.get("kept", 0))
        ratio = float(fi.get("allocated_ratio", 0.0)) * 100.0
        line = f"  Frame {frame:2d} ({frame_type}): keep={kept} ({ratio:5.1f}%)"
        if frame_type == "P":
            energy = fi.get("energy_mean")
            if energy is not None:
                line += f"  energy={float(energy):.4f}"
        return line

    def prune_chunk(
        self,
        features: torch.Tensor,
        chunk_grid_thw: torch.Tensor,
        state: Dict[str, Any],
        *,
        global_start_t: int,
        video_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        del video_id
        self._log_config_once()
        total_tokens, D = features.shape
        T, H_grid, W_grid = [int(x) for x in chunk_grid_thw.tolist()]
        merge_sq = self.spatial_merge_size ** 2
        N = H_grid * W_grid // merge_sq
        H_spatial = H_grid // self.spatial_merge_size
        W_spatial = W_grid // self.spatial_merge_size
        if total_tokens != T * N:
            raise AssertionError(
                f"[VTC_Dy] chunk shape mismatch: features={total_tokens}, T*N={T*N}"
            )
        if self._call_count == 0 and self.log_stats:
            print(
                f"[VTC_Dy] first streaming chunk: grid={chunk_grid_thw.tolist()} "
                f"features=({total_tokens},{D}) T={T} N={N} ratio={self.retention_ratio}"
            )
        self._call_count += 1

        feats = features.view(T, N, D)
        energy_window = state["energy_window"]
        F_prev: Optional[torch.Tensor] = state["prev_full_frame"]
        chunk_energy: List[Optional[torch.Tensor]] = []
        chunk_energy_means: List[Optional[float]] = []
        chunk_frame_info: List[Dict[str, Any]] = []
        codec_ip_active = (
            self.semantic_frame_type_source == "codec_ip"
            and bool(state.get("codec_ip_alignment_valid"))
        )

        for local_t in range(T):
            global_t = global_start_t + local_t
            F_t = feats[local_t]
            codec_stats = self._step_stats_for(state, global_t)
            codec_fields = self._codec_fields(codec_stats)
            if global_t == 0:
                chunk_energy.append(None)
                chunk_energy_means.append(None)
                chunk_frame_info.append({
                    "frame": global_t,
                    "semantic_temporal_step": global_t,
                    "type": "I",
                    "semantic_frame_type": "I",
                    "semantic_frame_type_source": "forced_first",
                    "kept": N,
                    "energy_mean": None,
                    "semantic_residual_mean": None,
                    **codec_fields,
                })
                F_prev = F_t
                continue
            if F_prev is None:
                raise AssertionError(f"[VTC_Dy] missing previous full frame for {global_t}")
            energy = torch.linalg.vector_norm(F_t - F_prev, dim=-1)
            energy_values = energy.detach().float()
            energy_mean = energy_values.mean().item()
            chunk_energy.append(energy)
            chunk_energy_means.append(float(energy_mean))
            if codec_ip_active:
                if codec_stats is None or not codec_stats.codec_frame_type_valid:
                    raise AssertionError(
                        "[VTC_Dy] codec-IP alignment was marked valid "
                        f"but temporal step {global_t} has no valid frame type"
                    )
                semantic_type = str(codec_stats.codec_semantic_frame_type)
                frame_type_source = "codec_ip"
            else:
                is_i = self.is_keyframe(energy_mean, energy_window)
                energy_window.append(energy_mean)
                if is_i:
                    energy_window.clear()
                    semantic_type = "I"
                else:
                    semantic_type = "P"
                frame_type_source = (
                    "residual_fallback"
                    if self.semantic_frame_type_source == "codec_ip"
                    else "residual"
                )
            kept = N if semantic_type == "I" else 0
            chunk_frame_info.append({
                "frame": global_t,
                "semantic_temporal_step": global_t,
                "type": semantic_type,
                "semantic_frame_type": semantic_type,
                "semantic_frame_type_source": frame_type_source,
                "kept": kept,
                "energy_mean": round(energy_mean, 4),
                "semantic_residual_mean": round(energy_mean, 6),
                **codec_fields,
            })
            F_prev = F_t

        num_i = sum(1 for fi in chunk_frame_info if fi["type"] == "I")
        num_p = sum(1 for fi in chunk_frame_info if fi["type"] == "P")
        _, target_kept_tokens, target_p_tokens_old = self._allocate_streaming_chunk_budget(
            state,
            global_end_t=global_start_t + T,
            num_i=num_i,
            num_p=num_p,
            tokens_per_frame=N,
            chunk_total_tokens=total_tokens,
        )

        prior_iframes = int(state.get("semantic_iframe_count", 0))
        iframe_flags: list[bool] = []
        seen_in_chunk = 0
        for fi in chunk_frame_info:
            if fi["type"] == "I":
                is_first = (prior_iframes + seen_in_chunk) == 0
                fi["is_first_iframe"] = bool(is_first)
                iframe_flags.append(is_first)
                seen_in_chunk += 1

        iframe_budget = self.iframe_budget_allocator.allocate(
            mode=self.iframe_compression_mode,
            tokens_per_frame=N,
            is_first_iframe=iframe_flags,
            num_p_frames=num_p,
            p_budget_old=int(target_p_tokens_old),
            overall_keep_ratio=float(self.retention_ratio),
            compress_when_ratio_one=self.iframe_compress_when_ratio_one,
            keep_ratio=self.i_frame_retention_ratio,
            first_keep_ratio=self.first_iframe_keep_ratio,
            dynamic_keep_ratio=self.dynamic_iframe_keep_ratio,
            min_keep_ratio=self.iframe_min_keep_ratio,
        )
        if iframe_budget.warning:
            print(f"[VTC_Dy][I-frame] WARNING: {iframe_budget.warning}")
            state["iframe_compression_warnings"].append(iframe_budget.warning)
        target_p_tokens = int(iframe_budget.p_budget_new)
        assert 0 <= target_p_tokens <= num_p * N

        alloc_start = time.perf_counter()
        min_keep_relaxed = False
        p_allocations = self._uniform_p_allocations(num_p, target_p_tokens, N)
        codec_fallback_count = 0
        residual_mean_fallback_count = 0
        if num_p > 0 and self.token_allocation_mode == "residual_mean":
            p_complexities = []
            for fi, energy_mean in zip(chunk_frame_info, chunk_energy_means):
                if fi["type"] != "P":
                    continue
                if energy_mean is None or not math.isfinite(float(energy_mean)):
                    complexity = 1.0
                    residual_mean_fallback_count += 1
                else:
                    # Residual-mean allocation uses the semantic residual
                    # before applying codec_budget_power.
                    complexity = math.sqrt(max(0.0, float(energy_mean)) + 1e-6)
                fi["p_budget_complexity"] = round(float(complexity), 6)
                p_complexities.append(float(complexity))
            allocation = CodecBitcostAllocator().allocate(
                p_complexities,
                total_budget=target_p_tokens,
                min_keep=min(N, max(0, int(self.p_frame_min_tokens))),
                max_keep=N,
                budget_power=self.codec_budget_power,
                max_multiplier=self.codec_budget_max_multiplier,
            )
            p_allocations = allocation.keep_counts
            min_keep_relaxed = bool(allocation.min_keep_relaxed)
        elif (
            num_p > 0
            and self.token_allocation_mode == "dyta"
            and state.get("codec_alignment_valid")
            and state.get("codec_step_stats") is not None
        ):
            p_complexities = []
            for fi in chunk_frame_info:
                if fi["type"] == "P":
                    complexity = fi.get("codec_complexity")
                    if complexity is None or not math.isfinite(float(complexity)):
                        complexity = 1.0
                        codec_fallback_count += 1
                    p_complexities.append(float(complexity))
            allocation = CodecBitcostAllocator().allocate(
                p_complexities,
                total_budget=target_p_tokens,
                min_keep=min(N, max(0, int(self.p_frame_min_tokens))),
                max_keep=N,
                budget_power=self.codec_budget_power,
                max_multiplier=self.codec_budget_max_multiplier,
            )
            p_allocations = allocation.keep_counts
            min_keep_relaxed = bool(allocation.min_keep_relaxed)
        elif self.token_allocation_mode == "dyta":
            codec_fallback_count = num_p
        state["codec_allocation_time_ms"] += (time.perf_counter() - alloc_start) * 1000.0
        state["total_codec_fallback_count"] += int(codec_fallback_count)
        state["total_residual_mean_fallback_count"] += int(residual_mean_fallback_count)
        assert len(p_allocations) == num_p
        assert sum(p_allocations) == target_p_tokens
        assert all(0 <= k <= N for k in p_allocations)

        masks: List[torch.Tensor] = []
        p_idx = 0
        i_idx = 0
        iframe_kept_tokens = 0
        iframe_released_tokens = 0
        for local_t, (fi, energy) in enumerate(zip(chunk_frame_info, chunk_energy)):
            global_t = int(fi["frame"])
            if fi["type"] == "I":
                k_i = int(iframe_budget.keep_counts[i_idx])
                selection = self.iframe_token_selector.select(
                    feats[local_t],
                    keep_tokens=k_i,
                    token_grid_h=H_spatial,
                    token_grid_w=W_spatial,
                    residual_energy=energy,
                    is_first_iframe=bool(fi.get("is_first_iframe", False)),
                    mode=self.iframe_selection_mode,
                    anchor_ratio=self.iframe_anchor_ratio,
                    anchor_grid=self.iframe_anchor_grid,
                    salient_residual_weight=self.i_frame_residual_weight,
                    global_frame_idx=global_t,
                )
                keep_indices = selection.keep_indices.to(features.device)
                mask = torch.zeros(N, dtype=torch.bool, device=features.device)
                mask[keep_indices] = True
                kept = int(keep_indices.numel())
                if kept != k_i:
                    raise AssertionError(f"[VTC_Dy] I-frame keep mismatch frame={global_t}, kept={kept}, K={k_i}")
                release = N - k_i
                fi.update({
                    "kept": k_i,
                    "allocated_tokens": k_i,
                    "iframe_compression_mode": self.iframe_compression_mode,
                    "iframe_selection_mode": self.iframe_selection_mode,
                    "original_tokens": N,
                    "kept_tokens": k_i,
                    "released_tokens": release,
                    "keep_ratio": round(k_i / float(N), 6) if N else 0.0,
                    "num_anchor_tokens": selection.num_anchor_tokens,
                    "num_salient_tokens": selection.num_salient_tokens,
                    "anchor_grid": self.iframe_anchor_grid,
                    "iframe_compression_skipped_reason": iframe_budget.skipped_reason,
                })
                iframe_kept_tokens += k_i
                iframe_released_tokens += release
                i_idx += 1
            else:
                k_p = int(p_allocations[p_idx])
                p_idx += 1
                if energy is None:
                    raise AssertionError("[VTC_Dy] missing energy for P-frame")
                mask, kept, token_select_stats = self._select_p_frame_tokens(
                    F_t=feats[local_t],
                    Energy=energy,
                    N=N,
                    K=k_p,
                    frame_idx=global_t,
                    H_spatial=H_spatial,
                    W_spatial=W_spatial,
                )
                if kept != k_p:
                    raise AssertionError(f"[VTC_Dy] P-frame keep mismatch frame={global_t}, kept={kept}, K={k_p}")
                fi["kept"] = kept
                fi["allocated_tokens"] = kept
                fi.update(token_select_stats)
            fi["allocated_ratio"] = round(float(fi["kept"]) / float(N), 6) if N else 0.0
            masks.append(mask)

        if self.log_frame_allocations:
            for fi in chunk_frame_info:
                print(self._format_frame_allocation_line(fi))

        keep_mask = torch.cat(masks, dim=0)
        pruned_features = features[keep_mask]
        kept_tokens = int(keep_mask.sum().item())
        expected_tokens = iframe_kept_tokens + sum(p_allocations)
        assert kept_tokens == expected_tokens

        state["prev_full_frame"] = feats[-1].detach().clone()
        state["semantic_iframe_count"] = prior_iframes + num_i
        state["frame_info"].extend(chunk_frame_info)
        state["keep_masks"].append(keep_mask.detach())
        state["original_tokens"] += int(total_tokens)
        state["kept_tokens"] += kept_tokens
        state["target_kept_tokens"] += int(target_kept_tokens)
        state["target_p_frame_kept_tokens"] += int(target_p_tokens)
        state["total_iframe_original_tokens"] += int(num_i * N)
        state["total_iframe_kept_tokens"] += int(iframe_kept_tokens)
        state["total_iframe_released_tokens"] += int(iframe_released_tokens)
        state["p_budget_total_increase_from_iframe"] += int(target_p_tokens - target_p_tokens_old)

        p_values = torch.tensor(p_allocations, dtype=torch.float32)
        chunk_summary = {
            "chunk_start_t": int(global_start_t),
            "chunk_end_t": int(global_start_t + T),
            "chunk_grid_thw": [T, H_grid, W_grid],
            "original_tokens": int(total_tokens),
            "kept_tokens": kept_tokens,
            "dropped_tokens": int(total_tokens - kept_tokens),
            "target_kept_tokens": int(target_kept_tokens),
            "target_p_frame_kept_tokens": int(target_p_tokens),
            "full_target_kept_tokens": state.get("full_target_kept_tokens"),
            "codec_budget_mode": self.token_allocation_mode,
            "token_allocation_mode": self.token_allocation_mode,
            "semantic_frame_type_source": (
                "codec_ip" if codec_ip_active else (
                    "residual_fallback"
                    if self.semantic_frame_type_source == "codec_ip"
                    else "residual"
                )
            ),
            "iframe_compression_mode": self.iframe_compression_mode,
            "num_iframes": int(num_i),
            "num_pframes": int(num_p),
            "iframe_original_tokens": int(num_i * N),
            "iframe_kept_tokens": int(iframe_kept_tokens),
            "iframe_released_tokens": int(iframe_released_tokens),
            "p_budget_old": int(target_p_tokens_old),
            "p_budget_new": int(target_p_tokens),
            "p_budget_increase_from_iframe": int(target_p_tokens - target_p_tokens_old),
            "p_budget_chunk": int(target_p_tokens),
            "p_budget_sum": int(sum(p_allocations)),
            "num_semantic_i_frames": int(num_i),
            "num_semantic_p_frames": int(num_p),
            "num_i_frames": int(num_i),
            "num_p_frames": int(num_p),
            "p_frame_min_tokens": int(p_values.min().item()) if num_p else None,
            "p_frame_max_tokens": int(p_values.max().item()) if num_p else None,
            "p_frame_mean_tokens": round(float(p_values.mean().item()), 6) if num_p else None,
            "p_frame_std_tokens": round(float(p_values.std(unbiased=False).item()), 6) if num_p else None,
            "min_keep_relaxed": bool(min_keep_relaxed),
            "codec_fallback_count": int(codec_fallback_count),
            "residual_mean_fallback_count": int(residual_mean_fallback_count),
            "iframe_compression_skipped_reason": iframe_budget.skipped_reason,
            "target_retention_ratio": self.retention_ratio,
            "actual_overall_keep_ratio": round(kept_tokens / float(total_tokens), 6) if total_tokens else 0.0,
            "last_full_frame_retained": True,
        }
        state["chunk_summaries"].append(chunk_summary)
        return pruned_features, keep_mask, chunk_summary

    def forward(
        self,
        features: torch.Tensor,
        grid_thw: torch.Tensor,
        video_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = self.init_stream_state(grid_thw)
        pruned, _, _ = self.prune_chunk(features, grid_thw, state, global_start_t=0, video_id=video_id)
        _, stats = self.finalize_stream_stats(state, grid_thw, video_id=video_id)
        return pruned, stats

    def _select_p_frame_tokens(
        self,
        *,
        F_t: torch.Tensor,
        Energy: torch.Tensor,
        N: int,
        K: int,
        frame_idx: int,
        H_spatial: int,
        W_spatial: int,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        del F_t, frame_idx
        K = min(int(N), max(0, int(K)))
        if self.token_selection_mode == "global_topk":
            mask, kept = self._select_tokens_with_k(Energy, N, K)
            return mask, kept, {
                "token_selection_mode": self.token_selection_mode,
            }
        indices, select_stats = self._select_sc_topk(
            Energy,
            keep_count=K,
            spatial_shape=(H_spatial, W_spatial),
        )
        if indices.numel() != K:
            raise AssertionError(f"[VTC_Dy] selected token count mismatch: selected={indices.numel()}, K={K}")
        mask = torch.zeros(N, dtype=torch.bool, device=Energy.device)
        mask[indices] = True
        kept = int(indices.numel())
        return mask, kept, select_stats

    def _infer_spatial_layout(self, N: int, spatial_shape: Tuple[int, int]) -> Tuple[int, int]:
        H, W = int(spatial_shape[0]), int(spatial_shape[1])
        if H <= 0 or W <= 0 or H * W != int(N):
            raise ValueError(f"Cannot map tokens to 2D grid: H={H}, W={W}, N={N}")
        return H, W

    def _build_regions(self, H: int, W: int, device: torch.device) -> List[torch.Tensor]:
        key = (
            int(H),
            int(W),
            self.region_grid_h,
            self.region_grid_w,
            str(device),
        )
        cached = self._spatial_region_cache.get(key)
        if cached is not None:
            return cached
        row_chunks = torch.tensor_split(torch.arange(H, device=device), self.region_grid_h)
        col_chunks = torch.tensor_split(torch.arange(W, device=device), self.region_grid_w)
        regions: List[torch.Tensor] = []
        for rows in row_chunks:
            for cols in col_chunks:
                if rows.numel() == 0 or cols.numel() == 0:
                    regions.append(torch.empty(0, dtype=torch.long, device=device))
                    continue
                tokens = (rows[:, None] * W + cols[None, :]).reshape(-1).long()
                regions.append(tokens)
        self._spatial_region_cache[key] = regions
        return regions

    def _allocate_region_budget(self, region_sizes: List[int], coverage_budget: int) -> List[int]:
        valid = [idx for idx, size in enumerate(region_sizes) if size > 0]
        budgets = [0 for _ in region_sizes]
        if coverage_budget <= 0 or not valid:
            return budgets
        remaining = min(int(coverage_budget), sum(region_sizes))
        q = remaining // len(valid)
        r = remaining % len(valid)
        for order, idx in enumerate(valid):
            keep = min(region_sizes[idx], q + (1 if order < r else 0))
            budgets[idx] = keep
        remaining -= sum(budgets)
        while remaining > 0:
            progressed = False
            for idx in valid:
                if budgets[idx] < region_sizes[idx]:
                    budgets[idx] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
        return budgets

    def _select_from_candidates(self, scores: torch.Tensor, candidates: torch.Tensor, keep_count: int) -> torch.Tensor:
        K = min(int(candidates.numel()), max(0, int(keep_count)))
        if K == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        if K == candidates.numel():
            return candidates.sort().values
        cand_scores = torch.nan_to_num(
            scores[candidates].detach().float(),
            nan=-float("inf"),
            posinf=float("inf"),
            neginf=-float("inf"),
        )
        if not torch.isfinite(cand_scores).any() or cand_scores.max().item() < 1e-8:
            return candidates[:K].sort().values
        _, local = torch.topk(cand_scores, K, largest=True, sorted=False)
        return candidates[local].sort().values

    def _select_sc_topk(
        self,
        Energy: torch.Tensor,
        *,
        keep_count: int,
        spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        N = int(Energy.numel())
        K = min(N, max(0, int(keep_count)))
        if K == 0:
            return torch.empty(0, device=Energy.device, dtype=torch.long), {
                "token_selection_mode": self.token_selection_mode,
            }
        if K >= N:
            all_indices = torch.arange(N, device=Energy.device, dtype=torch.long)
            return all_indices, {"token_selection_mode": self.token_selection_mode}
        H, W = self._infer_spatial_layout(N, spatial_shape)
        regions = self._build_regions(H, W, Energy.device)
        valid_region_count = sum(1 for region in regions if region.numel() > 0)
        coverage_budget = min(
            K,
            max(
                valid_region_count * self.region_min_tokens,
                int(math.floor(max(0.0, self.coverage_ratio) * K)),
            ),
        )
        region_sizes = [int(region.numel()) for region in regions]
        region_budgets = self._allocate_region_budget(region_sizes, coverage_budget)

        selected_mask = torch.zeros(N, dtype=torch.bool, device=Energy.device)
        for region, budget in zip(regions, region_budgets):
            if budget <= 0 or region.numel() == 0:
                continue
            local_selected = self._select_from_candidates(Energy, region, budget)
            selected_mask[local_selected] = True
        coverage_count = int(sum(region_budgets))

        global_count = K - coverage_count
        if global_count > 0:
            remaining = (~selected_mask).nonzero(as_tuple=False).flatten()
            global_indices = self._select_from_candidates(Energy, remaining, global_count)
            selected_mask[global_indices] = True

        indices = selected_mask.nonzero(as_tuple=False).flatten().sort().values
        return indices, {"token_selection_mode": self.token_selection_mode}

    def finalize_stream_stats(
        self,
        state: Dict[str, Any],
        full_grid_thw: torch.Tensor,
        *,
        video_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        T, H_grid, W_grid = [int(x) for x in full_grid_thw.tolist()]
        N = H_grid * W_grid // (self.spatial_merge_size ** 2)
        keep_mask = torch.cat(state["keep_masks"], dim=0)
        original_tokens = int(state["original_tokens"])
        kept_tokens = int(state["kept_tokens"])
        frame_info = state["frame_info"]
        if original_tokens != T * N:
            raise AssertionError(f"[VTC_Dy] stream token mismatch: original={original_tokens}, expected={T*N}")
        if keep_mask.numel() != original_tokens:
            raise AssertionError(f"[VTC_Dy] keep mask mismatch: mask={keep_mask.numel()}, expected={original_tokens}")

        num_i = sum(1 for fi in frame_info if fi["type"] == "I")
        num_p = sum(1 for fi in frame_info if fi["type"] == "P")
        p_original = num_p * N
        p_kept = sum(int(fi.get("kept", 0)) for fi in frame_info if fi["type"] == "P")
        p_ratio = (p_kept / p_original) if p_original > 0 else 1.0
        p_tokens_per_frame = (p_kept / num_p) if num_p > 0 else 0.0
        target_kept_tokens = int(state.get("full_target_kept_tokens", state["target_kept_tokens"]))
        target_p_frame_kept_tokens = max(0, target_kept_tokens - num_i * N)

        step_stats = state.get("codec_step_stats") or []
        complexities = torch.tensor(
            [float(s.codec_complexity) for s in step_stats],
            dtype=torch.float32,
        ) if step_stats else torch.empty(0)
        iframe_rows = [fi for fi in frame_info if fi.get("type") == "I"]
        first_rows = [fi for fi in iframe_rows if fi.get("is_first_iframe")]
        dynamic_rows = [fi for fi in iframe_rows if not fi.get("is_first_iframe")]
        total_iframe_original = int(len(iframe_rows) * N)
        total_iframe_kept = int(sum(int(fi.get("kept", 0)) for fi in iframe_rows))
        total_iframe_released = int(total_iframe_original - total_iframe_kept)

        stats = {
            "video_id": video_id,
            "grid_thw": [T, H_grid, W_grid],
            "frame_info": frame_info,
            "original_tokens": original_tokens,
            "kept_tokens": kept_tokens,
            "dropped_tokens": original_tokens - kept_tokens,
            "target_retention_ratio": self.retention_ratio,
            "target_kept_tokens": target_kept_tokens,
            "target_p_frame_kept_tokens": target_p_frame_kept_tokens,
            "overall_keep_ratio": round(kept_tokens / original_tokens, 4) if original_tokens else 1.0,
            "actual_overall_keep_ratio": round(kept_tokens / original_tokens, 4) if original_tokens else 1.0,
            "p_frame_keep_ratio": round(p_ratio, 4),
            "num_i_frames": num_i,
            "i_frame_tokens_per_frame": N,
            "num_p_frames": num_p,
            "p_frame_tokens_per_frame": round(p_tokens_per_frame, 4),
            "fill_mode": "drop",
            "keep_mask": keep_mask,
            "chunk_pruning": True,
            "rate_control_scope": "per_chunk_streaming",
            "chunk_summaries": state["chunk_summaries"],
            "semantic_version": self.semantic_version,
            "semantic_frame_type_source": self.semantic_frame_type_source,
            "semantic_frame_type_source_effective": (
                "codec_ip"
                if self.semantic_frame_type_source == "codec_ip"
                and state.get("codec_ip_alignment_valid")
                else (
                    "residual_fallback"
                    if self.semantic_frame_type_source == "codec_ip"
                    else "residual"
                )
            ),
            "token_selection_mode": self.token_selection_mode,
            "method_stack": self._method_stack(),
            "codec_name": state.get("codec_name"),
            "codec_frame_count": int(state.get("codec_frame_count", 0)),
            "codec_i_count": int(state.get("codec_i_count", 0)),
            "codec_p_count": int(state.get("codec_p_count", 0)),
            "codec_b_count": int(state.get("codec_b_count", 0)),
            "codec_alignment_valid": bool(state.get("codec_alignment_valid", False)),
            "codec_fallback_reason": state.get("codec_fallback_reason"),
            "codec_ip_alignment_valid": bool(
                state.get("codec_ip_alignment_valid", False)
            ),
            "codec_ip_fallback_reason": state.get("codec_ip_fallback_reason"),
            "codec_budget_mode": self.token_allocation_mode,
            "token_allocation_mode": self.token_allocation_mode,
            "codec_aggregation": self.codec_aggregation,
            "codec_normalization": self.codec_normalization,
            "codec_complexity_mean": round(float(complexities.mean().item()), 6) if complexities.numel() else None,
            "codec_complexity_std": round(float(complexities.std(unbiased=False).item()), 6) if complexities.numel() else None,
            "codec_complexity_min": round(float(complexities.min().item()), 6) if complexities.numel() else None,
            "codec_complexity_max": round(float(complexities.max().item()), 6) if complexities.numel() else None,
            "total_codec_fallback_count": int(state.get("total_codec_fallback_count", 0)),
            "total_residual_mean_fallback_count": int(
                state.get("total_residual_mean_fallback_count", 0)
            ),
            "codec_extraction_time_ms": state.get("codec_extraction_time_ms", 0.0),
            "codec_alignment_time_ms": state.get("codec_alignment_time_ms", 0.0),
            "codec_summary_time_ms": state.get("codec_summary_time_ms", 0.0),
            "codec_allocation_time_ms": round(float(state.get("codec_allocation_time_ms", 0.0)), 6),
            "codec_runtime_cache_hit": bool(
                state.get("codec_runtime_cache_hit", False)
            ),
            "codec_metadata_summary": state.get("codec_metadata_summary") or {},
            "iframe_compression_mode": self.iframe_compression_mode,
            "iframe_selection_mode": self.iframe_selection_mode,
            "i_frame_retention_ratio": self.i_frame_retention_ratio,
            "semantic_first_iframe_keep_ratio": self.first_iframe_keep_ratio,
            "semantic_dynamic_iframe_keep_ratio": self.dynamic_iframe_keep_ratio,
            "semantic_iframe_min_keep_ratio": self.iframe_min_keep_ratio,
            "semantic_iframe_anchor_ratio": self.iframe_anchor_ratio,
            "semantic_iframe_anchor_grid": self.iframe_anchor_grid,
            "i_frame_residual_weight": self.i_frame_residual_weight,
            "semantic_iframe_compress_when_ratio_one": self.iframe_compress_when_ratio_one,
            "total_iframe_original_tokens": total_iframe_original,
            "total_iframe_kept_tokens": total_iframe_kept,
            "total_iframe_released_tokens": total_iframe_released,
            "mean_iframe_keep_ratio": round(total_iframe_kept / total_iframe_original, 6)
            if total_iframe_original else 1.0,
            "first_iframe_keep_ratio_actual": round(
                sum(int(fi.get("kept", 0)) for fi in first_rows) / float(max(1, len(first_rows) * N)),
                6,
            ) if first_rows else None,
            "dynamic_iframe_keep_ratio_actual": round(
                sum(int(fi.get("kept", 0)) for fi in dynamic_rows) / float(max(1, len(dynamic_rows) * N)),
                6,
            ) if dynamic_rows else None,
            "p_budget_total_increase_from_iframe": int(state.get("p_budget_total_increase_from_iframe", 0)),
            "iframe_compression_warnings": list(state.get("iframe_compression_warnings", [])),
            "token_selection_mode": self.token_selection_mode,
            "region_grid_h": self.region_grid_h,
            "region_grid_w": self.region_grid_w,
            "coverage_ratio": self.coverage_ratio,
            "region_min_tokens": self.region_min_tokens,
        }
        return keep_mask, stats

    def format_stats(self, stats: Dict[str, Any]) -> str:
        grid = stats["grid_thw"]
        N = grid[1] * grid[2] // self.spatial_merge_size ** 2
        budget_mode = str(stats.get("token_allocation_mode") or stats.get("codec_budget_mode"))
        if budget_mode == "dyta":
            budget_detail = (
                f"codec_alignment={stats.get('codec_alignment_valid')} "
                f"fallback={stats.get('codec_fallback_reason')}"
            )
        elif budget_mode == "residual_mean":
            budget_detail = (
                f"mean_fallbacks={stats.get('total_residual_mean_fallback_count', 0)}"
            )
        else:
            budget_detail = "uniform"
        lines = [
            "=" * 58,
            f"=== Visual Token Coding Dynamic ({self.semantic_version}) ===",
            (
                f"  T={grid[0]} N={N} ratio={self.retention_ratio:.2f} "
                f"token_select={self.token_selection_mode}"
            ),
            (
                "  Semantic I/P: "
                f"{stats.get('semantic_frame_type_source_effective')} "
                f"codec_ip_alignment={stats.get('codec_ip_alignment_valid')} "
                f"fallback={stats.get('codec_ip_fallback_reason')}"
            ),
            (
                f"  P budget: {budget_mode} {budget_detail}"
            ),
            (
                f"  I-frame: {stats.get('iframe_compression_mode')} "
                f"released={stats.get('total_iframe_released_tokens')} "
                f"mean_keep={stats.get('mean_iframe_keep_ratio')}"
            ),
            (
                f"  Tokens: original={stats['original_tokens']} kept={stats['kept_tokens']} "
                f"dropped={stats['dropped_tokens']} overall={stats['overall_keep_ratio'] * 100:.1f}%"
            ),
            (
                f"  Frames: I={stats['num_i_frames']} P={stats['num_p_frames']} "
                f"P_keep={stats['p_frame_keep_ratio'] * 100:.1f}%"
            ),
            "=" * 58,
        ]
        return "\n".join(lines)


__all__ = ["VisualTokenCoding"]
