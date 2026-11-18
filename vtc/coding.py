"""Semantic I/P coding with codec-guided integer budgets."""

from __future__ import annotations

import collections
from pathlib import Path
import subprocess
import time
import warnings

import torch

from .codec_priors import (
    CodecBitcostExtractor,
    CodecTemporalAligner,
    VideoSamplingContext,
)
from .dynamic_budget import CodecGuidedBudgetAllocator
from .reference_budget import ReferenceBudgetAllocator
from .frame_redundancy import FrameRedundancySuppression, ReferenceTokenSelector


class VisualTokenCoding:
    """VTC and VTCdy use the same residual decisions and video token target.

    Feature tensors and selected indices stay on their device. Only the
    residual means required by sequential I/P decisions move to CPU once
    per chunk. Grid metadata is supplied on CPU by the Qwen3 adapter.
    """

    def __init__(
        self,
        method="vtcdy",
        retention_ratio=0.5,
        keyframe_threshold=2.0,
        window_size=8,
        min_p_tokens=8,
        spatial_merge_size=2,
        iframe_keep_ratio=0.9,
        codec_cache_dir=None,
        codec_ffprobe_path="ffprobe",
        codec_strict=False,
        codec_mean_weight=0.5,
        codec_budget_power=0.5,
        codec_budget_max_multiplier=2.0,
        log_stats=False,
    ):
        if method not in ("vtc", "vtcdy"):
            raise ValueError("method must be vtc or vtcdy")
        if not 0 < retention_ratio <= 1 or not 0 <= codec_mean_weight <= 1:
            raise ValueError(
                "retention_ratio must be in (0,1], codec_mean_weight in [0,1]"
            )
        self.method, self.retention_ratio = method, retention_ratio
        self.keyframe_threshold, self.window_size = keyframe_threshold, window_size
        self.min_p_tokens, self.spatial_merge_size = min_p_tokens, spatial_merge_size
        self.iframe_keep_ratio = iframe_keep_ratio
        self.temporal_patch_size = 2
        self.codec_cache_dir, self.codec_ffprobe_path = (
            codec_cache_dir,
            codec_ffprobe_path,
        )
        self.codec_strict, self.codec_mean_weight = codec_strict, codec_mean_weight
        self.codec_budget_power = codec_budget_power
        self.codec_budget_max_multiplier = codec_budget_max_multiplier
        self.log_stats = log_stats
        self._sampling_contexts = []
        self._codec_cache = None
        self._warned_codec_errors = set()
        self._budget = CodecGuidedBudgetAllocator()
        self._references = ReferenceBudgetAllocator()
        self._reference_selection = ReferenceTokenSelector()
        self._frs = FrameRedundancySuppression()

    def set_temporal_patch_size(self, size):
        self.temporal_patch_size = size

    def set_video_sampling_contexts(self, contexts):
        self._sampling_contexts = [
            VideoSamplingContext(**c) if isinstance(c, dict) else c for c in contexts
        ]

    def clear_video_sampling_contexts(self):
        self._sampling_contexts.clear()

    def _codec_data(self, context):
        path = Path(context.video_path).resolve()
        stat = path.stat()
        key = (
            str(path),
            stat.st_size,
            stat.st_mtime_ns,
            context.grid_t,
            context.temporal_patch_size,
            tuple(context.sampled_frame_indices),
            tuple(context.sampled_timestamps or []),
            context.source_fps,
            self.codec_mean_weight,
            self.codec_ffprobe_path,
            self.codec_strict,
        )
        if self._codec_cache is not None and self._codec_cache[0] == key:
            return self._codec_cache[1], True
        started = time.perf_counter()
        metadata = CodecBitcostExtractor(
            ffprobe_path=self.codec_ffprobe_path,
            cache_dir=self.codec_cache_dir,
            strict=self.codec_strict,
        ).extract(path)
        extracted = time.perf_counter()
        alpha = self.codec_mean_weight
        steps = CodecTemporalAligner(
            aggregation="mean" if alpha == 1 else "mean_p90",
            mean_weight=alpha,
            p90_weight=1 - alpha,
        ).align_to_temporal_steps(metadata, context)
        aligned = time.perf_counter()
        data = {
            "steps": steps,
            "summary": {
                "codec_name": metadata.codec_name,
                "width": metadata.width,
                "height": metadata.height,
                "total_frames": metadata.total_frames,
            },
            "extraction_ms": (extracted - started) * 1000,
            "alignment_ms": (aligned - extracted) * 1000,
        }
        self._codec_cache = key, data
        return data, False

    def preload_codec(self):
        """Explicit preprocessing; its wall time must be reported separately."""
        if self.method == "vtcdy":
            self._codec_data(self._sampling_contexts[0])

    def init_stream_state(self, full_grid_thw):
        total_t, height, width = map(int, full_grid_thw.tolist())
        per_frame = height * width // self.spatial_merge_size**2
        state = {
            "energy_window": collections.deque(maxlen=self.window_size),
            "prev_full_frame": None,
            "full_T": total_t,
            "full_target_kept_tokens": round(
                total_t * per_frame * self.retention_ratio
            ),
            "frame_info": [],
            "keep_masks": [],
            "keep_indices": [],
            "chunk_summaries": [],
            "original_tokens": 0,
            "kept_tokens": 0,
            "codec_alignment_valid": False,
            "codec_runtime_cache_hit": False,
            "codec_fallback_reason": None,
            "codec_step_stats": [],
            "codec_metadata_summary": {},
            "codec_extraction_time_ms": 0.0,
            "codec_alignment_time_ms": 0.0,
            "codec_allocation_time_ms": 0.0,
            "total_codec_fallback_count": 0,
            "total_iframe_released_tokens": 0,
        }
        context = self._sampling_contexts.pop(0) if self._sampling_contexts else None
        if self.method != "vtcdy":
            return state
        try:
            if context is None:
                raise ValueError("missing video sampling context for codec budgets")
            context.grid_t, context.temporal_patch_size = (
                total_t,
                self.temporal_patch_size,
            )
            if len(context.sampled_frame_indices) < total_t * self.temporal_patch_size:
                raise ValueError(
                    "sampled frame count is inconsistent with the visual grid"
                )
            data, hit = self._codec_data(context)
            if len(data["steps"]) != total_t:
                raise ValueError(
                    "codec alignment length is inconsistent with the visual grid"
                )
            state.update(
                codec_alignment_valid=True,
                codec_runtime_cache_hit=hit,
                codec_step_stats=data["steps"],
                codec_metadata_summary=data["summary"],
                codec_extraction_time_ms=0.0 if hit else data["extraction_ms"],
                codec_alignment_time_ms=0.0 if hit else data["alignment_ms"],
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            if self.codec_strict:
                raise
            state["codec_fallback_reason"] = str(exc)
            if str(exc) not in self._warned_codec_errors:
                warnings.warn(
                    f"Codec budget unavailable; using uniform P budgets: {exc}"
                )
                self._warned_codec_errors.add(str(exc))
        return state

    def is_keyframe(self, mean, history):
        if len(history) < 2:
            return False
        baseline = sum(history) / len(history)
        if baseline < 1e-8:
            return mean > 1e-6
        return mean > self.keyframe_threshold * baseline

    def _p_budget(self, state, end_t, num_i, num_p, per_frame):
        if num_p == 0:
            return 0
        future = max(0, state["full_T"] - end_t)
        remaining = max(
            0,
            state["full_target_kept_tokens"] - state["kept_tokens"] - num_i * per_frame,
        )
        return min(num_p * per_frame, round(remaining * (num_p / (num_p + future))))

    @staticmethod
    def _topk(energy, count):
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=energy.device)
        clean = torch.nan_to_num(
            energy.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        indices = torch.topk(clean, count, largest=True).indices
        fallback = torch.arange(count, device=energy.device)
        return torch.where(clean.max() < 1e-8, fallback, indices).sort().values

    def prune_chunk(
        self, features, chunk_grid_thw, state, *, global_start_t, video_id=None
    ):
        total_t, height, width = map(int, chunk_grid_thw.tolist())
        height //= self.spatial_merge_size
        width //= self.spatial_merge_size
        per_frame = height * width
        if features.shape[0] != total_t * per_frame:
            raise ValueError("visual feature count does not match the chunk grid")
        frames = features.view(total_t, per_frame, -1)
        previous = state["prev_full_frame"]
        # Keep each original per-frame reduction shape. Stack scalar means
        # before copying, so classification needs one D2H transfer per chunk.
        energies = []
        means = []
        for local_t in range(total_t):
            if global_start_t + local_t == 0:
                energies.append(None)
            else:
                reference = previous if local_t == 0 else frames[local_t - 1]
                energy = torch.linalg.vector_norm(frames[local_t] - reference, dim=-1)
                energies.append(energy)
                means.append(energy.detach().float().mean())
        cpu_means = iter(torch.stack(means).cpu().tolist() if means else [])
        rows = []
        history = state["energy_window"]
        for local_t, energy in enumerate(energies):
            index = global_start_t + local_t
            mean = None if energy is None else next(cpu_means)
            if index == 0:
                kind = "I"
            else:
                kind = "I" if self.is_keyframe(mean, history) else "P"
                history.append(mean)
                if kind == "I":
                    history.clear()
            rows.append(
                {
                    "frame": index,
                    "type": kind,
                    "semantic_frame_type": kind,
                    "energy_mean": round(mean, 4) if mean is not None else None,
                }
            )
        num_i = sum(row["type"] == "I" for row in rows)
        num_p = total_t - num_i
        old_p_budget = self._p_budget(
            state, global_start_t + total_t, num_i, num_p, per_frame
        )
        first_i = not any(row["type"] == "I" for row in state["frame_info"])
        flags = []
        for row in rows:
            if row["type"] == "I":
                row["is_first_iframe"] = first_i
                flags.append(first_i)
                first_i = False
        i_budget = self._references.allocate(
            mode="budgeted" if self.method == "vtcdy" else "none",
            tokens_per_frame=per_frame,
            is_first_iframe=flags,
            num_p_frames=num_p,
            p_budget_old=old_p_budget,
            overall_keep_ratio=self.retention_ratio,
            keep_ratio=self.iframe_keep_ratio,
        )
        if i_budget.warning:
            warnings.warn(i_budget.warning)
        p_budget = i_budget.p_budget_new
        min_relaxed = False
        start = time.perf_counter()
        if num_p and self.method == "vtcdy" and state["codec_alignment_valid"]:
            complexities = [
                round(
                    float(state["codec_step_stats"][row["frame"]].codec_complexity), 6
                )
                for row in rows
                if row["type"] == "P"
            ]
            result = self._budget.allocate(
                complexities,
                p_budget,
                min(per_frame, self.min_p_tokens),
                per_frame,
                self.codec_budget_power,
                self.codec_budget_max_multiplier,
            )
            p_counts, min_relaxed = result.keep_counts, result.min_keep_relaxed
        else:
            q, r = divmod(p_budget, num_p) if num_p else (0, 0)
            p_counts = [q + (i < r) for i in range(num_p)]
            if self.method == "vtcdy" and not state["codec_alignment_valid"]:
                state["total_codec_fallback_count"] += num_p
        state["codec_allocation_time_ms"] += (time.perf_counter() - start) * 1000
        i_counts, p_counts = iter(i_budget.keep_counts), iter(p_counts)
        selected = []
        for local_t, (row, energy) in enumerate(zip(rows, energies)):
            if row["type"] == "I":
                count = next(i_counts)
                indices = self._reference_selection.select(
                    frames[local_t], count, height, width, energy
                )
            else:
                count = next(p_counts)
                indices = (
                    self._frs.select(energy, count, height, width)
                    if self.method == "vtcdy"
                    else self._topk(energy, count)
                )
            row.update(
                original_tokens=per_frame,
                allocated_tokens=count,
                kept_tokens=count,
                kept=count,
            )
            selected.append(indices + local_t * per_frame)
        keep_indices = torch.cat(selected)
        keep_mask = torch.zeros(
            total_t * per_frame, dtype=torch.bool, device=features.device
        )
        keep_mask[keep_indices] = True
        # Shape is known from integer budgets; reuse indices in DeepStack too.
        pruned = features.index_select(0, keep_indices)
        kept = pruned.shape[0]
        if kept != sum(i_budget.keep_counts) + p_budget:
            raise AssertionError("selected indices do not match the allocated budget")
        state["last_keep_indices"] = keep_indices
        state["keep_indices"].append(keep_indices + global_start_t * per_frame)
        state["keep_masks"].append(keep_mask)
        state["prev_full_frame"] = frames[-1].detach().clone()
        state["frame_info"].extend(rows)
        state["original_tokens"] += total_t * per_frame
        state["kept_tokens"] += kept
        state["total_iframe_released_tokens"] += i_budget.released_tokens
        summary = {
            "chunk_start_t": global_start_t,
            "chunk_end_t": global_start_t + total_t,
            "original_tokens": total_t * per_frame,
            "kept_tokens": kept,
            "num_iframes": num_i,
            "num_pframes": num_p,
            "min_keep_relaxed": min_relaxed,
        }
        state["chunk_summaries"].append(summary)
        return pruned, keep_mask, summary

    def finalize_stream_stats(self, state, full_grid_thw, video_id=None):
        rows = state["frame_info"]
        original, kept = state["original_tokens"], state["kept_tokens"]
        i_rows = [r for r in rows if r["type"] == "I"]
        p_rows = [r for r in rows if r["type"] == "P"]
        mask = torch.cat(state["keep_masks"])
        indices = torch.cat(state["keep_indices"])
        stats = {
            "video_id": video_id,
            "method": self.method,
            "grid_thw": full_grid_thw.tolist(),
            "frame_info": rows,
            "original_tokens": original,
            "kept_tokens": kept,
            "dropped_tokens": original - kept,
            "keep_mask": mask,
            "keep_indices": indices,
            "target_retention_ratio": self.retention_ratio,
            "overall_keep_ratio": kept / original,
            "num_i_frames": len(i_rows),
            "num_p_frames": len(p_rows),
            "i_frame_tokens_per_frame": sum(r["kept"] for r in i_rows) / len(i_rows),
            "p_frame_tokens_per_frame": (
                sum(r["kept"] for r in p_rows) / len(p_rows) if p_rows else 0
            ),
            "total_iframe_released_tokens": state["total_iframe_released_tokens"],
            "chunk_summaries": state["chunk_summaries"],
        }
        stats.update(
            {
                key: value
                for key, value in state.items()
                if key.startswith("codec_") and key != "codec_step_stats"
            }
        )
        stats["total_codec_fallback_count"] = state["total_codec_fallback_count"]
        return mask, stats

    def forward(self, features, grid_thw, video_id=None):
        state = self.init_stream_state(grid_thw)
        pruned, _, _ = self.prune_chunk(features, grid_thw, state, global_start_t=0)
        _, stats = self.finalize_stream_stats(state, grid_thw, video_id)
        return pruned, stats

    def format_stats(self, stats):
        return (
            f"[{self.method}] tokens={stats['kept_tokens']}/{stats['original_tokens']} "
            f"I={stats['num_i_frames']} P={stats['num_p_frames']} "
            f"codec_cache_hit={stats['codec_runtime_cache_hit']}"
        )
