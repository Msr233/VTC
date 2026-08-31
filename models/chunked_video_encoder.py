from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class ChunkedVideoEncoderConfig:
    enabled: bool = False
    chunk_frames: int = 32
    staging_device: str = "cuda"
    collect_stats: bool = False
    verify_shapes: bool = True


def _to_stage(tensor: torch.Tensor, staging_device: str) -> torch.Tensor:
    if staging_device == "cpu":
        return tensor.detach().to("cpu", non_blocking=True)
    return tensor


def _maybe_sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mb(device: torch.device, enabled: bool) -> float | None:
    if not enabled or device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 6)


def _validate_config(
    cfg: ChunkedVideoEncoderConfig,
    temporal_patch_size: int,
) -> int:
    if cfg.chunk_frames <= 0:
        raise ValueError("--vit_chunk_frames must be > 0")
    if cfg.staging_device not in ("cuda", "cpu"):
        raise ValueError("--vit_chunk_staging_device must be 'cuda' or 'cpu'")
    if cfg.chunk_frames % temporal_patch_size != 0:
        raise ValueError(
            "--vit_chunk_frames must be divisible by the model temporal "
            f"patch size: chunk_frames={cfg.chunk_frames}, "
            f"temporal_patch_size={temporal_patch_size}"
        )
    return cfg.chunk_frames // temporal_patch_size


def _cat_to_device(
    chunks: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if not chunks:
        raise ValueError("Cannot concatenate an empty chunk list")
    merged = torch.cat(chunks, dim=0)
    if merged.device != device:
        merged = merged.to(device, non_blocking=True)
    return merged


def encode_video_in_chunks(
    visual_encoder,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    *,
    temporal_patch_size: int,
    spatial_merge_size: int,
    chunk_frames: int = 32,
    staging_device: str = "cuda",
    collect_stats: bool = False,
    verify_shapes: bool = True,
) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor], List[Dict[str, Any]]]:
    """Encode logical videos in temporal chunks, then concatenate features.

    ``pixel_values_videos`` must be the processor output for the full prompt.
    Each logical video is sliced by temporal patch rows from its original
    ``video_grid_thw`` entry; the processor and LLM still see one video.
    """
    cfg = ChunkedVideoEncoderConfig(
        enabled=True,
        chunk_frames=chunk_frames,
        staging_device=staging_device,
        collect_stats=collect_stats,
        verify_shapes=verify_shapes,
    )
    chunk_t = _validate_config(cfg, int(temporal_patch_size))

    if video_grid_thw is None:
        raise ValueError("video_grid_thw is required for ViT chunking")
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
        raise ValueError(
            "video_grid_thw must have shape (num_videos, 3), got "
            f"{tuple(video_grid_thw.shape)}"
        )

    expected_patch_rows = int(video_grid_thw.prod(dim=1).sum().item())
    if int(pixel_values_videos.shape[0]) != expected_patch_rows:
        raise ValueError(
            "pixel_values_videos row count does not match video_grid_thw: "
            f"rows={pixel_values_videos.shape[0]}, expected={expected_patch_rows}"
        )

    original_device = pixel_values_videos.device
    stats: List[Dict[str, Any]] = []
    full_main_by_video: List[torch.Tensor] = []
    deepstack_by_layer: List[List[torch.Tensor]] | None = None

    video_offset = 0
    for video_idx, grid in enumerate(video_grid_thw):
        T, H, W = [int(x) for x in grid.tolist()]
        if H % spatial_merge_size != 0 or W % spatial_merge_size != 0:
            raise ValueError(
                f"Grid H/W must be divisible by spatial_merge_size={spatial_merge_size}: "
                f"video_idx={video_idx}, grid={[T, H, W]}"
            )

        patches_per_t = H * W
        merged_tokens_per_t = patches_per_t // (spatial_merge_size ** 2)
        video_patch_rows = T * patches_per_t
        video_end = video_offset + video_patch_rows
        if video_end > pixel_values_videos.shape[0]:
            raise ValueError(
                f"Patch slice exceeds pixel_values_videos rows for video_idx={video_idx}"
            )

        if collect_stats and original_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(original_device)
        video_start_time = time.perf_counter()

        main_chunks: List[torch.Tensor] = []
        video_deepstack_chunks: List[List[torch.Tensor]] | None = None
        chunk_rows: List[Dict[str, Any]] = []
        prev_end_t = 0

        for chunk_idx, start_t in enumerate(range(0, T, chunk_t)):
            end_t = min(start_t + chunk_t, T)
            if start_t != prev_end_t:
                raise ValueError(
                    f"Chunk temporal order is not continuous: expected {prev_end_t}, got {start_t}"
                )
            prev_end_t = end_t

            start_patch = video_offset + start_t * patches_per_t
            end_patch = video_offset + end_t * patches_per_t
            input_patch_rows = end_patch - start_patch
            if input_patch_rows != (end_t - start_t) * patches_per_t:
                raise ValueError(
                    "Computed chunk patch rows do not match chunk grid: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}"
                )

            chunk_pixel_values = pixel_values_videos[start_patch:end_patch]
            chunk_grid_thw = video_grid_thw.new_tensor([[end_t - start_t, H, W]])

            _maybe_sync(original_device, collect_stats)
            start_time = time.perf_counter()
            with torch.inference_mode():
                chunk_main, chunk_deepstack = visual_encoder(
                    chunk_pixel_values.type(visual_encoder.dtype),
                    grid_thw=chunk_grid_thw,
                )
            _maybe_sync(original_device, collect_stats)
            vit_time_ms = (time.perf_counter() - start_time) * 1000.0

            expected_main_tokens = (end_t - start_t) * merged_tokens_per_t
            if chunk_main.shape[0] != expected_main_tokens:
                raise ValueError(
                    "Visual Encoder output token count mismatch: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                    f"got={chunk_main.shape[0]}, expected={expected_main_tokens}"
                )

            if video_deepstack_chunks is None:
                video_deepstack_chunks = [[] for _ in chunk_deepstack]
            if len(chunk_deepstack) != len(video_deepstack_chunks):
                raise ValueError(
                    "DeepStack layer count changed across chunks: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}"
                )

            deepstack_tokens = []
            for layer_idx, layer_tensor in enumerate(chunk_deepstack):
                if layer_tensor.shape[0] != chunk_main.shape[0]:
                    raise ValueError(
                        "DeepStack token count does not match main features: "
                        f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                        f"layer={layer_idx}, deepstack={layer_tensor.shape[0]}, "
                        f"main={chunk_main.shape[0]}"
                    )
                video_deepstack_chunks[layer_idx].append(
                    _to_stage(layer_tensor, cfg.staging_device)
                )
                deepstack_tokens.append(int(layer_tensor.shape[0]))

            main_chunks.append(_to_stage(chunk_main, cfg.staging_device))
            chunk_rows.append({
                "chunk_idx": chunk_idx,
                "start_t": start_t,
                "end_t": end_t,
                "chunk_grid_thw": [end_t - start_t, H, W],
                "input_patch_rows": int(input_patch_rows),
                "output_main_tokens": int(chunk_main.shape[0]),
                "output_deepstack_tokens": deepstack_tokens,
                "vit_time_ms": round(vit_time_ms, 6),
                "peak_memory_allocated_mb": _peak_mb(original_device, collect_stats),
            })

            del chunk_pixel_values
            del chunk_main
            del chunk_deepstack

        concat_start = time.perf_counter()
        full_main = _cat_to_device(main_chunks, original_device)
        expected_full_tokens = T * merged_tokens_per_t
        if full_main.shape[0] != expected_full_tokens:
            raise ValueError(
                "Concatenated main token count mismatch: "
                f"video_idx={video_idx}, got={full_main.shape[0]}, expected={expected_full_tokens}"
            )

        full_deepstack_for_video: List[torch.Tensor] = []
        if video_deepstack_chunks is not None:
            for layer_idx, layer_chunks in enumerate(video_deepstack_chunks):
                layer_full = _cat_to_device(layer_chunks, original_device)
                if layer_full.shape[0] != full_main.shape[0]:
                    raise ValueError(
                        "Concatenated DeepStack token count mismatch: "
                        f"video_idx={video_idx}, layer={layer_idx}, "
                        f"deepstack={layer_full.shape[0]}, main={full_main.shape[0]}"
                    )
                full_deepstack_for_video.append(layer_full)

        concat_time_ms = (time.perf_counter() - concat_start) * 1000.0

        if deepstack_by_layer is None:
            deepstack_by_layer = [[] for _ in full_deepstack_for_video]
        if len(full_deepstack_for_video) != len(deepstack_by_layer):
            raise ValueError(
                f"DeepStack layer count changed across videos at video_idx={video_idx}"
            )
        for layer_idx, layer_full in enumerate(full_deepstack_for_video):
            deepstack_by_layer[layer_idx].append(layer_full)

        num_chunks = len(chunk_rows)
        video_stats = {
            "video_idx": video_idx,
            "full_grid": [T, H, W],
            "raw_frames": T * int(temporal_patch_size),
            "temporal_steps": T,
            "chunk_frames": cfg.chunk_frames,
            "chunk_temporal_steps": chunk_t,
            "num_chunks": num_chunks,
            "main_tokens": int(full_main.shape[0]),
            "deepstack_layers": len(full_deepstack_for_video),
            "staging_device": cfg.staging_device,
            "chunks": chunk_rows,
            "chunked_vit_total_time_ms": round(
                sum(row["vit_time_ms"] for row in chunk_rows), 6
            ),
            "feature_concat_time_ms": round(concat_time_ms, 6),
            "chunked_vit_peak_memory_mb": _peak_mb(original_device, collect_stats),
            "full_main_token_count": int(full_main.shape[0]),
            "full_deepstack_token_count": [
                int(layer.shape[0]) for layer in full_deepstack_for_video
            ],
        }
        stats.append(video_stats)
        print(
            "[ViTChunk] "
            f"video_idx={video_idx} full_grid={[T,H,W]} "
            f"raw_frames={video_stats['raw_frames']} temporal_steps={T} "
            f"chunk_frames={cfg.chunk_frames} chunk_temporal_steps={chunk_t} "
            f"num_chunks={num_chunks} main_tokens={full_main.shape[0]} "
            f"deepstack_layers={len(full_deepstack_for_video)} "
            f"staging_device={cfg.staging_device}"
        )

        full_main_by_video.append(full_main)
        video_offset = video_end
        video_total_time_ms = (time.perf_counter() - video_start_time) * 1000.0
        video_stats["total_time_ms"] = round(video_total_time_ms, 6)

    if video_offset != pixel_values_videos.shape[0]:
        raise ValueError(
            "Not all pixel_values_videos rows were consumed: "
            f"consumed={video_offset}, rows={pixel_values_videos.shape[0]}"
        )

    if deepstack_by_layer is None:
        full_deepstack = []
    else:
        full_deepstack = [
            _cat_to_device(layer_parts, original_device)
            for layer_parts in deepstack_by_layer
        ]

    return tuple(full_main_by_video), full_deepstack, stats


def encode_and_prune_video_in_chunks(
    visual_encoder,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    pruner,
    *,
    temporal_patch_size: int,
    spatial_merge_size: int,
    chunk_frames: int = 32,
    staging_device: str = "cuda",
    collect_stats: bool = False,
    verify_shapes: bool = True,
) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
    """Encode video chunks and prune each chunk before keeping its outputs.

    This is the V4.2 streaming path.  It never materializes full unpruned video
    main/DeepStack features.  The pruner carries the previous full frame across
    chunks so residual prediction is continuous at chunk boundaries.
    """
    cfg = ChunkedVideoEncoderConfig(
        enabled=True,
        chunk_frames=chunk_frames,
        staging_device=staging_device,
        collect_stats=collect_stats,
        verify_shapes=verify_shapes,
    )
    chunk_t = _validate_config(cfg, int(temporal_patch_size))

    if video_grid_thw is None:
        raise ValueError("video_grid_thw is required for streaming ViT chunk pruning")
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
        raise ValueError(
            "video_grid_thw must have shape (num_videos, 3), got "
            f"{tuple(video_grid_thw.shape)}"
        )

    expected_patch_rows = int(video_grid_thw.prod(dim=1).sum().item())
    if int(pixel_values_videos.shape[0]) != expected_patch_rows:
        raise ValueError(
            "pixel_values_videos row count does not match video_grid_thw: "
            f"rows={pixel_values_videos.shape[0]}, expected={expected_patch_rows}"
        )

    original_device = pixel_values_videos.device
    all_stats: List[Dict[str, Any]] = []
    keep_masks: List[torch.Tensor] = []
    pruned_main_by_video: List[torch.Tensor] = []
    pruned_deepstack_by_layer: List[List[torch.Tensor]] | None = None

    video_offset = 0
    for video_idx, grid in enumerate(video_grid_thw):
        T, H, W = [int(x) for x in grid.tolist()]
        if H % spatial_merge_size != 0 or W % spatial_merge_size != 0:
            raise ValueError(
                f"Grid H/W must be divisible by spatial_merge_size={spatial_merge_size}: "
                f"video_idx={video_idx}, grid={[T, H, W]}"
            )

        patches_per_t = H * W
        merged_tokens_per_t = patches_per_t // (spatial_merge_size ** 2)
        video_patch_rows = T * patches_per_t
        video_end = video_offset + video_patch_rows
        if video_end > pixel_values_videos.shape[0]:
            raise ValueError(
                f"Patch slice exceeds pixel_values_videos rows for video_idx={video_idx}"
            )

        if collect_stats and original_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(original_device)
        video_start_time = time.perf_counter()

        if hasattr(pruner, "set_temporal_patch_size"):
            pruner.set_temporal_patch_size(int(temporal_patch_size))
        stream_state = pruner.init_stream_state(grid)
        pruned_main_parts: List[torch.Tensor] = []
        pruned_deepstack_parts: List[List[torch.Tensor]] | None = None
        chunk_rows: List[Dict[str, Any]] = []
        prev_end_t = 0

        for chunk_idx, start_t in enumerate(range(0, T, chunk_t)):
            end_t = min(start_t + chunk_t, T)
            if start_t != prev_end_t:
                raise ValueError(
                    f"Chunk temporal order is not continuous: expected {prev_end_t}, got {start_t}"
                )
            prev_end_t = end_t

            start_patch = video_offset + start_t * patches_per_t
            end_patch = video_offset + end_t * patches_per_t
            input_patch_rows = end_patch - start_patch
            if input_patch_rows != (end_t - start_t) * patches_per_t:
                raise ValueError(
                    "Computed chunk patch rows do not match chunk grid: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}"
                )

            chunk_pixel_values = pixel_values_videos[start_patch:end_patch]
            chunk_grid_thw = video_grid_thw.new_tensor([[end_t - start_t, H, W]])

            _maybe_sync(original_device, collect_stats)
            start_time = time.perf_counter()
            with torch.inference_mode():
                chunk_main, chunk_deepstack = visual_encoder(
                    chunk_pixel_values.type(visual_encoder.dtype),
                    grid_thw=chunk_grid_thw,
                )
            _maybe_sync(original_device, collect_stats)
            vit_time_ms = (time.perf_counter() - start_time) * 1000.0

            expected_main_tokens = (end_t - start_t) * merged_tokens_per_t
            if chunk_main.shape[0] != expected_main_tokens:
                raise ValueError(
                    "Visual Encoder output token count mismatch: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                    f"got={chunk_main.shape[0]}, expected={expected_main_tokens}"
                )

            prune_start = time.perf_counter()
            if hasattr(pruner, "prune_chunk_with_deepstack"):
                (
                    pruned_main,
                    pruned_chunk_deepstack,
                    chunk_keep_mask,
                    chunk_summary,
                ) = pruner.prune_chunk_with_deepstack(
                    chunk_main,
                    chunk_deepstack,
                    chunk_grid_thw[0],
                    stream_state,
                    global_start_t=start_t,
                )
            else:
                pruned_main, chunk_keep_mask, chunk_summary = pruner.prune_chunk(
                    chunk_main,
                    chunk_grid_thw[0],
                    stream_state,
                    global_start_t=start_t,
                )
                pruned_chunk_deepstack = [
                    layer_tensor[chunk_keep_mask]
                    for layer_tensor in chunk_deepstack
                ]
            _maybe_sync(original_device, collect_stats)
            prune_time_ms = (time.perf_counter() - prune_start) * 1000.0

            if chunk_keep_mask.numel() != chunk_main.shape[0]:
                raise ValueError(
                    "Chunk keep-mask length does not match chunk main tokens: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                    f"mask={chunk_keep_mask.numel()}, main={chunk_main.shape[0]}"
                )

            if pruned_deepstack_parts is None:
                pruned_deepstack_parts = [[] for _ in chunk_deepstack]
            if len(chunk_deepstack) != len(pruned_deepstack_parts):
                raise ValueError(
                    "DeepStack layer count changed across chunks: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}"
                )
            if len(pruned_chunk_deepstack) != len(pruned_deepstack_parts):
                raise ValueError(
                    "Pruned DeepStack layer count does not match original layers: "
                    f"video_idx={video_idx}, chunk_idx={chunk_idx}"
                )

            deepstack_tokens = []
            pruned_deepstack_tokens = []
            for layer_idx, (layer_tensor, pruned_layer) in enumerate(
                zip(chunk_deepstack, pruned_chunk_deepstack)
            ):
                if layer_tensor.shape[0] != chunk_main.shape[0]:
                    raise ValueError(
                        "DeepStack token count does not match main features: "
                        f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                        f"layer={layer_idx}, deepstack={layer_tensor.shape[0]}, "
                        f"main={chunk_main.shape[0]}"
                    )
                if pruned_layer.shape[0] != pruned_main.shape[0]:
                    raise ValueError(
                        "Pruned DeepStack token count does not match pruned main: "
                        f"video_idx={video_idx}, chunk_idx={chunk_idx}, "
                        f"layer={layer_idx}, deepstack={pruned_layer.shape[0]}, "
                        f"main={pruned_main.shape[0]}"
                    )
                pruned_deepstack_parts[layer_idx].append(
                    _to_stage(pruned_layer, cfg.staging_device)
                )
                deepstack_tokens.append(int(layer_tensor.shape[0]))
                pruned_deepstack_tokens.append(int(pruned_layer.shape[0]))

            pruned_main_parts.append(_to_stage(pruned_main, cfg.staging_device))

            row = {
                "chunk_idx": chunk_idx,
                "start_t": start_t,
                "end_t": end_t,
                "chunk_grid_thw": [end_t - start_t, H, W],
                "input_patch_rows": int(input_patch_rows),
                "output_main_tokens": int(chunk_main.shape[0]),
                "output_deepstack_tokens": deepstack_tokens,
                "pruned_main_tokens": int(pruned_main.shape[0]),
                "pruned_deepstack_tokens": pruned_deepstack_tokens,
                "vit_time_ms": round(vit_time_ms, 6),
                "prune_time_ms": round(prune_time_ms, 6),
                "peak_memory_allocated_mb": _peak_mb(original_device, collect_stats),
            }
            row.update(chunk_summary)
            chunk_rows.append(row)

            del chunk_pixel_values
            del chunk_main
            del chunk_deepstack
            del pruned_main

        concat_start = time.perf_counter()
        full_pruned_main = _cat_to_device(pruned_main_parts, original_device)
        full_keep_mask, video_stats = pruner.finalize_stream_stats(
            stream_state,
            grid,
        )
        if full_pruned_main.shape[0] != int(full_keep_mask.sum().item()):
            raise ValueError(
                "Pruned main token count does not match stream keep mask: "
                f"video_idx={video_idx}, main={full_pruned_main.shape[0]}, "
                f"mask_kept={int(full_keep_mask.sum().item())}"
            )

        full_pruned_deepstack_for_video: List[torch.Tensor] = []
        if pruned_deepstack_parts is not None:
            for layer_idx, layer_parts in enumerate(pruned_deepstack_parts):
                layer_full = _cat_to_device(layer_parts, original_device)
                if layer_full.shape[0] != full_pruned_main.shape[0]:
                    raise ValueError(
                        "Pruned DeepStack token count mismatch: "
                        f"video_idx={video_idx}, layer={layer_idx}, "
                        f"deepstack={layer_full.shape[0]}, main={full_pruned_main.shape[0]}"
                    )
                full_pruned_deepstack_for_video.append(layer_full)

        concat_time_ms = (time.perf_counter() - concat_start) * 1000.0

        if pruned_deepstack_by_layer is None:
            pruned_deepstack_by_layer = [
                [] for _ in full_pruned_deepstack_for_video
            ]
        if len(full_pruned_deepstack_for_video) != len(pruned_deepstack_by_layer):
            raise ValueError(
                f"DeepStack layer count changed across videos at video_idx={video_idx}"
            )
        for layer_idx, layer_full in enumerate(full_pruned_deepstack_for_video):
            pruned_deepstack_by_layer[layer_idx].append(layer_full)

        video_total_time_ms = (time.perf_counter() - video_start_time) * 1000.0
        vit_stats = {
            "video_idx": video_idx,
            "full_grid": [T, H, W],
            "raw_frames": T * int(temporal_patch_size),
            "temporal_steps": T,
            "chunk_frames": cfg.chunk_frames,
            "chunk_temporal_steps": chunk_t,
            "num_chunks": len(chunk_rows),
            "original_main_tokens": int(T * merged_tokens_per_t),
            "pruned_main_tokens": int(full_pruned_main.shape[0]),
            "deepstack_layers": len(full_pruned_deepstack_for_video),
            "staging_device": cfg.staging_device,
            "chunks": chunk_rows,
            "chunked_vit_total_time_ms": round(
                sum(row["vit_time_ms"] for row in chunk_rows), 6
            ),
            "chunked_prune_total_time_ms": round(
                sum(row["prune_time_ms"] for row in chunk_rows), 6
            ),
            "feature_concat_time_ms": round(concat_time_ms, 6),
            "total_time_ms": round(video_total_time_ms, 6),
            "chunked_vit_peak_memory_mb": _peak_mb(original_device, collect_stats),
        }
        video_stats["vit_chunk_stats"] = vit_stats
        all_stats.append(video_stats)
        keep_masks.append(full_keep_mask)
        pruned_main_by_video.append(full_pruned_main)
        video_offset = video_end

        pruner_class = pruner.__class__.__name__
        chunk_version = "Streaming"
        for suffix in ("V44", "V43", "V42", "V34", "V41", "V4", "V31", "V3"):
            if pruner_class.endswith(suffix):
                chunk_version = suffix
                break
        chunk_label = f"[ViTChunk{chunk_version or 'Streaming'}]"
        print(
            f"{chunk_label} "
            f"video_idx={video_idx} full_grid={[T,H,W]} "
            f"raw_frames={vit_stats['raw_frames']} temporal_steps={T} "
            f"chunk_frames={cfg.chunk_frames} chunk_temporal_steps={chunk_t} "
            f"num_chunks={len(chunk_rows)} "
            f"original_tokens={vit_stats['original_main_tokens']} "
            f"pruned_tokens={full_pruned_main.shape[0]} "
            f"deepstack_layers={len(full_pruned_deepstack_for_video)} "
            f"staging_device={cfg.staging_device}"
        )

    if video_offset != pixel_values_videos.shape[0]:
        raise ValueError(
            "Not all pixel_values_videos rows were consumed: "
            f"consumed={video_offset}, rows={pixel_values_videos.shape[0]}"
        )

    if pruned_deepstack_by_layer is None:
        full_pruned_deepstack = []
    else:
        full_pruned_deepstack = [
            _cat_to_device(layer_parts, original_device)
            for layer_parts in pruned_deepstack_by_layer
        ]

    return (
        tuple(pruned_main_by_video),
        full_pruned_deepstack,
        keep_masks,
        all_stats,
    )
