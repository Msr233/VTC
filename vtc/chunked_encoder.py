"""Qwen3 streaming encoding; gather main and DeepStack with the same indices."""

from dataclasses import dataclass
import torch


@dataclass
class ChunkedVideoEncoderConfig:
    enabled: bool = False
    chunk_frames: int = 16
    staging_device: str = "cuda"
    collect_stats: bool = False


def encode_and_prune_video_in_chunks(
    visual_encoder,
    pixel_values_videos,
    video_grid_thw,
    pruner,
    *,
    temporal_patch_size,
    spatial_merge_size,
    chunk_frames=16,
    staging_device="cuda",
    collect_stats=False,
):
    if chunk_frames <= 0 or chunk_frames % temporal_patch_size:
        raise ValueError(
            "chunk_frames must be a positive multiple of temporal_patch_size"
        )
    if staging_device not in ("cuda", "cpu"):
        raise ValueError("staging_device must be cuda or cpu")
    device = pixel_values_videos.device
    # Grid dimensions are control metadata. Transfer them once, not per chunk.
    grids = video_grid_thw.detach().cpu()
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError("video_grid_thw must have shape [videos, 3]")
    if sum(t * h * w for t, h, w in grids.tolist()) != pixel_values_videos.shape[0]:
        raise ValueError("pixel_values_videos does not match video_grid_thw")
    chunk_t = chunk_frames // temporal_patch_size
    pruner.set_temporal_patch_size(temporal_patch_size)
    videos, masks, all_stats = [], [], []
    deepstack_videos = None
    offset = 0

    def stage(tensor):
        return tensor.to("cpu") if staging_device == "cpu" else tensor

    def collect(parts):
        result = torch.cat(parts)
        return result.to(device) if result.device != device else result

    for video_idx, grid in enumerate(grids):
        total_t, height, width = map(int, grid.tolist())
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError("visual grid must be divisible by spatial_merge_size")
        per_t = height * width
        tokens_per_t = per_t // spatial_merge_size**2
        state = pruner.init_stream_state(grid)
        main_parts, deep_parts, chunks, timing_events = [], None, [], []
        for start_t in range(0, total_t, chunk_t):
            end_t = min(total_t, start_t + chunk_t)
            chunk_grid = video_grid_thw.new_tensor([[end_t - start_t, height, width]])
            chunk_pixels = pixel_values_videos[
                offset + start_t * per_t : offset + end_t * per_t
            ]
            if collect_stats and device.type == "cuda":
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
            main, deepstack = visual_encoder(
                chunk_pixels.type(visual_encoder.dtype), grid_thw=chunk_grid
            )
            if collect_stats and device.type == "cuda":
                events[1].record()
            expected = (end_t - start_t) * tokens_per_t
            if main.shape[0] != expected or any(
                layer.shape[0] != expected for layer in deepstack
            ):
                raise ValueError("main/DeepStack features do not match the chunk grid")
            control_grid = grid.new_tensor([end_t - start_t, height, width])
            pruned, _, row = pruner.prune_chunk(
                main, control_grid, state, global_start_t=start_t
            )
            indices = state["last_keep_indices"]
            pruned_layers = [layer.index_select(0, indices) for layer in deepstack]
            if collect_stats and device.type == "cuda":
                events[2].record()
                timing_events.append(events)
            if deep_parts is None:
                deep_parts = [[] for _ in deepstack]
            main_parts.append(stage(pruned))
            for parts, layer in zip(deep_parts, pruned_layers):
                parts.append(stage(layer))
            chunks.append(row)
        main = collect(main_parts)
        mask, stats = pruner.finalize_stream_stats(state, grid)
        layers = [collect(parts) for parts in deep_parts]
        if main.shape[0] != stats["kept_tokens"] or any(
            layer.shape[0] != main.shape[0] for layer in layers
        ):
            raise ValueError("main, DeepStack and allocated token counts disagree")
        if timing_events:
            # Profiling is explicit. No per-chunk profiling barriers in normal runs.
            timing_events[-1][-1].synchronize()
            for row, (begin, encoded, selected) in zip(chunks, timing_events):
                row["vision_cuda_ms"] = begin.elapsed_time(encoded)
                row["pruning_cuda_ms"] = encoded.elapsed_time(selected)
        stats["vit_chunk_stats"] = {
            "video_idx": video_idx,
            "chunk_frames": chunk_frames,
            "profiling_enabled": collect_stats,
            "chunks": chunks,
        }
        videos.append(main)
        masks.append(mask)
        all_stats.append(stats)
        if deepstack_videos is None:
            deepstack_videos = [[] for _ in layers]
        for parts, layer in zip(deepstack_videos, layers):
            parts.append(layer)
        offset += total_t * per_t
    return (
        tuple(videos),
        [torch.cat(parts) for parts in deepstack_videos],
        masks,
        all_stats,
    )
