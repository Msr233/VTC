def _to_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def _get_input_value(inputs, key):
    if inputs is None:
        return None
    if isinstance(inputs, dict):
        return inputs.get(key)
    if hasattr(inputs, key):
        return getattr(inputs, key)
    try:
        return inputs[key]
    except Exception:
        return None


def get_vtc_pruner(model):
    inner_model = getattr(model, "model", None)
    return getattr(inner_model, "_vtc", None)


def set_vtc_video_contexts(model, video_paths, video_metadatas, inputs):
    pruner = get_vtc_pruner(model)
    if pruner is None or not hasattr(pruner, "set_video_sampling_contexts"):
        return None

    if video_metadatas is None:
        pruner.set_video_sampling_contexts([])
        return pruner

    if isinstance(video_paths, (str, bytes)):
        video_paths = [video_paths]
    video_paths = list(video_paths or [])
    video_metadatas = list(video_metadatas)

    grids = _get_input_value(inputs, "video_grid_thw")
    grids = _to_list(grids)
    contexts = []
    for i, meta in enumerate(video_metadatas):
        meta = dict(meta or {})
        grid_t = 0
        if i < len(grids):
            grid = grids[i]
            if isinstance(grid, (list, tuple)) and grid:
                grid_t = int(grid[0])

        indices = (
            meta.get("frames_indices")
            or meta.get("sampled_frame_indices")
            or meta.get("frame_indices")
            or []
        )
        indices = [int(x) for x in _to_list(indices)]

        timestamps = (
            meta.get("sampled_timestamps")
            or meta.get("timestamps")
            or meta.get("pts")
            or []
        )
        timestamps = [float(x) for x in _to_list(timestamps)]

        temporal_patch_size = 2
        if grid_t > 0 and len(indices) >= grid_t and len(indices) % grid_t == 0:
            temporal_patch_size = max(1, len(indices) // grid_t)

        video_path = video_paths[i] if i < len(video_paths) else meta.get("video_path", "")
        contexts.append(
            {
                "video_path": str(video_path),
                "source_fps": float(meta.get("fps") or meta.get("source_fps") or 0.0),
                "source_total_frames": int(
                    meta.get("total_num_frames")
                    or meta.get("source_total_frames")
                    or len(indices)
                    or 0
                ),
                "sampled_frame_indices": indices,
                "sampled_timestamps": timestamps,
                "sampled_source_frame_indices": [
                    [int(index) for index in _to_list(group)]
                    for group in _to_list(meta.get("sampled_source_frame_indices"))
                ],
                "input_types": [
                    str(value) for value in _to_list(meta.get("input_types"))
                ],
                "temporal_patch_size": temporal_patch_size,
                "grid_t": grid_t,
            }
        )

    pruner.set_video_sampling_contexts(contexts)
    return pruner


def clear_vtc_video_contexts(model):
    pruner = get_vtc_pruner(model)
    if pruner is not None and hasattr(pruner, "clear_video_sampling_contexts"):
        pruner.clear_video_sampling_contexts()
