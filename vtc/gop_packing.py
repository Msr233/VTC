"""GoP input packing: one full image and four half-size images per unit.

Qwen3 temporal fusion uses pairs, so each group has two full images followed
by two 2x2 canvases (ten decoded source frames, four model input images).
This input grouping is independent of the semantic I/P decisions in VTC.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .video_io import (
    smart_resize,
    SPATIAL_MERGE_SIZE,
    VIDEO_MIN_TOKEN_NUM,
    VIDEO_MAX_TOKEN_NUM,
    FRAME_FACTOR,
    MODEL_SEQ_LEN,
    extract_vision_info,
    fetch_image,
    _resize_video_frames_like_image_inputs,
)

logger = logging.getLogger(__name__)


def build_gop_plan(total_num_frames, nframes):
    """Preserve the sampled logical timestamps used by codec alignment."""
    frames_indices = uniform_sample_indices(total_num_frames, nframes)
    plans, required = [], []
    for base in range(0, nframes, 4):
        logical = frames_indices[base : base + 4]
        full = logical[:2]
        start = frames_indices[base + 1]
        end = frames_indices[min(base + 4, nframes - 1)]
        source = uniform_sample_between(start, end, 8, total_frames=total_num_frames)
        canvas_a = _ensure_four_frame_indices(source[:4], start)
        canvas_b = _ensure_four_frame_indices(source[4:8], end)
        plans.append(
            {
                "logical_indices": logical,
                "full_frames": full,
                "canvas_a_frames": canvas_a,
                "canvas_b_frames": canvas_b,
            }
        )
        required.extend(full + canvas_a + canvas_b)
    return frames_indices, plans, required


def load_gop_video(ele, image_patch_size=16, read_batch_size=256):
    import decord

    video_path = str(ele["video"]).removeprefix("file://")
    nframes = int(ele["nframes"])
    if nframes <= 0 or nframes % 4:
        raise ValueError("GoP input requires a positive multiple of four input images")
    reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(reader)
    fps = float(reader.get_avg_fps())
    first = read_frame_decord(reader, 0)
    if first is None:
        raise ValueError(f"Cannot decode first frame: {video_path}")
    height, width = first.shape[:2]
    sampled, plans, required = build_gop_plan(total, nframes)
    cache = read_frames_decord_batch(reader, required, batch_size=read_batch_size)
    black = np.zeros((height, width, 3), dtype=np.uint8)

    def get_frame(index):
        frame = cache.get(clamp_frame_idx(index, total))
        if frame is None:
            logger.warning(
                "Cannot read frame %s in %s; using black frame", index, video_path
            )
            return black
        return frame

    output, indices, types = [], [], []
    for plan in plans:
        full = plan["full_frames"]
        output.extend(
            [
                get_frame(full[0]),
                get_frame(full[1]),
                make_mosaic(
                    [get_frame(i) for i in plan["canvas_a_frames"]], width, height
                ),
                make_mosaic(
                    [get_frame(i) for i in plan["canvas_b_frames"]], width, height
                ),
            ]
        )
        indices.extend([full[0], full[1], *plan["logical_indices"][2:]])
        types.extend(["full", "full", "mosaic", "mosaic"])
    video = torch.from_numpy(np.stack(output)).permute(0, 3, 1, 2)
    metadata = {
        "fps": fps,
        "frames_indices": indices,
        "input_types": types,
        "uniform_frames_indices": sampled,
        "gop_plans": plans,
        "total_num_frames": total,
        "video_backend": "decord_gop",
    }
    sample_fps = len(output) / max(total, 1e-6) * fps
    return _resize_gop_video(video, ele, image_patch_size), metadata, sample_fps


def process_gop_vision_info(
    conversations,
    return_video_kwargs=False,
    return_video_metadata=False,
    image_patch_size=16,
    read_batch_size=256,
):
    images, videos, sample_fps = [], [], []
    for info in extract_vision_info(conversations):
        if "image" in info or "image_url" in info:
            images.append(fetch_image(info, image_patch_size=image_patch_size))
        elif "video" in info:
            video, metadata, fps = load_gop_video(
                info, image_patch_size, read_batch_size
            )
            videos.append((video, metadata) if return_video_metadata else video)
            sample_fps.append(fps)
        else:
            raise ValueError("Expected image or video input")
    kwargs = {"do_sample_frames": False}
    if not return_video_metadata:
        kwargs["fps"] = sample_fps
    result = (images or None, videos or None)
    return (*result, kwargs) if return_video_kwargs else result


def clamp_frame_idx(frame_idx: int, total_frames: int) -> int:
    return max(0, min(int(frame_idx), total_frames - 1))


def uniform_sample_indices(total_frames: int, nframes: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if nframes <= 0:
        raise ValueError("nframes must be positive")
    indices = np.linspace(0, total_frames - 1, nframes)
    return [clamp_frame_idx(round(float(idx)), total_frames) for idx in indices]


def uniform_sample_between(
    start_frame: int,
    end_frame: int,
    num_samples: int,
    total_frames: Optional[int] = None,
) -> List[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    start = int(start_frame)
    end = int(end_frame)
    if start > end:
        start, end = end, start

    if total_frames is not None:
        start = clamp_frame_idx(start, total_frames)
        end = clamp_frame_idx(end, total_frames)

    sampled = [round(float(idx)) for idx in np.linspace(start, end, num_samples)]
    if total_frames is not None:
        sampled = [clamp_frame_idx(idx, total_frames) for idx in sampled]
    return [int(idx) for idx in sampled]


def _ensure_four_frame_indices(indices: List[int], fallback_idx: int) -> List[int]:
    if not indices:
        indices = [fallback_idx]
    return (indices + [indices[-1]] * 4)[:4]


def _resize_rgb(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(Image.fromarray(frame_rgb).resize((width, height), resampling))


def make_mosaic(frames: List[np.ndarray], width: int, height: int) -> np.ndarray:
    if not frames:
        raise ValueError("make_mosaic requires at least one source frame")
    while len(frames) < 4:
        frames.append(frames[-1])

    tile_width = max(1, width // 2)
    tile_height = max(1, height // 2)
    resized = [_resize_rgb(frame, tile_width, tile_height) for frame in frames[:4]]
    top = np.concatenate([resized[0], resized[1]], axis=1)
    bottom = np.concatenate([resized[2], resized[3]], axis=1)
    mosaic = np.concatenate([top, bottom], axis=0)
    if mosaic.shape[1] != width or mosaic.shape[0] != height:
        mosaic = _resize_rgb(mosaic, width, height)
    return mosaic


def read_frame_decord(
    vr: Any,
    frame_idx: int,
    max_neighbor_distance: int = 2,
) -> Optional[np.ndarray]:
    total_frames = len(vr)
    requested = clamp_frame_idx(frame_idx, total_frames)
    candidate_indices = [requested]
    for distance in range(1, max_neighbor_distance + 1):
        candidate_indices.append(clamp_frame_idx(requested - distance, total_frames))
        candidate_indices.append(clamp_frame_idx(requested + distance, total_frames))

    seen: set[int] = set()
    for candidate in candidate_indices:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            frame = vr[candidate].asnumpy()
            if frame is not None and frame.ndim == 3 and frame.shape[2] >= 3:
                return frame[:, :, :3]
        except Exception as exc:
            logger.warning(
                "failed to read frame %s while requesting %s: %s",
                candidate,
                requested,
                exc,
            )
    return None


def read_frames_decord_batch(
    vr: Any,
    frame_indices: List[int],
    batch_size: int = 64,
    max_neighbor_distance: int = 2,
) -> Dict[int, np.ndarray]:
    total_frames = len(vr)
    unique_indices = sorted(
        {clamp_frame_idx(idx, total_frames) for idx in frame_indices}
    )
    if not unique_indices:
        return {}

    frames: Dict[int, np.ndarray] = {}
    batch_size = max(1, int(batch_size))
    for start in range(0, len(unique_indices), batch_size):
        chunk = unique_indices[start : start + batch_size]
        try:
            batch = vr.get_batch(chunk).asnumpy()
            for frame_idx, frame in zip(chunk, batch):
                if frame is not None and frame.ndim == 3 and frame.shape[2] >= 3:
                    frames[frame_idx] = frame[:, :, :3]
        except Exception as exc:
            logger.warning(
                "batch read failed for frame chunk [%s, ..., %s] (%s frames): %s; "
                "falling back to single-frame reads",
                chunk[0],
                chunk[-1],
                len(chunk),
                exc,
            )

        for frame_idx in chunk:
            if frame_idx in frames:
                continue
            frame = read_frame_decord(
                vr,
                frame_idx,
                max_neighbor_distance=max_neighbor_distance,
            )
            if frame is not None:
                frames[frame_idx] = frame
    return frames


def _resize_gop_video(
    video: torch.Tensor,
    ele: Dict[str, Any],
    image_patch_size: int,
) -> torch.Tensor:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    video = _resize_video_frames_like_image_inputs(video, ele, image_factor)
    nframes, _, height, width = video.shape

    min_pixels = ele.get("min_pixels", video_frame_min_pixels)
    total_pixels = ele.get(
        "total_pixels",
        MODEL_SEQ_LEN * image_factor * image_factor * 0.9,
    )
    max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            "The given max_pixels[%s] exceeds limit[%s].",
            max_pixels_supposed,
            max_pixels,
        )
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
