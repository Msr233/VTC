import base64
import copy
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime acceleration.
    cv2 = None

from utils.hevc_bitcost import HEVCBitcostExtractor


MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8
MOSAIC_PACKING_ORDERS = {"qwen3_2i2p", "miniset_1i1p"}
MOSAIC_PAIR_PACK_ORDERS = {"sequential", "pair_aligned"}
# The submission reproduces the v4.7.2 DyRSO input.  Codec-guided
# full-pair selection from v5 is intentionally not a supported mode.
DYRSO_I_PAIR_MODES = {"fixed", "hevc_bitcost", "hevc_tile_sort"}
FRAME_EXTRACT_VERSION_TO_PACKING = {
    "v1": "qwen3_2i2p",
    "v1.1": "miniset_1i1p",
}

MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalFramePair:
    first: Any
    second: Any
    first_frame_index: int
    second_frame_index: int
    first_sample_rank: int
    second_sample_rank: int


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "keyframes" in ele:
        nframes = len(ele['keyframes'])

    elif "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        # print('nframes',nframes)
        # exit(0)
    else:
        fps = ele.get("fps", FPS)
        # print('fps', fps)
        # exit(0)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}")
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = ele.get("nframes", 0)
    if nframes <= 0:
        logger.warning(f"Invalid nframes: {nframes}, using smart_nframes")
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps


def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None


def _read_video_torchcodec(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec",
    )
    return video, video_metadata, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    source_is_video_path = isinstance(ele["video"], str)
    if source_is_video_path:
        video_reader_backend = get_video_reader_backend()
        # try:
        # st = time.time()
        # print(333)
        video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        # end = time.time()
            # print(end-st)

        # except Exception as e:
        #     logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
        #     video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)
    else:
        # The input is a list of frames (e.g. pre-extracted codec canvases)
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)

        # use ThreadPoolExecutor to parallel process frames
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element, **process_info}, image_factor)
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        # Use real metadata when the caller passes codec-derived fields
        has_real_metadata = all(
            k in ele for k in ("fps", "frames_indices", "total_num_frames")
        )
        if has_real_metadata:
            video_fps = float(ele["fps"])
            total_num_frames = int(ele["total_num_frames"])
            sample_fps = (len(image_list) / max(total_num_frames, 1e-6)) * video_fps
        else:
            sample_fps = float(ele.get("sample_fps", 2.0))
            video_fps = sample_fps
            total_num_frames = int((nframes / sample_fps) * video_fps)

        video = torch.stack([
            torch.from_numpy(np.array(image).transpose(2, 0, 1))
            for image in image_list
        ])

        if has_real_metadata:
            video_metadata = dict(
                fps=video_fps,
                frames_indices=ele["frames_indices"],
                total_num_frames=total_num_frames,
            )
            if ele.get("width") is not None and ele.get("height") is not None:
                video_metadata.update(width=int(ele["width"]), height=int(ele["height"]))
        else:
            raw_fps = process_info.pop("raw_fps", sample_fps)
            video_metadata = dict(
                fps=float(raw_fps),
                frames_indices=[i for i in range(len(video))],
                total_num_frames=int((nframes / sample_fps) * float(raw_fps)),
            )

    if source_is_video_path:
        video = _resize_video_frames_like_image_inputs(video, ele, image_factor)

    nframes, _, height, width = video.shape
    resize_basis_height, resize_basis_width = height, width
    if not source_is_video_path and ele.get("width") is not None and ele.get("height") is not None:
        resize_basis_height, resize_basis_width = _image_input_resized_size(
            int(ele["height"]),
            int(ele["width"]),
            ele,
            image_factor,
        )
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
            )
    else:
        resized_height, resized_width = smart_resize(
            resize_basis_height,
            resize_basis_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


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


def build_temporal_frame_pairs(
    frames: List[Any],
    frame_indices: Optional[List[int]] = None,
) -> List[TemporalFramePair]:
    if len(frames) % 2 != 0:
        raise ValueError(
            f"temporal frame pairing requires an even number of frames, got {len(frames)}"
        )
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    if len(frame_indices) != len(frames):
        raise ValueError(
            f"frame_indices length mismatch: frames={len(frames)} indices={len(frame_indices)}"
        )

    pairs: List[TemporalFramePair] = []
    for rank in range(0, len(frames), 2):
        pairs.append(
            TemporalFramePair(
                first=frames[rank],
                second=frames[rank + 1],
                first_frame_index=int(frame_indices[rank]),
                second_frame_index=int(frame_indices[rank + 1]),
                first_sample_rank=rank,
                second_sample_rank=rank + 1,
            )
        )
    return pairs


def build_pair_aligned_mosaic_pair(
    temporal_pairs: List[TemporalFramePair],
    tile_layout: Tuple[int, int] = (2, 2),
) -> Tuple[List[Any], List[Any], List[Dict[str, int]]]:
    tile_count = int(tile_layout[0]) * int(tile_layout[1])
    if len(temporal_pairs) != tile_count:
        raise ValueError(
            f"pair-aligned mosaic requires {tile_count} temporal pairs, got {len(temporal_pairs)}"
        )

    canvas_a = [pair.first for pair in temporal_pairs]
    canvas_b = [pair.second for pair in temporal_pairs]
    tile_pairs = [
        {
            "tile_id": tile_id,
            "first_sample_rank": pair.first_sample_rank,
            "second_sample_rank": pair.second_sample_rank,
            "first_frame_index": pair.first_frame_index,
            "second_frame_index": pair.second_frame_index,
        }
        for tile_id, pair in enumerate(temporal_pairs)
    ]
    return canvas_a, canvas_b, tile_pairs


def validate_homogeneous_temporal_pairs(input_types: List[str]) -> int:
    if len(input_types) % 2 != 0:
        raise AssertionError(
            f"Qwen3 temporal pairs require an even number of inputs, got {len(input_types)}"
        )

    heterogeneous = 0
    for index in range(0, len(input_types), 2):
        left = input_types[index]
        right = input_types[index + 1]
        if left != right:
            heterogeneous += 1
            raise AssertionError(
                f"Heterogeneous Qwen3 temporal pair at slots "
                f"{index}/{index + 1}: {left} + {right}"
            )
    return heterogeneous


def count_heterogeneous_temporal_pairs(input_types: List[str]) -> int:
    if len(input_types) % 2 != 0:
        return 1
    return sum(
        1
        for index in range(0, len(input_types), 2)
        if input_types[index] != input_types[index + 1]
    )


def _validate_mosaic_pair_pack_order(pair_pack_order: str) -> str:
    order = str(pair_pack_order or "sequential")
    if order not in MOSAIC_PAIR_PACK_ORDERS:
        raise ValueError(
            f"Unsupported mosaic pair pack order: {pair_pack_order!r}. "
            f"Expected one of {sorted(MOSAIC_PAIR_PACK_ORDERS)}"
        )
    return order


def _pair_alignment_misaligned_count(tile_pairs: List[Dict[str, int]]) -> int:
    return sum(
        1
        for pair in tile_pairs
        if int(pair["second_sample_rank"]) != int(pair["first_sample_rank"]) + 1
        or int(pair["second_frame_index"]) < int(pair["first_frame_index"])
    )


def _median_float(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _validate_dyrso_i_pair_mode(mode: str) -> str:
    value = str(mode or "fixed")
    if value not in DYRSO_I_PAIR_MODES:
        raise ValueError(
            f"Unsupported DyRSO I-pair mode: {mode!r}. "
            f"Expected one of {sorted(DYRSO_I_PAIR_MODES)}"
        )
    return value


def default_dyrso_i_pair_candidates(num_pairs: int, low_pair_count: int) -> List[int]:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if low_pair_count < 0 or low_pair_count >= num_pairs:
        raise ValueError(
            f"low_pair_count must be in [0, num_pairs), got {low_pair_count} for {num_pairs}"
        )
    middle = num_pairs // 2
    candidates = [0, middle, num_pairs - 1]
    return validate_dyrso_i_pair_candidates(candidates, num_pairs, low_pair_count)


def parse_dyrso_i_pair_candidates(
    text: Optional[str],
    *,
    num_pairs: int,
    low_pair_count: int,
) -> List[int]:
    if text is None or str(text).strip() == "":
        return default_dyrso_i_pair_candidates(num_pairs, low_pair_count)
    raw = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        try:
            raw.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid DyRSO I-pair candidate {item!r} in {text!r}") from exc
    return validate_dyrso_i_pair_candidates(raw, num_pairs, low_pair_count)


def validate_dyrso_i_pair_candidates(
    candidates: List[int],
    num_pairs: int,
    low_pair_count: int,
) -> List[int]:
    if num_pairs - 1 != int(low_pair_count):
        raise ValueError(
            f"DyRSO group shape is incompatible with IIPP scheduling: "
            f"num_pairs={num_pairs}, low_pair_count={low_pair_count}"
        )
    unique = sorted(set(int(x) for x in candidates))
    if not unique:
        raise ValueError("DyRSO I-pair candidates must contain at least one index")
    invalid = [idx for idx in unique if idx < 0 or idx >= num_pairs]
    if invalid:
        raise ValueError(
            f"DyRSO I-pair candidates out of range for {num_pairs} pairs: {invalid}"
        )
    return unique


def select_hevc_guided_full_pair(
    frame_bitcosts: List[float],
    candidate_pair_indices: List[int],
    pair_max_weight: float = 0.3,
) -> Dict[str, Any]:
    weight = float(pair_max_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"DYRSO_HEVC_PAIR_MAX_WEIGHT must be in [0,1], got {pair_max_weight}")
    if len(frame_bitcosts) < 2 or len(frame_bitcosts) % 2 != 0:
        raise ValueError(
            f"frame_bitcosts must contain an even positive number of values, got {len(frame_bitcosts)}"
        )
    if any(not math.isfinite(float(v)) for v in frame_bitcosts):
        raise ValueError(f"frame_bitcosts contains NaN/Inf: {frame_bitcosts}")

    num_pairs = len(frame_bitcosts) // 2
    candidates = sorted(set(int(x) for x in candidate_pair_indices))
    if not candidates:
        raise ValueError("candidate_pair_indices must contain at least one index")
    invalid = [idx for idx in candidates if idx < 0 or idx >= num_pairs]
    if invalid:
        raise ValueError(f"candidate pair indices out of range for {num_pairs} pairs: {invalid}")
    logs = [math.log1p(max(0.0, float(v))) for v in frame_bitcosts]
    median = _median_float(logs)
    mad = _median_float([abs(v - median) for v in logs])
    if mad < 1e-6:
        normalized = [0.0 for _ in logs]
    else:
        normalized = [(v - median) / (mad + 1e-6) for v in logs]
    if any(not math.isfinite(float(v)) for v in normalized):
        raise ValueError(f"normalized frame scores contain NaN/Inf: {normalized}")

    pair_scores = []
    for pair_idx in range(num_pairs):
        left = normalized[2 * pair_idx]
        right = normalized[2 * pair_idx + 1]
        pair_scores.append((1.0 - weight) * ((left + right) * 0.5) + weight * max(left, right))

    center = (num_pairs - 1) / 2.0
    selected = max(candidates, key=lambda idx: (pair_scores[idx], -abs(idx - center), -idx))
    return {
        "selected_pair_index": int(selected),
        "pair_scores": [float(x) for x in pair_scores],
        "normalized_frame_scores": [float(x) for x in normalized],
        "log_frame_bitcosts": [float(x) for x in logs],
    }


def build_dyrso_iipp_sources(
    group_frame_indices: List[int],
    *,
    selected_pair_index: int,
    tiles_per_mosaic: int = 4,
    num_mosaics: int = 2,
) -> Dict[str, Any]:
    if len(group_frame_indices) % 2 != 0:
        raise ValueError(
            f"DyRSO group requires an even number of source frames, got {len(group_frame_indices)}"
        )
    pair_start = int(selected_pair_index) * 2
    pair_end = pair_start + 2
    if pair_start < 0 or pair_end > len(group_frame_indices):
        raise ValueError(
            f"selected_pair_index={selected_pair_index} is invalid for group size {len(group_frame_indices)}"
        )
    full_frames = list(group_frame_indices[pair_start:pair_end])
    low_frames = list(group_frame_indices[:pair_start] + group_frame_indices[pair_end:])
    expected_low = int(tiles_per_mosaic) * int(num_mosaics)
    if len(low_frames) != expected_low:
        raise ValueError(
            f"DyRSO low-frame budget mismatch: got {len(low_frames)}, expected {expected_low}"
        )
    mosaics = [
        low_frames[i * tiles_per_mosaic : (i + 1) * tiles_per_mosaic]
        for i in range(num_mosaics)
    ]
    input_types = ["full", "full"] + ["mosaic"] * num_mosaics
    validate_homogeneous_temporal_pairs(input_types)
    return {
        "full_frames": full_frames,
        "low_frames": low_frames,
        "mosaic_frames": mosaics,
        "input_types": input_types,
    }


def sort_mosaic_tiles_by_bitcost(
    frame_indices: List[int],
    frame_bitcost_by_index: Dict[int, float],
    total_num_frames: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    scored_tiles: List[Dict[str, Any]] = []
    for source_rank, frame_idx in enumerate(frame_indices):
        clamped_idx = int(clamp_frame_idx(frame_idx, total_num_frames))
        if clamped_idx not in frame_bitcost_by_index:
            raise ValueError(f"HEVC frame bitcost missing for frame index {clamped_idx}")
        scored_tiles.append(
            {
                "source_rank": int(source_rank),
                "frame_index": int(frame_idx),
                "bitcost": float(frame_bitcost_by_index[clamped_idx]),
            }
        )

    ordered = sorted(scored_tiles, key=lambda item: (-float(item["bitcost"]), int(item["source_rank"])))
    tile_plan: List[Dict[str, Any]] = []
    for tile_id, item in enumerate(ordered):
        tile_plan.append(
            {
                "tile_id": int(tile_id),
                "source_rank": int(item["source_rank"]),
                "frame_index": int(item["frame_index"]),
                "bitcost": float(item["bitcost"]),
            }
        )
    return [int(item["frame_index"]) for item in ordered], tile_plan


def _validate_mosaic_packing_order(packing_order: str) -> str:
    order = str(packing_order or "qwen3_2i2p")
    if order not in MOSAIC_PACKING_ORDERS:
        raise ValueError(
            f"Unsupported mosaic packing order: {packing_order!r}. "
            f"Expected one of {sorted(MOSAIC_PACKING_ORDERS)}"
        )
    return order


def _validate_frame_extract_version(frame_extract_version: str) -> str:
    version = str(frame_extract_version or "v1")
    if version not in FRAME_EXTRACT_VERSION_TO_PACKING:
        raise ValueError(
            f"Unsupported frame extract version: {frame_extract_version!r}. "
            f"Expected one of {sorted(FRAME_EXTRACT_VERSION_TO_PACKING)}"
        )
    return version


def resolve_mosaic_packing_order(
    frame_extract_version: str = "v1",
    mosaic_packing_order: Optional[str] = None,
) -> Tuple[str, str]:
    version = _validate_frame_extract_version(frame_extract_version)
    if mosaic_packing_order:
        return version, _validate_mosaic_packing_order(mosaic_packing_order)
    return version, FRAME_EXTRACT_VERSION_TO_PACKING[version]


def _mosaic_output_order(packing_order: str) -> List[str]:
    packing_order = _validate_mosaic_packing_order(packing_order)
    if packing_order == "qwen3_2i2p":
        return ["original_0", "original_1", "mosaic_0", "mosaic_1"]
    return ["original_0", "mosaic_0", "original_1", "mosaic_1"]


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


def _save_first_mosaic_group_dump(
    *,
    dump_dir: str,
    video_path: str,
    ele: Dict[str, Any],
    plan: Dict[str, Any],
    packed_items: Dict[str, Tuple[np.ndarray, int]],
    output_order: List[str],
    mosaic_packing_order: str,
    mosaic_pair_pack_order: str,
    dyrso_i_pair_mode: str,
) -> bool:
    """Save one online mosaic sample per output directory for debugging."""
    if not dump_dir:
        return False
    try:
        os.makedirs(dump_dir, exist_ok=True)
        lock_path = os.path.join(dump_dir, ".saving.lock")
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

        saved_files: List[Dict[str, Any]] = []
        for slot, item_key in enumerate(output_order):
            frame, frame_idx = packed_items[item_key]
            filename = f"{slot:02d}_{item_key}_f{int(frame_idx)}.png"
            Image.fromarray(frame).save(os.path.join(dump_dir, filename))
            saved_files.append({
                "slot": int(slot),
                "item_key": item_key,
                "frame_index": int(frame_idx),
                "file": filename,
            })

        meta = {
            "video": str(video_path),
            "question_id": ele.get("question_id", ele.get("sample_id")),
            "sample_id": ele.get("sample_id"),
            "group_id": int(plan.get("group_id", 0)),
            "logical_indices": list(plan.get("logical_indices") or []),
            "original_frames": list(plan.get("original_frames") or []),
            "mosaic_range": list(plan.get("mosaic_range") or []),
            "mosaic_source_frames": list(plan.get("mosaic_source_frames") or []),
            "mosaic_0_frames": list(plan.get("mosaic_0_frames") or []),
            "mosaic_1_frames": list(plan.get("mosaic_1_frames") or []),
            "mosaic_packing_order": mosaic_packing_order,
            "mosaic_pair_pack_order": mosaic_pair_pack_order,
            "dyrso_i_pair_mode": dyrso_i_pair_mode,
            "dyrso_i_pair_selection": plan.get("dyrso_i_pair_selection"),
            "saved_files": saved_files,
        }
        meta_path = os.path.join(dump_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(dump_dir, ".saved"), "w", encoding="utf-8") as f:
            f.write("1\n")
        return True
    except Exception as exc:  # pragma: no cover - debug path should not break evals.
        logger.warning("failed to dump first mosaic group to %s: %s", dump_dir, exc)
        return False


def _resize_video_frames_like_image_inputs(
    video: torch.Tensor,
    ele: Dict[str, Any],
    image_factor: int,
) -> torch.Tensor:
    """Match the initial per-frame resize that list-of-image video inputs use."""
    _, _, height, width = video.shape
    resized_height, resized_width = _image_input_resized_size(height, width, ele, image_factor)
    if resized_height == height and resized_width == width:
        return video
    return transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _image_input_resized_size(
    height: int,
    width: int,
    ele: Dict[str, Any],
    image_factor: int,
) -> Tuple[int, int]:
    if "resized_height" in ele and "resized_width" in ele:
        return smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    return smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=IMAGE_MIN_TOKEN_NUM * image_factor * image_factor,
        max_pixels=ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * image_factor * image_factor),
    )


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
    unique_indices = sorted({clamp_frame_idx(idx, total_frames) for idx in frame_indices})
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


def build_mosaic_frame_plan(
    total_num_frames: int,
    nframes: int,
    mosaic_source_frames: int = 8,
    mosaic_pair_pack_order: str = "sequential",
    strict_pairing: bool = True,
    dyrso_i_pair_mode: str = "fixed",
    frame_bitcost_by_index: Optional[Dict[int, float]] = None,
    dyrso_i_pair_candidates: Optional[str] = None,
    dyrso_hevc_pair_max_weight: float = 0.3,
    dyrso_i_pair_strict: bool = True,
) -> Tuple[List[int], List[Dict[str, Any]], List[int]]:
    mosaic_pair_pack_order = _validate_mosaic_pair_pack_order(mosaic_pair_pack_order)
    dyrso_i_pair_mode = _validate_dyrso_i_pair_mode(dyrso_i_pair_mode)
    if dyrso_i_pair_mode in ("hevc_bitcost", "hevc_tile_sort") and mosaic_pair_pack_order != "sequential":
        raise ValueError(f"DyRSO {dyrso_i_pair_mode} scheduling requires sequential mosaic packing")
    frames_indices = uniform_sample_indices(total_num_frames, nframes)
    plans: List[Dict[str, Any]] = []
    required_frame_indices: List[int] = []

    num_groups = (len(frames_indices) + 3) // 4
    for group_id in range(num_groups):
        base = group_id * 4
        if base + 3 >= len(frames_indices):
            logger.warning(
                "[GROUP %03d] skipped incomplete group: %s",
                group_id,
                frames_indices[base:],
            )
            continue

        logical_indices = frames_indices[base : base + 4]
        original_frames = logical_indices[:2]
        mosaic_start = frames_indices[base + 1]
        mosaic_end = frames_indices[min(base + 4, len(frames_indices) - 1)]
        if mosaic_start > mosaic_end:
            mosaic_start, mosaic_end = mosaic_end, mosaic_start

        mosaic_source = uniform_sample_between(
            mosaic_start,
            mosaic_end,
            mosaic_source_frames,
            total_frames=total_num_frames,
        )
        dyrso_selection: Dict[str, Any] = {
            "mode": "fixed",
            "selected_pair_index": 0,
            "fallback_used": False,
            "fallback_reason": None,
        }
        pair_aligned_fallback = False
        pair_tile_pairs: List[Dict[str, int]] = []
        pair_alignment_valid = True
        output_order: Optional[List[str]] = None
        if dyrso_i_pair_mode == "hevc_bitcost":
            group_frame_indices = list(original_frames) + list(mosaic_source)
            num_mosaics = 2
            tiles_per_mosaic = 4
            num_pairs = len(group_frame_indices) // 2
            low_pair_count = num_mosaics * tiles_per_mosaic // 2
            try:
                if len(group_frame_indices) != 2 + num_mosaics * tiles_per_mosaic:
                    raise ValueError(
                        f"expected {2 + num_mosaics * tiles_per_mosaic} DyRSO source frames, "
                        f"got {len(group_frame_indices)}"
                    )
                if frame_bitcost_by_index is None:
                    raise ValueError("HEVC frame bitcosts are unavailable")
                frame_bitcosts = [
                    float(frame_bitcost_by_index[int(clamp_frame_idx(idx, total_num_frames))])
                    for idx in group_frame_indices
                ]
                candidates = parse_dyrso_i_pair_candidates(
                    dyrso_i_pair_candidates,
                    num_pairs=num_pairs,
                    low_pair_count=low_pair_count,
                )
                selected = select_hevc_guided_full_pair(
                    frame_bitcosts,
                    candidates,
                    pair_max_weight=dyrso_hevc_pair_max_weight,
                )
                sources = build_dyrso_iipp_sources(
                    group_frame_indices,
                    selected_pair_index=int(selected["selected_pair_index"]),
                    tiles_per_mosaic=tiles_per_mosaic,
                    num_mosaics=num_mosaics,
                )
                original_frames = sources["full_frames"]
                mosaic_0_frames = sources["mosaic_frames"][0]
                mosaic_1_frames = sources["mosaic_frames"][1]
                dyrso_selection = {
                    "mode": "hevc_bitcost",
                    "selected_pair_index": int(selected["selected_pair_index"]),
                    "candidate_pairs": candidates,
                    "group_frame_indices": group_frame_indices,
                    "sample_ranks": list(range(len(group_frame_indices))),
                    "raw_bitcost": frame_bitcosts,
                    "normalized_bitcost": selected["normalized_frame_scores"],
                    "pair_scores": selected["pair_scores"],
                    "selected_frames": original_frames,
                    "low_frames": sources["low_frames"],
                    "mosaic0_frames": mosaic_0_frames,
                    "mosaic1_frames": mosaic_1_frames,
                    "final_input_types": sources["input_types"],
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            except Exception as exc:
                if dyrso_i_pair_strict:
                    raise
                logger.warning(
                    "[GROUP %03d] HEVC-guided I-pair selection failed; "
                    "falling back to fixed I-pair. reason=%s",
                    group_id,
                    exc,
                )
                mosaic_0_frames = _ensure_four_frame_indices(mosaic_source[:4], mosaic_start)
                mosaic_1_seed = mosaic_source[4:8] if len(mosaic_source) <= 8 else mosaic_source[-4:]
                mosaic_1_frames = _ensure_four_frame_indices(mosaic_1_seed, mosaic_end)
                dyrso_selection = {
                    "mode": "fixed",
                    "selected_pair_index": 0,
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                }
        elif dyrso_i_pair_mode == "hevc_tile_sort":
            mosaic_0_seed = _ensure_four_frame_indices(mosaic_source[:4], mosaic_start)
            mosaic_1_seed = mosaic_source[4:8] if len(mosaic_source) <= 8 else mosaic_source[-4:]
            mosaic_1_seed = _ensure_four_frame_indices(mosaic_1_seed, mosaic_end)
            try:
                if frame_bitcost_by_index is None:
                    raise ValueError("HEVC frame bitcosts are unavailable")
                mosaic_0_frames, mosaic_0_tile_sort = sort_mosaic_tiles_by_bitcost(
                    mosaic_0_seed,
                    frame_bitcost_by_index,
                    total_num_frames,
                )
                mosaic_1_frames, mosaic_1_tile_sort = sort_mosaic_tiles_by_bitcost(
                    mosaic_1_seed,
                    frame_bitcost_by_index,
                    total_num_frames,
                )
                dyrso_selection = {
                    "mode": "hevc_tile_sort",
                    "selected_pair_index": 0,
                    "selected_frames": list(original_frames),
                    "mosaic0_original_order": list(mosaic_0_seed),
                    "mosaic1_original_order": list(mosaic_1_seed),
                    "mosaic0_frames": list(mosaic_0_frames),
                    "mosaic1_frames": list(mosaic_1_frames),
                    "mosaic0_tile_sort": mosaic_0_tile_sort,
                    "mosaic1_tile_sort": mosaic_1_tile_sort,
                    "final_input_types": ["full", "full", "mosaic", "mosaic"],
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            except Exception as exc:
                if dyrso_i_pair_strict:
                    raise
                logger.warning(
                    "[GROUP %03d] HEVC tile sorting failed; "
                    "falling back to fixed sequential tiles. reason=%s",
                    group_id,
                    exc,
                )
                mosaic_0_frames = list(mosaic_0_seed)
                mosaic_1_frames = list(mosaic_1_seed)
                dyrso_selection = {
                    "mode": "fixed",
                    "selected_pair_index": 0,
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                }
        elif mosaic_pair_pack_order == "pair_aligned":
            if len(mosaic_source) < 8:
                message = (
                    f"[GROUP {group_id:03d}] pair_aligned requires at least 8 "
                    f"mosaic source frames, got {len(mosaic_source)}"
                )
                if strict_pairing:
                    raise ValueError(message)
                logger.warning("%s; falling back to sequential packing", message)
                pair_aligned_fallback = True

            if not pair_aligned_fallback:
                temporal_pairs = build_temporal_frame_pairs(
                    mosaic_source[:8],
                    frame_indices=mosaic_source[:8],
                )
                mosaic_0_frames, mosaic_1_frames, pair_tile_pairs = build_pair_aligned_mosaic_pair(
                    temporal_pairs,
                    tile_layout=(2, 2),
                )
                misaligned = _pair_alignment_misaligned_count(pair_tile_pairs)
                pair_alignment_valid = misaligned == 0
                if misaligned and strict_pairing:
                    raise AssertionError(
                        f"[GROUP {group_id:03d}] pair-aligned mosaic has "
                        f"{misaligned} misaligned tile pairs: {pair_tile_pairs}"
                    )

        if dyrso_i_pair_mode == "fixed" and (mosaic_pair_pack_order == "sequential" or pair_aligned_fallback):
            mosaic_0_frames = _ensure_four_frame_indices(mosaic_source[:4], mosaic_start)
            mosaic_1_seed = mosaic_source[4:8] if len(mosaic_source) <= 8 else mosaic_source[-4:]
            mosaic_1_frames = _ensure_four_frame_indices(mosaic_1_seed, mosaic_end)
            pair_tile_pairs = [
                {
                    "tile_id": tile_id,
                    "first_sample_rank": tile_id,
                    "second_sample_rank": tile_id + 4,
                    "first_frame_index": int(mosaic_0_frames[tile_id]),
                    "second_frame_index": int(mosaic_1_frames[tile_id]),
                }
                for tile_id in range(4)
            ]
            pair_alignment_valid = False if mosaic_pair_pack_order == "pair_aligned" else True

        plans.append(
            {
                "group_id": group_id,
                "logical_indices": logical_indices,
                "original_frames": original_frames,
                "mosaic_range": [mosaic_start, mosaic_end],
                "mosaic_source_frames": mosaic_source,
                "mosaic_0_frames": mosaic_0_frames,
                "mosaic_1_frames": mosaic_1_frames,
                "mosaic_pair_pack_order": mosaic_pair_pack_order,
                "pair_tile_pairs": pair_tile_pairs,
                "pair_alignment_valid": pair_alignment_valid,
                "pair_aligned_fallback": pair_aligned_fallback,
                "dyrso_i_pair_selection": dyrso_selection,
                "output_order": output_order,
            }
        )
        required_frame_indices.extend(original_frames)
        required_frame_indices.extend(mosaic_0_frames)
        required_frame_indices.extend(mosaic_1_frames)

    return frames_indices, plans, required_frame_indices


def _resize_mosaic_video_for_qwen(
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


def load_mosaic_video_for_qwen(
    ele: Dict[str, Any],
    image_patch_size: int = 16,
    mosaic_source_frames: int = 8,
    read_batch_size: int = 64,
    frame_extract_version: str = "v1",
    mosaic_packing_order: Optional[str] = None,
    mosaic_pair_pack_order: str = "sequential",
    mosaic_require_homogeneous_pair: bool = False,
    mosaic_strict_pairing: bool = True,
    mosaic_dump_pair_mapping: bool = False,
    dyrso_i_pair_mode: str = "fixed",
    dyrso_hevc_pair_max_weight: float = 0.3,
    dyrso_i_pair_candidates: Optional[str] = None,
    dyrso_i_pair_strict: bool = True,
    dyrso_dump_i_pair_selection: bool = False,
    dyrso_codec_cache_dir: Optional[str] = None,
    dyrso_codec_ffprobe_path: str = "ffprobe",
    dyrso_codec_z_clip: float = 5.0,
    mosaic_first_group_dump_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    import decord

    video_path = ele["video"]
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[7:]
    if not isinstance(video_path, str):
        raise TypeError("mosaic video loader expects ele['video'] to be a video path string")

    nframes = int(ele.get("nframes") or 0)
    if nframes <= 0:
        raise ValueError("mosaic video loader requires a positive nframes value")

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_num_frames = len(vr)
    if total_num_frames <= 0:
        raise ValueError(f"video contains no frames: {video_path}")
    fps = float(vr.get_avg_fps())

    first_frame = read_frame_decord(vr, 0)
    if first_frame is None:
        raise ValueError(f"could not read first frame to infer dimensions: {video_path}")
    height, width = first_frame.shape[:2]
    frame_extract_version, mosaic_packing_order = resolve_mosaic_packing_order(
        frame_extract_version,
        mosaic_packing_order,
    )
    mosaic_pair_pack_order = _validate_mosaic_pair_pack_order(mosaic_pair_pack_order)
    dyrso_i_pair_mode = _validate_dyrso_i_pair_mode(dyrso_i_pair_mode)
    dyrso_frame_bitcosts: Optional[Dict[int, float]] = None
    dyrso_alignment_failure_count = 0
    dyrso_extraction_fallback = False
    if dyrso_i_pair_mode in ("hevc_bitcost", "hevc_tile_sort"):
        try:
            extractor = HEVCBitcostExtractor(
                ffprobe_path=dyrso_codec_ffprobe_path,
                cache_dir=dyrso_codec_cache_dir,
                strict=dyrso_i_pair_strict,
                z_clip=dyrso_codec_z_clip,
            )
            codec_metadata = extractor.extract(video_path)
            dyrso_frame_bitcosts = {
                int(frame.display_frame_id): float(frame.bitcost_bits)
                for frame in codec_metadata.frames
            }
            if not dyrso_frame_bitcosts:
                raise ValueError("empty HEVC frame bitcost metadata")
        except Exception as exc:
            dyrso_alignment_failure_count += 1
            if dyrso_i_pair_strict:
                raise
            logger.warning(
                "HEVC-guided mosaic scheduling disabled for %s; falling back to fixed. reason=%s",
                video_path,
                exc,
            )
            dyrso_i_pair_mode = "fixed"
            dyrso_extraction_fallback = True

    frames_indices, plans, required_frame_indices = build_mosaic_frame_plan(
        total_num_frames,
        nframes,
        mosaic_source_frames=mosaic_source_frames,
        mosaic_pair_pack_order=mosaic_pair_pack_order,
        strict_pairing=mosaic_strict_pairing,
        dyrso_i_pair_mode=dyrso_i_pair_mode,
        frame_bitcost_by_index=dyrso_frame_bitcosts,
        dyrso_i_pair_candidates=dyrso_i_pair_candidates,
        dyrso_hevc_pair_max_weight=dyrso_hevc_pair_max_weight,
        dyrso_i_pair_strict=dyrso_i_pair_strict,
    )
    frame_cache = read_frames_decord_batch(
        vr,
        required_frame_indices,
        batch_size=read_batch_size,
    )
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def get_cached_frame(frame_idx: int, group_id: int, frame_role: str) -> np.ndarray:
        clamped_idx = clamp_frame_idx(frame_idx, total_num_frames)
        frame = frame_cache.get(clamped_idx)
        if frame is None:
            logger.warning(
                "[GROUP %03d] could not read %s frame %s; using black frame",
                group_id,
                frame_role,
                frame_idx,
            )
            return black_frame
        return frame

    output_frames: List[np.ndarray] = []
    output_frame_indices: List[int] = []
    output_types: List[str] = []
    output_source_frame_indices: List[List[int]] = []
    mosaic_pair_stats = {
        "total_mosaic_pairs": 0,
        "total_tile_pairs": 0,
        "misaligned_tile_pairs": 0,
        "heterogeneous_temporal_pairs": 0,
        "pair_aligned_fallbacks": 0,
    }
    dyrso_i_pair_stats = {
        "selected_pair_histogram": {},
        "output_pattern_counts": {"IIPP": 0},
        "alignment_failure_count": int(dyrso_alignment_failure_count),
        "fallback_count": int(dyrso_extraction_fallback),
        "invalid_candidate_count": 0,
        "heterogeneous_pair_count": 0,
    }
    dumped_first_group = False
    for plan in plans:
        group_id = plan["group_id"]
        original_frames = plan["original_frames"]
        mosaic_0_frames = plan["mosaic_0_frames"]
        mosaic_1_frames = plan["mosaic_1_frames"]
        logical_indices = plan["logical_indices"]

        original_images = [
            get_cached_frame(frame_idx, group_id, "original")
            for frame_idx in original_frames
        ]

        mosaic_images: List[np.ndarray] = []
        for mosaic_frame_indices in (mosaic_0_frames, mosaic_1_frames):
            source_images = [
                get_cached_frame(frame_idx, group_id, "mosaic source")
                for frame_idx in mosaic_frame_indices
            ]
            mosaic_images.append(make_mosaic(source_images, width, height))

        packed_items = {
            "original_0": (original_images[0], int(original_frames[0])),
            "original_1": (original_images[1], int(original_frames[1])),
            "mosaic_0": (mosaic_images[0], int(logical_indices[2])),
            "mosaic_1": (mosaic_images[1], int(logical_indices[3])),
        }
        output_order = list(plan.get("output_order") or _mosaic_output_order(mosaic_packing_order))
        if mosaic_first_group_dump_dir and not dumped_first_group:
            dumped_first_group = _save_first_mosaic_group_dump(
                dump_dir=mosaic_first_group_dump_dir,
                video_path=video_path,
                ele=ele,
                plan=plan,
                packed_items=packed_items,
                output_order=output_order,
                mosaic_packing_order=mosaic_packing_order,
                mosaic_pair_pack_order=mosaic_pair_pack_order,
                dyrso_i_pair_mode=dyrso_i_pair_mode,
            )
        for item_key in output_order:
            frame, frame_idx = packed_items[item_key]
            output_frames.append(frame)
            output_frame_indices.append(frame_idx)
            output_types.append("mosaic" if item_key.startswith("mosaic") else "full")
            if item_key == "original_0":
                output_source_frame_indices.append([int(original_frames[0])])
            elif item_key == "original_1":
                output_source_frame_indices.append([int(original_frames[1])])
            elif item_key == "mosaic_0":
                output_source_frame_indices.append([int(value) for value in mosaic_0_frames])
            else:
                output_source_frame_indices.append([int(value) for value in mosaic_1_frames])

        tile_pairs = list(plan.get("pair_tile_pairs") or [])
        mosaic_pair_stats["total_mosaic_pairs"] += 1
        mosaic_pair_stats["total_tile_pairs"] += len(tile_pairs)
        mosaic_pair_stats["misaligned_tile_pairs"] += _pair_alignment_misaligned_count(tile_pairs)
        if plan.get("pair_aligned_fallback"):
            mosaic_pair_stats["pair_aligned_fallbacks"] += 1
        selection = dict(plan.get("dyrso_i_pair_selection") or {})
        selected_pair = int(selection.get("selected_pair_index", 0))
        hist_key = str(selected_pair)
        dyrso_i_pair_stats["selected_pair_histogram"][hist_key] = (
            int(dyrso_i_pair_stats["selected_pair_histogram"].get(hist_key, 0)) + 1
        )
        if selection.get("fallback_used"):
            dyrso_i_pair_stats["fallback_count"] += 1
        pattern = str(selection.get("output_pattern") or "IIPP")
        if pattern in dyrso_i_pair_stats["output_pattern_counts"]:
            dyrso_i_pair_stats["output_pattern_counts"][pattern] += 1
        final_types = list(selection.get("final_input_types") or ["full", "full", "mosaic", "mosaic"])
        dyrso_i_pair_stats["heterogeneous_pair_count"] += count_heterogeneous_temporal_pairs(final_types)

        if dyrso_dump_i_pair_selection and selection.get("mode") == "hevc_bitcost":
            print("[v4.6.1 HEVC-Guided IIPP]")
            print(f"question_id={ele.get('question_id', ele.get('sample_id', 'unknown'))}")
            print(f"video={video_path}")
            print(f"group_id={group_id}")
            print(f"sample_ranks={selection.get('sample_ranks')}")
            print(f"frame_indices={selection.get('group_frame_indices')}")
            print(f"raw_bitcost={selection.get('raw_bitcost')}")
            print(f"normalized_bitcost={selection.get('normalized_bitcost')}")
            print(
                "pair_scores="
                f"{[(idx, score) for idx, score in enumerate(selection.get('pair_scores') or [])]}"
            )
            print(f"candidate_pairs={selection.get('candidate_pairs')}")
            print(f"selected_pair={selection.get('selected_pair_index')}")
            print(f"selected_frames={selection.get('selected_frames')}")
            print(f"low_frames={selection.get('low_frames')}")
            print(f"mosaic0={selection.get('mosaic0_frames')}")
            print(f"mosaic1={selection.get('mosaic1_frames')}")
            print(f"final_input_types={selection.get('final_input_types')}")
            print(
                "homogeneous_temporal_pairs="
                f"{count_heterogeneous_temporal_pairs(selection.get('final_input_types') or []) == 0}"
            )
        elif dyrso_dump_i_pair_selection and selection.get("mode") == "hevc_tile_sort":
            print("[v4.6.2 HEVC Tile Sort]")
            print(f"question_id={ele.get('question_id', ele.get('sample_id', 'unknown'))}")
            print(f"video={video_path}")
            print(f"group_id={group_id}")
            print(f"selected_frames={selection.get('selected_frames')}")
            print(f"mosaic0_original={selection.get('mosaic0_original_order')}")
            print(f"mosaic0_sorted={selection.get('mosaic0_frames')}")
            print(f"mosaic0_tile_sort={selection.get('mosaic0_tile_sort')}")
            print(f"mosaic1_original={selection.get('mosaic1_original_order')}")
            print(f"mosaic1_sorted={selection.get('mosaic1_frames')}")
            print(f"mosaic1_tile_sort={selection.get('mosaic1_tile_sort')}")
            print(f"final_input_types={selection.get('final_input_types')}")

    if not output_frames:
        raise ValueError(f"no mosaic frames were generated for video: {video_path}")

    mosaic_pair_stats["heterogeneous_temporal_pairs"] = count_heterogeneous_temporal_pairs(output_types)
    if mosaic_require_homogeneous_pair:
        validate_homogeneous_temporal_pairs(output_types)
    if mosaic_strict_pairing and mosaic_pair_pack_order == "pair_aligned":
        if mosaic_pair_stats["misaligned_tile_pairs"] != 0:
            raise AssertionError(
                f"pair_aligned mosaic produced {mosaic_pair_stats['misaligned_tile_pairs']} "
                f"misaligned tile pairs for video={video_path}"
            )
        if mosaic_pair_stats["pair_aligned_fallbacks"] != 0:
            raise AssertionError(
                f"pair_aligned mosaic fell back to sequential for "
                f"{mosaic_pair_stats['pair_aligned_fallbacks']} groups in video={video_path}"
            )
    if mosaic_strict_pairing and mosaic_require_homogeneous_pair:
        if mosaic_pair_stats["heterogeneous_temporal_pairs"] != 0:
            raise AssertionError(
                f"mosaic output produced {mosaic_pair_stats['heterogeneous_temporal_pairs']} "
                f"heterogeneous Qwen3 temporal pairs for video={video_path}"
            )
    if dyrso_i_pair_strict and dyrso_i_pair_stats["heterogeneous_pair_count"] != 0:
        raise AssertionError(
            f"DyRSO I-pair scheduling produced {dyrso_i_pair_stats['heterogeneous_pair_count']} "
            f"heterogeneous Qwen3 temporal pairs for video={video_path}"
        )

    if mosaic_dump_pair_mapping:
        first_plan = plans[0] if plans else {}
        tile_pairs = list(first_plan.get("pair_tile_pairs") or [])
        print("[v4.6 Pair-Aligned Mosaic]")
        print(f"version={ele.get('semantic_version', 'unknown')}")
        print(f"pack_order={mosaic_pair_pack_order}")
        print(f"input_pattern={''.join('I' if t == 'full' else 'P' for t in output_types[:4])}")
        print(f"homogeneous_temporal_pairs={mosaic_pair_stats['heterogeneous_temporal_pairs'] == 0}")
        print(f"heterogeneous_pair_count={mosaic_pair_stats['heterogeneous_temporal_pairs']}")
        print(f"canvas_a_sample_ranks={[p.get('first_sample_rank') for p in tile_pairs]}")
        print(f"canvas_b_sample_ranks={[p.get('second_sample_rank') for p in tile_pairs]}")
        print(
            "tile_pairs="
            f"{[(p.get('first_sample_rank'), p.get('second_sample_rank')) for p in tile_pairs]}"
        )
        print(f"canvas_a_frame_indices={list(first_plan.get('mosaic_0_frames') or [])}")
        print(f"canvas_b_frame_indices={list(first_plan.get('mosaic_1_frames') or [])}")
        print(f"pair_alignment_valid={mosaic_pair_stats['misaligned_tile_pairs'] == 0}")

    video = torch.from_numpy(np.stack(output_frames, axis=0)).permute(0, 3, 1, 2)
    sample_fps = len(output_frames) / max(total_num_frames, 1e-6) * fps
    metadata = {
        "fps": fps,
        "frame_extract_version": frame_extract_version,
        "frames_indices": output_frame_indices,
        "input_types": output_types,
        "sampled_source_frame_indices": output_source_frame_indices,
        "uniform_frames_indices": frames_indices,
        "mosaic_packing_order": mosaic_packing_order,
        "mosaic_pair_pack_order": mosaic_pair_pack_order,
        "mosaic_pair_stats": mosaic_pair_stats,
        "dyrso_i_pair_mode": dyrso_i_pair_mode,
        "dyrso_i_pair_stats": dyrso_i_pair_stats,
        "mosaic_pair_plans": [
            {
                "group_id": plan.get("group_id"),
                "mosaic_0_frames": plan.get("mosaic_0_frames"),
                "mosaic_1_frames": plan.get("mosaic_1_frames"),
                "pair_tile_pairs": plan.get("pair_tile_pairs"),
                "pair_alignment_valid": plan.get("pair_alignment_valid"),
                "pair_aligned_fallback": plan.get("pair_aligned_fallback"),
                "dyrso_i_pair_selection": plan.get("dyrso_i_pair_selection"),
                "output_order": plan.get("output_order"),
            }
            for plan in plans
        ],
        "total_num_frames": total_num_frames,
        "video_backend": "decord_mosaic",
    }
    video = _resize_mosaic_video_for_qwen(video, ele, image_patch_size=image_patch_size)
    logger.info(
        "decord_mosaic: video_path=%s total_frames=%s fps=%s output_frames=%s",
        video_path,
        total_num_frames,
        fps,
        len(output_frames),
    )
    return video, metadata, sample_fps


def process_mosaic_video_vision_info(
    conversations,
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 16,
    mosaic_source_frames: int = 8,
    read_batch_size: int = 64,
    frame_extract_version: str = "v1",
    mosaic_packing_order: Optional[str] = None,
    mosaic_pair_pack_order: str = "sequential",
    mosaic_require_homogeneous_pair: bool = False,
    mosaic_strict_pairing: bool = True,
    mosaic_dump_pair_mapping: bool = False,
    dyrso_i_pair_mode: str = "fixed",
    dyrso_hevc_pair_max_weight: float = 0.3,
    dyrso_i_pair_candidates: Optional[str] = None,
    dyrso_i_pair_strict: bool = True,
    dyrso_dump_i_pair_selection: bool = False,
    dyrso_codec_cache_dir: Optional[str] = None,
    dyrso_codec_ffprobe_path: str = "ffprobe",
    dyrso_codec_z_clip: float = 5.0,
    mosaic_first_group_dump_dir: Optional[str] = None,
):
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            video, video_metadata, sample_fps = load_mosaic_video_for_qwen(
                vision_info,
                image_patch_size=image_patch_size,
                mosaic_source_frames=mosaic_source_frames,
                read_batch_size=read_batch_size,
                frame_extract_version=frame_extract_version,
                mosaic_packing_order=mosaic_packing_order,
                mosaic_pair_pack_order=mosaic_pair_pack_order,
                mosaic_require_homogeneous_pair=mosaic_require_homogeneous_pair,
                mosaic_strict_pairing=mosaic_strict_pairing,
                mosaic_dump_pair_mapping=mosaic_dump_pair_mapping,
                dyrso_i_pair_mode=dyrso_i_pair_mode,
                dyrso_hevc_pair_max_weight=dyrso_hevc_pair_max_weight,
                dyrso_i_pair_candidates=dyrso_i_pair_candidates,
                dyrso_i_pair_strict=dyrso_i_pair_strict,
                dyrso_dump_i_pair_selection=dyrso_dump_i_pair_selection,
                dyrso_codec_cache_dir=dyrso_codec_cache_dir,
                dyrso_codec_ffprobe_path=dyrso_codec_ffprobe_path,
                dyrso_codec_z_clip=dyrso_codec_z_clip,
                mosaic_first_group_dump_dir=mosaic_first_group_dump_dir,
            )
            video_sample_fps_list.append(sample_fps)
            video_inputs.append((video, video_metadata) if return_video_metadata else video)
        else:
            raise ValueError("image, image_url or video should in content.")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {"do_sample_frames": False}
    if not return_video_metadata:
        video_kwargs.update({"fps": video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs


def extract_vision_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

    vision_infos = extract_vision_info(conversations)
    # print(vision_infos)
   
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            # print(1)
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True,
                        image_patch_size=image_patch_size, return_video_metadata=return_video_metadata)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {'do_sample_frames': False}
    if not return_video_metadata: # BC for qwen2.5vl
        video_kwargs.update({'fps': video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
