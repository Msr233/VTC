from __future__ import annotations

import bisect
import fcntl
import gzip
import hashlib
import heapq
import json
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 1
PACKED_SCHEMA_VERSION = 2
SUPPORTED_CACHE_SCHEMA_VERSIONS = {
    LEGACY_SCHEMA_VERSION,
    PACKED_SCHEMA_VERSION,
    SCHEMA_VERSION,
}
SUPPORTED_HEVC_NAMES = {"hevc", "h265"}
SUPPORTED_H264_NAMES = {"h264", "avc", "avc1", "libx264"}
SUPPORTED_CODEC_NAMES = SUPPORTED_HEVC_NAMES | SUPPORTED_H264_NAMES
EPS = 1e-6


class UnsupportedCodecError(ValueError):
    """Raised when frame metadata is valid but the video codec is unsupported."""


@dataclass
class HEVCFrameBitcost:
    display_frame_id: int
    coded_frame_id: int | None
    pts_time: float
    dts_time: float | None
    duration_time: float | None
    pict_type: str
    key_frame: bool
    packet_size_bytes: int
    bitcost_bits: float
    bitcost_bpp: float
    log_bitcost: float
    robust_pb_score: float = 0.0
    positive_bitcost: float = 1.0


@dataclass
class HEVCBitcostMetadata:
    video_path: str
    codec_name: str
    width: int
    height: int
    avg_frame_rate: float
    duration: float
    total_frames: int
    frames: list[HEVCFrameBitcost]


@dataclass
class VideoSamplingContext:
    video_path: str
    source_fps: float
    source_total_frames: int
    sampled_frame_indices: list[int]
    sampled_timestamps: list[float]
    temporal_patch_size: int
    grid_t: int
    sampled_source_frame_indices: list[list[int]] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)


@dataclass
class TemporalStepCodecStats:
    temporal_step_id: int
    left_time: float
    right_time: float
    center_time: float
    sampled_frame_indices: list[int]
    sampled_timestamps: list[float]
    sampled_frame_pict_types: list[str]
    sampled_frame_type_sources: list[str]
    codec_semantic_frame_type: str | None
    codec_frame_type_valid: bool
    codec_frame_type_fallback_reason: str | None
    codec_pb_frame_count: int
    codec_p_frame_count: int
    codec_b_frame_count: int
    codec_i_frame_count: int
    pb_mean: float
    pb_p90: float
    pb_max: float
    pb_sum: float
    codec_complexity: float
    used_fallback: bool
    fallback_reason: str | None


def _parse_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "N/A", ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, "N/A", ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_fraction(value: Any) -> float:
    if not value or value == "0/0":
        return 0.0
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        return 0.0 if den_f == 0 else float(num) / den_f
    return float(text)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _cache_key(video_path: str | Path) -> tuple[str, dict[str, Any]]:
    path = Path(video_path).resolve()
    stat = path.stat()
    payload = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return digest, payload


def _metadata_to_json(metadata: HEVCBitcostMetadata, cache_identity: dict[str, Any] | None = None) -> dict:
    frames = metadata.frames
    timeline_end = float(metadata.duration)
    if frames:
        last = frames[-1]
        timeline_end = max(timeline_end, float(last.pts_time) + float(last.duration_time or 0.0))
    pts_us = [int(round(float(frame.pts_time) * 1_000_000)) for frame in frames]
    pts_us_delta = (
        [pts_us[0]]
        + [pts_us[index] - pts_us[index - 1] for index in range(1, len(pts_us))]
        if pts_us
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_identity": cache_identity,
        "codec_name": metadata.codec_name,
        "width": int(metadata.width),
        "height": int(metadata.height),
        "avg_frame_rate": float(metadata.avg_frame_rate),
        "duration": float(metadata.duration),
        "timeline_end": timeline_end,
        "frames": {
            "pts_us_delta": pts_us_delta,
            "types": "".join(frame.pict_type for frame in frames),
            "packet_sizes": [int(frame.packet_size_bytes) for frame in frames],
        },
    }


def metadata_from_json(data: dict, *, z_clip: float = 5.0) -> HEVCBitcostMetadata:
    schema_version = int(data.get("schema_version") or LEGACY_SCHEMA_VERSION)
    if schema_version in {PACKED_SCHEMA_VERSION, SCHEMA_VERSION}:
        packed = data.get("frames") or {}
        if schema_version == SCHEMA_VERSION:
            pts_us_delta = [int(value) for value in packed.get("pts_us_delta") or []]
            pts_us: list[int] = []
            for delta in pts_us_delta:
                pts_us.append(delta + (pts_us[-1] if pts_us else 0))
            pts_values = [value / 1_000_000.0 for value in pts_us]
        else:
            pts_values = list(packed.get("pts") or [])
        pict_types = str(packed.get("types") or "")
        packet_sizes = list(packed.get("packet_sizes") or [])
        if not (len(pts_values) == len(pict_types) == len(packet_sizes)):
            raise ValueError(
                "Invalid codec cache v2: pts/types/packet_sizes lengths do not match"
            )
        area = max(1, int(data.get("width") or 0) * int(data.get("height") or 0))
        timeline_end = float(data.get("timeline_end") or data.get("duration") or 0.0)
        frames = []
        for frame_id, (pts, pict_type, packet_size) in enumerate(
            zip(pts_values, pict_types, packet_sizes)
        ):
            packet_size = int(packet_size)
            bitcost_bits = float(packet_size * 8)
            bitcost_bpp = bitcost_bits / float(area)
            frames.append(
                HEVCFrameBitcost(
                    display_frame_id=frame_id,
                    coded_frame_id=None,
                    pts_time=float(pts),
                    dts_time=None,
                    duration_time=(
                        max(0.0, timeline_end - float(pts))
                        if frame_id == len(pts_values) - 1
                        else None
                    ),
                    pict_type=pict_type,
                    key_frame=pict_type == "I",
                    packet_size_bytes=packet_size,
                    bitcost_bits=bitcost_bits,
                    bitcost_bpp=bitcost_bpp,
                    log_bitcost=math.log1p(bitcost_bpp),
                )
            )
        identity = data.get("cache_identity") or {}
        video_path = str(identity.get("path") or "")
    elif schema_version == LEGACY_SCHEMA_VERSION:
        frames = [HEVCFrameBitcost(**frame) for frame in data.get("frames", [])]
        video_path = str(data.get("video_path") or (data.get("cache_identity") or {}).get("path") or "")
    else:
        raise ValueError(f"Unsupported codec cache schema_version={schema_version}")

    metadata = HEVCBitcostMetadata(
        video_path=video_path,
        codec_name=str(data.get("codec_name", "")),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        avg_frame_rate=float(data.get("avg_frame_rate") or 0.0),
        duration=float(data.get("duration") or 0.0),
        total_frames=int(data.get("total_frames") or len(frames)),
        frames=frames,
    )
    # Normalization is configuration-dependent and must never be persisted as
    # authoritative cache state.
    apply_robust_pb_normalization(metadata, z_clip=z_clip)
    return metadata


def build_metadata_from_ffprobe_json(video_path: str | Path, data: dict, *, z_clip: float = 5.0) -> HEVCBitcostMetadata:
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    codec_name = str(stream.get("codec_name") or "").lower()
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    avg_frame_rate = _parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = _parse_float(stream.get("duration"), 0.0) or 0.0
    area = max(1, width * height)

    raw_frames = []
    fallback_pts_count = 0
    for raw_idx, item in enumerate(data.get("frames") or []):
        pts = (
            _parse_float(item.get("best_effort_timestamp_time"))
            if item.get("best_effort_timestamp_time") not in (None, "N/A")
            else _parse_float(item.get("pts_time"))
        )
        if pts is None:
            pts = raw_idx / avg_frame_rate if avg_frame_rate > 0 else float(raw_idx)
            fallback_pts_count += 1
        dts = _parse_float(item.get("pkt_dts_time"))
        dur = _parse_float(item.get("pkt_duration_time"))
        pkt = _parse_int(item.get("pkt_size"), 0) or 0
        pict = str(item.get("pict_type") or "?").upper()
        if pict not in {"I", "P", "B"}:
            pict = "?"
        bitcost_bits = float(pkt * 8)
        bitcost_bpp = bitcost_bits / float(area)
        log_bitcost = math.log1p(bitcost_bpp)
        coded = _parse_int(
            item.get("coded_picture_number"),
            _parse_int(item.get("display_picture_number")),
        )
        raw_frames.append({
            "pts": float(pts),
            "coded": coded,
            "dts": dts,
            "dur": dur,
            "pict": pict,
            "key": bool(_parse_int(item.get("key_frame"), 0)),
            "pkt": pkt,
            "bits": bitcost_bits,
            "bpp": bitcost_bpp,
            "log": log_bitcost,
            "raw_idx": raw_idx,
        })

    raw_frames.sort(key=lambda x: (x["pts"], x["raw_idx"]))
    frames = [
        HEVCFrameBitcost(
            display_frame_id=i,
            coded_frame_id=item["coded"],
            pts_time=float(item["pts"]),
            dts_time=item["dts"],
            duration_time=item["dur"],
            pict_type=item["pict"],
            key_frame=bool(item["key"]),
            packet_size_bytes=int(item["pkt"]),
            bitcost_bits=float(item["bits"]),
            bitcost_bpp=float(item["bpp"]),
            log_bitcost=float(item["log"]),
        )
        for i, item in enumerate(raw_frames)
    ]
    metadata = HEVCBitcostMetadata(
        video_path=str(video_path),
        codec_name=codec_name,
        width=width,
        height=height,
        avg_frame_rate=avg_frame_rate,
        duration=duration if duration > 0 else (frames[-1].pts_time if frames else 0.0),
        total_frames=len(frames),
        frames=frames,
    )
    apply_robust_pb_normalization(metadata, z_clip=z_clip)
    if fallback_pts_count:
        print(f"[CodecBitcost] WARNING: {fallback_pts_count} frame(s) missing PTS; used frame_id/fps fallback.")
    return metadata


def apply_robust_pb_normalization(metadata: HEVCBitcostMetadata, *, z_clip: float = 5.0) -> None:
    for pict in ("P", "B"):
        values = [frame.log_bitcost for frame in metadata.frames if frame.pict_type == pict]
        med = _median(values)
        mad = _median([abs(v - med) for v in values])
        for frame in metadata.frames:
            if frame.pict_type != pict:
                continue
            if mad < EPS:
                z = 0.0
            else:
                z = (frame.log_bitcost - med) / (mad + EPS)
                z = max(-float(z_clip), min(float(z_clip), z))
            frame.robust_pb_score = float(z)
            frame.positive_bitcost = float(_softplus(z) + EPS)
    for frame in metadata.frames:
        if frame.pict_type not in ("P", "B"):
            frame.robust_pb_score = 0.0
            frame.positive_bitcost = 0.0


class HEVCBitcostExtractor:
    def __init__(
        self,
        *,
        ffprobe_path: str = "ffprobe",
        cache_dir: str | Path | None = None,
        strict: bool = False,
        z_clip: float = 5.0,
        supported_codecs: Iterable[str] | None = None,
    ) -> None:
        self.ffprobe_path = ffprobe_path
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.strict = bool(strict)
        self.z_clip = float(z_clip)
        self.supported_codecs = {str(x).lower() for x in (supported_codecs or SUPPORTED_CODEC_NAMES)}

    def extract(self, video_path: str | Path) -> HEVCBitcostMetadata:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(str(video_path))
        if self.cache_dir is not None:
            cached = self._read_cache(path)
            if cached is not None:
                return cached
            with self._cache_lock(path):
                # Another worker may have populated the cache while this
                # process was waiting for the per-video lock.
                cached = self._read_cache(path)
                if cached is not None:
                    return cached
                return self._extract_uncached(path)
        return self._extract_uncached(path)

    def _extract_uncached(self, path: Path) -> HEVCBitcostMetadata:
        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            (
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration:"
                "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,pict_type,pkt_size"
            ),
            "-of",
            "json=compact=1",
            str(path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        metadata = build_metadata_from_ffprobe_json(str(path), json.loads(proc.stdout), z_clip=self.z_clip)
        if self.strict:
            validate_metadata(metadata, strict=True, supported_codecs=self.supported_codecs)
        else:
            if metadata.codec_name not in self.supported_codecs:
                print(
                    f"[CodecBitcost] WARNING: codec_name={metadata.codec_name!r}; "
                    f"expected one of {sorted(self.supported_codecs)}. "
                    "Continuing in non-strict mode."
                )
            validate_metadata(metadata, strict=False, supported_codecs=self.supported_codecs)
        if self.cache_dir is not None:
            self._write_cache(path, metadata)
        return metadata

    @contextmanager
    def _cache_lock(self, video_path: Path):
        assert self.cache_dir is not None
        cache_path, _ = self._cache_path(video_path)
        lock_dir = self.cache_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{cache_path.name}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _cache_path(self, video_path: Path) -> tuple[Path, dict[str, Any]]:
        digest, identity = _cache_key(video_path)
        stem = video_path.stem.replace(os.sep, "_")
        assert self.cache_dir is not None
        return self.cache_dir / f"{stem}_{digest}.json.gz", identity

    def _read_cache(self, video_path: Path) -> HEVCBitcostMetadata | None:
        cache_path, identity = self._cache_path(video_path)
        if not cache_path.exists():
            return self._read_compatible_cache(video_path, identity, cache_path)
        try:
            data = self._load_cache_json(cache_path)
            if int(data.get("schema_version") or 0) not in {
                PACKED_SCHEMA_VERSION,
                SCHEMA_VERSION,
            }:
                return None
            if data.get("cache_identity") != identity:
                return None
            return metadata_from_json(data, z_clip=self.z_clip)
        except Exception:
            return None

    def _read_compatible_cache(
        self,
        video_path: Path,
        identity: dict[str, Any],
        preferred_cache_path: Path,
    ) -> HEVCBitcostMetadata | None:
        if self.cache_dir is None:
            return None
        current_name = Path(identity["path"]).name
        current_size = int(identity["size"])
        stem = video_path.stem.replace(os.sep, "_")
        candidates = list(self.cache_dir.glob(f"{stem}_*.json.gz"))
        candidates.extend(self.cache_dir.glob(f"{stem}_*.json"))
        for candidate in sorted(candidates):
            if candidate == preferred_cache_path:
                continue
            try:
                data = self._load_cache_json(candidate)
            except Exception:
                continue
            if int(data.get("schema_version") or 0) not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
                continue
            candidate_identity = data.get("cache_identity") or {}
            candidate_path = str(candidate_identity.get("path") or data.get("video_path") or "")
            if Path(candidate_path).name != current_name:
                continue
            if int(candidate_identity.get("size") or -1) != current_size:
                continue
            metadata = metadata_from_json(data, z_clip=self.z_clip)
            metadata.video_path = str(video_path)
            self._write_cache(video_path, metadata)
            return metadata
        return None

    @staticmethod
    def _load_cache_json(cache_path: Path) -> dict:
        if cache_path.suffix == ".gz":
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    def _write_cache(self, video_path: Path, metadata: HEVCBitcostMetadata) -> None:
        cache_path, identity = self._cache_path(video_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = _metadata_to_json(metadata, identity)
        fd, tmp_name = tempfile.mkstemp(prefix=cache_path.name, suffix=".tmp", dir=str(cache_path.parent))
        try:
            encoded = json.dumps(
                data,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
            with os.fdopen(fd, "wb") as f:
                f.write(compressed)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, cache_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def validate_metadata(
    metadata: HEVCBitcostMetadata,
    *,
    strict: bool,
    supported_codecs: Iterable[str] | None = None,
) -> None:
    supported = {str(x).lower() for x in (supported_codecs or SUPPORTED_CODEC_NAMES)}
    if strict and metadata.codec_name not in supported:
        raise UnsupportedCodecError(
            f"Expected codec in {sorted(supported)}, got codec_name={metadata.codec_name!r}"
        )
    if not metadata.frames:
        raise ValueError("No frame metadata found")
    if not any(frame.packet_size_bytes > 0 for frame in metadata.frames):
        raise ValueError("No valid pkt_size found in frame metadata")
    if not any(math.isfinite(frame.pts_time) for frame in metadata.frames):
        raise ValueError("No valid PTS found in frame metadata")


class HEVCBitcostAligner:
    def __init__(
        self,
        *,
        aggregation: str = "mean_p90",
        normalization: str = "robust_pb",
        i_policy: str = "exclude",
        mean_weight: float = 0.5,
        p90_weight: float = 0.5,
        z_clip: float = 5.0,
        eps: float = EPS,
    ) -> None:
        if aggregation not in {"mean", "p90", "mean_p90", "max", "sum"}:
            raise ValueError(f"Unsupported codec aggregation mode: {aggregation}")
        if normalization not in {"raw_bits", "bpp", "log_bpp", "robust_pb"}:
            raise ValueError(f"Unsupported codec normalization: {normalization}")
        if i_policy not in {"exclude", "nearest_pb", "clip_as_pb"}:
            raise ValueError(f"Unsupported codec I-frame policy: {i_policy}")
        self.aggregation = aggregation
        self.normalization = normalization
        self.i_policy = i_policy
        self.mean_weight = float(mean_weight)
        self.p90_weight = float(p90_weight)
        self.z_clip = float(z_clip)
        self.eps = float(eps)

    def align_to_temporal_steps(
        self,
        metadata: HEVCBitcostMetadata,
        sampling_context: VideoSamplingContext,
    ) -> list[TemporalStepCodecStats]:
        tps = max(1, int(sampling_context.temporal_patch_size))
        grid_t = int(sampling_context.grid_t)
        indices = list(sampling_context.sampled_frame_indices)
        timestamps = list(sampling_context.sampled_timestamps or [])
        if not timestamps or len(timestamps) != len(indices):
            timestamps = self._timestamps_from_metadata(metadata, indices, sampling_context.source_fps)
        if len(indices) < grid_t * tps:
            pad_count = grid_t * tps - len(indices)
            if indices:
                indices = indices + [indices[-1]] * pad_count
                timestamps = timestamps + [timestamps[-1]] * pad_count
            else:
                indices = [0] * (grid_t * tps)
                timestamps = [0.0] * (grid_t * tps)
        if len(indices) < grid_t * tps:
            raise ValueError("Unable to build sampled frame groups for temporal steps")

        centers = []
        groups = []
        for step_id in range(grid_t):
            start = step_id * tps
            end = start + tps
            group_indices = [int(x) for x in indices[start:end]]
            group_ts = [float(x) for x in timestamps[start:end]]
            groups.append((group_indices, group_ts))
            centers.append(sum(group_ts) / max(1, len(group_ts)))

        video_start = metadata.frames[0].pts_time if metadata.frames else 0.0
        video_end = metadata.duration
        if metadata.frames:
            last = metadata.frames[-1]
            video_end = max(video_end, last.pts_time + (last.duration_time or 0.0))
        pb_frames = [f for f in metadata.frames if f.pict_type in {"P", "B"}]
        p_logs = [f.log_bitcost for f in metadata.frames if f.pict_type == "P"]
        p_median = _median(p_logs)
        p_mad = _median([abs(v - p_median) for v in p_logs])
        frame_pts = [frame.pts_time for frame in metadata.frames]
        results = []
        for step_id, center in enumerate(centers):
            left = video_start if step_id == 0 else 0.5 * (centers[step_id - 1] + center)
            right = video_end if step_id == grid_t - 1 else 0.5 * (center + centers[step_id + 1])
            window_start = bisect.bisect_left(frame_pts, left)
            window_end = bisect.bisect_left(frame_pts, right, lo=window_start)
            window = metadata.frames[window_start:window_end]
            usable = [f for f in window if f.pict_type in {"P", "B"}]
            i_count = sum(1 for f in window if f.pict_type == "I")
            used_fallback = False
            fallback_reason = None
            if not usable and self.i_policy == "clip_as_pb":
                usable = [f for f in window if f.pict_type == "I"]
            if not usable:
                used_fallback = True
                if pb_frames:
                    nearest = min(pb_frames, key=lambda f: abs(f.pts_time - center))
                    usable = [nearest]
                    fallback_reason = "nearest_pb"
                else:
                    fallback_reason = "neutral_no_pb"
            values = []
            for frame in usable:
                value = self._frame_value(frame, p_median=p_median, p_mad=p_mad)
                if math.isfinite(value):
                    values.append(value)
            if not values:
                values = [1.0]
                used_fallback = True
                fallback_reason = fallback_reason or "neutral_invalid_values"
            pb_mean = sum(values) / len(values)
            pb_p90 = _percentile(values, 0.9)
            pb_max = max(values)
            pb_sum = sum(values)
            complexity = self._aggregate(pb_mean, pb_p90, pb_max, pb_sum)
            group_indices, group_ts = groups[step_id]
            (
                sampled_pict_types,
                sampled_type_sources,
                codec_semantic_type,
                codec_frame_type_valid,
                codec_frame_type_fallback_reason,
            ) = self._align_sampled_frame_types(
                metadata,
                group_indices,
                group_ts,
            )
            results.append(
                TemporalStepCodecStats(
                    temporal_step_id=step_id,
                    left_time=float(left),
                    right_time=float(right),
                    center_time=float(center),
                    sampled_frame_indices=group_indices,
                    sampled_timestamps=group_ts,
                    sampled_frame_pict_types=sampled_pict_types,
                    sampled_frame_type_sources=sampled_type_sources,
                    codec_semantic_frame_type=codec_semantic_type,
                    codec_frame_type_valid=codec_frame_type_valid,
                    codec_frame_type_fallback_reason=codec_frame_type_fallback_reason,
                    codec_pb_frame_count=sum(1 for f in usable if f.pict_type in {"P", "B"}),
                    codec_p_frame_count=sum(1 for f in usable if f.pict_type == "P"),
                    codec_b_frame_count=sum(1 for f in usable if f.pict_type == "B"),
                    codec_i_frame_count=i_count,
                    pb_mean=float(pb_mean),
                    pb_p90=float(pb_p90),
                    pb_max=float(pb_max),
                    pb_sum=float(pb_sum),
                    codec_complexity=max(float(complexity), self.eps),
                    used_fallback=used_fallback,
                    fallback_reason=fallback_reason,
                )
            )
        return results

    @staticmethod
    def _align_sampled_frame_types(
        metadata: HEVCBitcostMetadata,
        sampled_indices: Sequence[int],
        sampled_timestamps: Sequence[float],
    ) -> tuple[list[str], list[str], str | None, bool, str | None]:
        """Map sampled display frames to codec picture types.

        Display-frame indices are authoritative when available. Timestamps are
        only used when an index falls outside the extracted display-frame list.
        """
        frames = metadata.frames
        pict_types: list[str] = []
        sources: list[str] = []
        for position, raw_index in enumerate(sampled_indices):
            frame = None
            source = "unavailable"
            frame_index = int(raw_index)
            if 0 <= frame_index < len(frames):
                frame = frames[frame_index]
                source = "display_index"
            elif frames and position < len(sampled_timestamps):
                timestamp = float(sampled_timestamps[position])
                if math.isfinite(timestamp):
                    frame = min(frames, key=lambda item: abs(item.pts_time - timestamp))
                    source = "nearest_pts"
            pict_types.append(frame.pict_type if frame is not None else "?")
            sources.append(source)

        invalid_positions = [
            index for index, pict_type in enumerate(pict_types)
            if pict_type not in {"I", "P", "B"}
        ]
        if invalid_positions:
            reason = "invalid_or_missing_pict_type:" + ",".join(
                str(index) for index in invalid_positions
            )
            return pict_types, sources, None, False, reason
        if not pict_types:
            return pict_types, sources, None, False, "no_sampled_frames"
        semantic_type = "I" if "I" in pict_types else "P"
        return pict_types, sources, semantic_type, True, None

    def _timestamps_from_metadata(
        self,
        metadata: HEVCBitcostMetadata,
        indices: Sequence[int],
        source_fps: float,
    ) -> list[float]:
        by_id = {frame.display_frame_id: frame.pts_time for frame in metadata.frames}
        out = []
        for idx in indices:
            if int(idx) in by_id:
                out.append(float(by_id[int(idx)]))
            elif source_fps > 0:
                out.append(float(idx) / float(source_fps))
            else:
                out.append(float(idx))
        return out

    def _frame_value(
        self,
        frame: HEVCFrameBitcost,
        *,
        p_median: float = 0.0,
        p_mad: float = 0.0,
    ) -> float:
        if self.normalization == "raw_bits":
            return max(float(frame.bitcost_bits), self.eps)
        if self.normalization == "bpp":
            return max(float(frame.bitcost_bpp), self.eps)
        if self.normalization == "log_bpp":
            return max(float(frame.log_bitcost), self.eps)
        if frame.pict_type == "I" and self.i_policy == "clip_as_pb":
            if p_mad < self.eps:
                z = 0.0
            else:
                z = (float(frame.log_bitcost) - float(p_median)) / (float(p_mad) + self.eps)
                z = max(-self.z_clip, min(self.z_clip, z))
            return max(float(_softplus(z) + self.eps), self.eps)
        return max(float(frame.positive_bitcost), self.eps)

    def _aggregate(self, mean: float, p90: float, max_v: float, sum_v: float) -> float:
        if self.aggregation == "mean":
            return mean
        if self.aggregation == "p90":
            return p90
        if self.aggregation == "max":
            return max_v
        if self.aggregation == "sum":
            return sum_v
        return self.mean_weight * mean + self.p90_weight * p90


def summarize_metadata(metadata: HEVCBitcostMetadata) -> dict[str, Any]:
    counts = {typ: sum(1 for f in metadata.frames if f.pict_type == typ) for typ in ("I", "P", "B", "?")}
    out: dict[str, Any] = {
        "video_path": metadata.video_path,
        "codec_name": metadata.codec_name,
        "width": metadata.width,
        "height": metadata.height,
        "avg_frame_rate": metadata.avg_frame_rate,
        "duration": metadata.duration,
        "total_frames": metadata.total_frames,
        "frame_counts": counts,
        "top20_bitcost_frame_ids": [
            f.display_frame_id
            for f in heapq.nlargest(20, metadata.frames, key=lambda x: x.bitcost_bits)
        ],
    }
    for typ in ("I", "P", "B"):
        vals = [f.bitcost_bits for f in metadata.frames if f.pict_type == typ]
        out[f"{typ}_mean_bitcost_bits"] = sum(vals) / len(vals) if vals else None
        out[f"{typ}_median_bitcost_bits"] = _median(vals) if vals else None
    return out
