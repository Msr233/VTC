from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_vtc_token_stats(
    stats: Dict[str, Any],
    fallback_video_id: Optional[str] = None,
) -> Dict[str, Any]:
    frame_info = stats.get("frame_info") or []
    i_frames = [f for f in frame_info if f.get("type") == "I"]
    p_frames = [f for f in frame_info if f.get("type") == "P"]

    num_i_frames = _as_int(stats.get("num_i_frames"))
    if num_i_frames is None:
        num_i_frames = len(i_frames)

    num_p_frames = _as_int(stats.get("num_p_frames"))
    if num_p_frames is None:
        num_p_frames = len(p_frames)

    i_tokens_per_frame = _as_float(stats.get("i_frame_tokens_per_frame"))
    if i_tokens_per_frame is None:
        i_tokens_per_frame = _mean(f.get("kept") for f in i_frames)
    if i_tokens_per_frame is None and num_i_frames:
        i_original = _as_int(stats.get("i_frame_original_tokens"))
        if i_original is not None:
            i_tokens_per_frame = i_original / num_i_frames

    p_tokens_per_frame = _as_float(stats.get("p_frame_tokens_per_frame"))
    if p_tokens_per_frame is None:
        p_tokens_per_frame = _mean(f.get("kept") for f in p_frames)
    if p_tokens_per_frame is None and num_p_frames:
        p_kept = _as_int(stats.get("p_frame_kept_tokens"))
        if p_kept is not None:
            p_tokens_per_frame = p_kept / num_p_frames

    original_tokens = _as_int(
        stats.get("original_tokens", stats.get("original_visual_tokens"))
    )
    kept_tokens = _as_int(
        stats.get("kept_tokens", stats.get("kept_visual_tokens"))
    )
    overall_keep = _as_float(
        stats.get("overall_keep_ratio", stats.get("actual_keep_ratio"))
    )

    return {
        "video_id": stats.get("video_id") or fallback_video_id,
        "num_i_frames": num_i_frames,
        "i_frame_tokens_per_frame": (
            round(i_tokens_per_frame, 4)
            if i_tokens_per_frame is not None
            else None
        ),
        "num_p_frames": num_p_frames,
        "p_frame_tokens_per_frame": (
            round(p_tokens_per_frame, 4)
            if p_tokens_per_frame is not None
            else None
        ),
        "original_visual_tokens": original_tokens,
        "kept_visual_tokens": kept_tokens,
        "overall_keep": round(overall_keep, 6)
        if overall_keep is not None
        else None,
    }


def latest_vtc_token_stats(
    model: Any,
    fallback_video_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    stats_list = getattr(model, "_vtc_last_stats", None)
    if not stats_list:
        return None
    return summarize_vtc_token_stats(stats_list[0], fallback_video_id)


def print_vtc_token_report(records: List[Dict[str, Any]]) -> None:
    rows_by_video: Dict[str, Dict[str, Any]] = {}
    for record in records:
        row = record.get("vtc_token_stats")
        if not row:
            continue
        video_id = row.get("video_id") or record.get("video_id")
        if video_id is None:
            continue
        rows_by_video.setdefault(str(video_id), row)

    if not rows_by_video:
        return

    rows = [rows_by_video[key] for key in sorted(rows_by_video)]
    print("")
    print("=" * 120)
    print(f"{'Visual Token Coding Report':^120}")
    print("=" * 120)
    print(
        "Video ID\tI-Frames\tI Tokens/Frame\tP-Frames\tP Tokens/Frame\t"
        "Original\tKept\tOverall Keep"
    )
    print("-" * 120)

    keep_ratios = []
    kept_values = []
    original_total = 0
    kept_total = 0
    for row in rows:
        original = row.get("original_visual_tokens")
        kept = row.get("kept_visual_tokens")
        keep = row.get("overall_keep")
        if keep is not None:
            keep_ratios.append(float(keep))
        if kept is not None:
            kept_values.append(float(kept))
            kept_total += int(kept)
        if original is not None:
            original_total += int(original)

        keep_text = f"{float(keep) * 100:.2f}%" if keep is not None else "NA"
        print(
            f"{row.get('video_id')}\t"
            f"{row.get('num_i_frames')}\t"
            f"{row.get('i_frame_tokens_per_frame')}\t"
            f"{row.get('num_p_frames')}\t"
            f"{row.get('p_frame_tokens_per_frame')}\t"
            f"{original}\t"
            f"{kept}\t"
            f"{keep_text}"
        )

    mean_keep = _mean(keep_ratios)
    avg_kept = _mean(kept_values)
    weighted_keep = kept_total / original_total if original_total else None

    print("-" * 120)
    if mean_keep is not None:
        print(f"Dataset Overall Keep (mean of videos): {mean_keep * 100:.2f}%")
    if weighted_keep is not None:
        print(f"Dataset Overall Keep (token-weighted): {weighted_keep * 100:.2f}%")
    if avg_kept is not None:
        print(f"Average Kept Tokens per Video: {avg_kept:.2f}")
    print("=" * 120)
