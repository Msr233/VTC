"""Compact per-question visual token statistics."""


def latest_vtc_token_stats(model, fallback_video_id=None):
    records = getattr(model, "_vtc_last_stats", None)
    if not records:
        return None
    stats = records[0]
    return {
        "video_id": stats["video_id"] or fallback_video_id,
        "num_i_frames": stats["num_i_frames"],
        "num_p_frames": stats["num_p_frames"],
        "i_frame_tokens_per_frame": stats["i_frame_tokens_per_frame"],
        "p_frame_tokens_per_frame": stats["p_frame_tokens_per_frame"],
        "original_visual_tokens": stats["original_tokens"],
        "kept_visual_tokens": stats["kept_tokens"],
        "overall_keep": stats["overall_keep_ratio"],
        "codec_runtime_cache_hit": stats["codec_runtime_cache_hit"],
        "codec_alignment_valid": stats["codec_alignment_valid"],
        "codec_fallback_reason": stats["codec_fallback_reason"],
    }
