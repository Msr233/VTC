from utils.hevc_bitcost import (
    SCHEMA_VERSION,
    _metadata_to_json,
    build_metadata_from_ffprobe_json,
    metadata_from_json,
)


def _metadata():
    return build_metadata_from_ffprobe_json(
        "/tmp/example.mp4",
        {
            "streams": [
                {
                    "codec_name": "hevc",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "duration": "0.1001",
                }
            ],
            "frames": [
                {
                    "best_effort_timestamp_time": "0.000000",
                    "pkt_duration_time": "0.033367",
                    "pict_type": "I",
                    "pkt_size": "1000",
                },
                {
                    "best_effort_timestamp_time": "0.033367",
                    "pkt_duration_time": "0.033366",
                    "pict_type": "B",
                    "pkt_size": "250",
                },
                {
                    "best_effort_timestamp_time": "0.066733",
                    "pkt_duration_time": "0.033367",
                    "pict_type": "P",
                    "pkt_size": "500",
                },
            ],
        },
    )


def test_schema_v3_round_trip_uses_pts_deltas():
    source = _metadata()
    packed = _metadata_to_json(
        source,
        {"path": "/tmp/example.mp4", "size": 1, "mtime_ns": 2},
    )
    assert packed["schema_version"] == SCHEMA_VERSION
    assert packed["frames"]["pts_us_delta"] == [0, 33367, 33366]
    assert "pts" not in packed["frames"]
    assert "timeline_start" not in packed

    restored = metadata_from_json(packed)
    assert [frame.pts_time for frame in restored.frames] == [
        frame.pts_time for frame in source.frames
    ]
    assert [frame.pict_type for frame in restored.frames] == ["I", "B", "P"]
    assert [frame.packet_size_bytes for frame in restored.frames] == [1000, 250, 500]


def test_schema_v2_remains_readable():
    restored = metadata_from_json(
        {
            "schema_version": 2,
            "cache_identity": {"path": "/tmp/legacy.mp4"},
            "codec_name": "hevc",
            "width": 16,
            "height": 16,
            "avg_frame_rate": 25.0,
            "duration": 0.08,
            "timeline_end": 0.08,
            "frames": {
                "pts": [0.0, 0.04],
                "types": "IP",
                "packet_sizes": [100, 50],
            },
        }
    )
    assert [frame.pts_time for frame in restored.frames] == [0.0, 0.04]
    assert [frame.packet_size_bytes for frame in restored.frames] == [100, 50]
