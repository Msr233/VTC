from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

from utils.hevc_bitcost import (
    SCHEMA_VERSION,
    _metadata_to_json,
    metadata_from_json,
)


def _frame_signature(metadata):
    return [
        (
            float(frame.pts_time),
            str(frame.pict_type),
            int(frame.packet_size_bytes),
        )
        for frame in metadata.frames
    ]


def migrate_file(path: Path) -> tuple[int, int, bool]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        source = json.load(stream)
    if int(source.get("schema_version") or 1) == SCHEMA_VERSION:
        size = path.stat().st_size
        return size, size, False

    before = metadata_from_json(source)
    migrated = _metadata_to_json(before, source.get("cache_identity"))
    after = metadata_from_json(migrated)
    if _frame_signature(before) != _frame_signature(after):
        raise ValueError(f"round-trip frame mismatch: {path}")

    encoded = json.dumps(
        migrated,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
    old_size = path.stat().st_size
    fd, temporary = tempfile.mkstemp(
        prefix=path.name,
        suffix=".migrate.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return old_size, len(compressed), True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically migrate codec cache files to compact schema v3."
    )
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.cache_dir.glob("*.json.gz"))
    old_total = 0
    new_total = 0
    migrated_count = 0
    for index, path in enumerate(paths, start=1):
        old_size, new_size, migrated = migrate_file(path)
        old_total += old_size
        new_total += new_size
        migrated_count += int(migrated)
        if index % 100 == 0 or index == len(paths):
            print(
                f"[{index}/{len(paths)}] migrated={migrated_count} "
                f"size={new_total / 1024 / 1024:.2f} MiB",
                flush=True,
            )
    print(
        f"complete: migrated={migrated_count}, "
        f"before={old_total / 1024 / 1024:.2f} MiB, "
        f"after={new_total / 1024 / 1024:.2f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
