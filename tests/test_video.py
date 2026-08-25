import json
import os
from pathlib import Path

import pytest

from text_localizer.video import (
    _load_matching_sample_manifest,
    evenly_spaced_timestamps,
    interval_timestamps,
    parse_frame_rate,
)


def test_parse_fractional_frame_rate() -> None:
    assert parse_frame_rate("30000/1001") == pytest.approx(29.97002997)


def test_evenly_spaced_timestamps_are_bounded() -> None:
    timestamps = evenly_spaced_timestamps(10.0, 4)
    assert len(timestamps) == 4
    assert timestamps == sorted(timestamps)
    assert 0.0 <= timestamps[0] < timestamps[-1] <= 10.0


def test_timestamp_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        evenly_spaced_timestamps(10.0, 0)


def test_interval_timestamps_cover_video_without_exceeding_duration() -> None:
    assert interval_timestamps(3.2, 1.0) == [0.0, 1.0, 2.0, 3.0]
    assert interval_timestamps(0.2, 1.0) == [0.0]


def test_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        interval_timestamps(10.0, 0.0)


def test_sample_manifest_is_invalidated_when_video_timestamp_changes(tmp_path: Path) -> None:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    frame_path = tmp_path / "frame-000000.png"
    frame_path.write_bytes(b"frame")
    modified_ns = video_path.stat().st_mtime_ns
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_path": str(video_path.resolve()),
                "video_size_bytes": video_path.stat().st_size,
                "video_modified_ns": modified_ns,
                "interval_seconds": 1.0,
                "frame_count": 1,
                "frames": [{"index": 0, "timestamp_seconds": 0.0, "path": str(frame_path)}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        _load_matching_sample_manifest(manifest_path, video_path=video_path, interval_seconds=1.0)
        is not None
    )

    os.utime(video_path, ns=(modified_ns + 1_000_000, modified_ns + 1_000_000))

    assert (
        _load_matching_sample_manifest(manifest_path, video_path=video_path, interval_seconds=1.0)
        is None
    )
