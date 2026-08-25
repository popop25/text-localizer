import json
import os
import subprocess
from pathlib import Path

import pytest

from text_localizer import video as video_module
from text_localizer.video import (
    VideoInfo,
    _load_matching_sample_manifest,
    evenly_spaced_timestamps,
    extract_timestamped_frames,
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


def test_timestamped_extraction_uses_round_up_and_exact_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "frames"
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> object:
        assert check is True
        commands.append(command)
        pattern = command[-1]
        for index in range(4):
            Path(pattern.replace("%06d", f"{index:06d}")).write_bytes(b"frame")
        return object()

    monkeypatch.setattr(
        video_module,
        "resolve_executable",
        lambda *_args, **_kwargs: Path("ffmpeg.exe"),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    info = VideoInfo(
        str(video_path.resolve()),
        1920,
        1080,
        30.0,
        3.2,
        "h264",
    )

    frames, _, reused = extract_timestamped_frames(
        video_path,
        output_dir,
        1.0,
        video_info=info,
    )
    resumed, _, resumed_reused = extract_timestamped_frames(
        video_path,
        output_dir,
        1.0,
        video_info=info,
    )

    assert reused is False
    assert resumed_reused is True
    assert [frame.timestamp_seconds for frame in frames] == [0.0, 1.0, 2.0, 3.0]
    assert resumed == frames
    assert len(commands) == 1
    assert any("round=up" in argument for argument in commands[0])
    payload = json.loads((output_dir / "samples.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["sampling_policy"] == "fps-round-up-v1"


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
                "schema_version": 2,
                "sampling_policy": "fps-round-up-v1",
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
        _load_matching_sample_manifest(
            manifest_path, video_path=video_path, interval_seconds=1.0, expected_timestamps=[0.0]
        )
        is not None
    )

    os.utime(video_path, ns=(modified_ns + 1_000_000, modified_ns + 1_000_000))

    assert (
        _load_matching_sample_manifest(
            manifest_path, video_path=video_path, interval_seconds=1.0, expected_timestamps=[0.0]
        )
        is None
    )
