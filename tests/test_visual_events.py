from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from text_localizer.domain import NormalizedBox
from text_localizer.profile import VisualEventOptions
from text_localizer.video import TimestampedFrame, VideoInfo
from text_localizer.visual_events import index_visual_events


def _frame(path: Path, rectangle: tuple[int, int, int, int] | None = None) -> Path:
    image = Image.new("L", (120, 80), 0)
    if rectangle is not None:
        ImageDraw.Draw(image).rectangle(rectangle, fill=255)
    image.save(path)
    return path


def _video_info(path: Path, duration: float) -> VideoInfo:
    return VideoInfo(
        path=str(path),
        width=120,
        height=80,
        fps=30.0,
        duration_seconds=duration,
        codec="h264",
    )


def test_visual_events_separate_partial_and_global_changes(tmp_path: Path) -> None:
    paths = [
        _frame(tmp_path / "frame-0.png"),
        _frame(tmp_path / "frame-1.png", (0, 0, 25, 25)),
        _frame(tmp_path / "frame-2.png", (0, 0, 25, 25)),
        _frame(tmp_path / "frame-3.png", (0, 0, 119, 79)),
    ]
    frames = tuple(
        TimestampedFrame(index, index * 0.2, str(path)) for index, path in enumerate(paths)
    )
    options = VisualEventOptions(
        analysis_width=120,
        tile_columns=4,
        tile_rows=2,
        tile_change_threshold=0.05,
        minimum_changed_tiles=1,
        global_change_ratio=0.5,
    )

    payload = index_visual_events(
        frames,
        _video_info(tmp_path / "video.mp4", 0.8),
        profile_name="synthetic",
        roi=None,
        options=options,
    )

    assert payload["event_count"] == 2
    assert [item["classification"] for item in payload["events"]] == [
        "partial",
        "global",
    ]
    assert payload["events"][0]["start_seconds"] == 0.0
    assert payload["events"][0]["end_seconds"] == 0.2
    assert payload["events"][1]["start_seconds"] == 0.4
    assert payload["events"][1]["end_seconds"] == pytest.approx(0.6)
    assert payload["metrics"]["changed_interval_count"] == 2
    assert payload["metrics"]["stable_interval_count"] == 1


def test_visual_event_exclusion_masks_changes(tmp_path: Path) -> None:
    first = _frame(tmp_path / "frame-0.png")
    second = _frame(tmp_path / "frame-1.png", (0, 0, 59, 39))
    frames = (
        TimestampedFrame(0, 0.0, str(first)),
        TimestampedFrame(1, 0.2, str(second)),
    )
    options = VisualEventOptions(
        analysis_width=120,
        tile_columns=4,
        tile_rows=2,
        tile_change_threshold=0.01,
        exclude_regions=(NormalizedBox(0.0, 0.0, 0.5, 0.5),),
    )

    payload = index_visual_events(
        frames,
        _video_info(tmp_path / "video.mp4", 0.4),
        profile_name=None,
        roi=None,
        options=options,
    )

    assert payload["event_count"] == 0
    assert payload["metrics"]["stable_interval_count"] == 1


def test_visual_event_roi_ignores_outside_changes(tmp_path: Path) -> None:
    first = _frame(tmp_path / "frame-0.png")
    second = _frame(tmp_path / "frame-1.png", (0, 0, 49, 79))
    frames = (
        TimestampedFrame(0, 0.0, str(first)),
        TimestampedFrame(1, 0.2, str(second)),
    )

    payload = index_visual_events(
        frames,
        _video_info(tmp_path / "video.mp4", 0.4),
        profile_name=None,
        roi=NormalizedBox(0.5, 0.0, 0.5, 1.0),
        options=VisualEventOptions(
            analysis_width=120,
            tile_columns=4,
            tile_rows=2,
            tile_change_threshold=0.01,
        ),
    )

    assert payload["event_count"] == 0
    assert payload["analysis"]["roi"] == {
        "x": 0.5,
        "y": 0.0,
        "width": 0.5,
        "height": 1.0,
    }


def test_visual_event_timestamps_must_be_strictly_increasing(tmp_path: Path) -> None:
    first = _frame(tmp_path / "frame-0.png")
    second = _frame(tmp_path / "frame-1.png")
    frames = (
        TimestampedFrame(0, 0.2, str(first)),
        TimestampedFrame(1, 0.2, str(second)),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        index_visual_events(
            frames,
            _video_info(tmp_path / "video.mp4", 0.4),
            profile_name=None,
            roi=None,
            options=VisualEventOptions(),
        )


def test_visual_event_options_reject_invalid_grid() -> None:
    with pytest.raises(ValueError, match="tile dimensions"):
        VisualEventOptions(tile_columns=1)
