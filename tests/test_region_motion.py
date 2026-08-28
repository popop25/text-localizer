from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from text_localizer.region_motion import RegionMotionOptions, refine_region_motion


def _manifest(video_path: Path, *, end_y: float = 0.2, source: str = "scene") -> dict[str, Any]:
    bbox = {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.05}
    region = {
        "bbox": bbox,
        "source_text": "강의소개",
        "translated_text": "講義紹介",
        "action": "translate",
        "needs_review": False,
        "translation_quality_flags": [],
    }
    return {
        "schema_version": 2,
        "video": {
            "path": str(video_path.resolve()),
            "width": 1000,
            "height": 500,
            "fps": 30.0,
            "duration_seconds": 20.0,
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "regions": [region],
            }
        ],
        "metrics": {"needs_review_count": 0},
        "region_track_count": 1,
        "region_tracks": [
            {
                "region_track_id": "region-0001",
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "region": region,
                "keyframes": [
                    {
                        "timestamp_seconds": 0.0,
                        "bbox": bbox,
                        "interpolation": "hold",
                        "source": source,
                    },
                    {
                        "timestamp_seconds": 20.0,
                        "bbox": {"x": 0.2, "y": end_y, "width": 0.2, "height": 0.05},
                        "interpolation": "hold",
                        "source": source,
                    },
                ],
            }
        ],
        "region_tracking": {"algorithm": "event-ocr-keyframes-v2"},
    }


def _write_manifest(tmp_path: Path, **kwargs: Any) -> Path:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "tracks.json"
    manifest_path.write_text(json.dumps(_manifest(video_path, **kwargs)), encoding="utf-8")
    return manifest_path


def test_refine_region_motion_marks_slow_scene_drift_linear(tmp_path: Path) -> None:
    result = refine_region_motion(_write_manifest(tmp_path), tmp_path / "output")

    first = result["region_tracks"][0]["keyframes"][0]
    assert first["interpolation"] == "linear"
    assert result["region_motion"]["algorithm"] == "scene-anchor-linear-v1"
    assert result["region_motion"]["refined_segment_count"] == 1
    assert (tmp_path / "output" / "motion-quality.json").is_file()
    assert "region_motion" in (tmp_path / "output" / "viewer.html").read_text(encoding="utf-8")


def test_refine_region_motion_leaves_fast_jump_held(tmp_path: Path) -> None:
    result = refine_region_motion(
        _write_manifest(tmp_path),
        tmp_path / "output",
        options=RegionMotionOptions(maximum_vertical_speed_ratio_per_second=0.001),
    )

    assert result["region_tracks"][0]["keyframes"][0]["interpolation"] == "hold"
    assert result["region_motion"]["refined_segment_count"] == 0


def test_refine_region_motion_does_not_override_visual_event_segments(tmp_path: Path) -> None:
    result = refine_region_motion(
        _write_manifest(tmp_path, source="visual_event"), tmp_path / "output"
    )

    assert result["region_tracks"][0]["keyframes"][0]["interpolation"] == "hold"
    assert result["region_motion"]["refined_segment_count"] == 0
