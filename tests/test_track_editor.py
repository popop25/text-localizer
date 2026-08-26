from pathlib import Path
from typing import Any

import pytest

from text_localizer.track_editor import validate_track_manifest, write_track_editor


def _manifest() -> dict[str, Any]:
    region = {
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.1},
        "source_text": "LangGraph 기초",
        "translated_text": "LangGraph 基礎",
        "ocr_confidence": 0.95,
        "action": "translate",
        "translation_quality_flags": ["manual_review"],
        "needs_review": True,
    }
    return {
        "schema_version": 1,
        "video": {
            "path": "lecture.mp4",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration_seconds": 10.0,
            "codec": "h264",
        },
        "profile_name": "test",
        "sampling": {"interval_seconds": 1.0, "transition_precision_seconds": 1.0},
        "track_count": 2,
        "tracks": [
            {
                "track_id": "scene-0001",
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "representative_frame": "frame-1.png",
                "analysis_path": None,
                "regions": [region],
            },
            {
                "track_id": "scene-0002",
                "scene_id": 2,
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "representative_frame": "frame-2.png",
                "analysis_path": None,
                "regions": [],
            },
        ],
        "metrics": {"needs_review_count": 1},
    }


def test_track_editor_contains_edit_review_time_and_export_controls(tmp_path: Path) -> None:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "editor.html"

    write_track_editor(output_path, video_path, _manifest())

    html = output_path.read_text(encoding="utf-8")
    assert "Export edited JSON" in html
    assert "Approved for overlay" in html
    assert "Start seconds" in html
    assert "tracks.edited.json" in html
    assert "human_edited" in html
    assert video_path.resolve().as_uri() in html
    assert "requestVideoFrameCallback" in html
    assert "renderPreview(metadata.mediaTime)" in html
    assert "requestAnimationFrame" in html


def test_track_manifest_rejects_non_contiguous_ranges() -> None:
    manifest = _manifest()
    manifest["tracks"][1]["start_seconds"] = 6.0

    with pytest.raises(ValueError, match="not contiguous"):
        validate_track_manifest(manifest)


def test_track_manifest_must_cover_full_video_duration() -> None:
    manifest = _manifest()
    manifest["tracks"][0]["start_seconds"] = 0.5

    with pytest.raises(ValueError, match="start at zero"):
        validate_track_manifest(manifest)


def test_track_manifest_review_metric_must_match_regions() -> None:
    manifest = _manifest()
    manifest["metrics"]["needs_review_count"] = 0

    with pytest.raises(ValueError, match="does not match"):
        validate_track_manifest(manifest)


def test_track_manifest_rejects_non_finite_times() -> None:
    manifest = _manifest()
    manifest["tracks"][0]["end_seconds"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        validate_track_manifest(manifest)


def test_track_editor_escapes_script_terminators(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["tracks"][0]["regions"][0]["translated_text"] = "</script><script>bad()</script>"
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "editor.html"

    write_track_editor(output_path, video_path, manifest)

    html = output_path.read_text(encoding="utf-8")
    assert "</script><script>bad()" not in html
    assert "\\u003c/script>" in html
