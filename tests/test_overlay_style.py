from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

import text_localizer.overlay_style as overlay_style
from text_localizer.overlay_style import (
    OverlayStyleOptions,
    estimate_overlay_style,
    style_region_tracks,
)
from text_localizer.track_editor import validate_track_manifest


def _manifest(video_path: Path) -> dict[str, Any]:
    bbox = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2}
    region = {
        "bbox": bbox,
        "source_text": "테스트",
        "translated_text": "テスト",
        "ocr_confidence": 0.95,
        "action": "translate",
        "translation_quality_flags": [],
        "needs_review": False,
    }
    return {
        "schema_version": 2,
        "video": {
            "path": str(video_path.resolve()),
            "width": 200,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": 2.0,
            "codec": "h264",
        },
        "sampling": {"interval_seconds": 1.0, "transition_precision_seconds": 0.1},
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "regions": [region],
            }
        ],
        "metrics": {"needs_review_count": 0},
        "region_track_count": 1,
        "region_tracks": [
            {
                "region_track_id": "region-0001",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "region": region,
                "keyframes": [
                    {
                        "timestamp_seconds": 0.0,
                        "bbox": bbox,
                        "interpolation": "hold",
                        "source_text": "테스트",
                        "ocr_confidence": 0.9,
                    },
                    {
                        "timestamp_seconds": 1.0,
                        "bbox": bbox,
                        "interpolation": "hold",
                        "source_text": "테스트",
                        "ocr_confidence": 0.99,
                        "phase": "post",
                    },
                ],
            }
        ],
        "region_tracking": {"algorithm": "event-ocr-keyframes-v2"},
    }


def _source_frame() -> Image.Image:
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 34, 112, 37), fill=(20, 30, 40))
    draw.rectangle((55, 42, 108, 45), fill=(20, 30, 40))
    return image


def test_estimate_overlay_style_matches_uniform_source_palette() -> None:
    style = estimate_overlay_style(
        _source_frame(),
        {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2},
    )

    assert style["background_color"] == "#ffffff"
    assert style["foreground_color"] == "#141e28"
    assert style["background_opacity"] == 1.0
    assert style["fallback_used"] is False
    assert style["fallback_reason"] is None
    assert style["contrast_ratio"] > 10.0
    assert 0.11 <= style["font_size_ratio"] <= 0.19


def test_estimate_overlay_style_falls_back_without_foreground() -> None:
    image = Image.new("RGB", (100, 100), (230, 230, 230))

    style = estimate_overlay_style(
        image,
        {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.1},
    )

    assert style["fallback_used"] is True
    assert style["fallback_reason"] == "no-distinct-foreground"
    assert style["background_color"] == "#e6e6e6"
    assert style["foreground_color"] == "#000000"
    assert style["background_opacity"] == 1.0


def test_estimate_overlay_style_ignores_tall_decorative_icon() -> None:
    image = Image.new("RGB", (300, 100), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((34, 31, 40, 58), fill=(220, 220, 220))
    draw.rectangle((50, 39, 245, 52), fill=(220, 220, 220))

    style = estimate_overlay_style(
        image,
        {"x": 0.1, "y": 0.3, "width": 0.75, "height": 0.3},
    )

    assert style["algorithm"] == "source-palette-v2"
    assert style["glyph_height_method"] == "column-span-p80-v1"
    assert 0.15 <= style["font_size_ratio"] <= 0.2


def test_style_region_tracks_writes_valid_manifest_and_adaptive_viewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "tracks.json"
    manifest_path.write_text(
        json.dumps(_manifest(video_path), ensure_ascii=False),
        encoding="utf-8",
    )
    source_frame = _source_frame()

    def fake_extract(
        _video_path: Path,
        output_path: Path,
        timestamp_seconds: float,
    ) -> tuple[Path, bool]:
        assert timestamp_seconds == 1.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_frame.save(output_path)
        return output_path.resolve(), False

    monkeypatch.setattr(overlay_style, "extract_frame_at_timestamp", fake_extract)
    output_dir = tmp_path / "output"
    result = style_region_tracks(manifest_path, output_dir, tmp_path / "work")

    validate_track_manifest(result)
    style = result["region_tracks"][0]["overlay_style"]
    assert style["sample_timestamp_seconds"] == 1.0
    assert style["fallback_used"] is False
    assert result["region_styling"]["styled_region_track_count"] == 1
    assert result["region_styling"]["extracted_frame_count"] == 1
    assert result["region_styling"]["video_size_bytes"] == video_path.stat().st_size
    assert (output_dir / "tracks.json").is_file()
    viewer = (output_dir / "viewer.html").read_text(encoding="utf-8")
    assert "style.background_color" in viewer
    assert "style.font_size_ratio" in viewer
    assert "fitOverlayText(node," in viewer
    assert "source-palette-v2" in viewer


def test_overlay_style_options_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="contrast ratio"):
        OverlayStyleOptions(minimum_contrast_ratio=0.5)


def test_track_manifest_rejects_invalid_overlay_color(tmp_path: Path) -> None:
    video_path = tmp_path / "lecture.mp4"
    manifest = _manifest(video_path)
    style = estimate_overlay_style(
        _source_frame(),
        {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2},
    )
    style["sample_timestamp_seconds"] = 1.0
    style["background_color"] = "white"
    manifest["region_tracks"][0]["overlay_style"] = style

    with pytest.raises(ValueError, match="background_color"):
        validate_track_manifest(manifest)
