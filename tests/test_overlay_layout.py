from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from text_localizer.overlay_layout import (
    OverlayLayoutOptions,
    layout_region_tracks,
)
from text_localizer.rendering import find_japanese_font
from text_localizer.track_editor import validate_track_manifest


def _font_path() -> Path:
    return find_japanese_font()


def _style() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "source-palette-v1",
        "foreground_color": "#eeeeee",
        "background_color": "#202020",
        "background_opacity": 1.0,
        "font_size_ratio": 0.06,
        "font_weight": 500,
        "line_height": 1.15,
        "border_radius_px": 2.0,
        "text_shadow": "0 1px 1px #000000",
        "confidence": 0.9,
        "background_uniformity": 0.95,
        "foreground_pixel_ratio": 0.1,
        "contrast_ratio": 12.0,
        "foreground_adjusted": False,
        "fallback_used": False,
        "fallback_reason": None,
        "sample_timestamp_seconds": 0.0,
    }


def _manifest(video_path: Path, translated_text: str) -> dict[str, Any]:
    bbox = {"x": 0.1, "y": 0.2, "width": 0.18, "height": 0.08}
    region = {
        "action": "translate",
        "source_text": "기업용 AI 서비스 개발",
        "translated_text": translated_text,
        "ocr_confidence": 0.95,
        "translation_quality_flags": [],
        "needs_review": False,
        "bbox": bbox,
    }
    return {
        "schema_version": 2,
        "video": {
            "path": str(video_path.resolve()),
            "width": 800,
            "height": 400,
            "fps": 30.0,
            "duration_seconds": 2.0,
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
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
                    }
                ],
                "overlay_style": _style(),
            }
        ],
        "region_tracking": {"algorithm": "event-ocr-keyframes-v2"},
        "region_styling": {"algorithm": "source-palette-v1"},
    }


def _write_manifest(tmp_path: Path, translated_text: str) -> Path:
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "tracks.json"
    manifest_path.write_text(
        json.dumps(_manifest(video_path, translated_text), ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def test_layout_regions_preserves_source_size_and_eliminates_clipping(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, "企業向けAI開発")
    output_dir = tmp_path / "output"

    result = layout_region_tracks(
        manifest_path,
        output_dir,
        font_path=_font_path(),
    )

    validate_track_manifest(result)
    layout = result["region_tracks"][0]["overlay_layout"]
    replacement = result["region_tracks"][0]["overlay_replacement"]
    assert layout["content_fits"] is True
    assert layout["layout_needs_review"] is False
    assert layout["font_scale"] >= 0.95
    assert result["region_layout"]["baseline_source_box_overflow_count"] == 1
    assert result["region_layout"]["clipped_region_count"] == 0
    assert result["region_layout"]["gates_passed"] is True
    assert replacement["algorithm"] == "padded-source-box-v1"
    assert replacement["composite_order"] == ["erase-source", "draw-translation"]
    assert replacement["erase_padding_top_ratio"] > 0
    assert result["region_replacement"]["replacement_region_count"] == 1
    assert (output_dir / "layout-quality.json").is_file()
    viewer = (output_dir / "viewer.html").read_text(encoding="utf-8")
    assert "adaptive-fit-v1" in viewer
    assert "applyOverlayLayout(node, layout, bbox)" in viewer
    assert "appendSourceEraser(item)" in viewer
    assert 'node.style.backgroundColor = "transparent"' in viewer
    assert 'layout.lines.join("\\n")' in viewer
    assert 'layout.lines.join("\n")' not in viewer
    assert "binary-shrink" in viewer


def test_layout_regions_flags_content_that_requires_emergency_fit(tmp_path: Path) -> None:
    text = "非常に長い日本語の説明文" * 12
    manifest_path = _write_manifest(tmp_path, text)

    result = layout_region_tracks(
        manifest_path,
        tmp_path / "output",
        options=OverlayLayoutOptions(
            maximum_width_scale=1.0,
            maximum_frame_width_ratio=0.2,
            maximum_lines=1,
            minimum_font_scale=0.8,
        ),
        font_path=_font_path(),
    )

    layout = result["region_tracks"][0]["overlay_layout"]
    assert layout["layout_needs_review"] is True
    assert result["region_layout"]["layout_review_count"] == 1
    assert result["region_layout"]["gates_passed"] is False


def test_track_validation_rejects_stale_layout_text(tmp_path: Path) -> None:
    result = layout_region_tracks(
        _write_manifest(tmp_path, "企業向けAI開発"),
        tmp_path / "output",
        font_path=_font_path(),
    )
    result["region_tracks"][0]["region"]["translated_text"] = "変更された翻訳"

    with pytest.raises(ValueError, match="stale"):
        validate_track_manifest(result)


@pytest.mark.parametrize(
    "options",
    [
        OverlayLayoutOptions(maximum_width_scale=1.0),
        OverlayLayoutOptions(maximum_lines=1),
        OverlayLayoutOptions(minimum_font_scale=0.5),
    ],
)
def test_layout_options_accept_supported_boundaries(options: OverlayLayoutOptions) -> None:
    assert isinstance(options, OverlayLayoutOptions)


def test_layout_options_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="width scale"):
        OverlayLayoutOptions(maximum_width_scale=0.9)
