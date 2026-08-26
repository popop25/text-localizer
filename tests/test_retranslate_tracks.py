import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from text_localizer.glossary import Glossary, GlossaryEntry
from text_localizer.retranslate_tracks import retranslate_region_tracks


class _FakeTranslationProvider:
    name = "fake-translator"

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        return tuple(value.replace("개발", "開発") for value in source_texts)


def _manifest() -> dict[str, Any]:
    bbox = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}
    region = {
        "action": "translate",
        "source_text": "Agent 개발",
        "translation_source_text": "Agent 개발",
        "translated_text": "Agent 実施",
        "needs_review": False,
        "translation_quality_flags": [],
        "bbox": bbox,
    }
    return {
        "schema_version": 2,
        "video": {
            "path": "lecture.mp4",
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": 1.0,
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "regions": [region],
            }
        ],
        "region_track_count": 1,
        "region_tracks": [
            {
                "region_track_id": "region-0001",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "region": region,
                "keyframes": [
                    {
                        "timestamp_seconds": 0.0,
                        "bbox": bbox,
                        "interpolation": "hold",
                    }
                ],
                "overlay_style": {
                    "schema_version": 1,
                    "algorithm": "source-palette-v1",
                    "foreground_color": "#ffffff",
                    "background_color": "#000000",
                    "background_opacity": 1.0,
                    "font_size_ratio": 0.03,
                    "font_weight": 500,
                    "line_height": 1.15,
                    "border_radius_px": 2.0,
                    "text_shadow": "0 1px 1px #000000",
                    "confidence": 0.9,
                    "background_uniformity": 1.0,
                    "foreground_pixel_ratio": 0.1,
                    "contrast_ratio": 21.0,
                    "foreground_adjusted": False,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "sample_timestamp_seconds": 0.0,
                },
            }
        ],
        "metrics": {"needs_review_count": 0},
    }


def test_retranslate_tracks_preserves_timeline_and_style(tmp_path: Path) -> None:
    source_path = tmp_path / "input" / "tracks.json"
    output_dir = tmp_path / "output"
    source_path.parent.mkdir()
    source_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    glossary = Glossary((GlossaryEntry(source="Agent", target="Agent"),))

    report = retranslate_region_tracks(
        source_path,
        output_dir,
        _FakeTranslationProvider(),
        glossary,
        batch_size=2,
    )

    saved = json.loads((output_dir / "tracks.json").read_text(encoding="utf-8"))
    assert saved["region_tracks"][0]["region"]["translated_text"] == "Agent 開発"
    assert saved["region_tracks"][0]["overlay_style"]["foreground_color"] == "#ffffff"
    assert saved["region_tracks"][0]["keyframes"] == _manifest()["region_tracks"][0]["keyframes"]
    assert saved["translation_provider"] == "fake-translator"
    assert report["summary"]["needs_review_count"] == 0
    assert Path(report["viewer"]).is_file()
