import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from text_localizer import video_renderer
from text_localizer.video import VideoInfo
from text_localizer.video_renderer import (
    VideoRenderOptions,
    build_ass_document,
    render_localized_video,
)


def _manifest(video_path: Path, *, needs_review: bool = False) -> dict[str, Any]:
    regions = [
        {
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.1},
            "source_text": "LangGraph",
            "translated_text": "LangGraph \u57fa\u790e {x}\\y",
            "ocr_confidence": 0.95,
            "action": "translate",
            "translation_quality_flags": [],
            "needs_review": False,
        }
    ]
    if needs_review:
        regions.append(
            {
                "bbox": {"x": 0.2, "y": 0.4, "width": 0.4, "height": 0.1},
                "source_text": "review",
                "translated_text": "\u8981\u78ba\u8a8d",
                "ocr_confidence": 0.9,
                "action": "translate",
                "translation_quality_flags": ["manual_review"],
                "needs_review": True,
            }
        )
    return {
        "schema_version": 1,
        "video": {
            "path": str(video_path),
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "duration_seconds": 10.0,
            "codec": "h264",
        },
        "profile_name": "test",
        "sampling": {
            "interval_seconds": 1.0,
            "transition_precision_seconds": 0.125,
        },
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "representative_frame": "frame.png",
                "analysis_path": None,
                "regions": regions,
            }
        ],
        "metrics": {"needs_review_count": int(needs_review)},
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_ass_document_draws_box_and_escaped_approved_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path / "lecture.mp4", needs_review=True)
    monkeypatch.setattr(video_renderer, "_fit_font_size", lambda *_: 24)

    ass_document, overlay_count = build_ass_document(
        manifest,
        font_path=tmp_path / "font.ttc",
        font_family="Yu Gothic",
    )

    assert overlay_count == 1
    assert "PlayResX: 1920" in ass_document
    assert "PlayResY: 1080" in ass_document
    assert ass_document.count("Dialogue:") == 2
    assert r"\clip(192,216,1152,324)" in ass_document
    assert r"\{x\}" in ass_document
    assert r"\\y" in ass_document
    assert "\u8981\u78ba\u8a8d" not in ass_document
    assert "0:00:10.00" in ass_document


def test_render_options_validate_crf_and_preset() -> None:
    with pytest.raises(ValueError, match="CRF"):
        VideoRenderOptions(crf=52)
    with pytest.raises(ValueError, match="preset"):
        VideoRenderOptions(preset="turbo")


def test_final_render_refuses_review_flags(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "tracks.json"
    _write_manifest(manifest_path, _manifest(source, needs_review=True))

    with pytest.raises(ValueError, match="zero review flags"):
        render_localized_video(manifest_path, tmp_path / "localized.mp4")


def test_final_render_refuses_original_or_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"source")
    manifest_path = tmp_path / "tracks.json"
    _write_manifest(manifest_path, _manifest(source))

    with pytest.raises(ValueError, match="must not overwrite"):
        render_localized_video(manifest_path, source)

    existing = tmp_path / "localized.mp4"
    existing.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        render_localized_video(manifest_path, existing)


def test_final_render_writes_video_ass_and_audit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"original-source")
    manifest_path = tmp_path / "tracks.json"
    _write_manifest(manifest_path, _manifest(source))
    output = tmp_path / "localized.mp4"
    calls: list[list[str]] = []

    def fake_inspect(path: Path) -> VideoInfo:
        return VideoInfo(
            path=str(path.resolve()),
            width=1920,
            height=1080,
            fps=30.0,
            duration_seconds=10.0,
            codec="h264",
        )

    def fake_run(
        command: list[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"rendered-video")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_renderer, "inspect_video", fake_inspect)
    monkeypatch.setattr(
        video_renderer,
        "find_japanese_font",
        lambda: tmp_path / "YuGothM.ttc",
    )
    monkeypatch.setattr(video_renderer, "_fit_font_size", lambda *_: 24)
    monkeypatch.setattr(
        video_renderer,
        "resolve_executable",
        lambda *_args, **_kwargs: tmp_path / "ffmpeg.exe",
    )
    monkeypatch.setattr(video_renderer, "_probe_audio_codecs", lambda _: ("aac",))
    monkeypatch.setattr(subprocess, "run", fake_run)

    report = render_localized_video(manifest_path, output)

    assert source.read_bytes() == b"original-source"
    assert output.read_bytes() == b"rendered-video"
    assert output.with_suffix(".ass").is_file()
    assert output.with_suffix(".render.json").is_file()
    assert report["overlay_region_count"] == 1
    assert report["source_unchanged"] is True
    assert report["output_audio_codecs"] == ["aac"]
    assert len(calls) == 1
    assert "libx264" in calls[0]
    assert "copy" in calls[0]
    assert any(argument.startswith("ass=filename=") for argument in calls[0])
