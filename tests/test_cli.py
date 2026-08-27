from pathlib import Path
from typing import Any

import pytest

import text_localizer.cli as cli
from text_localizer.cli import build_parser


def test_required_commands_are_registered() -> None:
    parser = build_parser()
    command_arguments = {
        "doctor": ["doctor"],
        "inspect-video": ["inspect-video", "placeholder"],
        "extract-frames": ["extract-frames", "placeholder"],
        "analyze-image": ["analyze-image", "placeholder"],
        "deduplicate-frames": ["deduplicate-frames", "placeholder"],
        "index-video": ["index-video", "placeholder"],
        "localize-video": ["localize-video", "placeholder"],
        "index-visual-events": ["index-visual-events", "placeholder"],
        "track-regions": ["track-regions", "tracks.json", "visual-events.json"],
        "style-regions": ["style-regions", "tracks.json"],
        "layout-regions": ["layout-regions", "tracks.json"],
        "retranslate-tracks": ["retranslate-tracks", "tracks.json"],
        "render-tracks": ["render-tracks", "placeholder"],
        "evaluate-timeline": ["evaluate-timeline", "tracks.json", "benchmark.json"],
        "evaluate-translations": [
            "evaluate-translations",
            "tracks.json",
            "benchmark.json",
        ],
        "benchmark-translator": ["benchmark-translator", "benchmark.json"],
        "setup-models": ["setup-models", "--ocr"],
        "render-video": ["render-video", "placeholder"],
    }
    for command, arguments in command_arguments.items():
        namespace = parser.parse_args(arguments)
        assert namespace.command == command


def test_translation_commands_default_to_quality_selected_provider() -> None:
    parser = build_parser()

    for arguments in (
        ["analyze-image", "frame.png"],
        ["localize-video", "video.mp4"],
        ["retranslate-tracks", "tracks.json"],
        ["benchmark-translator", "benchmark.json"],
        ["setup-models", "--translation"],
    ):
        namespace = parser.parse_args(arguments)
        assert namespace.translation_provider == "gemma3-270m"


def test_track_regions_parses_repeatable_include_windows() -> None:
    namespace = build_parser().parse_args(
        [
            "track-regions",
            "tracks.json",
            "visual-events.json",
            "--include-window",
            "8:9.4",
            "--include-window",
            "120:166.8",
        ]
    )

    assert namespace.include_window == [(8.0, 9.4), (120.0, 166.8)]


def test_style_regions_parses_quality_thresholds() -> None:
    namespace = build_parser().parse_args(
        [
            "style-regions",
            "tracks.json",
            "--background-padding-ratio",
            "0.5",
            "--minimum-contrast-ratio",
            "4.5",
            "--minimum-style-confidence",
            "0.6",
        ]
    )

    assert namespace.background_padding_ratio == 0.5
    assert namespace.minimum_contrast_ratio == 4.5
    assert namespace.minimum_style_confidence == 0.6


def test_layout_regions_parses_fit_policy() -> None:
    namespace = build_parser().parse_args(
        [
            "layout-regions",
            "tracks.json",
            "--maximum-width-scale",
            "1.4",
            "--maximum-frame-width-ratio",
            "0.5",
            "--maximum-lines",
            "2",
            "--minimum-font-scale",
            "0.82",
            "--require-pass",
        ]
    )

    assert namespace.maximum_width_scale == 1.4
    assert namespace.maximum_frame_width_ratio == 0.5
    assert namespace.maximum_lines == 2
    assert namespace.minimum_font_scale == 0.82
    assert namespace.require_pass is True


def test_render_tracks_command_writes_viewer_and_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest: dict[str, Any] = {"video": {"path": "source.mp4"}}
    writes: list[tuple[Path, Path, dict[str, Any]]] = []
    monkeypatch.setattr(cli, "load_track_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        cli,
        "write_video_viewer",
        lambda path, video, payload: writes.append((path, video, payload)),
    )
    monkeypatch.setattr(
        cli,
        "write_track_editor",
        lambda path, video, payload: writes.append((path, video, payload)),
    )

    status = cli.main(["render-tracks", "tracks.json", "--output", str(tmp_path)])

    assert status == 0
    assert [item[0].name for item in writes] == ["viewer.html", "editor.html"]
    assert all(item[1] == Path("source.mp4") for item in writes)
    assert all(item[2] is manifest for item in writes)


def test_render_tracks_schema_v2_skips_legacy_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest: dict[str, Any] = {"schema_version": 2, "video": {"path": "source.mp4"}}
    viewer_writes: list[Path] = []
    editor_writes: list[Path] = []
    monkeypatch.setattr(cli, "load_track_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        cli,
        "write_video_viewer",
        lambda path, _video, _payload: viewer_writes.append(path),
    )
    monkeypatch.setattr(
        cli,
        "write_track_editor",
        lambda path, _video, _payload: editor_writes.append(path),
    )

    result = cli.main(["render-tracks", str(tmp_path / "tracks.json"), "--output", str(tmp_path)])

    assert result == 0
    assert viewer_writes == [tmp_path / "viewer.html"]
    assert editor_writes == []


def test_evaluate_timeline_command_can_require_passing_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "evaluate_temporal_quality_files",
        lambda *_args, **_kwargs: {"summary": {"gates_passed": False}},
    )

    status = cli.main(
        [
            "evaluate-timeline",
            "tracks.json",
            "benchmark.json",
            "--require-pass",
        ]
    )

    assert status == 1
