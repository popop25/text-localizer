from pathlib import Path
from typing import Any

from text_localizer.video_viewer import write_video_viewer


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "region_tracks": [],
        "region_tracking": {"algorithm": "test"},
        "region_styling": {"algorithm": "test"},
    }


def test_viewer_uses_relative_video_url_inside_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "preview"
    output_dir.mkdir()
    video_path = output_dir / "source.mp4"
    video_path.write_bytes(b"video")
    viewer_path = output_dir / "viewer.html"

    write_video_viewer(viewer_path, video_path, _manifest())

    viewer = viewer_path.read_text(encoding="utf-8")
    assert 'const videoSource = "source.mp4";' in viewer


def test_viewer_keeps_file_url_for_video_outside_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "preview"
    output_dir.mkdir()
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    viewer_path = output_dir / "viewer.html"

    write_video_viewer(viewer_path, video_path, _manifest())

    viewer = viewer_path.read_text(encoding="utf-8")
    assert video_path.resolve().as_uri() in viewer
