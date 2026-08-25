from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import ImageFont

from text_localizer.environment import resolve_executable
from text_localizer.rendering import find_japanese_font
from text_localizer.track_editor import load_track_manifest
from text_localizer.video import VideoInfo, inspect_video

_X264_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
}


@dataclass(frozen=True, slots=True)
class VideoRenderOptions:
    crf: int = 18
    preset: str = "medium"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.crf, bool) or not 0 <= self.crf <= 51:
            raise ValueError("CRF must be an integer between 0 and 51.")
        if self.preset not in _X264_PRESETS:
            raise ValueError(f"Unsupported libx264 preset: {self.preset}")


def render_localized_video(
    track_manifest_path: Path,
    output_path: Path,
    *,
    video_path: Path | None = None,
    options: VideoRenderOptions | None = None,
) -> dict[str, Any]:
    active_options = options or VideoRenderOptions()
    manifest = load_track_manifest(track_manifest_path)
    review_count = int(manifest["metrics"]["needs_review_count"])
    if review_count:
        raise ValueError(f"Final video rendering requires zero review flags; found {review_count}.")

    source_path = video_path or Path(str(manifest["video"]["path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Final video output must use the .mp4 extension.")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Final video output must not overwrite the original video.")

    ass_output = output_path.with_suffix(".ass")
    report_output = output_path.with_suffix(".render.json")
    existing = [path for path in (output_path, ass_output, report_output) if path.exists()]
    if existing and not active_options.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise ValueError(f"Render output already exists: {names}. Use --overwrite to replace it.")

    input_info = inspect_video(source_path)
    _validate_source_against_manifest(input_info, manifest)
    font_path = find_japanese_font()
    font_family = _font_family(font_path)
    ass_document, overlay_region_count = build_ass_document(
        manifest,
        font_path=font_path,
        font_family=font_family,
    )
    if overlay_region_count == 0:
        raise ValueError("Track manifest contains no approved translated regions to render.")

    ffmpeg = resolve_executable("ffmpeg", environment_variable="FFMPEG_BIN")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Run doctor and set FFMPEG_BIN if needed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _file_fingerprint(source_path)
    input_audio_codecs = _probe_audio_codecs(source_path)
    started = perf_counter()

    with tempfile.TemporaryDirectory(
        prefix=".text-localizer-render-",
        dir=output_path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_ass = temporary_root / "overlays.ass"
        temporary_output = temporary_root / "localized.mp4"
        temporary_report = temporary_root / "render.json"
        temporary_ass.write_text(ass_document, encoding="utf-8-sig")

        ass_filter = _ass_filter(
            temporary_ass,
            fonts_dir=font_path.parent,
            width=input_info.width,
            height=input_info.height,
        )
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vf",
            ass_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            active_options.preset,
            "-crf",
            str(active_options.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-movflags",
            "+faststart",
            "-y",
            str(temporary_output),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            detail = result.stderr.strip()[-4000:]
            raise RuntimeError(f"FFmpeg final video rendering failed: {detail}")
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("FFmpeg completed without producing a final video.")

        output_info = inspect_video(temporary_output)
        _validate_rendered_video(input_info, output_info)
        output_audio_codecs = _probe_audio_codecs(temporary_output)
        if input_audio_codecs != output_audio_codecs:
            raise RuntimeError(
                "Rendered audio streams do not match the original: "
                f"{input_audio_codecs!r} != {output_audio_codecs!r}"
            )
        if _file_fingerprint(source_path) != source_fingerprint:
            raise RuntimeError("The original video changed during final rendering.")

        latency_ms = (perf_counter() - started) * 1000.0
        report: dict[str, Any] = {
            "schema_version": 1,
            "track_manifest": str(track_manifest_path.resolve()),
            "input_video": input_info.to_dict(),
            "input_size_bytes": source_fingerprint[0],
            "input_modified_ns": source_fingerprint[1],
            "input_audio_codecs": list(input_audio_codecs),
            "output_video": {
                **output_info.to_dict(),
                "path": str(output_path.resolve()),
                "size_bytes": temporary_output.stat().st_size,
            },
            "output_audio_codecs": list(output_audio_codecs),
            "encoder": {
                "video": "libx264",
                "preset": active_options.preset,
                "crf": active_options.crf,
                "pixel_format": "yuv420p",
                "audio": "copy",
            },
            "font": {
                "family": font_family,
                "path": str(font_path.resolve()),
            },
            "overlay_region_count": overlay_region_count,
            "ass_dialogue_count": overlay_region_count * 2,
            "review_count": 0,
            "render_latency_ms": latency_ms,
            "source_unchanged": True,
            "artifacts": {
                "video": str(output_path.resolve()),
                "ass": str(ass_output.resolve()),
                "report": str(report_output.resolve()),
            },
        }
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temporary_output.replace(output_path)
        temporary_ass.replace(ass_output)
        temporary_report.replace(report_output)

    return report


def build_ass_document(
    manifest: dict[str, Any],
    *,
    font_path: Path,
    font_family: str,
) -> tuple[str, int]:
    video = manifest["video"]
    width = int(video["width"])
    height = int(video["height"])
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Text,{font_family},42,&H00FFFFFF,&H00FFFFFF,&H00101010,"
            "&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
        ),
        (
            "Style: Box,Arial,10,&H00121212,&H00121212,&H00121212,"
            "&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    overlay_region_count = 0

    for track in manifest["tracks"]:
        start = _format_ass_time(float(track["start_seconds"]))
        end = _format_ass_time(float(track["end_seconds"]))
        for region in track["regions"]:
            translated_text = region.get("translated_text")
            if (
                region.get("action") != "translate"
                or not isinstance(translated_text, str)
                or not translated_text.strip()
                or bool(region.get("needs_review"))
            ):
                continue

            bbox = region["bbox"]
            left = round(float(bbox["x"]) * width)
            top = round(float(bbox["y"]) * height)
            right = round((float(bbox["x"]) + float(bbox["width"])) * width)
            bottom = round((float(bbox["y"]) + float(bbox["height"])) * height)
            box_width = max(1, right - left)
            box_height = max(1, bottom - top)
            font_size = _fit_font_size(
                translated_text,
                font_path,
                max(1, box_width - 8),
                max(1, box_height - 4),
            )
            escaped_text = _escape_ass_text(translated_text)
            background_tags = (
                rf"{{\an7\pos({left},{top})\p1\bord0\shad0"
                rf"\1c&H121212&\1a&H08&}}"
            )
            background = (
                f"{background_tags}m 0 0 l {box_width} 0 "
                f"l {box_width} {box_height} l 0 {box_height}"
            )
            text_tags = (
                rf"{{\an7\pos({left + 4},{top + 2})\fs{font_size}"
                rf"\clip({left},{top},{right},{bottom})\bord0\shad0\q2}}"
            )
            events.append(f"Dialogue: 0,{start},{end},Box,,0,0,0,,{background}")
            events.append(f"Dialogue: 1,{start},{end},Text,,0,0,0,,{text_tags}{escaped_text}")
            overlay_region_count += 1

    return "\n".join([*header, *events, ""]), overlay_region_count


def _fit_font_size(
    text: str,
    font_path: Path,
    box_width: int,
    box_height: int,
) -> int:
    start_size = max(10, min(72, int(box_height * 0.9)))
    for size in range(start_size, 9, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= box_width and bottom - top <= box_height:
            return size
    return 10


def _format_ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _escape_ass_text(text: str) -> str:
    escaped = text.replace("\\", r"\\")
    escaped = escaped.replace("{", r"\{").replace("}", r"\}")
    return escaped.replace("\r\n", r"\N").replace("\r", r"\N").replace("\n", r"\N")


def _ass_filter(
    ass_path: Path,
    *,
    fonts_dir: Path,
    width: int,
    height: int,
) -> str:
    ass_value = _escape_filter_value(ass_path)
    fonts_value = _escape_filter_value(fonts_dir)
    return f"ass=filename='{ass_value}':original_size={width}x{height}:fontsdir='{fonts_value}'"


def _escape_filter_value(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def _font_family(font_path: Path) -> str:
    return "Meiryo" if font_path.name.lower().startswith("meiryo") else "Yu Gothic"


def _validate_source_against_manifest(
    source: VideoInfo,
    manifest: dict[str, Any],
) -> None:
    expected = manifest["video"]
    if source.width != int(expected["width"]) or source.height != int(expected["height"]):
        raise ValueError("Input video dimensions do not match the track manifest.")
    expected_fps = float(expected["fps"])
    if not math.isclose(source.fps, expected_fps, rel_tol=0.001, abs_tol=0.001):
        raise ValueError("Input video frame rate does not match the track manifest.")
    duration_tolerance = max(0.1, 2.0 / max(source.fps, 1.0))
    if abs(source.duration_seconds - float(expected["duration_seconds"])) > duration_tolerance:
        raise ValueError("Input video duration does not match the track manifest.")


def _validate_rendered_video(source: VideoInfo, output: VideoInfo) -> None:
    if output.codec != "h264":
        raise RuntimeError(f"Rendered video codec is not H.264: {output.codec}")
    if (output.width, output.height) != (source.width, source.height):
        raise RuntimeError("Rendered video dimensions do not match the original.")
    if not math.isclose(output.fps, source.fps, rel_tol=0.001, abs_tol=0.001):
        raise RuntimeError("Rendered video frame rate does not match the original.")
    duration_tolerance = max(0.1, 2.0 / max(source.fps, 1.0))
    if abs(output.duration_seconds - source.duration_seconds) > duration_tolerance:
        raise RuntimeError("Rendered video duration does not match the original.")


def _probe_audio_codecs(path: Path) -> tuple[str, ...]:
    ffprobe = resolve_executable("ffprobe", environment_variable="FFPROBE_BIN")
    if ffprobe is None:
        raise RuntimeError("FFprobe was not found. Run doctor and set FFPROBE_BIN if needed.")
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise RuntimeError(f"FFprobe returned invalid audio stream data for {path}")
    return tuple(str(stream.get("codec_name", "unknown")) for stream in streams)


def _file_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns
