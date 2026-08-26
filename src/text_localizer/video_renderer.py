from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
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
    if manifest.get("schema_version") == 2:
        raise ValueError(
            "Schema-v2 region keyframes cannot yet be burned into MP4; review viewer.html."
        )
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
    ass_document, layout_metrics = _build_ass_document(
        manifest,
        font_path=font_path,
        font_family=font_family,
    )
    overlay_region_count = layout_metrics.overlay_region_count
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
            "schema_version": 2,
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
            "overlay_layout": layout_metrics.to_dict(),
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


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+&-]*|[ \t]+|.", re.DOTALL)
_PROHIBITED_LINE_START = frozenset("、。，．…!?！？)]}」』】》〉")


@dataclass(frozen=True, slots=True)
class OverlayLayoutMetrics:
    overlay_region_count: int
    multiline_region_count: int
    expanded_region_count: int
    repositioned_region_count: int
    minimum_font_size: int
    maximum_font_size: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "mode": "readable-spatial-v1",
            "multiline_region_count": self.multiline_region_count,
            "expanded_region_count": self.expanded_region_count,
            "repositioned_region_count": self.repositioned_region_count,
            "minimum_font_size": self.minimum_font_size,
            "maximum_font_size": self.maximum_font_size,
        }


@dataclass(frozen=True, slots=True)
class _RegionLayout:
    original_left: int
    original_top: int
    original_right: int
    original_bottom: int
    left: int
    top: int
    right: int
    bottom: int
    lines: tuple[str, ...]
    font_size: int

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def is_expanded(self) -> bool:
        return (
            self.right - self.left > self.original_right - self.original_left
            or self.bottom - self.top > self.original_bottom - self.original_top
        )

    @property
    def is_repositioned(self) -> bool:
        return self.left != self.original_left or self.top != self.original_top


def build_ass_document(
    manifest: dict[str, Any],
    *,
    font_path: Path,
    font_family: str,
) -> tuple[str, int]:
    document, metrics = _build_ass_document(
        manifest,
        font_path=font_path,
        font_family=font_family,
    )
    return document, metrics.overlay_region_count


def _build_ass_document(
    manifest: dict[str, Any],
    *,
    font_path: Path,
    font_family: str,
) -> tuple[str, OverlayLayoutMetrics]:
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
        "; LayoutMode: readable-spatial-v1",
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
    all_layouts: list[_RegionLayout] = []

    for track in manifest["tracks"]:
        start = _format_ass_time(float(track["start_seconds"]))
        end = _format_ass_time(float(track["end_seconds"]))
        track_layouts: list[_RegionLayout] = []
        for region in track["regions"]:
            translated_text = region.get("translated_text")
            if (
                region.get("action") != "translate"
                or not isinstance(translated_text, str)
                or not translated_text.strip()
                or bool(region.get("needs_review"))
            ):
                continue
            track_layouts.append(
                _layout_region(
                    translated_text.strip(),
                    region["bbox"],
                    font_path=font_path,
                    frame_width=width,
                    frame_height=height,
                )
            )

        resolved_layouts = _resolve_layout_collisions(track_layouts, frame_height=height)
        all_layouts.extend(resolved_layouts)
        for layout in resolved_layouts:
            background_left = min(layout.original_left, layout.left)
            background_top = min(layout.original_top, layout.top)
            background_right = max(layout.original_right, layout.right)
            background_bottom = max(layout.original_bottom, layout.bottom)
            background_width = max(1, background_right - background_left)
            background_height = max(1, background_bottom - background_top)
            escaped_text = _escape_ass_text("\n".join(layout.lines))
            background_tags = (
                rf"{{\an7\pos({background_left},{background_top})\p1\bord0\shad0"
                rf"\1c&H121212&\1a&H08&}}"
            )
            background = (
                f"{background_tags}m 0 0 l {background_width} 0 "
                f"l {background_width} {background_height} l 0 {background_height}"
            )
            text_tags = (
                rf"{{\an7\pos({layout.left + 4},{layout.top + 2})\fs{layout.font_size}"
                rf"\clip({background_left},{background_top},{background_right},"
                rf"{background_bottom})\bord0\shad0\q2}}"
            )
            events.append(f"Dialogue: 0,{start},{end},Box,,0,0,0,,{background}")
            events.append(f"Dialogue: 1,{start},{end},Text,,0,0,0,,{text_tags}{escaped_text}")

    font_sizes = [layout.font_size for layout in all_layouts]
    metrics = OverlayLayoutMetrics(
        overlay_region_count=len(all_layouts),
        multiline_region_count=sum(layout.line_count > 1 for layout in all_layouts),
        expanded_region_count=sum(layout.is_expanded for layout in all_layouts),
        repositioned_region_count=sum(layout.is_repositioned for layout in all_layouts),
        minimum_font_size=min(font_sizes, default=0),
        maximum_font_size=max(font_sizes, default=0),
    )
    return "\n".join([*header, *events, ""]), metrics


def _layout_region(
    text: str,
    bbox: dict[str, Any],
    *,
    font_path: Path,
    frame_width: int,
    frame_height: int,
) -> _RegionLayout:
    original_left = max(0, round(float(bbox["x"]) * frame_width))
    original_top = max(0, round(float(bbox["y"]) * frame_height))
    original_right = min(
        frame_width,
        round((float(bbox["x"]) + float(bbox["width"])) * frame_width),
    )
    original_bottom = min(
        frame_height,
        round((float(bbox["y"]) + float(bbox["height"])) * frame_height),
    )
    original_width = max(1, original_right - original_left)
    original_height = max(1, original_bottom - original_top)
    padding_x = max(4, round(frame_width / 480))
    padding_y = max(2, round(frame_height / 540))
    minimum_font_size = max(12, round(16 * frame_height / 1080))
    preferred_font_size = max(
        minimum_font_size,
        min(72, round(original_height * 0.9)),
    )
    expanded_width = max(
        original_width,
        min(
            frame_width - original_left,
            round(frame_width * 0.45),
            round(original_width * 1.5),
        ),
    )
    candidate_widths = [original_width]
    if expanded_width >= original_width + padding_x * 2:
        candidate_widths.append(expanded_width)

    selected_lines: tuple[str, ...] | None = None
    selected_font_size = minimum_font_size
    selected_width = original_width
    normalized_single_line = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in normalized_single_line:
        for candidate_width in candidate_widths:
            for font_size in range(preferred_font_size, minimum_font_size - 1, -1):
                font = ImageFont.truetype(str(font_path), size=font_size)
                if _text_width(font, text) <= max(
                    1,
                    candidate_width - padding_x * 2,
                ):
                    selected_lines = (text,)
                    selected_font_size = font_size
                    selected_width = candidate_width
                    break
            if selected_lines is not None:
                break

    if selected_lines is None:
        for candidate_width in candidate_widths:
            for font_size in range(preferred_font_size, minimum_font_size - 1, -1):
                font = ImageFont.truetype(str(font_path), size=font_size)
                lines = _wrap_text(
                    text,
                    font,
                    max_width=max(1, candidate_width - padding_x * 2),
                    max_lines=3,
                )
                if lines is not None:
                    selected_lines = lines
                    selected_font_size = font_size
                    selected_width = candidate_width
                    break
            if selected_lines is not None:
                break

    if selected_lines is None:
        selected_width = expanded_width
        selected_font_size = minimum_font_size
        font = ImageFont.truetype(str(font_path), size=selected_font_size)
        selected_lines = _wrap_text(
            text,
            font,
            max_width=max(1, selected_width - padding_x * 2),
            max_lines=None,
        )
        if selected_lines is None:
            raise RuntimeError("Unable to lay out translated overlay text.")

    line_height = max(selected_font_size + 2, math.ceil(selected_font_size * 1.2))
    required_height = original_height
    if len(selected_lines) > 1:
        required_height = max(
            original_height,
            padding_y * 2 + line_height * len(selected_lines),
        )
    layout_top = original_top
    if layout_top + required_height > frame_height:
        layout_top = max(0, frame_height - required_height)

    return _RegionLayout(
        original_left=original_left,
        original_top=original_top,
        original_right=original_right,
        original_bottom=original_bottom,
        left=original_left,
        top=layout_top,
        right=min(frame_width, original_left + selected_width),
        bottom=min(frame_height, layout_top + required_height),
        lines=selected_lines,
        font_size=selected_font_size,
    )


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_lines: int | None,
) -> tuple[str, ...] | None:
    lines: list[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for paragraph in normalized.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(_wrap_paragraph(paragraph.strip(), font, max_width=max_width))
    if max_lines is not None and len(lines) > max_lines:
        return None
    return tuple(lines)


def _wrap_paragraph(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    pending_space = False
    for token in _TOKEN_PATTERN.findall(text):
        if token.isspace():
            pending_space = bool(current)
            continue

        separator = " " if pending_space and current else ""
        candidate = f"{current}{separator}{token}"
        if not current or _text_width(font, candidate) <= max_width:
            if not current and _text_width(font, token) > max_width:
                chunks = _split_overlong_token(token, font, max_width=max_width)
                lines.extend(chunks[:-1])
                current = chunks[-1]
            else:
                current = candidate
            pending_space = False
            continue

        if token in _PROHIBITED_LINE_START:
            current += token
            pending_space = False
            continue

        lines.append(current.rstrip())
        pending_space = False
        if _text_width(font, token) <= max_width:
            current = token
        else:
            chunks = _split_overlong_token(token, font, max_width=max_width)
            lines.extend(chunks[:-1])
            current = chunks[-1]

    if current:
        lines.append(current.rstrip())
    return lines or [""]


def _split_overlong_token(
    token: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for character in token:
        candidate = f"{current}{character}"
        if current and _text_width(font, candidate) > max_width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [token]


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text))


def _resolve_layout_collisions(
    layouts: list[_RegionLayout],
    *,
    frame_height: int,
) -> list[_RegionLayout]:
    resolved: list[_RegionLayout] = []
    for layout in sorted(layouts, key=lambda item: (item.original_top, item.original_left)):
        active = layout
        for _ in range(len(resolved) + 1):
            collisions = [
                previous
                for previous in resolved
                if _horizontally_overlaps(active, previous)
                and _vertically_overlaps(active, previous)
            ]
            if not collisions:
                break
            height = active.bottom - active.top
            candidate_top = max(previous.bottom for previous in collisions) + 2
            if candidate_top + height > frame_height:
                break
            active = replace(
                active,
                top=candidate_top,
                bottom=candidate_top + height,
            )
        resolved.append(active)
    return resolved


def _horizontally_overlaps(first: _RegionLayout, second: _RegionLayout) -> bool:
    return min(first.right, second.right) - max(first.left, second.left) > 2


def _vertically_overlaps(first: _RegionLayout, second: _RegionLayout) -> bool:
    return min(first.bottom, second.bottom) - max(first.top, second.top) > 0


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
