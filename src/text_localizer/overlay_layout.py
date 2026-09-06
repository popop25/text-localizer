from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from PIL import ImageFont

from text_localizer.rendering import find_japanese_font
from text_localizer.track_editor import load_track_manifest, validate_track_manifest
from text_localizer.video_viewer import write_video_viewer

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+&-]*|[ \t]+|.", re.DOTALL)
_PROHIBITED_LINE_START = frozenset("、。，．…!?！？)]}」』】》〉")


@dataclass(frozen=True, slots=True)
class OverlayLayoutOptions:
    maximum_width_scale: float = 1.8
    maximum_frame_width_ratio: float = 0.6
    maximum_lines: int = 2
    minimum_font_scale: float = 0.8
    font_match_threshold: float = 0.95
    minimum_font_match_ratio: float = 0.9

    def __post_init__(self) -> None:
        if not 1.0 <= self.maximum_width_scale <= 3.0:
            raise ValueError("Layout maximum width scale must be between 1 and 3.")
        if not 0.1 <= self.maximum_frame_width_ratio <= 1.0:
            raise ValueError("Layout maximum frame width ratio must be between 0.1 and 1.")
        if isinstance(self.maximum_lines, bool) or not 1 <= self.maximum_lines <= 4:
            raise ValueError("Layout maximum lines must be an integer between 1 and 4.")
        if not 0.5 <= self.minimum_font_scale <= 1.0:
            raise ValueError("Layout minimum font scale must be between 0.5 and 1.")
        if not self.minimum_font_scale <= self.font_match_threshold <= 1.0:
            raise ValueError("Layout font match threshold must not be below its minimum scale.")
        if not 0.0 <= self.minimum_font_match_ratio <= 1.0:
            raise ValueError("Layout minimum font match ratio must be between 0 and 1.")


def layout_region_tracks(
    manifest_path: Path,
    output_dir: Path,
    *,
    options: OverlayLayoutOptions | None = None,
    font_path: Path | None = None,
) -> dict[str, Any]:
    resolved_options = options or OverlayLayoutOptions()
    source_manifest = load_track_manifest(manifest_path)
    if source_manifest.get("schema_version") != 2:
        raise ValueError("Adaptive overlay layout requires a schema-v2 region-track manifest.")

    result = deepcopy(source_manifest)
    video = cast(dict[str, Any], result["video"])
    frame_width = int(video["width"])
    frame_height = int(video["height"])
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Adaptive overlay layout requires positive video dimensions.")
    resolved_font_path = font_path or find_japanese_font()
    raw_tracks = cast(list[Any], result["region_tracks"])
    tracks = [cast(dict[str, Any], item) for item in raw_tracks]
    records: list[dict[str, Any]] = []
    started = perf_counter()

    for track in tracks:
        region = cast(dict[str, Any], track["region"])
        translated_text = region.get("translated_text")
        if (
            region.get("action") != "translate"
            or not isinstance(translated_text, str)
            or not translated_text.strip()
            or bool(region.get("needs_review"))
        ):
            track.pop("overlay_layout", None)
            track.pop("overlay_replacement", None)
            continue
        layout, record = _resolve_layout(
            translated_text.strip(),
            cast(dict[str, Any], region["bbox"]),
            cast(dict[str, Any] | None, track.get("overlay_style")),
            frame_width=frame_width,
            frame_height=frame_height,
            font_path=resolved_font_path,
            options=resolved_options,
        )
        track["overlay_layout"] = layout
        replacement = _resolve_replacement_geometry(
            layout,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        track["overlay_replacement"] = replacement
        record.update(
            {
                "erase_padding_left_px": round(
                    float(replacement["erase_padding_left_ratio"]) * frame_width
                ),
                "erase_padding_right_px": round(
                    float(replacement["erase_padding_right_ratio"]) * frame_width
                ),
                "erase_padding_top_px": round(
                    float(replacement["erase_padding_top_ratio"]) * frame_height
                ),
                "erase_padding_bottom_px": round(
                    float(replacement["erase_padding_bottom_ratio"]) * frame_height
                ),
            }
        )
        records.append({"region_track_id": track["region_track_id"], **record})

    strategy_counts = Counter(str(record["fit_strategy"]) for record in records)
    approved_count = len(records)
    source_match_count = sum(
        float(record["font_scale"]) >= resolved_options.font_match_threshold for record in records
    )
    source_match_ratio = source_match_count / approved_count if approved_count else 0.0
    baseline_overflow_count = sum(bool(record["baseline_overflow"]) for record in records)
    clipped_count = sum(not bool(record["content_fits"]) for record in records)
    boundary_violation_count = sum(not bool(record["within_frame"]) for record in records)
    below_minimum_count = sum(
        float(record["font_scale"]) + 1e-9 < resolved_options.minimum_font_scale
        for record in records
    )
    review_count = sum(bool(record["layout_needs_review"]) for record in records)
    gate_results = {
        "no_clipped_regions": clipped_count == 0,
        "no_boundary_violations": boundary_violation_count == 0,
        "no_subminimum_font_sizes": below_minimum_count == 0,
        "source_font_match_ratio": source_match_ratio >= resolved_options.minimum_font_match_ratio,
        "no_layout_review_flags": review_count == 0,
    }
    current_run_latency_ms = (perf_counter() - started) * 1000.0
    summary: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "adaptive-fit-v1",
        "font_path": str(resolved_font_path.resolve()),
        "options": {
            "maximum_width_scale": resolved_options.maximum_width_scale,
            "maximum_frame_width_ratio": resolved_options.maximum_frame_width_ratio,
            "maximum_lines": resolved_options.maximum_lines,
            "minimum_font_scale": resolved_options.minimum_font_scale,
            "font_match_threshold": resolved_options.font_match_threshold,
            "minimum_font_match_ratio": resolved_options.minimum_font_match_ratio,
        },
        "approved_region_count": approved_count,
        "baseline_source_box_overflow_count": baseline_overflow_count,
        "source_font_match_count": source_match_count,
        "source_font_match_ratio": source_match_ratio,
        "clipped_region_count": clipped_count,
        "boundary_violation_count": boundary_violation_count,
        "subminimum_font_size_count": below_minimum_count,
        "layout_review_count": review_count,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "gate_results": gate_results,
        "gates_passed": all(gate_results.values()),
        "current_run_latency_ms": current_run_latency_ms,
    }
    result["region_layout"] = summary
    result["region_replacement"] = {
        "schema_version": 1,
        "algorithm": "padded-source-box-v1",
        "composite_order": ["erase-source", "draw-translation"],
        "replacement_region_count": approved_count,
    }
    metrics = cast(dict[str, Any], result["metrics"])
    metrics.update(
        {
            "layout_baseline_overflow_count": baseline_overflow_count,
            "layout_clipped_region_count": clipped_count,
            "layout_source_font_match_ratio": source_match_ratio,
            "layout_review_count": review_count,
            "layout_current_run_latency_ms": current_run_latency_ms,
        }
    )
    validate_track_manifest(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.json"
    tracks_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {"schema_version": 1, "summary": summary, "regions": records}
    (output_dir / "layout-quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    video_path = _resolve_video_path(result, manifest_path)
    write_video_viewer(output_dir / "viewer.html", video_path, result)
    return result


def _resolve_replacement_geometry(
    layout: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    source_font_size = max(1, round(float(layout["source_font_size_ratio"]) * frame_height))
    horizontal_padding = max(2, math.ceil(source_font_size * 0.18))
    top_padding = max(2, math.ceil(source_font_size * 0.35))
    bottom_padding = max(2, math.ceil(source_font_size * 0.35))
    return {
        "schema_version": 1,
        "algorithm": "padded-source-box-v1",
        "composite_order": ["erase-source", "draw-translation"],
        "erase_padding_left_ratio": horizontal_padding / frame_width,
        "erase_padding_right_ratio": horizontal_padding / frame_width,
        "erase_padding_top_ratio": top_padding / frame_height,
        "erase_padding_bottom_ratio": bottom_padding / frame_height,
    }


def _resolve_layout(
    text: str,
    bbox: dict[str, Any],
    style: dict[str, Any] | None,
    *,
    frame_width: int,
    frame_height: int,
    font_path: Path,
    options: OverlayLayoutOptions,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_left, original_top, original_right, original_bottom = _pixel_box(
        bbox, frame_width, frame_height
    )
    original_width = max(1, original_right - original_left)
    original_height = max(1, original_bottom - original_top)
    source_font_size = max(
        10,
        round(
            float(style["font_size_ratio"]) * frame_height
            if style is not None
            else original_height * 0.8
        ),
    )
    line_height_ratio = float(style["line_height"]) if style is not None else 1.15
    padding_x = max(3, round(frame_width / 640))
    padding_y = max(2, round(frame_height / 540))
    maximum_width = max(
        original_width,
        min(
            frame_width,
            round(frame_width * options.maximum_frame_width_ratio),
            round(original_width * options.maximum_width_scale),
        ),
    )
    baseline_font = ImageFont.truetype(str(font_path), size=source_font_size)
    baseline_width = math.ceil(_text_width(baseline_font, text)) + padding_x * 2
    baseline_height = math.ceil(source_font_size * line_height_ratio) + padding_y * 2
    baseline_overflow = baseline_width > original_width or baseline_height > original_height

    minimum_font_size = max(10, math.ceil(source_font_size * options.minimum_font_scale))
    resolved = _find_layout(
        text,
        font_path=font_path,
        source_font_size=source_font_size,
        minimum_font_size=minimum_font_size,
        original_width=original_width,
        original_height=original_height,
        maximum_width=maximum_width,
        padding_x=padding_x,
        padding_y=padding_y,
        line_height_ratio=line_height_ratio,
        maximum_lines=options.maximum_lines,
    )
    layout_needs_review = False
    if resolved is None:
        layout_needs_review = True
        emergency_minimum = max(10, math.ceil(source_font_size * 0.6))
        resolved = _find_layout(
            text,
            font_path=font_path,
            source_font_size=minimum_font_size - 1,
            minimum_font_size=emergency_minimum,
            original_width=original_width,
            original_height=original_height,
            maximum_width=maximum_width,
            padding_x=padding_x,
            padding_y=padding_y,
            line_height_ratio=line_height_ratio,
            maximum_lines=max(3, options.maximum_lines),
        )
    if resolved is None:
        layout_needs_review = True
        emergency_font = ImageFont.truetype(str(font_path), size=10)
        lines = _wrap_text(
            text,
            emergency_font,
            max_width=max(1, maximum_width - padding_x * 2),
            max_lines=None,
        )
        assert lines is not None
        resolved = _ResolvedLayout(
            lines=lines,
            font_size=10,
            width=max(
                original_width,
                min(
                    maximum_width,
                    math.ceil(max(_text_width(emergency_font, line) for line in lines))
                    + padding_x * 2,
                ),
            ),
            height=max(
                original_height,
                math.ceil(10 * line_height_ratio) * len(lines) + padding_y * 2,
            ),
            strategy="review",
        )

    target_width = min(frame_width, resolved.width)
    target_height = min(frame_height, resolved.height)
    left, right = _place_axis(
        original_left,
        original_right,
        target_width,
        frame_width,
        anchor_start=True,
    )
    top, bottom = _place_axis(original_top, original_bottom, target_height, frame_height)
    content_width = max(1, target_width - padding_x * 2)
    content_height = max(1, target_height - padding_y * 2)
    resolved_font = ImageFont.truetype(str(font_path), size=resolved.font_size)
    measured_width = math.ceil(max(_text_width(resolved_font, line) for line in resolved.lines))
    measured_height = math.ceil(resolved.font_size * line_height_ratio) * len(resolved.lines)
    content_fits = measured_width <= content_width and measured_height <= content_height
    within_frame = 0 <= left < right <= frame_width and 0 <= top < bottom <= frame_height
    font_scale = resolved.font_size / source_font_size
    strategy = "review" if layout_needs_review else resolved.strategy
    normalized_text = _normalize_text(text)
    layout = {
        "schema_version": 1,
        "algorithm": "adaptive-fit-v1",
        "text": normalized_text,
        "text_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
        "lines": list(resolved.lines),
        "line_count": len(resolved.lines),
        "fit_strategy": strategy,
        "source_font_size_ratio": source_font_size / frame_height,
        "resolved_font_size_ratio": resolved.font_size / frame_height,
        "font_scale": font_scale,
        "line_height": line_height_ratio,
        "padding_x_ratio": padding_x / frame_width,
        "padding_y_ratio": padding_y / frame_height,
        "left_extension_ratio": (original_left - left) / frame_width,
        "right_extension_ratio": (right - original_right) / frame_width,
        "top_extension_ratio": (original_top - top) / frame_height,
        "bottom_extension_ratio": (bottom - original_bottom) / frame_height,
        "target_width_ratio": target_width / frame_width,
        "target_height_ratio": target_height / frame_height,
        "content_fits": content_fits,
        "layout_needs_review": layout_needs_review or not content_fits or not within_frame,
    }
    record = {
        "baseline_overflow": baseline_overflow,
        "fit_strategy": strategy,
        "line_count": len(resolved.lines),
        "source_font_size_px": source_font_size,
        "resolved_font_size_px": resolved.font_size,
        "font_scale": font_scale,
        "source_box_width_px": original_width,
        "source_box_height_px": original_height,
        "target_box_width_px": target_width,
        "target_box_height_px": target_height,
        "measured_text_width_px": measured_width,
        "measured_text_height_px": measured_height,
        "content_fits": content_fits,
        "within_frame": within_frame,
        "layout_needs_review": bool(layout["layout_needs_review"]),
    }
    return layout, record


@dataclass(frozen=True, slots=True)
class _ResolvedLayout:
    lines: tuple[str, ...]
    font_size: int
    width: int
    height: int
    strategy: str


def _find_layout(
    text: str,
    *,
    font_path: Path,
    source_font_size: int,
    minimum_font_size: int,
    original_width: int,
    original_height: int,
    maximum_width: int,
    padding_x: int,
    padding_y: int,
    line_height_ratio: float,
    maximum_lines: int,
) -> _ResolvedLayout | None:
    if source_font_size < minimum_font_size:
        return None
    for font_size in range(source_font_size, minimum_font_size - 1, -1):
        font = ImageFont.truetype(str(font_path), size=font_size)
        single_width = math.ceil(_text_width(font, text)) + padding_x * 2
        single_height = math.ceil(font_size * line_height_ratio) + padding_y * 2
        if single_width <= maximum_width:
            width = max(original_width, single_width)
            height = max(original_height, single_height)
            if font_size < source_font_size:
                strategy = "shrink"
            elif width == original_width and height == original_height:
                strategy = "source-box"
            else:
                strategy = "expand"
            return _ResolvedLayout((text,), font_size, width, height, strategy)

    for font_size in range(source_font_size, minimum_font_size - 1, -1):
        font = ImageFont.truetype(str(font_path), size=font_size)
        lines = _wrap_text(
            text,
            font,
            max_width=max(1, maximum_width - padding_x * 2),
            max_lines=maximum_lines,
        )
        if lines is None:
            continue
        measured_width = math.ceil(max(_text_width(font, line) for line in lines))
        if measured_width + padding_x * 2 > maximum_width:
            continue
        width = max(original_width, min(maximum_width, measured_width + padding_x * 2))
        height = max(
            original_height,
            math.ceil(font_size * line_height_ratio) * len(lines) + padding_y * 2,
        )
        return _ResolvedLayout(
            lines,
            font_size,
            width,
            height,
            "wrap" if font_size == source_font_size else "shrink",
        )
    return None


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_lines: int | None,
) -> tuple[str, ...] | None:
    lines: list[str] = []
    for paragraph in _normalize_text(text).split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(_wrap_paragraph(paragraph, font, max_width=max_width))
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


def _pixel_box(
    bbox: dict[str, Any], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    left = max(0, round(float(bbox["x"]) * frame_width))
    top = max(0, round(float(bbox["y"]) * frame_height))
    right = min(
        frame_width,
        round((float(bbox["x"]) + float(bbox["width"])) * frame_width),
    )
    bottom = min(
        frame_height,
        round((float(bbox["y"]) + float(bbox["height"])) * frame_height),
    )
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _place_axis(
    start: int,
    end: int,
    target_size: int,
    frame_size: int,
    *,
    anchor_start: bool = False,
) -> tuple[int, int]:
    target_size = min(frame_size, max(end - start, target_size))
    extra = target_size - (end - start)
    before = 0 if anchor_start else min(start, extra // 2)
    resolved_start = start - before
    resolved_end = resolved_start + target_size
    if resolved_end > frame_size:
        resolved_end = frame_size
        resolved_start = frame_size - target_size
    return resolved_start, resolved_end


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text))


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _resolve_video_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    raw_path = Path(str(cast(dict[str, Any], manifest["video"])["path"]))
    if raw_path.is_absolute():
        return raw_path
    return (manifest_path.parent / raw_path).resolve()
