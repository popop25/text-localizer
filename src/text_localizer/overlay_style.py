from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from PIL import Image

from text_localizer.track_editor import load_track_manifest, validate_track_manifest
from text_localizer.video import extract_frame_at_timestamp
from text_localizer.video_viewer import write_video_viewer

_COLOR_QUANTIZATION = 16


@dataclass(frozen=True, slots=True)
class OverlayStyleOptions:
    background_padding_ratio: float = 0.35
    minimum_contrast_ratio: float = 3.0
    minimum_style_confidence: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.background_padding_ratio <= 2.0:
            raise ValueError("Style background padding ratio must be between 0 and 2.")
        if not 1.0 <= self.minimum_contrast_ratio <= 21.0:
            raise ValueError("Style minimum contrast ratio must be between 1 and 21.")
        if not 0.0 <= self.minimum_style_confidence <= 1.0:
            raise ValueError("Style minimum confidence must be between 0 and 1.")


def style_region_tracks(
    manifest_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    options: OverlayStyleOptions | None = None,
) -> dict[str, Any]:
    resolved_options = options or OverlayStyleOptions()
    source_manifest = load_track_manifest(manifest_path)
    if source_manifest.get("schema_version") != 2:
        raise ValueError("Adaptive overlay styling requires a schema-v2 region-track manifest.")
    video_path = _resolve_video_path(source_manifest, manifest_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Track source video does not exist: {video_path}")

    result = deepcopy(source_manifest)
    result.pop("region_layout", None)
    raw_tracks = result.get("region_tracks")
    assert isinstance(raw_tracks, list)
    region_tracks = [cast(dict[str, Any], item) for item in raw_tracks]
    video_stat = video_path.stat()
    video_fingerprint = f"{video_stat.st_size:x}-{video_stat.st_mtime_ns:x}"
    frame_dir = work_dir / "frames" / video_fingerprint
    frames: dict[int, tuple[Path, bool]] = {}
    images: dict[Path, Image.Image] = {}
    confidences: list[float] = []
    contrasts: list[float] = []
    fallback_count = 0
    sampled_palette_fallback_count = 0
    started = perf_counter()

    for track in region_tracks:
        track.pop("overlay_layout", None)
        timestamp, bbox = _select_style_sample(track)
        timestamp_key = round(timestamp * 1_000_000)
        frame_record = frames.get(timestamp_key)
        if frame_record is None:
            frame_path = frame_dir / f"frame-{timestamp_key:012d}.png"
            frame_record = extract_frame_at_timestamp(video_path, frame_path, timestamp)
            frames[timestamp_key] = frame_record
        frame_path, _reused = frame_record
        image = images.get(frame_path)
        if image is None:
            with Image.open(frame_path) as opened:
                image = opened.convert("RGB")
            images[frame_path] = image
        style = estimate_overlay_style(image, bbox, options=resolved_options)
        style["sample_timestamp_seconds"] = timestamp
        track["overlay_style"] = style
        confidences.append(float(style["confidence"]))
        contrasts.append(float(style["contrast_ratio"]))
        fallback_count += int(style["fallback_used"] is True)
        sampled_palette_fallback_count += int(
            style["fallback_used"] is True and float(style["background_opacity"]) == 1.0
        )

    current_run_latency_ms = (perf_counter() - started) * 1000.0
    extracted_count = sum(not reused for _path, reused in frames.values())
    reused_count = len(frames) - extracted_count
    result["region_styling"] = {
        "schema_version": 1,
        "algorithm": "source-palette-v2",
        "sample_policy": "highest-confidence-stable-keyframe-v1",
        "video_size_bytes": video_stat.st_size,
        "video_modified_ns": video_stat.st_mtime_ns,
        "options": {
            "background_padding_ratio": resolved_options.background_padding_ratio,
            "minimum_contrast_ratio": resolved_options.minimum_contrast_ratio,
            "minimum_style_confidence": resolved_options.minimum_style_confidence,
        },
        "styled_region_track_count": len(region_tracks),
        "fallback_style_count": fallback_count,
        "sampled_palette_fallback_count": sampled_palette_fallback_count,
        "fixed_palette_fallback_count": fallback_count - sampled_palette_fallback_count,
        "mean_style_confidence": _mean(confidences),
        "median_contrast_ratio": float(median(contrasts)) if contrasts else 0.0,
        "unique_sample_frame_count": len(frames),
        "extracted_frame_count": extracted_count,
        "reused_frame_count": reused_count,
        "current_run_latency_ms": current_run_latency_ms,
    }
    metrics = cast(dict[str, Any], result["metrics"])
    for key in tuple(metrics):
        if str(key).startswith("layout_"):
            metrics.pop(key)
    metrics["style_styled_region_track_count"] = len(region_tracks)
    metrics["style_fallback_count"] = fallback_count
    metrics["style_fixed_palette_fallback_count"] = fallback_count - sampled_palette_fallback_count
    metrics["style_mean_confidence"] = _mean(confidences)
    metrics["style_current_run_latency_ms"] = current_run_latency_ms
    validate_track_manifest(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tracks.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_video_viewer(output_dir / "viewer.html", video_path, result)
    return result


def estimate_overlay_style(
    image: Image.Image,
    bbox: dict[str, Any],
    *,
    options: OverlayStyleOptions | None = None,
) -> dict[str, Any]:
    resolved_options = options or OverlayStyleOptions()
    rgb = image.convert("RGB")
    width, height = rgb.size
    left, top, right, bottom = _pixel_box(bbox, width, height)
    box_height = bottom - top
    padding = max(2, round(box_height * resolved_options.background_padding_ratio))
    outer_left = max(0, left - padding)
    outer_top = max(0, top - padding)
    outer_right = min(width, right + padding)
    outer_bottom = min(height, bottom + padding)
    pixels = rgb.load()
    assert pixels is not None

    background_samples = [
        cast(tuple[int, int, int], pixels[x, y])
        for y in range(outer_top, outer_bottom)
        for x in range(outer_left, outer_right)
        if x < left or x >= right or y < top or y >= bottom
    ]
    if not background_samples:
        background_samples = _perimeter_pixels(rgb, left, top, right, bottom)
    background, dominant_fraction = _dominant_color(background_samples)
    background_distances = [_color_distance(item, background) for item in background_samples]
    # The padding ring can contain nearby labels or icons. Its dominant color is still the
    # useful background; the median distance avoids treating those neighboring foreground
    # pixels as background noise.
    background_noise = _percentile(background_distances, 0.5)
    foreground_threshold = max(28.0, background_noise * 1.75)

    core: list[tuple[int, int, tuple[int, int, int], float]] = []
    for y in range(top, bottom):
        for x in range(left, right):
            color = cast(tuple[int, int, int], pixels[x, y])
            core.append((x, y, color, _color_distance(color, background)))
    foreground = [item for item in core if item[3] >= foreground_threshold]
    foreground_ratio = len(foreground) / len(core) if core else 0.0
    valid_foreground = 0.005 <= foreground_ratio <= 0.55
    background_uniformity = max(0.0, 1.0 - min(1.0, background_noise / 80.0))
    occupancy_confidence = max(0.0, 1.0 - abs(foreground_ratio - 0.1) / 0.2)
    confidence = (
        background_uniformity * 0.55
        + min(1.0, dominant_fraction * 2.0) * 0.25
        + occupancy_confidence * 0.2
    )

    if not valid_foreground:
        return _fallback_style(
            bbox,
            confidence,
            reason="no-distinct-foreground",
            sampled_background=background,
            background_uniformity=background_uniformity,
            foreground_ratio=foreground_ratio,
            use_sampled_palette=confidence >= resolved_options.minimum_style_confidence,
        )

    strongest = sorted(foreground, key=lambda item: item[3], reverse=True)
    strongest_count = max(1, math.ceil(len(strongest) * 0.5))
    foreground_color = _median_color([item[2] for item in strongest[:strongest_count]])
    contrast = _contrast_ratio(foreground_color, background)
    foreground_adjusted = False
    if contrast < resolved_options.minimum_contrast_ratio:
        black = (0, 0, 0)
        white = (255, 255, 255)
        black_contrast = _contrast_ratio(black, background)
        white_contrast = _contrast_ratio(white, background)
        foreground_color, contrast = (
            (black, black_contrast) if black_contrast >= white_contrast else (white, white_contrast)
        )
        foreground_adjusted = True
        confidence *= 0.85

    glyph_height = _robust_glyph_height(foreground)
    glyph_height_ratio = glyph_height / height
    bbox_height_ratio = box_height / height
    font_size_ratio = min(
        bbox_height_ratio * 0.95,
        max(bbox_height_ratio * 0.55, glyph_height_ratio * 1.15),
    )
    fallback_used = confidence < resolved_options.minimum_style_confidence
    if fallback_used:
        return _fallback_style(
            bbox,
            confidence,
            reason="low-style-confidence",
            sampled_background=background,
            background_uniformity=background_uniformity,
            foreground_ratio=foreground_ratio,
            use_sampled_palette=False,
        )

    return {
        "schema_version": 1,
        "algorithm": "source-palette-v2",
        "foreground_color": _hex_color(foreground_color),
        "background_color": _hex_color(background),
        "background_opacity": 1.0,
        "font_size_ratio": font_size_ratio,
        "font_weight": 500 if foreground_ratio < 0.18 else 600,
        "line_height": 1.05,
        "border_radius_px": 0.0,
        "text_shadow": "none",
        "confidence": confidence,
        "background_uniformity": background_uniformity,
        "foreground_pixel_ratio": foreground_ratio,
        "contrast_ratio": contrast,
        "foreground_adjusted": foreground_adjusted,
        "fallback_used": False,
        "fallback_reason": None,
        "glyph_height_ratio": glyph_height_ratio,
        "glyph_height_method": "column-span-p80-v1",
    }


def _select_style_sample(track: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    raw_keyframes = track.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError(f"Region track {track.get('region_track_id')} requires style keyframes.")
    midpoint = (float(track["start_seconds"]) + float(track["end_seconds"])) / 2.0

    def score(item: dict[str, Any]) -> tuple[int, float, float]:
        phase = str(item.get("phase", ""))
        stable_rank = 2 if phase == "post" else 1 if phase in {"pre", ""} else 0
        confidence = float(item.get("ocr_confidence", 0.0))
        distance = abs(float(item["timestamp_seconds"]) - midpoint)
        return stable_rank, confidence, -distance

    selected = max((cast(dict[str, Any], item) for item in raw_keyframes), key=score)
    bbox = selected.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"Region track {track.get('region_track_id')} style sample has no bbox.")
    return float(selected["timestamp_seconds"]), cast(dict[str, Any], bbox)


def _resolve_video_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    video = cast(dict[str, Any], manifest["video"])
    path = Path(str(video["path"]))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _pixel_box(
    bbox: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(image_width - 1, math.floor(float(bbox["x"]) * image_width)))
    top = max(0, min(image_height - 1, math.floor(float(bbox["y"]) * image_height)))
    right = max(
        left + 1,
        min(image_width, math.ceil((float(bbox["x"]) + float(bbox["width"])) * image_width)),
    )
    bottom = max(
        top + 1,
        min(image_height, math.ceil((float(bbox["y"]) + float(bbox["height"])) * image_height)),
    )
    return left, top, right, bottom


def _perimeter_pixels(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> list[tuple[int, int, int]]:
    pixels = image.load()
    assert pixels is not None
    result: list[tuple[int, int, int]] = []
    for x in range(left, right):
        result.append(cast(tuple[int, int, int], pixels[x, top]))
        if bottom - 1 != top:
            result.append(cast(tuple[int, int, int], pixels[x, bottom - 1]))
    for y in range(top + 1, bottom - 1):
        result.append(cast(tuple[int, int, int], pixels[left, y]))
        if right - 1 != left:
            result.append(cast(tuple[int, int, int], pixels[right - 1, y]))
    return result


def _dominant_color(
    colors: list[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], float]:
    if not colors:
        return (11, 13, 17), 0.0
    bins = Counter(tuple(channel // _COLOR_QUANTIZATION for channel in color) for color in colors)
    dominant_bin, count = bins.most_common(1)[0]
    cluster = [
        color
        for color in colors
        if tuple(channel // _COLOR_QUANTIZATION for channel in color) == dominant_bin
    ]
    return _median_color(cluster), count / len(colors)


def _median_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return tuple(round(median(color[index] for color in colors)) for index in range(3))  # type: ignore[return-value]


def _color_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _contrast_ratio(foreground: tuple[int, int, int], background: tuple[int, int, int]) -> float:
    foreground_luminance = _relative_luminance(foreground)
    background_luminance = _relative_luminance(background)
    lighter = max(foreground_luminance, background_luminance)
    darker = min(foreground_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(color: tuple[int, int, int]) -> float:
    channels = []
    for value in color:
        normalized = value / 255.0
        channels.append(
            normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4
        )
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722


def _fallback_style(
    bbox: dict[str, Any],
    confidence: float,
    *,
    reason: str,
    sampled_background: tuple[int, int, int],
    background_uniformity: float,
    foreground_ratio: float,
    use_sampled_palette: bool,
) -> dict[str, Any]:
    background = sampled_background if use_sampled_palette else (11, 13, 17)
    black = (0, 0, 0)
    white = (255, 255, 255)
    black_contrast = _contrast_ratio(black, background)
    white_contrast = _contrast_ratio(white, background)
    foreground, contrast = (
        (black, black_contrast) if black_contrast >= white_contrast else (white, white_contrast)
    )
    return {
        "schema_version": 1,
        "algorithm": "source-palette-v2",
        "foreground_color": _hex_color(foreground),
        "background_color": _hex_color(background),
        "background_opacity": 1.0 if use_sampled_palette else 0.88,
        "font_size_ratio": float(bbox["height"]) * 0.8,
        "font_weight": 500,
        "line_height": 1.15,
        "border_radius_px": 2.0,
        "text_shadow": "0 1px 1px #000000",
        "confidence": max(0.0, min(1.0, confidence)),
        "background_uniformity": background_uniformity,
        "foreground_pixel_ratio": foreground_ratio,
        "contrast_ratio": contrast,
        "foreground_adjusted": False,
        "fallback_used": True,
        "fallback_reason": reason,
    }


def _robust_glyph_height(
    foreground: list[tuple[int, int, tuple[int, int, int], float]],
) -> float:
    vertical_extents: dict[int, tuple[int, int]] = {}
    for x, y, _color, _distance in foreground:
        current = vertical_extents.get(x)
        if current is None:
            vertical_extents[x] = (y, y)
        else:
            vertical_extents[x] = (min(current[0], y), max(current[1], y))
    spans = [float(bottom - top + 1) for top, bottom in vertical_extents.values()]
    return max(1.0, _percentile(spans, 0.8))


def _hex_color(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in color)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
