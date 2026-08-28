from __future__ import annotations

import json
import math
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from text_localizer.domain import NormalizedBox
from text_localizer.track_editor import validate_track_manifest


@dataclass(frozen=True, slots=True)
class ReplacementQualityGates:
    minimum_frame_pass_rate: float = 1.0
    minimum_region_recall: float = 1.0
    minimum_bbox_iou: float = 0.75
    minimum_source_coverage_ratio: float = 0.98
    maximum_active_region_count_error: int = 0
    maximum_overlay_collisions: int = 0


@dataclass(frozen=True, slots=True)
class ReplacementTextMatch:
    field: str
    mode: str
    patterns: tuple[str, ...]

    def matches(self, region: dict[str, Any]) -> bool:
        normalized_patterns = tuple(_normalize_text(pattern) for pattern in self.patterns)
        for candidate in _region_text_values(region, self.field):
            normalized_candidate = _normalize_text(candidate)
            for pattern in normalized_patterns:
                if self.mode == "exact" and normalized_candidate == pattern:
                    return True
                if self.mode == "contains" and pattern in normalized_candidate:
                    return True
        return False


@dataclass(frozen=True, slots=True)
class ExpectedReplacementRegion:
    region_id: str
    label: str
    text_match: ReplacementTextMatch
    source_bbox: NormalizedBox


@dataclass(frozen=True, slots=True)
class ReplacementFrame:
    frame_id: str
    label: str
    timestamp_seconds: float
    expected_active_region_count: int
    regions: tuple[ExpectedReplacementRegion, ...]


@dataclass(frozen=True, slots=True)
class ReplacementBenchmark:
    name: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    gates: ReplacementQualityGates
    frames: tuple[ReplacementFrame, ...]


@dataclass(frozen=True, slots=True)
class _ActiveRegion:
    track_id: str
    region: dict[str, Any]
    source_bbox: NormalizedBox
    erase_bbox: NormalizedBox
    target_bbox: NormalizedBox
    replacement_algorithm: str | None


def load_replacement_benchmark(path: Path) -> ReplacementBenchmark:
    if not path.is_file():
        raise FileNotFoundError(f"Replacement benchmark does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Replacement benchmark is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Replacement benchmark root must be a mapping.")
    return parse_replacement_benchmark(cast(dict[str, Any], payload))


def parse_replacement_benchmark(payload: dict[str, Any]) -> ReplacementBenchmark:
    if payload.get("schema_version") != 1:
        raise ValueError("Replacement benchmark schema_version must be 1.")
    name = _nonblank_string(payload.get("name"), "name")
    video = _mapping(payload.get("video"), "video")
    duration = _positive_number(video.get("duration_seconds"), "video.duration_seconds")
    fps = _positive_number(video.get("fps"), "video.fps")
    width = _positive_integer(video.get("width"), "video.width")
    height = _positive_integer(video.get("height"), "video.height")
    gates = _parse_gates(payload.get("gates"))

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("Replacement benchmark requires at least one frame.")
    frames: list[ReplacementFrame] = []
    frame_ids: set[str] = set()
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _parse_frame(raw_frame, frame_index, duration)
        if frame.frame_id in frame_ids:
            raise ValueError(f"Duplicate replacement frame id: {frame.frame_id}")
        frame_ids.add(frame.frame_id)
        frames.append(frame)
    return ReplacementBenchmark(name, duration, fps, width, height, gates, tuple(frames))


def evaluate_replacement_quality(
    manifest: dict[str, Any], benchmark: ReplacementBenchmark
) -> dict[str, Any]:
    validate_track_manifest(manifest)
    if manifest.get("schema_version") != 2:
        raise ValueError("Replacement quality requires schema-v2 region tracks.")
    _validate_video_identity(manifest, benchmark)

    frame_reports: list[dict[str, Any]] = []
    matched_regions = 0
    bbox_ious: list[float] = []
    source_coverages: list[float] = []
    active_region_count_error = 0
    collision_count = 0

    for frame in benchmark.frames:
        active = _active_regions(manifest, frame.timestamp_seconds)
        report = _evaluate_frame(frame, active, benchmark.gates)
        frame_reports.append(report)
        matched_regions += int(report["matched_region_count"])
        bbox_ious.extend(float(value) for value in report["matched_bbox_ious"])
        source_coverages.extend(float(value) for value in report["matched_source_coverages"])
        active_region_count_error += int(report["active_region_count_error"])
        collision_count += int(report["overlay_collision_count"])

    frame_count = len(benchmark.frames)
    passed_frame_count = sum(bool(report["frame_passed"]) for report in frame_reports)
    expected_region_count = sum(len(frame.regions) for frame in benchmark.frames)
    frame_pass_rate = passed_frame_count / frame_count
    region_recall = matched_regions / expected_region_count if expected_region_count else 0.0
    bbox_iou_minimum = min(bbox_ious) if bbox_ious else None
    bbox_iou_median = statistics.median(bbox_ious) if bbox_ious else None
    source_coverage_minimum = min(source_coverages) if source_coverages else None

    gate_results = {
        "frame_pass_rate": frame_pass_rate >= benchmark.gates.minimum_frame_pass_rate,
        "region_recall": region_recall >= benchmark.gates.minimum_region_recall,
        "bbox_iou": (
            bbox_iou_minimum is not None
            and bbox_iou_minimum >= benchmark.gates.minimum_bbox_iou
        ),
        "source_coverage": (
            source_coverage_minimum is not None
            and source_coverage_minimum >= benchmark.gates.minimum_source_coverage_ratio
        ),
        "active_region_count": (
            active_region_count_error <= benchmark.gates.maximum_active_region_count_error
        ),
        "overlay_collisions": (
            collision_count <= benchmark.gates.maximum_overlay_collisions
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": benchmark.name,
        "track_manifest_schema_version": manifest["schema_version"],
        "summary": {
            "frame_count": frame_count,
            "passed_frame_count": passed_frame_count,
            "frame_pass_rate": frame_pass_rate,
            "expected_region_count": expected_region_count,
            "matched_region_count": matched_regions,
            "region_recall": region_recall,
            "bbox_iou_minimum": bbox_iou_minimum,
            "bbox_iou_median": bbox_iou_median,
            "source_coverage_minimum": source_coverage_minimum,
            "active_region_count_error": active_region_count_error,
            "overlay_collision_count": collision_count,
            "gates_passed": all(gate_results.values()),
        },
        "gates": asdict(benchmark.gates),
        "gate_results": gate_results,
        "frames": frame_reports,
    }


def evaluate_replacement_quality_files(
    manifest_path: Path,
    benchmark_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Track manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Track manifest root must be a mapping.")
    report = evaluate_replacement_quality(
        cast(dict[str, Any], payload), load_replacement_benchmark(benchmark_path)
    )
    report["inputs"] = {
        "track_manifest": str(manifest_path.resolve()),
        "benchmark": str(benchmark_path.resolve()),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _evaluate_frame(
    frame: ReplacementFrame,
    active: tuple[_ActiveRegion, ...],
    gates: ReplacementQualityGates,
) -> dict[str, Any]:
    region_reports: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    matched_coverages: list[float] = []
    used_track_ids: set[str] = set()
    for expected in frame.regions:
        candidates = [
            item
            for item in active
            if item.track_id not in used_track_ids and expected.text_match.matches(item.region)
        ]
        if not candidates:
            region_reports.append(
                {
                    "id": expected.region_id,
                    "label": expected.label,
                    "matched": False,
                    "track_id": None,
                    "expected_source_bbox": _box_dict(expected.source_bbox),
                    "observed_source_bbox": None,
                    "erase_bbox": None,
                    "bbox_iou": None,
                    "source_coverage_ratio": None,
                    "replacement_algorithm": None,
                    "passed": False,
                }
            )
            continue
        match = max(candidates, key=lambda item: _box_iou(expected.source_bbox, item.source_bbox))
        used_track_ids.add(match.track_id)
        bbox_iou = _box_iou(expected.source_bbox, match.source_bbox)
        source_coverage = _box_coverage(expected.source_bbox, match.erase_bbox)
        matched_ious.append(bbox_iou)
        matched_coverages.append(source_coverage)
        passed = (
            bbox_iou >= gates.minimum_bbox_iou
            and source_coverage >= gates.minimum_source_coverage_ratio
            and match.replacement_algorithm is not None
        )
        region_reports.append(
            {
                "id": expected.region_id,
                "label": expected.label,
                "matched": True,
                "track_id": match.track_id,
                "expected_source_bbox": _box_dict(expected.source_bbox),
                "observed_source_bbox": _box_dict(match.source_bbox),
                "erase_bbox": _box_dict(match.erase_bbox),
                "bbox_iou": bbox_iou,
                "source_coverage_ratio": source_coverage,
                "replacement_algorithm": match.replacement_algorithm,
                "passed": passed,
            }
        )

    count_error = abs(len(active) - frame.expected_active_region_count)
    collisions = _overlay_collisions(active)
    frame_passed = (
        all(bool(report["passed"]) for report in region_reports)
        and count_error <= gates.maximum_active_region_count_error
        and len(collisions) <= gates.maximum_overlay_collisions
    )
    return {
        "id": frame.frame_id,
        "label": frame.label,
        "timestamp_seconds": frame.timestamp_seconds,
        "expected_active_region_count": frame.expected_active_region_count,
        "active_region_count": len(active),
        "active_region_count_error": count_error,
        "expected_region_count": len(frame.regions),
        "matched_region_count": len(matched_ious),
        "matched_bbox_ious": matched_ious,
        "matched_source_coverages": matched_coverages,
        "overlay_collision_count": len(collisions),
        "overlay_collisions": collisions,
        "frame_passed": frame_passed,
        "regions": region_reports,
    }


def _active_regions(manifest: dict[str, Any], timestamp: float) -> tuple[_ActiveRegion, ...]:
    active: list[_ActiveRegion] = []
    for raw_track in cast(list[Any], manifest["region_tracks"]):
        track = cast(dict[str, Any], raw_track)
        if not float(track["start_seconds"]) <= timestamp < float(track["end_seconds"]):
            continue
        region = cast(dict[str, Any], track["region"])
        if region.get("action") != "translate" or region.get("needs_review") is True:
            continue
        if (
            not isinstance(region.get("translated_text"), str)
            or not region["translated_text"].strip()
        ):
            continue
        source_bbox = _track_box(track, timestamp)
        replacement = track.get("overlay_replacement")
        replacement_mapping = replacement if isinstance(replacement, dict) else None
        erase_bbox = _erase_box(source_bbox, replacement_mapping)
        target_bbox = _target_box(source_bbox, track.get("overlay_layout"))
        algorithm = None
        if replacement_mapping is not None and isinstance(
            replacement_mapping.get("algorithm"), str
        ):
            algorithm = str(replacement_mapping["algorithm"])
        active.append(
            _ActiveRegion(
                str(track["region_track_id"]),
                region,
                source_bbox,
                erase_bbox,
                target_bbox,
                algorithm,
            )
        )
    return tuple(active)


def _track_box(track: dict[str, Any], timestamp: float) -> NormalizedBox:
    keyframes = [cast(dict[str, Any], item) for item in cast(list[Any], track["keyframes"])]
    current_index = 0
    for index, keyframe in enumerate(keyframes):
        if float(keyframe["timestamp_seconds"]) <= timestamp:
            current_index = index
        else:
            break
    current = keyframes[current_index]
    current_box = _parse_box(current.get("bbox"), "region keyframe bbox")
    if current.get("interpolation") != "linear" or current_index + 1 >= len(keyframes):
        return current_box
    following = keyframes[current_index + 1]
    start = float(current["timestamp_seconds"])
    end = float(following["timestamp_seconds"])
    if end <= start:
        return current_box
    fraction = min(1.0, max(0.0, (timestamp - start) / (end - start)))
    following_box = _parse_box(following.get("bbox"), "region keyframe bbox")
    return NormalizedBox(
        x=current_box.x + (following_box.x - current_box.x) * fraction,
        y=current_box.y + (following_box.y - current_box.y) * fraction,
        width=current_box.width + (following_box.width - current_box.width) * fraction,
        height=current_box.height + (following_box.height - current_box.height) * fraction,
    )


def _erase_box(source: NormalizedBox, replacement: dict[str, Any] | None) -> NormalizedBox:
    if replacement is None:
        return source
    left = _ratio(replacement.get("erase_padding_left_ratio", 0.0), "erase padding left")
    right = _ratio(replacement.get("erase_padding_right_ratio", 0.0), "erase padding right")
    top = _ratio(replacement.get("erase_padding_top_ratio", 0.0), "erase padding top")
    bottom = _ratio(replacement.get("erase_padding_bottom_ratio", 0.0), "erase padding bottom")
    x = max(0.0, source.x - left)
    y = max(0.0, source.y - top)
    right_edge = min(1.0, source.x + source.width + right)
    bottom_edge = min(1.0, source.y + source.height + bottom)
    return NormalizedBox(x, y, right_edge - x, bottom_edge - y)


def _target_box(source: NormalizedBox, raw_layout: Any) -> NormalizedBox:
    if not isinstance(raw_layout, dict):
        return source
    left = _ratio(raw_layout.get("left_extension_ratio", 0.0), "layout left extension")
    right = _ratio(raw_layout.get("right_extension_ratio", 0.0), "layout right extension")
    top = _ratio(raw_layout.get("top_extension_ratio", 0.0), "layout top extension")
    bottom = _ratio(raw_layout.get("bottom_extension_ratio", 0.0), "layout bottom extension")
    x = max(0.0, source.x - left)
    y = max(0.0, source.y - top)
    right_edge = min(1.0, source.x + source.width + right)
    bottom_edge = min(1.0, source.y + source.height + bottom)
    return NormalizedBox(x, y, right_edge - x, bottom_edge - y)


def _overlay_collisions(active: tuple[_ActiveRegion, ...]) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            overlap = _intersection_area(left.target_bbox, right.target_bbox)
            if overlap <= 1e-9:
                continue
            collisions.append(
                {
                    "left_track_id": left.track_id,
                    "right_track_id": right.track_id,
                    "intersection_ratio": overlap,
                }
            )
    return collisions


def _parse_gates(raw: Any) -> ReplacementQualityGates:
    values = {} if raw is None else _mapping(raw, "gates")
    return ReplacementQualityGates(
        minimum_frame_pass_rate=_ratio(
            values.get("minimum_frame_pass_rate", 1.0), "gates.minimum_frame_pass_rate"
        ),
        minimum_region_recall=_ratio(
            values.get("minimum_region_recall", 1.0), "gates.minimum_region_recall"
        ),
        minimum_bbox_iou=_ratio(
            values.get("minimum_bbox_iou", 0.75), "gates.minimum_bbox_iou"
        ),
        minimum_source_coverage_ratio=_ratio(
            values.get("minimum_source_coverage_ratio", 0.98),
            "gates.minimum_source_coverage_ratio",
        ),
        maximum_active_region_count_error=_nonnegative_integer(
            values.get("maximum_active_region_count_error", 0),
            "gates.maximum_active_region_count_error",
        ),
        maximum_overlay_collisions=_nonnegative_integer(
            values.get("maximum_overlay_collisions", 0),
            "gates.maximum_overlay_collisions",
        ),
    )


def _parse_frame(raw: Any, index: int, duration: float) -> ReplacementFrame:
    values = _mapping(raw, f"frames[{index}]")
    frame_id = _nonblank_string(values.get("id"), f"frames[{index}].id")
    label = _nonblank_string(values.get("label"), f"frames[{index}].label")
    timestamp = _nonnegative_number(
        values.get("timestamp_seconds"), f"frames[{index}].timestamp_seconds"
    )
    if timestamp > duration:
        raise ValueError(f"Replacement frame {frame_id} is outside the video duration.")
    expected_count = _nonnegative_integer(
        values.get("expected_active_region_count"),
        f"frames[{index}].expected_active_region_count",
    )
    raw_regions = values.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError(f"Replacement frame {frame_id} requires regions.")
    regions: list[ExpectedReplacementRegion] = []
    region_ids: set[str] = set()
    for region_index, raw_region in enumerate(raw_regions):
        region = _parse_region(raw_region, index, region_index)
        if region.region_id in region_ids:
            raise ValueError(f"Duplicate replacement region id in {frame_id}: {region.region_id}")
        region_ids.add(region.region_id)
        regions.append(region)
    return ReplacementFrame(frame_id, label, timestamp, expected_count, tuple(regions))


def _parse_region(raw: Any, frame_index: int, region_index: int) -> ExpectedReplacementRegion:
    prefix = f"frames[{frame_index}].regions[{region_index}]"
    values = _mapping(raw, prefix)
    region_id = _nonblank_string(values.get("id"), f"{prefix}.id")
    label = _nonblank_string(values.get("label"), f"{prefix}.label")
    raw_match = _mapping(values.get("match"), f"{prefix}.match")
    field = _nonblank_string(raw_match.get("field"), f"{prefix}.match.field")
    if field not in {"source_text", "translated_text", "any_text"}:
        raise ValueError(f"Replacement region {region_id} has an unsupported match field.")
    mode = _nonblank_string(raw_match.get("mode", "contains"), f"{prefix}.match.mode")
    if mode not in {"exact", "contains"}:
        raise ValueError(f"Replacement region {region_id} has an unsupported match mode.")
    raw_patterns = raw_match.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise ValueError(f"Replacement region {region_id} requires match.patterns.")
    patterns = tuple(_nonblank_string(item, f"{prefix}.match.patterns") for item in raw_patterns)
    return ExpectedReplacementRegion(
        region_id,
        label,
        ReplacementTextMatch(field, mode, patterns),
        _parse_box(values.get("source_bbox"), f"{prefix}.source_bbox"),
    )


def _validate_video_identity(manifest: dict[str, Any], benchmark: ReplacementBenchmark) -> None:
    video = cast(dict[str, Any], manifest["video"])
    comparisons = (
        (float(video["duration_seconds"]), benchmark.duration_seconds, 0.001, "duration"),
        (float(video["fps"]), benchmark.fps, 0.01, "FPS"),
        (int(video["width"]), benchmark.width, 0, "width"),
        (int(video["height"]), benchmark.height, 0, "height"),
    )
    for actual, expected, tolerance, label in comparisons:
        if not math.isclose(actual, expected, abs_tol=tolerance):
            raise ValueError(f"Track manifest {label} does not match replacement benchmark video.")


def _region_text_values(region: dict[str, Any], field: str) -> tuple[str, ...]:
    source_keys = ("source_text", "normalized_source_text", "raw_source_text")
    translated_keys = ("translated_text",)
    keys: tuple[str, ...] = source_keys if field == "source_text" else translated_keys
    if field == "any_text":
        keys = source_keys + translated_keys
    return tuple(value for key in keys if isinstance((value := region.get(key)), str))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(normalized.split())


def _box_iou(left: NormalizedBox, right: NormalizedBox) -> float:
    intersection = _intersection_area(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _box_coverage(source: NormalizedBox, cover: NormalizedBox) -> float:
    area = source.width * source.height
    return _intersection_area(source, cover) / area if area > 0 else 0.0


def _intersection_area(left: NormalizedBox, right: NormalizedBox) -> float:
    width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    height = max(
        0.0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    return width * height


def _parse_box(raw: Any, label: str) -> NormalizedBox:
    values = _mapping(raw, label)
    try:
        return NormalizedBox(
            x=float(values["x"]),
            y=float(values["y"]),
            width=float(values["width"]),
            height=float(values["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a normalized x/y/width/height box.") from exc


def _box_dict(box: NormalizedBox) -> dict[str, float]:
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], raw)


def _nonblank_string(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return raw.strip()


def _positive_number(raw: Any, label: str) -> float:
    value = _nonnegative_number(raw, label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return value


def _nonnegative_number(raw: Any, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"{label} must be a finite number.")
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return value


def _ratio(raw: Any, label: str) -> float:
    value = _nonnegative_number(raw, label)
    if value > 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return value


def _positive_integer(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(raw)


def _nonnegative_integer(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return int(raw)
