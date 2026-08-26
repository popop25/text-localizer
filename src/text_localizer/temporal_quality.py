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
class TemporalQualityGates:
    minimum_annotation_recall: float = 0.95
    minimum_keyframe_recall: float = 0.95
    maximum_boundary_error_ms_p95: float = 67.0
    minimum_bbox_iou_median: float = 0.75
    maximum_flicker_events: int = 0
    minimum_annotation_coverage_ratio: float = 0.95
    flicker_tolerance_seconds: float = 0.067
    temporal_search_margin_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class TextMatch:
    field: str
    mode: str
    patterns: tuple[str, ...]

    def matches(self, region: dict[str, Any]) -> bool:
        candidates = _region_text_values(region, self.field)
        normalized_patterns = tuple(_normalize_text(pattern) for pattern in self.patterns)
        for candidate in candidates:
            normalized_candidate = _normalize_text(candidate)
            for pattern in normalized_patterns:
                if self.mode == "exact" and normalized_candidate == pattern:
                    return True
                if self.mode == "contains" and pattern in normalized_candidate:
                    return True
        return False


@dataclass(frozen=True, slots=True)
class ExpectedKeyframe:
    timestamp_seconds: float
    bbox: NormalizedBox


@dataclass(frozen=True, slots=True)
class TemporalAnnotation:
    annotation_id: str
    label: str
    start_seconds: float
    end_seconds: float
    text_match: TextMatch
    keyframes: tuple[ExpectedKeyframe, ...]
    evaluate_boundaries: bool


@dataclass(frozen=True, slots=True)
class TemporalBenchmark:
    name: str
    duration_seconds: float
    fps: float
    gates: TemporalQualityGates
    annotations: tuple[TemporalAnnotation, ...]


@dataclass(frozen=True, slots=True)
class _Observation:
    track_id: str
    region_index: int
    start_seconds: float
    end_seconds: float
    bbox: NormalizedBox
    region: dict[str, Any]


def load_temporal_benchmark(path: Path) -> TemporalBenchmark:
    if not path.is_file():
        raise FileNotFoundError(f"Temporal benchmark does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Temporal benchmark is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Temporal benchmark root must be a mapping.")
    return parse_temporal_benchmark(cast(dict[str, Any], raw))


def parse_temporal_benchmark(payload: dict[str, Any]) -> TemporalBenchmark:
    if payload.get("schema_version") != 1:
        raise ValueError("Temporal benchmark schema_version must be 1.")
    name = _nonblank_string(payload.get("name"), "name")
    video = _mapping(payload.get("video"), "video")
    duration_seconds = _positive_number(video.get("duration_seconds"), "video.duration_seconds")
    fps = _positive_number(video.get("fps"), "video.fps")
    gates = _parse_gates(payload.get("gates"))

    raw_annotations = payload.get("annotations")
    if not isinstance(raw_annotations, list) or not raw_annotations:
        raise ValueError("Temporal benchmark requires at least one annotation.")
    annotations: list[TemporalAnnotation] = []
    annotation_ids: set[str] = set()
    for index, raw_annotation in enumerate(raw_annotations):
        annotation = _parse_annotation(raw_annotation, index, duration_seconds)
        if annotation.annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation id: {annotation.annotation_id}")
        annotation_ids.add(annotation.annotation_id)
        annotations.append(annotation)

    return TemporalBenchmark(
        name=name,
        duration_seconds=duration_seconds,
        fps=fps,
        gates=gates,
        annotations=tuple(annotations),
    )


def evaluate_temporal_quality(
    manifest: dict[str, Any],
    benchmark: TemporalBenchmark,
) -> dict[str, Any]:
    validate_track_manifest(manifest)
    _validate_video_identity(manifest, benchmark)
    observations = _manifest_observations(manifest)

    annotation_reports: list[dict[str, Any]] = []
    matched_annotations = 0
    matched_keyframes = 0
    keyframe_count = 0
    flicker_events = 0
    boundary_errors_ms: list[float] = []
    bbox_ious: list[float] = []

    for annotation in benchmark.annotations:
        report = _evaluate_annotation(annotation, observations, benchmark.gates)
        annotation_reports.append(report)
        matched_annotations += int(report["coverage_passed"])
        matched_keyframes += int(report["matched_keyframe_count"])
        keyframe_count += int(report["keyframe_count"])
        flicker_events += int(report["flicker_event_count"])
        bbox_ious.extend(float(item) for item in report["matched_bbox_ious"])
        for field in ("onset_error_ms", "offset_error_ms"):
            error = report[field]
            if error is not None:
                boundary_errors_ms.append(abs(float(error)))

    annotation_count = len(benchmark.annotations)
    annotation_recall = matched_annotations / annotation_count
    keyframe_recall = matched_keyframes / keyframe_count if keyframe_count else 0.0
    boundary_error_p95 = _percentile(boundary_errors_ms, 0.95)
    bbox_iou_median = statistics.median(bbox_ious) if bbox_ious else None

    gate_results = {
        "annotation_recall": (annotation_recall >= benchmark.gates.minimum_annotation_recall),
        "keyframe_recall": keyframe_recall >= benchmark.gates.minimum_keyframe_recall,
        "boundary_error_ms_p95": (
            boundary_error_p95 is None
            or boundary_error_p95 <= benchmark.gates.maximum_boundary_error_ms_p95
        ),
        "bbox_iou_median": (
            bbox_iou_median is not None
            and bbox_iou_median >= benchmark.gates.minimum_bbox_iou_median
        ),
        "flicker_events": flicker_events <= benchmark.gates.maximum_flicker_events,
    }
    gates_passed = all(gate_results.values())

    return {
        "schema_version": 1,
        "benchmark": benchmark.name,
        "track_manifest_schema_version": manifest["schema_version"],
        "summary": {
            "annotation_count": annotation_count,
            "matched_annotation_count": matched_annotations,
            "annotation_recall": annotation_recall,
            "keyframe_count": keyframe_count,
            "matched_keyframe_count": matched_keyframes,
            "keyframe_recall": keyframe_recall,
            "boundary_sample_count": len(boundary_errors_ms),
            "boundary_error_ms_p95": boundary_error_p95,
            "bbox_iou_median": bbox_iou_median,
            "flicker_event_count": flicker_events,
            "gates_passed": gates_passed,
        },
        "gates": asdict(benchmark.gates),
        "gate_results": gate_results,
        "annotations": annotation_reports,
    }


def evaluate_temporal_quality_files(
    manifest_path: Path,
    benchmark_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track manifest does not exist: {manifest_path}")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Track manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(raw_manifest, dict):
        raise ValueError("Track manifest root must be a mapping.")

    report = evaluate_temporal_quality(
        cast(dict[str, Any], raw_manifest),
        load_temporal_benchmark(benchmark_path),
    )
    report["inputs"] = {
        "track_manifest": str(manifest_path.resolve()),
        "benchmark": str(benchmark_path.resolve()),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _parse_gates(raw: Any) -> TemporalQualityGates:
    if raw is None:
        return TemporalQualityGates()
    values = _mapping(raw, "gates")
    return TemporalQualityGates(
        minimum_annotation_recall=_ratio(
            values.get("minimum_annotation_recall", 0.95),
            "gates.minimum_annotation_recall",
        ),
        minimum_keyframe_recall=_ratio(
            values.get("minimum_keyframe_recall", 0.95),
            "gates.minimum_keyframe_recall",
        ),
        maximum_boundary_error_ms_p95=_nonnegative_number(
            values.get("maximum_boundary_error_ms_p95", 67.0),
            "gates.maximum_boundary_error_ms_p95",
        ),
        minimum_bbox_iou_median=_ratio(
            values.get("minimum_bbox_iou_median", 0.75),
            "gates.minimum_bbox_iou_median",
        ),
        maximum_flicker_events=_nonnegative_integer(
            values.get("maximum_flicker_events", 0),
            "gates.maximum_flicker_events",
        ),
        minimum_annotation_coverage_ratio=_ratio(
            values.get("minimum_annotation_coverage_ratio", 0.95),
            "gates.minimum_annotation_coverage_ratio",
        ),
        flicker_tolerance_seconds=_nonnegative_number(
            values.get("flicker_tolerance_seconds", 0.067),
            "gates.flicker_tolerance_seconds",
        ),
        temporal_search_margin_seconds=_nonnegative_number(
            values.get("temporal_search_margin_seconds", 1.0),
            "gates.temporal_search_margin_seconds",
        ),
    )


def _parse_annotation(raw: Any, index: int, duration: float) -> TemporalAnnotation:
    values = _mapping(raw, f"annotations[{index}]")
    annotation_id = _nonblank_string(values.get("id"), f"annotations[{index}].id")
    label = _nonblank_string(values.get("label"), f"annotations[{index}].label")
    start = _nonnegative_number(values.get("start_seconds"), f"annotations[{index}].start_seconds")
    end = _positive_number(values.get("end_seconds"), f"annotations[{index}].end_seconds")
    if end <= start or end > duration + 1e-6:
        raise ValueError(f"Annotation {annotation_id} has an invalid time range.")

    raw_match = _mapping(values.get("match"), f"annotations[{index}].match")
    field = _nonblank_string(raw_match.get("field"), f"annotations[{index}].match.field")
    if field not in {"source_text", "translated_text", "any_text"}:
        raise ValueError(f"Annotation {annotation_id} has an unsupported match field: {field}")
    mode = _nonblank_string(raw_match.get("mode", "contains"), f"annotations[{index}].match.mode")
    if mode not in {"exact", "contains"}:
        raise ValueError(f"Annotation {annotation_id} has an unsupported match mode: {mode}")
    raw_patterns = raw_match.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise ValueError(f"Annotation {annotation_id} requires match.patterns.")
    patterns = tuple(
        _nonblank_string(item, f"annotations[{index}].match.patterns") for item in raw_patterns
    )

    raw_keyframes = values.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError(f"Annotation {annotation_id} requires at least one keyframe.")
    keyframes: list[ExpectedKeyframe] = []
    previous_timestamp: float | None = None
    for keyframe_index, raw_keyframe in enumerate(raw_keyframes):
        keyframe = _mapping(
            raw_keyframe,
            f"annotations[{index}].keyframes[{keyframe_index}]",
        )
        timestamp = _nonnegative_number(
            keyframe.get("timestamp_seconds"),
            f"annotations[{index}].keyframes[{keyframe_index}].timestamp_seconds",
        )
        if timestamp < start or timestamp > end:
            raise ValueError(f"Annotation {annotation_id} keyframe is outside its time range.")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(f"Annotation {annotation_id} keyframes must be strictly ordered.")
        previous_timestamp = timestamp
        keyframes.append(
            ExpectedKeyframe(
                timestamp_seconds=timestamp,
                bbox=_parse_box(
                    keyframe.get("bbox"),
                    f"annotations[{index}].keyframes[{keyframe_index}].bbox",
                ),
            )
        )

    evaluate_boundaries = values.get("evaluate_boundaries", True)
    if not isinstance(evaluate_boundaries, bool):
        raise ValueError(f"Annotation {annotation_id} evaluate_boundaries must be a boolean.")
    return TemporalAnnotation(
        annotation_id=annotation_id,
        label=label,
        start_seconds=start,
        end_seconds=end,
        text_match=TextMatch(field=field, mode=mode, patterns=patterns),
        keyframes=tuple(keyframes),
        evaluate_boundaries=evaluate_boundaries,
    )


def _evaluate_annotation(
    annotation: TemporalAnnotation,
    observations: tuple[_Observation, ...],
    gates: TemporalQualityGates,
) -> dict[str, Any]:
    candidates = tuple(
        observation
        for observation in observations
        if annotation.text_match.matches(observation.region)
        and observation.end_seconds
        >= annotation.start_seconds - gates.temporal_search_margin_seconds
        and observation.start_seconds
        <= annotation.end_seconds + gates.temporal_search_margin_seconds
    )
    clipped_intervals = [
        (
            max(annotation.start_seconds, observation.start_seconds),
            min(annotation.end_seconds, observation.end_seconds),
        )
        for observation in candidates
        if observation.end_seconds > annotation.start_seconds
        and observation.start_seconds < annotation.end_seconds
    ]
    merged_clipped = _merge_intervals(clipped_intervals, gates.flicker_tolerance_seconds)
    covered_seconds = sum(end - start for start, end in merged_clipped)
    coverage_ratio = covered_seconds / (annotation.end_seconds - annotation.start_seconds)
    flicker_count = max(0, len(merged_clipped) - 1)

    onset_error_ms: float | None = None
    offset_error_ms: float | None = None
    if annotation.evaluate_boundaries:
        raw_intervals = _merge_intervals(
            [(item.start_seconds, item.end_seconds) for item in candidates],
            gates.flicker_tolerance_seconds,
        )
        overlapping = [
            interval
            for interval in raw_intervals
            if interval[1] > annotation.start_seconds and interval[0] < annotation.end_seconds
        ]
        if overlapping:
            best = max(
                overlapping,
                key=lambda interval: _interval_overlap(
                    interval,
                    (annotation.start_seconds, annotation.end_seconds),
                ),
            )
            onset_error_ms = (best[0] - annotation.start_seconds) * 1000.0
            offset_error_ms = (best[1] - annotation.end_seconds) * 1000.0

    keyframe_reports: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    for keyframe in annotation.keyframes:
        active = [
            observation
            for observation in candidates
            if observation.start_seconds <= keyframe.timestamp_seconds < observation.end_seconds
        ]
        if active:
            best_observation, best_iou = max(
                ((item, _box_iou(keyframe.bbox, item.bbox)) for item in active),
                key=lambda pair: pair[1],
            )
            matched_ious.append(best_iou)
            keyframe_reports.append(
                {
                    "timestamp_seconds": keyframe.timestamp_seconds,
                    "matched": True,
                    "bbox_iou": best_iou,
                    "track_id": best_observation.track_id,
                    "region_index": best_observation.region_index,
                }
            )
        else:
            keyframe_reports.append(
                {
                    "timestamp_seconds": keyframe.timestamp_seconds,
                    "matched": False,
                    "bbox_iou": None,
                    "track_id": None,
                    "region_index": None,
                }
            )

    return {
        "id": annotation.annotation_id,
        "label": annotation.label,
        "start_seconds": annotation.start_seconds,
        "end_seconds": annotation.end_seconds,
        "coverage_ratio": coverage_ratio,
        "coverage_passed": coverage_ratio >= gates.minimum_annotation_coverage_ratio,
        "onset_error_ms": onset_error_ms,
        "offset_error_ms": offset_error_ms,
        "flicker_event_count": flicker_count,
        "keyframe_count": len(annotation.keyframes),
        "matched_keyframe_count": len(matched_ious),
        "matched_bbox_ious": matched_ious,
        "keyframes": keyframe_reports,
    }


def _manifest_observations(manifest: dict[str, Any]) -> tuple[_Observation, ...]:
    observations: list[_Observation] = []
    for raw_track in cast(list[Any], manifest["tracks"]):
        track = cast(dict[str, Any], raw_track)
        for region_index, raw_region in enumerate(cast(list[Any], track["regions"])):
            region = cast(dict[str, Any], raw_region)
            if region.get("action") != "translate":
                continue
            if region.get("needs_review") is True:
                continue
            if not isinstance(region.get("translated_text"), str):
                continue
            observations.append(
                _Observation(
                    track_id=cast(str, track["track_id"]),
                    region_index=region_index,
                    start_seconds=float(track["start_seconds"]),
                    end_seconds=float(track["end_seconds"]),
                    bbox=_parse_box(region.get("bbox"), "track region bbox"),
                    region=region,
                )
            )
    return tuple(observations)


def _validate_video_identity(manifest: dict[str, Any], benchmark: TemporalBenchmark) -> None:
    video = cast(dict[str, Any], manifest["video"])
    duration = float(video["duration_seconds"])
    fps = _positive_number(video.get("fps"), "track manifest video.fps")
    if not math.isclose(duration, benchmark.duration_seconds, abs_tol=0.001):
        raise ValueError("Track manifest duration does not match temporal benchmark video.")
    if not math.isclose(fps, benchmark.fps, abs_tol=0.01):
        raise ValueError("Track manifest FPS does not match temporal benchmark video.")


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


def _merge_intervals(
    intervals: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    valid = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in valid:
        if merged and start <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _interval_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _box_iou(left: NormalizedBox, right: NormalizedBox) -> float:
    intersection_width = max(
        0.0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    intersection_height = max(
        0.0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return min(1.0, intersection / union) if union else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _parse_box(raw: Any, label: str) -> NormalizedBox:
    values = _mapping(raw, label)
    return NormalizedBox(
        x=_nonnegative_number(values.get("x"), f"{label}.x"),
        y=_nonnegative_number(values.get("y"), f"{label}.y"),
        width=_positive_number(values.get("width"), f"{label}.width"),
        height=_positive_number(values.get("height"), f"{label}.height"),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return cast(dict[str, Any], value)


def _nonblank_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string.")
    return value.strip()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative.")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _ratio(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return int(value)
