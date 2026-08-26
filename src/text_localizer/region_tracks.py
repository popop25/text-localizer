from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from text_localizer.domain import NormalizedBox, RegionAction
from text_localizer.glossary import Glossary
from text_localizer.providers.base import OcrProvider
from text_localizer.quality import (
    AnalysisOptions,
    box_to_dict,
    ocr_input_for_roi,
    prepare_regions,
    remap_ocr_output,
)
from text_localizer.track_editor import validate_track_manifest
from text_localizer.video_viewer import write_video_viewer

_COMPACT_TEXT_RE = re.compile(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+")
_TIME_TOLERANCE_SECONDS = 1e-6
_LOW_CONSENSUS_AGREEMENT = 0.78
_MIN_TRAILING_MISSING_OBSERVATIONS = 2


@dataclass(frozen=True, slots=True)
class RegionTrackingOptions:
    minimum_event_ratio: float = 0.25
    maximum_event_duration_seconds: float = 1.5
    minimum_text_similarity: float = 0.68
    tile_padding: int = 0
    start_seconds: float | None = None
    end_seconds: float | None = None
    include_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_event_ratio <= 1.0:
            raise ValueError("Region-track minimum event ratio must be between 0 and 1.")
        if self.maximum_event_duration_seconds <= 0.0:
            raise ValueError("Region-track maximum event duration must be positive.")
        if not 0.0 <= self.minimum_text_similarity <= 1.0:
            raise ValueError("Region-track text similarity must be between 0 and 1.")
        if not 0 <= self.tile_padding <= 8:
            raise ValueError("Region-track tile padding must be between 0 and 8.")
        if self.start_seconds is not None and self.start_seconds < 0.0:
            raise ValueError("Region-track start time must be non-negative.")
        if self.end_seconds is not None and self.end_seconds <= 0.0:
            raise ValueError("Region-track end time must be positive.")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("Region-track end time must be later than its start time.")
        if self.include_windows and (
            self.start_seconds is not None or self.end_seconds is not None
        ):
            raise ValueError(
                "Region-track include windows cannot be combined with start/end bounds."
            )
        previous_end: float | None = None
        for start, end in self.include_windows:
            if start < 0.0 or end <= start:
                raise ValueError("Region-track include windows require 0 <= start < end.")
            if previous_end is not None and start <= previous_end:
                raise ValueError(
                    "Region-track include windows must be ordered and non-overlapping."
                )
            previous_end = end


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    event_id: str
    frame_index: int
    timestamp_seconds: float
    path: Path
    roi: NormalizedBox
    phase: str
    event_start_seconds: float


def localize_region_tracks(
    base_manifest_path: Path,
    visual_event_manifest_path: Path,
    output_dir: Path,
    work_dir: Path,
    ocr_provider: OcrProvider,
    *,
    glossary: Glossary,
    analysis_options: AnalysisOptions,
    tracking_exclude_regions: tuple[NormalizedBox, ...],
    options: RegionTrackingOptions,
) -> dict[str, Any]:
    """Add event-keyframed region tracks without re-running translation."""

    started = perf_counter()
    base_manifest = _load_json_mapping(base_manifest_path, "Base track manifest")
    validate_track_manifest(base_manifest)
    if base_manifest.get("schema_version") != 1:
        raise ValueError("Region tracking currently requires a schema-v1 base manifest.")
    visual_manifest = _load_json_mapping(visual_event_manifest_path, "Visual-event manifest")
    _validate_visual_manifest(visual_manifest, base_manifest)

    selected = select_tracking_frames(visual_manifest, options)
    observations: list[dict[str, Any]] = []
    reused_count = 0
    total_ocr_latency_ms = 0.0
    for position, frame in enumerate(selected, start=1):
        observation_path = work_dir / "observations" / f"frame-{frame.frame_index:06d}.json"
        observation = _load_cached_observation(
            observation_path,
            frame,
            ocr_provider_name=ocr_provider.name,
            minimum_region_height=analysis_options.min_region_height,
            group_adjacent=analysis_options.group_adjacent,
            exclude_regions=tracking_exclude_regions,
        )
        if observation is None:
            observation = _observe_tracking_frame(
                frame,
                observation_path.parent,
                ocr_provider,
                glossary=glossary,
                minimum_region_height=analysis_options.min_region_height,
                group_adjacent=analysis_options.group_adjacent,
                exclude_regions=tracking_exclude_regions,
            )
            _write_json(observation_path, observation)
        else:
            reused_count += 1
        total_ocr_latency_ms += float(observation["ocr_latency_ms"])
        observation["selection_position"] = position
        observations.append(observation)

    manifest = build_region_track_manifest(
        base_manifest,
        observations,
        visual_event_manifest_path=visual_event_manifest_path,
        options=options,
    )
    metrics = cast(dict[str, Any], manifest["metrics"])
    metrics.update(
        {
            "tracking_selected_frame_count": len(selected),
            "tracking_ocr_frame_count": len(selected) - reused_count,
            "tracking_reused_observation_count": reused_count,
            "tracking_total_ocr_latency_ms": total_ocr_latency_ms,
            "tracking_current_run_latency_ms": (perf_counter() - started) * 1000.0,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "tracks.json", manifest)
    write_video_viewer(
        output_dir / "viewer.html",
        Path(str(cast(dict[str, Any], manifest["video"])["path"])),
        manifest,
    )
    return manifest


def select_tracking_frames(
    visual_manifest: dict[str, Any],
    options: RegionTrackingOptions,
) -> tuple[TrackingFrame, ...]:
    """Choose one stable pre-frame, changed frames, and one stable post-frame per event."""

    _validate_visual_manifest_structure(visual_manifest)
    raw_frames = cast(list[Any], visual_manifest["frames"])
    frames_by_index = {
        int(cast(dict[str, Any], item)["frame_index"]): cast(dict[str, Any], item)
        for item in raw_frames
    }
    selected: dict[tuple[str, int], TrackingFrame] = {}
    for raw_event in cast(list[Any], visual_manifest["events"]):
        event = cast(dict[str, Any], raw_event)
        start = float(event["start_seconds"])
        end = float(event["end_seconds"])
        if float(event["peak_changed_ratio"]) < options.minimum_event_ratio:
            continue
        if float(event["duration_seconds"]) > options.maximum_event_duration_seconds:
            continue
        if options.start_seconds is not None and end < options.start_seconds:
            continue
        if options.end_seconds is not None and start > options.end_seconds:
            continue
        if options.include_windows and not any(
            end >= window_start and start <= window_end
            for window_start, window_end in options.include_windows
        ):
            continue

        event_id = str(event["event_id"])
        first_index = int(event["first_frame_index"])
        last_index = int(event["last_frame_index"])
        roi = _event_roi(visual_manifest, event, options.tile_padding)
        candidate_indices = {max(0, first_index - 1), min(last_index + 1, len(raw_frames) - 1)}
        candidate_indices.update(
            index
            for index in range(first_index, last_index + 1)
            if bool(frames_by_index[index].get("changed"))
        )
        ordered_indices = sorted(candidate_indices)
        for candidate_position, frame_index in enumerate(ordered_indices):
            raw_frame = frames_by_index[frame_index]
            phase = "changed"
            if candidate_position == 0:
                phase = "pre"
            elif candidate_position == len(ordered_indices) - 1:
                phase = "post"
            selected[(event_id, frame_index)] = TrackingFrame(
                event_id=event_id,
                frame_index=frame_index,
                timestamp_seconds=float(raw_frame["timestamp_seconds"]),
                path=Path(str(raw_frame["path"])),
                roi=roi,
                phase=phase,
                event_start_seconds=start,
            )
    return tuple(
        sorted(selected.values(), key=lambda item: (item.timestamp_seconds, item.frame_index))
    )


def build_region_track_manifest(
    base_manifest: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    visual_event_manifest_path: Path,
    options: RegionTrackingOptions,
) -> dict[str, Any]:
    """Compile scene regions into independent tracks and inject event OCR keyframes."""

    validate_track_manifest(base_manifest)
    if base_manifest.get("schema_version") != 1:
        raise ValueError("Region tracking currently requires a schema-v1 base manifest.")

    region_tracks = _compile_base_region_tracks(base_manifest, options.minimum_text_similarity)
    injection_metrics = _inject_event_keyframes(
        region_tracks,
        observations,
        options.minimum_text_similarity,
    )
    consensus_metrics = _finalize_track_text_consensus(
        region_tracks,
        options.minimum_text_similarity,
    )
    result = deepcopy(base_manifest)
    result["schema_version"] = 2
    result["region_track_count"] = len(region_tracks)
    result["region_tracks"] = region_tracks
    result["region_tracking"] = {
        "schema_version": 1,
        "algorithm": "event-ocr-keyframes-v2",
        "interpolation": "per-keyframe",
        "translation_policy": "reuse-consensus-nearest-v1-region",
        "text_consensus": "confidence-weighted-medoid-v1-anchor",
        "visibility_policy": "event-trailing-miss-v1",
        "visual_event_manifest": str(visual_event_manifest_path.resolve()),
        "options": {
            "minimum_event_ratio": options.minimum_event_ratio,
            "maximum_event_duration_seconds": options.maximum_event_duration_seconds,
            "minimum_text_similarity": options.minimum_text_similarity,
            "tile_padding": options.tile_padding,
            "start_seconds": options.start_seconds,
            "end_seconds": options.end_seconds,
            "include_windows": [
                {"start_seconds": start, "end_seconds": end}
                for start, end in options.include_windows
            ],
        },
        **injection_metrics,
        **consensus_metrics,
    }
    metrics = cast(dict[str, Any], result["metrics"])
    metrics["needs_review_count"] = sum(
        int(cast(dict[str, Any], item["region"])["needs_review"]) for item in region_tracks
    )
    return result


def _compile_base_region_tracks(
    manifest: dict[str, Any],
    minimum_text_similarity: float,
) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    next_id = 1
    for raw_scene in cast(list[Any], manifest["tracks"]):
        scene = cast(dict[str, Any], raw_scene)
        start = float(scene["start_seconds"])
        end = float(scene["end_seconds"])
        regions = [
            cast(dict[str, Any], raw_region)
            for raw_region in cast(list[Any], scene["regions"])
            if _is_trackable_region(cast(dict[str, Any], raw_region))
        ]
        matches = _match_regions_to_tracks(regions, active, minimum_text_similarity, start)
        next_active: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            matched = next(
                (track for candidate_index, track in matches if candidate_index == region_index),
                None,
            )
            if matched is None:
                if not _is_renderable_region(region):
                    continue
                matched = {
                    "region_track_id": f"region-{next_id:04d}",
                    "start_seconds": start,
                    "end_seconds": end,
                    "region": deepcopy(region),
                    "keyframes": [],
                    "_region_candidates": [deepcopy(region)],
                }
                next_id += 1
                compiled.append(matched)
            else:
                if _is_renderable_region(region):
                    cast(list[dict[str, Any]], matched["_region_candidates"]).append(
                        deepcopy(region)
                    )
                matched["end_seconds"] = end
                if _is_renderable_region(region):
                    _prefer_region_payload(cast(dict[str, Any], matched["region"]), region)
            _append_keyframe(
                cast(list[dict[str, Any]], matched["keyframes"]),
                timestamp=start,
                bbox=_box_mapping(region["bbox"]),
                source="scene",
                interpolation="hold",
                source_text=str(region.get("source_text", "")),
                confidence=float(region.get("ocr_confidence", 0.0)),
            )
            next_active.append(matched)
        active = next_active
    return compiled


def _inject_event_keyframes(
    region_tracks: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    minimum_text_similarity: float,
) -> dict[str, Any]:
    matched_observations = 0
    unmatched_observations = 0
    affected_ids: set[str] = set()
    visibility_adjusted_ids: set[str] = set()
    by_event: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_event.setdefault(str(observation["event_id"]), []).append(observation)

    for event_id, event_observations in by_event.items():
        ordered = sorted(event_observations, key=lambda item: float(item["timestamp_seconds"]))
        event_start = float(ordered[0]["timestamp_seconds"])
        event_end = float(ordered[-1]["timestamp_seconds"])
        additions: dict[str, list[dict[str, Any]]] = {}
        visibility_samples: dict[str, list[tuple[float, bool, str]]] = {}
        for observation_index, observation in enumerate(ordered):
            timestamp = float(observation["timestamp_seconds"])
            candidates = [
                track
                for track in region_tracks
                if float(track["start_seconds"]) <= timestamp + _TIME_TOLERANCE_SECONDS
                and float(track["end_seconds"]) >= timestamp - _TIME_TOLERANCE_SECONDS
            ]
            regions = [
                cast(dict[str, Any], item) for item in cast(list[Any], observation["regions"])
            ]
            matches = _match_observations_to_tracks(
                regions,
                candidates,
                minimum_text_similarity,
                timestamp,
            )
            matched_track_ids = {str(track["region_track_id"]) for _index, track in matches}
            for track in candidates:
                track_id = str(track["region_track_id"])
                if _observation_covers_track(observation, track, timestamp):
                    visibility_samples.setdefault(track_id, []).append(
                        (
                            timestamp,
                            track_id in matched_track_ids,
                            str(observation["phase"]),
                        )
                    )
            matched_observations += len(matches)
            unmatched_observations += len(regions) - len(matches)
            for region_index, track in matches:
                region = regions[region_index]
                track_id = str(track["region_track_id"])
                affected_ids.add(track_id)
                additions.setdefault(track_id, []).append(
                    {
                        "timestamp_seconds": timestamp,
                        "bbox": _box_mapping(region["bbox"]),
                        "source": "visual_event",
                        "event_id": event_id,
                        "frame_index": int(observation["frame_index"]),
                        "phase": str(observation["phase"]),
                        "event_start_seconds": float(
                            observation.get("event_start_seconds", ordered[0]["timestamp_seconds"])
                        ),
                        "interpolation": (
                            "hold" if observation_index == len(ordered) - 1 else "linear"
                        ),
                        "source_text": str(region["source_text"]),
                        "ocr_confidence": float(region["ocr_confidence"]),
                    }
                )

        for track in region_tracks:
            track_id = str(track["region_track_id"])
            pending = additions.get(track_id)
            if not pending:
                continue
            pending = _stabilize_event_geometry(pending)
            pending = _add_transition_start_keyframe(pending)
            keyframes = cast(list[dict[str, Any]], track["keyframes"])
            track_start = float(track["start_seconds"])
            keyframes[:] = [
                item
                for item in keyframes
                if math.isclose(float(item["timestamp_seconds"]), track_start, abs_tol=1e-6)
                or not event_start - _TIME_TOLERANCE_SECONDS
                <= float(item["timestamp_seconds"])
                <= event_end + _TIME_TOLERANCE_SECONDS
            ]
            for item in pending:
                _append_keyframe(
                    keyframes,
                    timestamp=float(item["timestamp_seconds"]),
                    bbox=cast(dict[str, float], item["bbox"]),
                    source="visual_event",
                    interpolation=str(item["interpolation"]),
                    source_text=str(item["source_text"]),
                    confidence=float(item["ocr_confidence"]),
                    event_id=str(item["event_id"]),
                    frame_index=int(item["frame_index"]),
                    phase=str(item["phase"]),
                )
            keyframes.sort(key=lambda item: float(item["timestamp_seconds"]))

        for track in region_tracks:
            track_id = str(track["region_track_id"])
            new_end = _trailing_visibility_end(
                track,
                visibility_samples.get(track_id, []),
            )
            if new_end is None:
                continue
            track["end_seconds"] = new_end
            keyframes = cast(list[dict[str, Any]], track["keyframes"])
            keyframes[:] = [
                item
                for item in keyframes
                if float(item["timestamp_seconds"]) < new_end - _TIME_TOLERANCE_SECONDS
            ]
            if not keyframes:
                raise ValueError(f"Visibility adjustment removed every keyframe from {track_id}.")
            keyframes[-1]["interpolation"] = "hold"
            visibility_adjusted_ids.add(track_id)

    for track in region_tracks:
        cast(list[dict[str, Any]], track["keyframes"]).sort(
            key=lambda item: float(item["timestamp_seconds"])
        )
    return {
        "observation_count": len(observations),
        "matched_ocr_region_count": matched_observations,
        "unmatched_ocr_region_count": unmatched_observations,
        "affected_region_track_count": len(affected_ids),
        "visibility_adjusted_track_count": len(visibility_adjusted_ids),
    }


def _observation_covers_track(
    observation: dict[str, Any],
    track: dict[str, Any],
    timestamp: float,
) -> bool:
    raw_roi = observation.get("roi")
    if not isinstance(raw_roi, dict):
        return True
    track_box = _nearest_keyframe_box(track, timestamp)
    roi = _box_mapping(raw_roi)
    return _box_coverage_ratio(track_box, roi) >= 0.5


def _trailing_visibility_end(
    track: dict[str, Any],
    samples: list[tuple[float, bool, str]],
) -> float | None:
    matched_indices = [
        index for index, (_timestamp, matched, _phase) in enumerate(samples) if matched
    ]
    if not matched_indices:
        return None
    trailing = samples[matched_indices[-1] + 1 :]
    if len(trailing) < _MIN_TRAILING_MISSING_OBSERVATIONS:
        return None
    if any(matched for _timestamp, matched, _phase in trailing):
        return None
    if trailing[-1][2] != "post":
        return None
    candidate = trailing[0][0]
    if candidate <= float(track["start_seconds"]) + _TIME_TOLERANCE_SECONDS:
        return None
    if candidate >= float(track["end_seconds"]) - _TIME_TOLERANCE_SECONDS:
        return None
    return candidate


def _observe_tracking_frame(
    frame: TrackingFrame,
    temporary_dir: Path,
    ocr_provider: OcrProvider,
    *,
    glossary: Glossary,
    minimum_region_height: float,
    group_adjacent: bool,
    exclude_regions: tuple[NormalizedBox, ...],
) -> dict[str, Any]:
    if not frame.path.is_file():
        raise FileNotFoundError(f"Visual-event frame does not exist: {frame.path}")
    options = AnalysisOptions(
        roi=frame.roi,
        exclude_regions=exclude_regions,
        min_region_height=minimum_region_height,
        group_adjacent=group_adjacent,
    )
    with ocr_input_for_roi(frame.path, temporary_dir, frame.roi) as (ocr_path, actual_roi):
        output = remap_ocr_output(ocr_provider.recognize(ocr_path), actual_roi)
    prepared = prepare_regions(output.regions, glossary, options)
    regions = [
        {
            "bbox": box_to_dict(item.bbox),
            "source_text": item.raw_source_text,
            "ocr_confidence": item.confidence,
        }
        for item in prepared
        if item.forced_action is not RegionAction.IGNORE
    ]
    return {
        "schema_version": 1,
        "event_id": frame.event_id,
        "frame_index": frame.frame_index,
        "timestamp_seconds": frame.timestamp_seconds,
        "frame_path": str(frame.path.resolve()),
        "frame_sha256": _file_sha256(frame.path),
        "phase": frame.phase,
        "event_start_seconds": frame.event_start_seconds,
        "roi": box_to_dict(frame.roi),
        "exclude_regions": [box_to_dict(item) for item in exclude_regions],
        "minimum_region_height": minimum_region_height,
        "group_adjacent": group_adjacent,
        "ocr_provider": ocr_provider.name,
        "ocr_latency_ms": output.latency_ms,
        "regions": regions,
    }


def _load_cached_observation(
    path: Path,
    frame: TrackingFrame,
    *,
    ocr_provider_name: str,
    minimum_region_height: float,
    group_adjacent: bool,
    exclude_regions: tuple[NormalizedBox, ...],
) -> dict[str, Any] | None:
    if not path.is_file() or not frame.path.is_file():
        return None
    try:
        payload = _load_json_mapping(path, "Region-track observation")
        if payload.get("schema_version") != 1:
            return None
        if payload.get("event_id") != frame.event_id:
            return None
        if payload.get("frame_index") != frame.frame_index:
            return None
        timestamp = payload.get("timestamp_seconds")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
            return None
        if not math.isclose(float(timestamp), frame.timestamp_seconds, abs_tol=1e-6):
            return None
        if payload.get("frame_sha256") != _file_sha256(frame.path):
            return None
        if payload.get("ocr_provider") != ocr_provider_name:
            return None
        if payload.get("roi") != box_to_dict(frame.roi):
            return None
        if payload.get("exclude_regions") != [box_to_dict(item) for item in exclude_regions]:
            return None
        if payload.get("minimum_region_height") != minimum_region_height:
            return None
        if payload.get("group_adjacent") is not group_adjacent:
            return None
        if not isinstance(payload.get("regions"), list):
            return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    payload["phase"] = frame.phase
    payload["event_start_seconds"] = frame.event_start_seconds
    return payload


def _stabilize_event_geometry(
    keyframes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep horizontal geometry stable when an event is clearly a vertical scroll."""

    if len(keyframes) < 2:
        return keyframes
    boxes = [cast(dict[str, float], item["bbox"]) for item in keyframes]
    x_span = max(item["x"] for item in boxes) - min(item["x"] for item in boxes)
    y_span = max(item["y"] for item in boxes) - min(item["y"] for item in boxes)
    if y_span <= 0.01 or y_span <= x_span * 2.0:
        return keyframes
    stable_x = statistics.median(item["x"] for item in boxes)
    stable_width = statistics.median(item["width"] for item in boxes)
    stable_height = statistics.median(item["height"] for item in boxes)
    result = deepcopy(keyframes)
    for item in result:
        bbox = cast(dict[str, float], item["bbox"])
        bbox["x"] = stable_x
        bbox["width"] = stable_width
        bbox["height"] = stable_height
    return result


def _add_transition_start_keyframe(
    keyframes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(keyframes, key=lambda item: float(item["timestamp_seconds"]))
    pre = next((item for item in ordered if item.get("phase") == "pre"), None)
    if pre is None:
        return ordered
    event_start = float(pre["event_start_seconds"])
    pre_timestamp = float(pre["timestamp_seconds"])
    if event_start <= pre_timestamp + _TIME_TOLERANCE_SECONDS:
        return ordered
    pre["interpolation"] = "hold"
    transition = deepcopy(pre)
    transition["timestamp_seconds"] = event_start
    transition["phase"] = "transition_start"
    transition["interpolation"] = "linear"
    ordered.append(transition)
    return sorted(ordered, key=lambda item: float(item["timestamp_seconds"]))


def _match_regions_to_tracks(
    regions: list[dict[str, Any]],
    active: list[dict[str, Any]],
    minimum_text_similarity: float,
    timestamp: float,
) -> list[tuple[int, dict[str, Any]]]:
    scored: list[tuple[float, int, int]] = []
    for region_index, region in enumerate(regions):
        for track_index, track in enumerate(active):
            if abs(float(track["end_seconds"]) - timestamp) > _TIME_TOLERANCE_SECONDS:
                continue
            text_score = _text_similarity(
                str(region.get("source_text", "")),
                _last_keyframe_source_text(track),
            )
            if text_score < minimum_text_similarity:
                continue
            spatial_score = _spatial_similarity(
                _box_mapping(region["bbox"]),
                _last_keyframe_box(track),
            )
            scored.append((text_score * 0.85 + spatial_score * 0.15, region_index, track_index))
    return _greedy_matches(scored, regions, active)


def _match_observations_to_tracks(
    regions: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    minimum_text_similarity: float,
    timestamp: float,
) -> list[tuple[int, dict[str, Any]]]:
    scored: list[tuple[float, int, int]] = []
    for region_index, region in enumerate(regions):
        for track_index, track in enumerate(tracks):
            text_score = _text_similarity(
                str(region.get("source_text", "")),
                _nearest_keyframe_source_text(track, timestamp),
            )
            if text_score < minimum_text_similarity:
                continue
            spatial_score = _spatial_similarity(
                _box_mapping(region["bbox"]),
                _nearest_keyframe_box(track, timestamp),
            )
            scored.append((text_score * 0.9 + spatial_score * 0.1, region_index, track_index))
    return _greedy_matches(scored, regions, tracks)


def _greedy_matches(
    scored: list[tuple[float, int, int]],
    regions: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    used_regions: set[int] = set()
    used_tracks: set[int] = set()
    result: list[tuple[int, dict[str, Any]]] = []
    for _score, region_index, track_index in sorted(scored, reverse=True):
        if region_index in used_regions or track_index in used_tracks:
            continue
        used_regions.add(region_index)
        used_tracks.add(track_index)
        result.append((region_index, tracks[track_index]))
    return result


def _event_roi(
    manifest: dict[str, Any],
    event: dict[str, Any],
    padding: int,
) -> NormalizedBox:
    grid = cast(dict[str, Any], manifest["grid"])
    columns = int(grid["columns"])
    rows = int(grid["rows"])
    tiles = {
        (int(cast(dict[str, Any], item)["column"]), int(cast(dict[str, Any], item)["row"])): (
            _box_mapping(cast(dict[str, Any], item)["bbox"])
        )
        for item in cast(list[Any], grid["tiles"])
    }
    changed = [cast(dict[str, Any], item) for item in cast(list[Any], event["changed_tiles"])]
    selected: list[dict[str, float]] = []
    for item in changed:
        column = int(item["column"])
        row = int(item["row"])
        for padded_column in range(max(0, column - padding), min(columns, column + padding + 1)):
            for padded_row in range(max(0, row - padding), min(rows, row + padding + 1)):
                selected.append(tiles[(padded_column, padded_row)])
    if not selected:
        raise ValueError(f"Visual event {event['event_id']} has no changed tile geometry.")
    left = min(item["x"] for item in selected)
    top = min(item["y"] for item in selected)
    right = max(item["x"] + item["width"] for item in selected)
    bottom = max(item["y"] + item["height"] for item in selected)
    return NormalizedBox(left, top, right - left, bottom - top)


def _validate_visual_manifest(
    visual_manifest: dict[str, Any],
    track_manifest: dict[str, Any],
) -> None:
    _validate_visual_manifest_structure(visual_manifest)
    visual_video = cast(dict[str, Any], visual_manifest["video"])
    track_video = cast(dict[str, Any], track_manifest["video"])
    for field, tolerance in (("duration_seconds", 0.001), ("fps", 0.01)):
        if not math.isclose(
            float(visual_video[field]),
            float(track_video[field]),
            abs_tol=tolerance,
        ):
            raise ValueError(f"Visual-event and track manifests disagree on video {field}.")


def _validate_visual_manifest_structure(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Visual-event manifest schema_version must be 1.")
    if not isinstance(manifest.get("video"), dict):
        raise ValueError("Visual-event manifest requires video metadata.")
    if not isinstance(manifest.get("frames"), list) or not manifest["frames"]:
        raise ValueError("Visual-event manifest requires frames.")
    if not isinstance(manifest.get("events"), list):
        raise ValueError("Visual-event manifest requires events.")
    grid = manifest.get("grid")
    if not isinstance(grid, dict) or not isinstance(grid.get("tiles"), list):
        raise ValueError("Visual-event manifest requires grid tile geometry.")


def _append_keyframe(
    keyframes: list[dict[str, Any]],
    *,
    timestamp: float,
    bbox: dict[str, float],
    source: str,
    interpolation: str,
    source_text: str,
    confidence: float,
    event_id: str | None = None,
    frame_index: int | None = None,
    phase: str | None = None,
) -> None:
    existing = next(
        (
            item
            for item in keyframes
            if math.isclose(float(item["timestamp_seconds"]), timestamp, abs_tol=1e-6)
        ),
        None,
    )
    payload: dict[str, Any] = {
        "timestamp_seconds": timestamp,
        "bbox": bbox,
        "interpolation": interpolation,
        "source": source,
        "source_text": source_text,
        "ocr_confidence": confidence,
    }
    if event_id is not None:
        payload["event_id"] = event_id
    if frame_index is not None:
        payload["frame_index"] = frame_index
    if phase is not None:
        payload["phase"] = phase
    if existing is None:
        keyframes.append(payload)
    else:
        existing.clear()
        existing.update(payload)


def _prefer_region_payload(current: dict[str, Any], candidate: dict[str, Any]) -> None:
    current_review = current.get("needs_review") is True
    candidate_review = candidate.get("needs_review") is True
    if current_review and not candidate_review:
        current.clear()
        current.update(deepcopy(candidate))
        return
    if current_review == candidate_review and float(candidate.get("ocr_confidence", 0.0)) > float(
        current.get("ocr_confidence", 0.0)
    ):
        current.clear()
        current.update(deepcopy(candidate))


def _finalize_track_text_consensus(
    region_tracks: list[dict[str, Any]],
    minimum_text_similarity: float,
) -> dict[str, int]:
    changed_track_count = 0
    low_agreement_count = 0
    translation_mismatch_count = 0
    for track in region_tracks:
        raw_candidates = track.pop("_region_candidates", None)
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(
                f"Region track {track['region_track_id']} has no translation candidates."
            )
        candidates = [
            cast(dict[str, Any], item)
            for item in raw_candidates
            if isinstance(item, dict) and _is_renderable_region(cast(dict[str, Any], item))
        ]
        if not candidates:
            raise ValueError(
                f"Region track {track['region_track_id']} has no renderable translation candidate."
            )

        keyframes = cast(list[dict[str, Any]], track["keyframes"])
        selected_text, selected_confidence, agreement_ratio, exact_agreement_ratio = (
            _select_consensus_observation(keyframes)
        )
        legacy_region = cast(dict[str, Any], track["region"])
        legacy_source_text = str(legacy_region.get("source_text", ""))
        best_candidate = max(
            candidates,
            key=lambda candidate: (
                _text_similarity(selected_text, str(candidate.get("source_text", ""))),
                int(candidate.get("needs_review") is not True),
                float(candidate.get("ocr_confidence", 0.0)),
            ),
        )
        region = deepcopy(best_candidate)
        translation_source_text = str(region.get("source_text", ""))
        translation_source_similarity = _text_similarity(
            selected_text,
            translation_source_text,
        )
        raw_flags = region.get("translation_quality_flags", [])
        flags = [str(item) for item in raw_flags] if isinstance(raw_flags, list) else []
        needs_review = region.get("needs_review") is True
        if agreement_ratio < _LOW_CONSENSUS_AGREEMENT:
            if "ocr_consensus_low_agreement" not in flags:
                flags.append("ocr_consensus_low_agreement")
            needs_review = True
            low_agreement_count += 1
        if translation_source_similarity < minimum_text_similarity:
            if "ocr_consensus_translation_mismatch" not in flags:
                flags.append("ocr_consensus_translation_mismatch")
            needs_review = True
            translation_mismatch_count += 1

        # Event observations choose the nearest already-translated v1 candidate, but the
        # candidate's source stays canonical. Repeated OCR mistakes are correlated (for
        # example, every transition frame can read ``OpenAI`` as ``OpenAl``), so allowing
        # the raw medoid to replace a compatible translated source would regress both
        # terminology and source/target alignment.
        canonical_source_text = translation_source_text
        region["source_text"] = canonical_source_text
        region["translation_source_text"] = translation_source_text
        region["translation_quality_flags"] = flags
        region["needs_review"] = needs_review
        track["region"] = region
        track["text_consensus"] = {
            "schema_version": 1,
            "algorithm": "confidence-weighted-medoid-v1-anchor",
            "observation_count": len(keyframes),
            "unique_text_count": len(
                {_compact_text(str(item.get("source_text", ""))) for item in keyframes}
            ),
            "selected_source_text": canonical_source_text,
            "observed_consensus_text": selected_text,
            "selected_ocr_confidence": selected_confidence,
            "agreement_ratio": agreement_ratio,
            "exact_agreement_ratio": exact_agreement_ratio,
            "translation_source_text": translation_source_text,
            "translation_source_similarity": translation_source_similarity,
        }
        if _compact_text(legacy_source_text) != _compact_text(canonical_source_text):
            changed_track_count += 1

    return {
        "text_consensus_track_count": len(region_tracks),
        "text_consensus_changed_track_count": changed_track_count,
        "text_consensus_low_agreement_count": low_agreement_count,
        "text_consensus_translation_mismatch_count": translation_mismatch_count,
    }


def _select_consensus_observation(
    keyframes: list[dict[str, Any]],
) -> tuple[str, float, float, float]:
    observations = [
        (
            str(item.get("source_text", "")).strip(),
            max(0.05, min(1.0, float(item.get("ocr_confidence", 0.0)))),
        )
        for item in keyframes
        if str(item.get("source_text", "")).strip()
    ]
    if not observations:
        raise ValueError("Region track text consensus requires at least one OCR observation.")
    total_weight = sum(weight for _text, weight in observations)
    scored: list[tuple[float, float, float, int, str, float]] = []
    for candidate_text, candidate_confidence in observations:
        agreement = (
            sum(
                weight * _text_similarity(candidate_text, observed_text)
                for observed_text, weight in observations
            )
            / total_weight
        )
        exact_agreement = (
            sum(
                weight
                for observed_text, weight in observations
                if _compact_text(candidate_text) == _compact_text(observed_text)
            )
            / total_weight
        )
        readability = _consensus_readability(candidate_text)
        score = agreement + exact_agreement * 0.08 + candidate_confidence * 0.02
        scored.append(
            (
                score,
                readability,
                candidate_confidence,
                len(candidate_text),
                candidate_text,
                exact_agreement,
            )
        )
    _score, _readability, confidence, _length, selected, exact_agreement = max(scored)
    agreement = (
        sum(
            weight * _text_similarity(selected, observed_text)
            for observed_text, weight in observations
        )
        / total_weight
    )
    return selected, confidence, agreement, exact_agreement


def _consensus_readability(value: str) -> float:
    stripped = value.strip()
    if not stripped:
        return 0.0
    score = 1.0 + min(stripped.count(" "), 8) * 0.01
    if stripped[0] in '^<>"»':
        score -= 0.2
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if stripped.count(opening) != stripped.count(closing):
            score -= 0.15
    if stripped.count('"') % 2:
        score -= 0.15
    return score


def _is_renderable_region(region: dict[str, Any]) -> bool:
    return (
        region.get("action") == "translate"
        and isinstance(region.get("translated_text"), str)
        and bool(str(region["translated_text"]).strip())
    )


def _is_trackable_region(region: dict[str, Any]) -> bool:
    return (
        isinstance(region.get("bbox"), dict)
        and isinstance(region.get("source_text"), str)
        and bool(str(region["source_text"]).strip())
    )


def _text_similarity(left: str, right: str) -> float:
    compact_left = _compact_text(left)
    compact_right = _compact_text(right)
    if not compact_left or not compact_right:
        return 0.0
    if compact_left == compact_right:
        return 1.0
    containment = min(len(compact_left), len(compact_right)) / max(
        len(compact_left), len(compact_right)
    )
    if compact_left in compact_right or compact_right in compact_left:
        return max(containment, 0.8)
    return SequenceMatcher(None, compact_left, compact_right, autojunk=False).ratio()


def _compact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _COMPACT_TEXT_RE.sub("", normalized)


def _spatial_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    left_x = left["x"] + left["width"] / 2
    left_y = left["y"] + left["height"] / 2
    right_x = right["x"] + right["width"] / 2
    right_y = right["y"] + right["height"] / 2
    distance = math.hypot(left_x - right_x, left_y - right_y)
    return max(0.0, 1.0 - distance / 0.5)


def _box_coverage_ratio(box: dict[str, float], container: dict[str, float]) -> float:
    left = max(box["x"], container["x"])
    top = max(box["y"], container["y"])
    right = min(box["x"] + box["width"], container["x"] + container["width"])
    bottom = min(box["y"] + box["height"], container["y"] + container["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = box["width"] * box["height"]
    return intersection / area if area > 0.0 else 0.0


def _last_keyframe_box(track: dict[str, Any]) -> dict[str, float]:
    keyframes = cast(list[dict[str, Any]], track["keyframes"])
    return _box_mapping(keyframes[-1]["bbox"])


def _last_keyframe_source_text(track: dict[str, Any]) -> str:
    keyframes = cast(list[dict[str, Any]], track["keyframes"])
    return str(keyframes[-1].get("source_text", ""))


def _nearest_keyframe_box(track: dict[str, Any], timestamp: float) -> dict[str, float]:
    keyframes = cast(list[dict[str, Any]], track["keyframes"])
    nearest = min(keyframes, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
    return _box_mapping(nearest["bbox"])


def _nearest_keyframe_source_text(track: dict[str, Any], timestamp: float) -> str:
    keyframes = cast(list[dict[str, Any]], track["keyframes"])
    nearest = min(keyframes, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
    return str(nearest.get("source_text", ""))


def _box_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("Region bbox must be a mapping.")
    return {
        "x": float(value["x"]),
        "y": float(value["y"]),
        "width": float(value["width"]),
        "height": float(value["height"]),
    }


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} root must be a mapping.")
    return cast(dict[str, Any], raw)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
