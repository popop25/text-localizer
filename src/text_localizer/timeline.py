from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from text_localizer.boundaries import refine_video_scene_boundaries
from text_localizer.cache import TranslationCache
from text_localizer.domain import NormalizedBox
from text_localizer.glossary import Glossary
from text_localizer.pipeline import (
    TRANSLATION_CACHE_REVISION,
    analyze_static_image,
    relocalize_static_analysis,
)
from text_localizer.profile import SceneDetectionOptions
from text_localizer.providers.base import OcrProvider, TranslationProvider
from text_localizer.quality import AnalysisOptions, box_to_dict
from text_localizer.scenes import index_frame_scenes
from text_localizer.track_editor import write_track_editor
from text_localizer.video import (
    TimestampedFrame,
    VideoInfo,
    extract_timestamped_frames,
    inspect_video,
)
from text_localizer.video_viewer import write_video_viewer

LOGGER = logging.getLogger(__name__)


def index_video_timeline(
    video_path: Path,
    work_dir: Path,
    *,
    interval_seconds: float,
    profile_name: str | None,
    roi: NormalizedBox | None,
    exclude_regions: tuple[NormalizedBox, ...],
    scene_options: SceneDetectionOptions,
    boundary_refinement_iterations: int = 0,
) -> dict[str, Any]:
    """Extract timestamped samples and turn consecutive hashes into scene ranges."""

    started = perf_counter()
    info = inspect_video(video_path)
    frames, extraction_latency_ms, reused_samples = extract_timestamped_frames(
        video_path,
        work_dir / "frames",
        interval_seconds,
        video_info=info,
    )
    payload = index_timestamped_frame_scenes(
        frames,
        info,
        interval_seconds=interval_seconds,
        profile_name=profile_name,
        roi=roi,
        exclude_regions=exclude_regions,
        scene_options=scene_options,
        extraction_latency_ms=extraction_latency_ms,
        reused_samples=reused_samples,
    )
    if boundary_refinement_iterations:
        payload = refine_video_scene_boundaries(
            payload,
            work_dir / "refined-boundaries",
            iterations=boundary_refinement_iterations,
        )
    payload["metrics"]["total_indexing_latency_ms"] = (perf_counter() - started) * 1000.0
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_json(work_dir / "scene-index.json", payload)
    return payload


def index_timestamped_frame_scenes(
    frames: tuple[TimestampedFrame, ...],
    video_info: VideoInfo,
    *,
    interval_seconds: float,
    profile_name: str | None,
    roi: NormalizedBox | None,
    exclude_regions: tuple[NormalizedBox, ...],
    scene_options: SceneDetectionOptions,
    extraction_latency_ms: float = 0.0,
    reused_samples: bool = False,
) -> dict[str, Any]:
    scene_index = index_frame_scenes(
        [Path(frame.path) for frame in frames],
        profile_name=profile_name,
        roi=roi,
        exclude_regions=exclude_regions,
        options=scene_options,
    )
    frame_records = _timestamped_scene_records(frames, scene_index["frames"])
    manifest = {
        "schema_version": 1,
        "video": video_info.to_dict(),
        "profile_name": profile_name,
        "sampling": {
            "interval_seconds": interval_seconds,
            "frame_count": len(frames),
            "transition_precision_seconds": interval_seconds,
            "samples_reused": reused_samples,
        },
        "scene_detection": {
            "hash_algorithm": scene_index["hash_algorithm"],
            "hash_size": scene_index["hash_size"],
            "hamming_threshold": scene_index["hamming_threshold"],
            "roi": scene_index["roi"],
            "exclude_regions": scene_index["exclude_regions"],
        },
        "total_frames": scene_index["total_frames"],
        "unique_scenes": scene_index["unique_scenes"],
        "duplicate_frames": scene_index["duplicate_frames"],
        "reuse_ratio": scene_index["reuse_ratio"],
        "frames": frame_records,
        "scenes": _build_scene_ranges(frame_records, video_info.duration_seconds),
        "metrics": {
            "frame_extraction_latency_ms": extraction_latency_ms,
            "scene_hashing_latency_ms": scene_index["processing_latency_ms"],
            "total_indexing_latency_ms": scene_index["processing_latency_ms"],
        },
    }
    return manifest


def build_track_manifest(
    scene_index: dict[str, Any],
    analysis_payloads: dict[int, dict[str, Any]],
    *,
    analysis_paths: dict[int, str] | None = None,
    analysis_reused_count: int = 0,
    current_run_latency_ms: float = 0.0,
    analysis_relocalized_count: int = 0,
) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    total_ocr_latency_ms = 0.0
    total_translation_latency_ms = 0.0
    total_scene_processing_latency_ms = 0.0
    needs_review_count = 0
    translation_reused_count = 0

    for raw_scene in scene_index["scenes"]:
        scene_id = int(raw_scene["scene_id"])
        if scene_id not in analysis_payloads:
            raise ValueError(f"Missing analysis for scene {scene_id}.")
        analysis = analysis_payloads[scene_id]
        metrics = analysis.get("metrics", {})
        total_ocr_latency_ms += float(metrics.get("ocr_latency_ms", 0.0))
        total_translation_latency_ms += float(metrics.get("total_translation_latency_ms", 0.0))
        total_scene_processing_latency_ms += float(metrics.get("total_processing_latency_ms", 0.0))
        needs_review_count += int(metrics.get("needs_review_count", 0))
        translation_reused_count += int(metrics.get("translation_reused_count", 0))
        tracks.append(
            {
                "track_id": f"scene-{scene_id:04d}",
                "scene_id": scene_id,
                "start_seconds": float(raw_scene["start_seconds"]),
                "end_seconds": float(raw_scene["end_seconds"]),
                "representative_frame": raw_scene["representative_path"],
                "analysis_path": (
                    analysis_paths.get(scene_id) if analysis_paths is not None else None
                ),
                "regions": analysis["regions"],
            }
        )

    manifest = {
        "schema_version": 1,
        "video": scene_index["video"],
        "profile_name": scene_index.get("profile_name"),
        "sampling": scene_index["sampling"],
        "track_count": len(tracks),
        "tracks": tracks,
        "metrics": {
            "sample_frame_count": scene_index["total_frames"],
            "unique_scene_count": scene_index["unique_scenes"],
            "duplicate_frame_count": scene_index["duplicate_frames"],
            "reuse_ratio": scene_index["reuse_ratio"],
            "analysis_reused_count": analysis_reused_count,
            "analysis_relocalized_count": analysis_relocalized_count,
            "translation_reused_count": translation_reused_count,
            "total_ocr_latency_ms": total_ocr_latency_ms,
            "total_translation_latency_ms": total_translation_latency_ms,
            "total_scene_processing_latency_ms": total_scene_processing_latency_ms,
            "current_run_latency_ms": current_run_latency_ms,
            "needs_review_count": needs_review_count,
        },
    }
    if "boundary_refinement" in scene_index:
        manifest["boundary_refinement"] = scene_index["boundary_refinement"]
    return manifest


def localize_scene_index(
    scene_index: dict[str, Any],
    output_dir: Path,
    ocr_provider: OcrProvider,
    translation_provider: TranslationProvider,
    cache: TranslationCache,
    *,
    glossary: Glossary,
    analysis_options: AnalysisOptions,
) -> dict[str, Any]:
    """Analyze only scene representatives and emit resumable time-range tracks."""

    started = perf_counter()
    analysis_payloads: dict[int, dict[str, Any]] = {}
    analysis_paths: dict[int, str] = {}
    reused_count = 0

    relocalized_count = 0
    scenes = scene_index["scenes"]
    for position, raw_scene in enumerate(scenes, start=1):
        scene_id = int(raw_scene["scene_id"])
        representative = Path(str(raw_scene["representative_path"]))
        scene_output = output_dir / "scenes" / f"scene-{scene_id:04d}"
        regions_path = scene_output / "regions.json"
        analysis = _load_matching_analysis(
            regions_path,
            representative,
            ocr_provider_name=ocr_provider.name,
            translation_provider_name=translation_provider.name,
            glossary_fingerprint=glossary.fingerprint,
            analysis_options=analysis_options,
        )
        if analysis is None:
            prior_analysis = _load_relocalizable_analysis(
                regions_path,
                representative,
                ocr_provider_name=ocr_provider.name,
                analysis_options=analysis_options,
            )
            if prior_analysis is not None:
                LOGGER.info(
                    "Relocalizing stored OCR for scene %d/%d",
                    position,
                    len(scenes),
                )
                analysis = relocalize_static_analysis(
                    representative,
                    scene_output,
                    prior_analysis,
                    ocr_provider,
                    translation_provider,
                    cache,
                    glossary=glossary,
                    options=analysis_options,
                )
                relocalized_count += 1
            else:
                LOGGER.info(
                    "Analyzing scene %d/%d at %.3fs",
                    position,
                    len(scenes),
                    float(raw_scene["start_seconds"]),
                )
                analysis = analyze_static_image(
                    representative,
                    scene_output,
                    ocr_provider,
                    translation_provider,
                    cache,
                    glossary=glossary,
                    options=analysis_options,
                )
        else:
            reused_count += 1
            LOGGER.info("Reusing completed analysis for scene %d/%d", position, len(scenes))
        analysis_payloads[scene_id] = analysis
        analysis_paths[scene_id] = str(regions_path.resolve())

    current_run_latency_ms = (perf_counter() - started) * 1000.0
    manifest = build_track_manifest(
        scene_index,
        analysis_payloads,
        analysis_paths=analysis_paths,
        analysis_reused_count=reused_count,
        current_run_latency_ms=current_run_latency_ms,
        analysis_relocalized_count=relocalized_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "tracks.json", manifest)
    video_path = Path(str(scene_index["video"]["path"]))
    write_video_viewer(output_dir / "viewer.html", video_path, manifest)
    write_track_editor(output_dir / "editor.html", video_path, manifest)
    return manifest


def _timestamped_scene_records(
    frames: tuple[TimestampedFrame, ...],
    scene_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(frames) != len(scene_records):
        raise ValueError("Timestamp and scene record counts do not match.")
    return [
        {
            **record,
            "sample_index": frame.index,
            "timestamp_seconds": frame.timestamp_seconds,
        }
        for frame, record in zip(frames, scene_records, strict=True)
    ]


def _build_scene_ranges(
    frames: list[dict[str, Any]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not frames:
        raise ValueError("At least one timestamped frame is required.")
    groups: list[list[dict[str, Any]]] = []
    for frame in frames:
        if not groups or frame["scene_id"] != groups[-1][0]["scene_id"]:
            groups.append([frame])
        else:
            groups[-1].append(frame)

    scenes: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        first = group[0]
        next_start = (
            float(groups[index + 1][0]["timestamp_seconds"])
            if index + 1 < len(groups)
            else duration_seconds
        )
        scenes.append(
            {
                "scene_id": int(first["scene_id"]),
                "start_seconds": float(first["timestamp_seconds"]),
                "end_seconds": max(float(first["timestamp_seconds"]), next_start),
                "first_sample_seconds": float(first["timestamp_seconds"]),
                "last_sample_seconds": float(group[-1]["timestamp_seconds"]),
                "sample_count": len(group),
                "representative_path": first["representative_path"],
            }
        )
    return scenes


def _load_matching_analysis(
    regions_path: Path,
    representative: Path,
    *,
    ocr_provider_name: str,
    translation_provider_name: str,
    glossary_fingerprint: str,
    analysis_options: AnalysisOptions,
) -> dict[str, Any] | None:
    if not regions_path.is_file():
        return None
    try:
        raw_payload = json.loads(regions_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            return None
        payload = cast(dict[str, Any], raw_payload)
        if payload.get("schema_version") != 2:
            return None
        if payload.get("translation_cache_revision") != TRANSLATION_CACHE_REVISION:
            return None
        if Path(str(payload["input_image"])).resolve() != representative.resolve():
            return None
        if payload.get("ocr_provider") != ocr_provider_name:
            return None
        if payload.get("translation_provider") != translation_provider_name:
            return None
        if payload.get("glossary_fingerprint") != glossary_fingerprint:
            return None
        if not _analysis_options_match(payload.get("analysis_options"), analysis_options):
            return None
        if not isinstance(payload.get("regions"), list):
            return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload


def _load_relocalizable_analysis(
    regions_path: Path,
    representative: Path,
    *,
    ocr_provider_name: str,
    analysis_options: AnalysisOptions,
) -> dict[str, Any] | None:
    if not regions_path.is_file():
        return None
    try:
        raw_payload = json.loads(regions_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            return None
        payload = cast(dict[str, Any], raw_payload)
        if payload.get("schema_version") != 2:
            return None
        if Path(str(payload["input_image"])).resolve() != representative.resolve():
            return None
        if payload.get("ocr_provider") != ocr_provider_name:
            return None
        if not _analysis_options_match(payload.get("analysis_options"), analysis_options):
            return None
        if not isinstance(payload.get("regions"), list):
            return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload


def _analysis_options_match(raw: Any, expected: AnalysisOptions) -> bool:
    if not isinstance(raw, dict):
        return False
    if raw.get("exclude_regions") != [box_to_dict(box) for box in expected.exclude_regions]:
        return False
    if raw.get("min_region_height") != expected.min_region_height:
        return False
    if raw.get("group_adjacent") is not expected.group_adjacent:
        return False
    if raw.get("translation_batch_size") != expected.translation_batch_size:
        return False
    return _boxes_close(raw.get("roi"), box_to_dict(expected.roi))


def _boxes_close(raw: Any, expected: dict[str, float] | None) -> bool:
    if expected is None:
        return raw is None
    if not isinstance(raw, dict):
        return False
    try:
        return all(abs(float(raw[key]) - value) <= 0.002 for key, value in expected.items())
    except (KeyError, TypeError, ValueError):
        return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
