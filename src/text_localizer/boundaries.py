from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from text_localizer.domain import NormalizedBox
from text_localizer.scenes import difference_hash, hamming_distance
from text_localizer.video import extract_frame_at_timestamp


@dataclass(frozen=True, slots=True)
class BoundaryHashSample:
    timestamp_seconds: float
    frame_path: str
    frame_hash: int
    reused: bool = False


BoundaryHashProvider = Callable[[float, int, int], BoundaryHashSample]


def refine_video_scene_boundaries(
    scene_index: dict[str, Any],
    output_dir: Path,
    *,
    iterations: int,
) -> dict[str, Any]:
    if not 1 <= iterations <= 8:
        raise ValueError("Boundary refinement iterations must be between 1 and 8.")
    video_path = Path(str(scene_index["video"]["path"]))
    video_stat = video_path.stat()
    source_identity = f"{video_path.resolve()}|{video_stat.st_size}|{video_stat.st_mtime_ns}"
    source_fingerprint = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:12]
    detection = scene_index["scene_detection"]
    roi = _optional_box(detection.get("roi"))
    exclusions = tuple(_required_box(item) for item in detection.get("exclude_regions", []))
    hash_size = int(detection["hash_size"])
    started = perf_counter()

    def sample_hash(timestamp: float, boundary_index: int, step: int) -> BoundaryHashSample:
        micros = round(timestamp * 1_000_000)
        output_path = (
            output_dir
            / source_fingerprint
            / f"boundary-{boundary_index:04d}-step-{step:02d}-{micros:012d}.png"
        )
        frame_path, reused = extract_frame_at_timestamp(video_path, output_path, timestamp)
        return BoundaryHashSample(
            timestamp_seconds=timestamp,
            frame_path=str(frame_path),
            frame_hash=difference_hash(
                frame_path,
                roi=roi,
                exclude_regions=exclusions,
                hash_size=hash_size,
            ),
            reused=reused,
        )

    payload = refine_scene_boundaries(
        scene_index,
        iterations=iterations,
        sample_hash=sample_hash,
    )
    payload["metrics"]["boundary_refinement_latency_ms"] = (perf_counter() - started) * 1000.0
    return payload


def refine_scene_boundaries(
    scene_index: dict[str, Any],
    *,
    iterations: int,
    sample_hash: BoundaryHashProvider,
) -> dict[str, Any]:
    if not 1 <= iterations <= 8:
        raise ValueError("Boundary refinement iterations must be between 1 and 8.")
    scenes = scene_index.get("scenes")
    frames = scene_index.get("frames")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Scene index does not contain scenes.")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Scene index does not contain frame decisions.")

    payload = copy.deepcopy(scene_index)
    refined_scenes = payload["scenes"]
    threshold = int(payload["scene_detection"]["hamming_threshold"])
    boundaries: list[dict[str, Any]] = []
    extracted_count = 0
    reused_count = 0

    for boundary_index in range(1, len(refined_scenes)):
        previous = refined_scenes[boundary_index - 1]
        following = refined_scenes[boundary_index]
        previous_scene_id = int(previous["scene_id"])
        following_scene_id = int(following["scene_id"])
        lower = float(previous["last_sample_seconds"])
        upper = float(following["first_sample_seconds"])
        if upper <= lower:
            raise ValueError("Scene sample ranges must increase at each boundary.")
        original_lower = lower
        original_upper = upper
        previous_hash = _representative_hash(frames, previous_scene_id)
        following_hash = _representative_hash(frames, following_scene_id)
        decisions: list[dict[str, Any]] = []

        for step in range(1, iterations + 1):
            timestamp = (lower + upper) / 2.0
            sample = sample_hash(timestamp, boundary_index, step)
            previous_distance = hamming_distance(sample.frame_hash, previous_hash)
            following_distance = hamming_distance(sample.frame_hash, following_hash)
            side, reason = _classify_boundary_sample(
                previous_distance,
                following_distance,
                threshold,
            )
            if side == "previous":
                lower = timestamp
            else:
                upper = timestamp
            extracted_count += int(not sample.reused)
            reused_count += int(sample.reused)
            decisions.append(
                {
                    **asdict(sample),
                    "frame_hash": _format_hash(
                        sample.frame_hash,
                        int(payload["scene_detection"]["hash_size"]),
                    ),
                    "previous_distance": previous_distance,
                    "following_distance": following_distance,
                    "classified_as": side,
                    "classification_reason": reason,
                }
            )

        estimated = (lower + upper) / 2.0
        previous["end_seconds"] = estimated
        following["start_seconds"] = estimated
        boundaries.append(
            {
                "previous_scene_id": previous_scene_id,
                "following_scene_id": following_scene_id,
                "previous_sample_seconds": original_lower,
                "following_sample_seconds": original_upper,
                "refined_seconds": estimated,
                "final_lower_seconds": lower,
                "final_upper_seconds": upper,
                "precision_seconds": upper - lower,
                "decisions": decisions,
            }
        )

    maximum_precision = max(
        (float(item["precision_seconds"]) for item in boundaries),
        default=float(payload["sampling"]["transition_precision_seconds"]),
    )
    payload["sampling"]["transition_precision_seconds"] = maximum_precision
    payload["boundary_refinement"] = {
        "schema_version": 1,
        "iterations": iterations,
        "boundary_count": len(boundaries),
        "extracted_frame_count": extracted_count,
        "reused_frame_count": reused_count,
        "boundaries": boundaries,
    }
    payload.setdefault("metrics", {})["boundary_refinement_latency_ms"] = 0.0
    return payload


def _classify_boundary_sample(
    previous_distance: int,
    following_distance: int,
    threshold: int,
) -> tuple[str, str]:
    previous_match = previous_distance <= threshold
    following_match = following_distance <= threshold
    if previous_match and not following_match:
        return "previous", "within_previous_threshold"
    if following_match and not previous_match:
        return "following", "within_following_threshold"
    if previous_distance <= following_distance:
        return "previous", "nearest_anchor"
    return "following", "nearest_anchor"


def _representative_hash(frames: list[dict[str, Any]], scene_id: int) -> int:
    for frame in frames:
        if int(frame["scene_id"]) == scene_id and bool(frame["is_representative"]):
            return int(str(frame["hash"]), 16)
    raise ValueError(f"Scene {scene_id} does not have a representative hash.")


def _format_hash(value: int, hash_size: int) -> str:
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def _optional_box(raw: Any) -> NormalizedBox | None:
    if raw is None:
        return None
    return _required_box(raw)


def _required_box(raw: Any) -> NormalizedBox:
    if not isinstance(raw, dict):
        raise ValueError("Scene detection box must be a mapping.")
    try:
        return NormalizedBox(
            x=float(raw["x"]),
            y=float(raw["y"]),
            width=float(raw["width"]),
            height=float(raw["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Scene detection box is invalid.") from exc
