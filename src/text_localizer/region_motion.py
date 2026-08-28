from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from text_localizer.track_editor import load_track_manifest, validate_track_manifest
from text_localizer.video_viewer import write_video_viewer


@dataclass(frozen=True, slots=True)
class RegionMotionOptions:
    minimum_segment_seconds: float = 1.0
    minimum_vertical_displacement_ratio: float = 0.005
    maximum_vertical_speed_ratio_per_second: float = 0.01
    maximum_horizontal_speed_ratio_per_second: float = 0.002
    maximum_size_change_ratio: float = 0.4

    def __post_init__(self) -> None:
        if self.minimum_segment_seconds <= 0.0:
            raise ValueError("Motion minimum segment duration must be positive.")
        for value, label in (
            (self.minimum_vertical_displacement_ratio, "minimum vertical displacement"),
            (self.maximum_vertical_speed_ratio_per_second, "maximum vertical speed"),
            (self.maximum_horizontal_speed_ratio_per_second, "maximum horizontal speed"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Motion {label} must be between 0 and 1.")
        if not 0.0 <= self.maximum_size_change_ratio <= 1.0:
            raise ValueError("Motion maximum size change must be between 0 and 1.")


def refine_region_motion(
    manifest_path: Path,
    output_dir: Path,
    *,
    options: RegionMotionOptions | None = None,
) -> dict[str, Any]:
    resolved_options = options or RegionMotionOptions()
    source = load_track_manifest(manifest_path)
    if source.get("schema_version") != 2:
        raise ValueError("Region motion refinement requires schema-v2 region tracks.")
    result = deepcopy(source)
    tracks = [cast(dict[str, Any], item) for item in cast(list[Any], result["region_tracks"])]
    refined_track_count = 0
    refined_segment_count = 0
    records: list[dict[str, Any]] = []
    started = perf_counter()

    for track in tracks:
        keyframes = [
            cast(dict[str, Any], item) for item in cast(list[Any], track["keyframes"])
        ]
        track_segments: list[dict[str, Any]] = []
        for index, current in enumerate(keyframes[:-1]):
            following = keyframes[index + 1]
            if not _is_continuous_scene_motion(current, following, resolved_options):
                continue
            current["interpolation"] = "linear"
            track_segments.append(
                {
                    "start_seconds": float(current["timestamp_seconds"]),
                    "end_seconds": float(following["timestamp_seconds"]),
                    "vertical_displacement_ratio": (
                        float(cast(dict[str, Any], following["bbox"])["y"])
                        - float(cast(dict[str, Any], current["bbox"])["y"])
                    ),
                }
            )
        if not track_segments:
            continue
        refined_track_count += 1
        refined_segment_count += len(track_segments)
        records.append(
            {
                "region_track_id": track["region_track_id"],
                "segments": track_segments,
            }
        )

    current_run_latency_ms = (perf_counter() - started) * 1000.0
    result["region_motion"] = {
        "schema_version": 1,
        "algorithm": "scene-anchor-linear-v1",
        "options": asdict(resolved_options),
        "refined_region_track_count": refined_track_count,
        "refined_segment_count": refined_segment_count,
        "current_run_latency_ms": current_run_latency_ms,
    }
    validate_track_manifest(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.json"
    tracks_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "motion-quality.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": result["region_motion"],
                "tracks": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_video_viewer(
        output_dir / "viewer.html",
        _resolve_video_path(result, manifest_path),
        result,
    )
    return result


def _is_continuous_scene_motion(
    current: dict[str, Any],
    following: dict[str, Any],
    options: RegionMotionOptions,
) -> bool:
    if current.get("source") != "scene" or following.get("source") != "scene":
        return False
    if current.get("interpolation") != "hold":
        return False
    start = float(current["timestamp_seconds"])
    end = float(following["timestamp_seconds"])
    duration = end - start
    if duration < options.minimum_segment_seconds:
        return False
    left = cast(dict[str, Any], current["bbox"])
    right = cast(dict[str, Any], following["bbox"])
    dx = float(right["x"]) - float(left["x"])
    dy = float(right["y"]) - float(left["y"])
    if abs(dy) < options.minimum_vertical_displacement_ratio:
        return False
    if abs(dy) / duration > options.maximum_vertical_speed_ratio_per_second:
        return False
    if abs(dx) / duration > options.maximum_horizontal_speed_ratio_per_second:
        return False
    if abs(dy) < abs(dx) * 2.0:
        return False
    for field in ("width", "height"):
        left_size = float(left[field])
        right_size = float(right[field])
        if left_size <= 0.0 or right_size <= 0.0:
            return False
        relative_change = abs(right_size - left_size) / max(left_size, right_size)
        if relative_change > options.maximum_size_change_ratio:
            return False
    return all(
        math.isfinite(float(box[field]))
        for box in (left, right)
        for field in ("x", "y", "width", "height")
    )


def _resolve_video_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    raw_path = Path(str(cast(dict[str, Any], manifest["video"])["path"]))
    if raw_path.is_absolute():
        return raw_path
    return (manifest_path.parent / raw_path).resolve()
