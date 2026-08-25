from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from PIL import Image, ImageDraw

from text_localizer.domain import NormalizedBox
from text_localizer.profile import SceneDetectionOptions
from text_localizer.quality import box_to_dict


@dataclass(frozen=True, slots=True)
class SceneFrame:
    path: str
    hash: str
    scene_id: int
    representative_path: str
    anchor_distance: int | None
    previous_frame_distance: int | None
    is_representative: bool


def index_frame_scenes(
    frame_paths: list[Path],
    *,
    profile_name: str | None,
    roi: NormalizedBox | None,
    exclude_regions: tuple[NormalizedBox, ...],
    options: SceneDetectionOptions,
) -> dict[str, Any]:
    if not frame_paths:
        raise ValueError("At least one frame is required for scene indexing.")
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frame does not exist: {missing[0]}")

    started = perf_counter()
    records: list[SceneFrame] = []
    representative_path: Path | None = None
    representative_hash: int | None = None
    previous_hash: int | None = None
    scene_id = 0
    width = math.ceil(options.hash_size * options.hash_size / 4)

    for path in frame_paths:
        frame_hash = difference_hash(
            path,
            roi=roi,
            exclude_regions=exclude_regions,
            hash_size=options.hash_size,
        )
        anchor_distance = (
            None
            if representative_hash is None
            else hamming_distance(frame_hash, representative_hash)
        )
        previous_frame_distance = (
            None if previous_hash is None else hamming_distance(frame_hash, previous_hash)
        )
        is_representative = anchor_distance is None or anchor_distance > options.hamming_threshold
        if is_representative:
            scene_id += 1
            representative_path = path.resolve()
            representative_hash = frame_hash
        if representative_path is None:
            raise RuntimeError("Scene representative was not initialized.")
        records.append(
            SceneFrame(
                path=str(path.resolve()),
                hash=f"{frame_hash:0{width}x}",
                scene_id=scene_id,
                representative_path=str(representative_path),
                anchor_distance=anchor_distance,
                previous_frame_distance=previous_frame_distance,
                is_representative=is_representative,
            )
        )
        previous_hash = frame_hash

    representatives = [item.path for item in records if item.is_representative]
    duplicate_frames = len(records) - len(representatives)
    return {
        "schema_version": 1,
        "profile_name": profile_name,
        "hash_algorithm": "dhash",
        "hash_size": options.hash_size,
        "hamming_threshold": options.hamming_threshold,
        "roi": box_to_dict(roi),
        "exclude_regions": [box_to_dict(box) for box in exclude_regions],
        "total_frames": len(records),
        "unique_scenes": len(representatives),
        "duplicate_frames": duplicate_frames,
        "reuse_ratio": duplicate_frames / len(records),
        "processing_latency_ms": (perf_counter() - started) * 1000.0,
        "representative_paths": representatives,
        "frames": [asdict(item) for item in records],
    }


def index_frame_directory(
    directory: Path,
    *,
    pattern: str,
    output_path: Path,
    profile_name: str | None,
    roi: NormalizedBox | None,
    exclude_regions: tuple[NormalizedBox, ...],
    options: SceneDetectionOptions,
) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {directory}")
    frame_paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    payload = index_frame_scenes(
        frame_paths,
        profile_name=profile_name,
        roi=roi,
        exclude_regions=exclude_regions,
        options=options,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def difference_hash(
    image_path: Path,
    *,
    roi: NormalizedBox | None,
    exclude_regions: tuple[NormalizedBox, ...],
    hash_size: int,
) -> int:
    if not image_path.is_file():
        raise FileNotFoundError(f"Frame does not exist: {image_path}")
    if not 4 <= hash_size <= 64:
        raise ValueError("Hash size must be between 4 and 64.")

    with Image.open(image_path) as source:
        grayscale = source.convert("L")
    width, height = grayscale.size
    left, top, right, bottom = _pixel_bounds(roi, width, height)
    cropped = grayscale.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(cropped)
    for excluded in exclude_regions:
        mask = _intersection_bounds(excluded, left, top, right, bottom, width, height)
        if mask is not None:
            draw.rectangle(mask, fill=0)

    resized = cropped.resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    flattened = getattr(resized, "get_flattened_data", None)
    if callable(flattened):
        pixel_values = cast(Callable[[], Iterable[int]], flattened)()
    else:
        pixel_values = cast(Iterable[int], resized.getdata())
    pixels = list(pixel_values)
    value = 0
    for y in range(hash_size):
        row = y * (hash_size + 1)
        for x in range(hash_size):
            value = (value << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _pixel_bounds(
    roi: NormalizedBox | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    left = max(0, math.floor(roi.x * width))
    top = max(0, math.floor(roi.y * height))
    right = min(width, math.ceil((roi.x + roi.width) * width))
    bottom = min(height, math.ceil((roi.y + roi.height) * height))
    if right <= left or bottom <= top:
        raise ValueError("ROI resolves to an empty pixel area.")
    return left, top, right, bottom


def _intersection_bounds(
    box: NormalizedBox,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    left = max(crop_left, math.floor(box.x * image_width))
    top = max(crop_top, math.floor(box.y * image_height))
    right = min(crop_right, math.ceil((box.x + box.width) * image_width))
    bottom = min(crop_bottom, math.ceil((box.y + box.height) * image_height))
    if right <= left or bottom <= top:
        return None
    return left - crop_left, top - crop_top, right - crop_left, bottom - crop_top
