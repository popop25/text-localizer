from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from text_localizer.domain import NormalizedBox
from text_localizer.profile import VisualEventOptions
from text_localizer.video import (
    TimestampedFrame,
    VideoInfo,
    extract_timestamped_frames,
    inspect_video,
)


@dataclass(slots=True)
class _EventAccumulator:
    event_id: str
    start_seconds: float
    end_seconds: float
    first_frame_index: int
    last_frame_index: int
    changed_interval_count: int = 0
    changed_tiles: set[tuple[int, int]] = field(default_factory=set)
    peak_changed_tile_count: int = 0
    peak_changed_ratio: float = 0.0
    peak_tile_score: float = 0.0

    def add(
        self,
        *,
        end_seconds: float,
        frame_index: int,
        changed_tiles: tuple[tuple[int, int, float], ...],
        tile_count: int,
    ) -> None:
        self.end_seconds = end_seconds
        self.last_frame_index = frame_index
        self.changed_interval_count += 1
        self.changed_tiles.update((column, row) for column, row, _score in changed_tiles)
        self.peak_changed_tile_count = max(
            self.peak_changed_tile_count,
            len(changed_tiles),
        )
        self.peak_changed_ratio = max(
            self.peak_changed_ratio,
            len(changed_tiles) / tile_count,
        )
        self.peak_tile_score = max(
            self.peak_tile_score,
            max((score for _column, _row, score in changed_tiles), default=0.0),
        )

    def to_dict(self, global_change_ratio: float) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.end_seconds - self.start_seconds,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "changed_interval_count": self.changed_interval_count,
            "classification": (
                "global" if self.peak_changed_ratio >= global_change_ratio else "partial"
            ),
            "changed_tiles": [
                {"column": column, "row": row}
                for column, row in sorted(self.changed_tiles, key=lambda item: (item[1], item[0]))
            ],
            "peak_changed_tile_count": self.peak_changed_tile_count,
            "peak_changed_ratio": self.peak_changed_ratio,
            "peak_tile_score": self.peak_tile_score,
        }


def index_video_visual_events(
    video_path: Path,
    work_dir: Path,
    *,
    profile_name: str | None,
    roi: NormalizedBox | None,
    options: VisualEventOptions,
) -> dict[str, Any]:
    info = inspect_video(video_path)
    frames, extraction_latency_ms, reused_samples = extract_timestamped_frames(
        video_path,
        work_dir / "frames",
        options.sample_interval_seconds,
        video_info=info,
    )
    payload = index_visual_events(
        frames,
        info,
        profile_name=profile_name,
        roi=roi,
        options=options,
        extraction_latency_ms=extraction_latency_ms,
        reused_samples=reused_samples,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "visual-events.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def index_visual_events(
    frames: tuple[TimestampedFrame, ...],
    video_info: VideoInfo,
    *,
    profile_name: str | None,
    roi: NormalizedBox | None,
    options: VisualEventOptions,
    extraction_latency_ms: float = 0.0,
    reused_samples: bool = False,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("Visual-event indexing requires at least one frame.")
    started = perf_counter()
    active_roi = roi or NormalizedBox(0.0, 0.0, 1.0, 1.0)
    tile_count = options.tile_columns * options.tile_rows
    frame_records: list[dict[str, Any]] = []
    accumulators: list[_EventAccumulator] = []
    active_event: _EventAccumulator | None = None
    previous_image: Image.Image | None = None
    previous_frame: TimestampedFrame | None = None
    analysis_size: tuple[int, int] | None = None
    changed_interval_count = 0

    for position, frame in enumerate(frames):
        if frame.index != position:
            raise ValueError("Visual-event frame indices must be contiguous and zero-based.")
        if (
            previous_frame is not None
            and frame.timestamp_seconds <= previous_frame.timestamp_seconds
        ):
            raise ValueError("Visual-event frame timestamps must be strictly increasing.")
        current_image = _prepare_frame(
            Path(frame.path),
            roi=active_roi,
            exclusions=options.exclude_regions,
            analysis_width=options.analysis_width,
            minimum_height=options.tile_rows,
        )
        if analysis_size is None:
            analysis_size = current_image.size
        elif current_image.size != analysis_size:
            raise ValueError("Visual-event frames must have the same dimensions.")

        record: dict[str, Any] = {
            "frame_index": frame.index,
            "timestamp_seconds": frame.timestamp_seconds,
            "path": frame.path,
            "interval_start_seconds": (
                previous_frame.timestamp_seconds if previous_frame is not None else None
            ),
            "event_id": None,
            "changed": False,
            "changed_tile_count": 0,
            "changed_ratio": 0.0,
            "mean_tile_score": 0.0,
            "peak_tile_score": 0.0,
            "changed_tiles": [],
        }
        if previous_image is not None and previous_frame is not None:
            scores = _tile_scores(
                previous_image,
                current_image,
                columns=options.tile_columns,
                rows=options.tile_rows,
            )
            changed_tiles = tuple(
                (
                    index % options.tile_columns,
                    index // options.tile_columns,
                    score,
                )
                for index, score in enumerate(scores)
                if score >= options.tile_change_threshold
            )
            is_changed = len(changed_tiles) >= options.minimum_changed_tiles
            record.update(
                {
                    "changed": is_changed,
                    "changed_tile_count": len(changed_tiles),
                    "changed_ratio": len(changed_tiles) / tile_count,
                    "mean_tile_score": statistics.fmean(scores),
                    "peak_tile_score": max(scores),
                    "changed_tiles": [
                        {
                            "column": column,
                            "row": row,
                            "score": round(score, 6),
                        }
                        for column, row, score in changed_tiles
                    ],
                }
            )
            if is_changed:
                changed_interval_count += 1
                if active_event is None:
                    active_event = _EventAccumulator(
                        event_id=f"visual-event-{len(accumulators) + 1:04d}",
                        start_seconds=previous_frame.timestamp_seconds,
                        end_seconds=frame.timestamp_seconds,
                        first_frame_index=previous_frame.index,
                        last_frame_index=frame.index,
                    )
                    accumulators.append(active_event)
                active_event.add(
                    end_seconds=frame.timestamp_seconds,
                    frame_index=frame.index,
                    changed_tiles=changed_tiles,
                    tile_count=tile_count,
                )
                record["event_id"] = active_event.event_id
            else:
                active_event = None

        frame_records.append(record)
        previous_image = current_image
        previous_frame = frame

    if analysis_size is None:
        raise RuntimeError("Visual-event analysis size was not initialized.")
    events = [item.to_dict(options.global_change_ratio) for item in accumulators]
    partial_event_count = sum(item["classification"] == "partial" for item in events)
    global_event_count = len(events) - partial_event_count
    processing_latency_ms = (perf_counter() - started) * 1000.0
    return {
        "schema_version": 1,
        "video": video_info.to_dict(),
        "profile_name": profile_name,
        "sampling": {
            "interval_seconds": options.sample_interval_seconds,
            "frame_count": len(frames),
            "reused": reused_samples,
        },
        "analysis": {
            "roi": asdict(active_roi),
            "exclude_regions": [asdict(item) for item in options.exclude_regions],
            "width": analysis_size[0],
            "height": analysis_size[1],
        },
        "grid": {
            "columns": options.tile_columns,
            "rows": options.tile_rows,
            "tile_count": tile_count,
            "tile_change_threshold": options.tile_change_threshold,
            "minimum_changed_tiles": options.minimum_changed_tiles,
            "global_change_ratio": options.global_change_ratio,
            "tiles": _grid_tiles(active_roi, options.tile_columns, options.tile_rows),
        },
        "event_count": len(events),
        "events": events,
        "frames": frame_records,
        "metrics": {
            "frame_extraction_latency_ms": extraction_latency_ms,
            "visual_analysis_latency_ms": processing_latency_ms,
            "total_indexing_latency_ms": extraction_latency_ms + processing_latency_ms,
            "comparison_count": max(0, len(frames) - 1),
            "changed_interval_count": changed_interval_count,
            "stable_interval_count": max(0, len(frames) - 1 - changed_interval_count),
            "partial_event_count": partial_event_count,
            "global_event_count": global_event_count,
        },
    }


def _prepare_frame(
    path: Path,
    *,
    roi: NormalizedBox,
    exclusions: tuple[NormalizedBox, ...],
    analysis_width: int,
    minimum_height: int,
) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Visual-event frame does not exist: {path}")
    with Image.open(path) as source:
        grayscale = source.convert("L")
    left, top, right, bottom = _pixel_bounds(roi, grayscale.width, grayscale.height)
    cropped = grayscale.crop((left, top, right, bottom))
    if exclusions:
        draw = ImageDraw.Draw(cropped)
        for exclusion in exclusions:
            intersection = _intersection_bounds(
                exclusion,
                crop_left=left,
                crop_top=top,
                crop_right=right,
                crop_bottom=bottom,
                frame_width=grayscale.width,
                frame_height=grayscale.height,
            )
            if intersection is not None:
                draw.rectangle(intersection, fill=0)
    target_height = max(
        minimum_height,
        round(cropped.height * analysis_width / cropped.width),
    )
    return cropped.resize(
        (analysis_width, target_height),
        Image.Resampling.BILINEAR,
    )


def _tile_scores(
    previous: Image.Image,
    current: Image.Image,
    *,
    columns: int,
    rows: int,
) -> tuple[float, ...]:
    difference = ImageChops.difference(previous, current)
    pooled = difference.resize((columns, rows), Image.Resampling.BOX)
    return tuple(value / 255.0 for value in pooled.tobytes())


def _grid_tiles(
    roi: NormalizedBox,
    columns: int,
    rows: int,
) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "row": row,
            "bbox": {
                "x": roi.x + roi.width * column / columns,
                "y": roi.y + roi.height * row / rows,
                "width": roi.width / columns,
                "height": roi.height / rows,
            },
        }
        for row in range(rows)
        for column in range(columns)
    ]


def _pixel_bounds(
    box: NormalizedBox,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = min(width - 1, max(0, round(box.x * width)))
    top = min(height - 1, max(0, round(box.y * height)))
    right = min(width, max(left + 1, round((box.x + box.width) * width)))
    bottom = min(height, max(top + 1, round((box.y + box.height) * height)))
    return left, top, right, bottom


def _intersection_bounds(
    box: NormalizedBox,
    *,
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    box_left, box_top, box_right, box_bottom = _pixel_bounds(
        box,
        frame_width,
        frame_height,
    )
    left = max(crop_left, box_left)
    top = max(crop_top, box_top)
    right = min(crop_right, box_right)
    bottom = min(crop_bottom, box_bottom)
    if right <= left or bottom <= top:
        return None
    return (
        left - crop_left,
        top - crop_top,
        right - crop_left - 1,
        bottom - crop_top - 1,
    )
