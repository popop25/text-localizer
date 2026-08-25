from typing import Any

import pytest

from text_localizer.boundaries import BoundaryHashSample, refine_scene_boundaries


def _scene_index() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "video": {
            "path": "lecture.mp4",
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": 3.0,
            "codec": "h264",
        },
        "sampling": {
            "interval_seconds": 1.0,
            "frame_count": 3,
            "transition_precision_seconds": 1.0,
            "samples_reused": False,
        },
        "scene_detection": {
            "hash_algorithm": "dhash",
            "hash_size": 4,
            "hamming_threshold": 0,
            "roi": None,
            "exclude_regions": [],
        },
        "total_frames": 3,
        "unique_scenes": 2,
        "duplicate_frames": 1,
        "reuse_ratio": 1 / 3,
        "frames": [
            {
                "path": "frame-0.png",
                "hash": "0000",
                "scene_id": 1,
                "representative_path": "frame-0.png",
                "anchor_distance": None,
                "previous_frame_distance": None,
                "is_representative": True,
                "sample_index": 0,
                "timestamp_seconds": 0.0,
            },
            {
                "path": "frame-1.png",
                "hash": "0000",
                "scene_id": 1,
                "representative_path": "frame-0.png",
                "anchor_distance": 0,
                "previous_frame_distance": 0,
                "is_representative": False,
                "sample_index": 1,
                "timestamp_seconds": 1.0,
            },
            {
                "path": "frame-2.png",
                "hash": "ffff",
                "scene_id": 2,
                "representative_path": "frame-2.png",
                "anchor_distance": 16,
                "previous_frame_distance": 16,
                "is_representative": True,
                "sample_index": 2,
                "timestamp_seconds": 2.0,
            },
        ],
        "scenes": [
            {
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "first_sample_seconds": 0.0,
                "last_sample_seconds": 1.0,
                "sample_count": 2,
                "representative_path": "frame-0.png",
            },
            {
                "scene_id": 2,
                "start_seconds": 2.0,
                "end_seconds": 3.0,
                "first_sample_seconds": 2.0,
                "last_sample_seconds": 2.0,
                "sample_count": 1,
                "representative_path": "frame-2.png",
            },
        ],
        "metrics": {},
    }


def test_boundary_refinement_bisects_transition_without_ocr() -> None:
    sampled: list[float] = []

    def sample_hash(timestamp: float, boundary_index: int, step: int) -> BoundaryHashSample:
        sampled.append(timestamp)
        assert boundary_index == 1
        return BoundaryHashSample(
            timestamp_seconds=timestamp,
            frame_path=f"step-{step}.png",
            frame_hash=0x0000 if timestamp < 1.625 else 0xFFFF,
        )

    payload = refine_scene_boundaries(
        _scene_index(),
        iterations=3,
        sample_hash=sample_hash,
    )

    boundary = payload["boundary_refinement"]["boundaries"][0]
    assert sampled == [1.5, 1.75, 1.625]
    assert boundary["final_lower_seconds"] == 1.5
    assert boundary["final_upper_seconds"] == 1.625
    assert boundary["refined_seconds"] == 1.5625
    assert boundary["precision_seconds"] == 0.125
    assert payload["scenes"][0]["end_seconds"] == 1.5625
    assert payload["scenes"][1]["start_seconds"] == 1.5625
    assert payload["sampling"]["transition_precision_seconds"] == 0.125
    assert payload["boundary_refinement"]["extracted_frame_count"] == 3


def test_boundary_refinement_iteration_limit_is_validated() -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        refine_scene_boundaries(
            _scene_index(),
            iterations=0,
            sample_hash=lambda *_: BoundaryHashSample(0.0, "unused.png", 0),
        )
