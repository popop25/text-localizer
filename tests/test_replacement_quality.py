from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from text_localizer.replacement_quality import (
    evaluate_replacement_quality,
    evaluate_replacement_quality_files,
    load_replacement_benchmark,
    parse_replacement_benchmark,
)

BOX = {"x": 0.2, "y": 0.25, "width": 0.2, "height": 0.05}


def _region(source: str = "강의소개", target: str = "講義紹介") -> dict[str, Any]:
    return {
        "bbox": BOX,
        "source_text": source,
        "translated_text": target,
        "action": "translate",
        "needs_review": False,
        "translation_quality_flags": [],
    }


def _manifest(*, interpolation: str = "linear", replacement: bool = True) -> dict[str, Any]:
    region = _region()
    track: dict[str, Any] = {
        "region_track_id": "region-0001",
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "region": region,
        "keyframes": [
            {
                "timestamp_seconds": 0.0,
                "bbox": {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.05},
                "interpolation": interpolation,
            },
            {
                "timestamp_seconds": 10.0,
                "bbox": {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.05},
                "interpolation": "hold",
            },
        ],
        "overlay_layout": {
            "schema_version": 1,
            "algorithm": "adaptive-fit-v1",
            "text": "講義紹介",
            "text_sha256": "90ee774bbfc413def5f263e0fcf97a70b3655d26b15122e23678cc918f18ef40",
            "lines": ["講義紹介"],
            "line_count": 1,
            "fit_strategy": "source-box",
            "source_font_size_ratio": 0.04,
            "resolved_font_size_ratio": 0.04,
            "font_scale": 1.0,
            "line_height": 1.05,
            "padding_x_ratio": 0.002,
            "padding_y_ratio": 0.002,
            "left_extension_ratio": 0.0,
            "right_extension_ratio": 0.0,
            "top_extension_ratio": 0.0,
            "bottom_extension_ratio": 0.0,
            "target_width_ratio": 0.2,
            "target_height_ratio": 0.05,
            "content_fits": True,
            "layout_needs_review": False,
        },
    }
    if replacement:
        track["overlay_replacement"] = {
            "schema_version": 1,
            "algorithm": "padded-source-box-v1",
            "composite_order": ["erase-source", "draw-translation"],
            "erase_padding_left_ratio": 0.005,
            "erase_padding_right_ratio": 0.005,
            "erase_padding_top_ratio": 0.01,
            "erase_padding_bottom_ratio": 0.005,
        }
    return {
        "schema_version": 2,
        "video": {
            "path": "lecture.mp4",
            "width": 1000,
            "height": 500,
            "fps": 30.0,
            "duration_seconds": 10.0,
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "regions": [region],
            }
        ],
        "metrics": {"needs_review_count": 0},
        "region_track_count": 1,
        "region_tracks": [track],
        "region_tracking": {"algorithm": "event-ocr-keyframes-v2"},
    }


def _benchmark() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "replacement fixture",
        "video": {"duration_seconds": 10.0, "fps": 30.0, "width": 1000, "height": 500},
        "gates": {
            "minimum_frame_pass_rate": 1.0,
            "minimum_region_recall": 1.0,
            "minimum_bbox_iou": 0.9,
            "minimum_source_coverage_ratio": 0.98,
            "maximum_active_region_count_error": 0,
            "maximum_overlay_collisions": 0,
        },
        "frames": [
            {
                "id": "scroll-midpoint",
                "label": "Scroll midpoint",
                "timestamp_seconds": 5.0,
                "expected_active_region_count": 1,
                "regions": [
                    {
                        "id": "title",
                        "label": "Lecture title",
                        "match": {
                            "field": "source_text",
                            "mode": "contains",
                            "patterns": ["강의소개"],
                        },
                        "source_bbox": BOX,
                    }
                ],
            }
        ],
    }


def test_replacement_quality_passes_interpolated_covered_region() -> None:
    report = evaluate_replacement_quality(
        _manifest(), parse_replacement_benchmark(_benchmark())
    )

    assert report["summary"]["frame_pass_rate"] == 1.0
    assert report["summary"]["bbox_iou_minimum"] == pytest.approx(1.0)
    assert report["summary"]["source_coverage_minimum"] == pytest.approx(1.0)
    assert report["summary"]["gates_passed"] is True


def test_replacement_quality_exposes_held_scroll_drift() -> None:
    report = evaluate_replacement_quality(
        _manifest(interpolation="hold"), parse_replacement_benchmark(_benchmark())
    )

    assert report["frames"][0]["regions"][0]["bbox_iou"] == 0.0
    assert report["gate_results"]["bbox_iou"] is False
    assert report["summary"]["gates_passed"] is False


def test_replacement_quality_requires_explicit_source_erasure() -> None:
    report = evaluate_replacement_quality(
        _manifest(replacement=False), parse_replacement_benchmark(_benchmark())
    )

    region = report["frames"][0]["regions"][0]
    assert region["source_coverage_ratio"] == pytest.approx(1.0)
    assert region["replacement_algorithm"] is None
    assert region["passed"] is False
    assert report["summary"]["gates_passed"] is False


def test_replacement_quality_file_runner_writes_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tracks.json"
    benchmark_path = tmp_path / "benchmark.json"
    report_path = tmp_path / "replacement-quality.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")

    report = evaluate_replacement_quality_files(
        manifest_path, benchmark_path, output_path=report_path
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["inputs"]["track_manifest"] == str(manifest_path.resolve())


def test_checked_in_zero_video_replacement_benchmark_is_valid() -> None:
    path = Path(__file__).parents[1] / "config" / "benchmarks" / "0-replacement-quality.json"

    benchmark = load_replacement_benchmark(path)

    assert benchmark.name == "0.mp4 visual replacement fidelity"
    assert benchmark.frames[0].timestamp_seconds == 76.0
    assert len(benchmark.frames[0].regions) >= 6
