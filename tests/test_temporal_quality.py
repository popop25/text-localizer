import json
from pathlib import Path
from typing import Any

import pytest

from text_localizer.temporal_quality import (
    evaluate_temporal_quality,
    evaluate_temporal_quality_files,
    load_temporal_benchmark,
    parse_temporal_benchmark,
)

BOX = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1}


def _region(*, bbox: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "action": "translate",
        "source_text": "강의 소개",
        "translated_text": "講義紹介",
        "needs_review": False,
        "bbox": bbox or BOX,
    }


def _manifest(region_sets: list[list[dict[str, Any]]]) -> dict[str, Any]:
    track_count = len(region_sets)
    return {
        "schema_version": 1,
        "video": {
            "path": "lecture.mp4",
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": float(track_count),
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
        "track_count": track_count,
        "tracks": [
            {
                "track_id": f"scene-{index + 1:04d}",
                "start_seconds": float(index),
                "end_seconds": float(index + 1),
                "regions": regions,
            }
            for index, regions in enumerate(region_sets)
        ],
        "metrics": {"needs_review_count": 0},
    }


def _benchmark(
    duration: float,
    *,
    bbox: dict[str, float] | None = None,
    evaluate_boundaries: bool = True,
) -> dict[str, Any]:
    keyframe_times = [0.5] if duration <= 1.0 else [0.5, duration - 0.5]
    return {
        "schema_version": 1,
        "name": "synthetic-temporal-quality",
        "video": {"duration_seconds": duration, "fps": 30.0},
        "gates": {
            "minimum_annotation_recall": 0.95,
            "minimum_keyframe_recall": 0.95,
            "maximum_boundary_error_ms_p95": 67.0,
            "minimum_bbox_iou_median": 0.75,
            "maximum_flicker_events": 0,
        },
        "annotations": [
            {
                "id": "title",
                "label": "Lecture title",
                "start_seconds": 0.0,
                "end_seconds": duration,
                "evaluate_boundaries": evaluate_boundaries,
                "match": {
                    "field": "source_text",
                    "mode": "contains",
                    "patterns": ["강의소개"],
                },
                "keyframes": [
                    {"timestamp_seconds": timestamp, "bbox": bbox or BOX}
                    for timestamp in keyframe_times
                ],
            }
        ],
    }


def test_temporal_quality_passes_contiguous_exact_observations() -> None:
    manifest = _manifest([[_region()], [_region()], [_region()]])
    benchmark = parse_temporal_benchmark(_benchmark(3.0))

    report = evaluate_temporal_quality(manifest, benchmark)

    assert report["summary"] == {
        "annotation_count": 1,
        "matched_annotation_count": 1,
        "annotation_recall": 1.0,
        "keyframe_count": 2,
        "matched_keyframe_count": 2,
        "keyframe_recall": 1.0,
        "boundary_sample_count": 2,
        "boundary_error_ms_p95": 0.0,
        "bbox_iou_median": 1.0,
        "flicker_event_count": 0,
        "gates_passed": True,
    }
    assert all(report["gate_results"].values())


def test_temporal_quality_reads_schema_v2_region_keyframes() -> None:
    manifest = _manifest([[_region()], [_region()], [_region()]])
    manifest["schema_version"] = 2
    manifest["region_track_count"] = 1
    manifest["region_tracks"] = [
        {
            "region_track_id": "region-0001",
            "start_seconds": 0.0,
            "end_seconds": 3.0,
            "region": _region(),
            "keyframes": [
                {
                    "timestamp_seconds": 0.0,
                    "bbox": BOX,
                    "interpolation": "hold",
                },
                {
                    "timestamp_seconds": 1.0,
                    "bbox": BOX,
                    "interpolation": "linear",
                },
            ],
        }
    ]

    report = evaluate_temporal_quality(manifest, parse_temporal_benchmark(_benchmark(3.0)))

    assert report["track_manifest_schema_version"] == 2
    assert report["summary"]["keyframe_recall"] == 1.0
    assert report["summary"]["gates_passed"] is True


def test_temporal_quality_exposes_missing_keyframe_and_flicker() -> None:
    manifest = _manifest([[_region()], [], [_region()]])
    payload = _benchmark(3.0)
    payload["annotations"][0]["keyframes"] = [
        {"timestamp_seconds": 0.5, "bbox": BOX},
        {"timestamp_seconds": 1.5, "bbox": BOX},
        {"timestamp_seconds": 2.5, "bbox": BOX},
    ]

    report = evaluate_temporal_quality(manifest, parse_temporal_benchmark(payload))

    assert report["summary"]["annotation_recall"] == 0.0
    assert report["summary"]["keyframe_recall"] == pytest.approx(2 / 3)
    assert report["summary"]["flicker_event_count"] == 1
    assert report["summary"]["gates_passed"] is False
    assert report["gate_results"]["annotation_recall"] is False
    assert report["gate_results"]["keyframe_recall"] is False
    assert report["gate_results"]["flicker_events"] is False


def test_temporal_quality_absence_annotation_detects_leaked_overlay() -> None:
    payload = _benchmark(3.0)
    payload["annotations"].append(
        {
            "id": "company-logo-absent",
            "label": "Company logo does not leak into the lecture page",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "expect_visible": False,
            "evaluate_boundaries": False,
            "match": {
                "field": "source_text",
                "mode": "contains",
                "patterns": ["주식회사"],
            },
        }
    )
    company = _region()
    company["source_text"] = "주식회사"
    company["translated_text"] = "株式会社"
    manifest = _manifest([[_region()], [_region(), company], [_region(), company]])

    report = evaluate_temporal_quality(manifest, parse_temporal_benchmark(payload))

    absence = report["annotations"][1]
    assert absence["expect_visible"] is False
    assert absence["observed_coverage_ratio"] == 1.0
    assert absence["coverage_ratio"] == 0.0
    assert absence["coverage_passed"] is False
    assert absence["flicker_event_count"] == 1
    assert report["summary"]["gates_passed"] is False


def test_temporal_quality_absence_annotation_passes_without_matching_overlay() -> None:
    payload = _benchmark(3.0)
    payload["annotations"].append(
        {
            "id": "company-logo-absent",
            "label": "Company logo does not leak into the lecture page",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "expect_visible": False,
            "evaluate_boundaries": False,
            "match": {
                "field": "source_text",
                "mode": "contains",
                "patterns": ["주식회사"],
            },
        }
    )

    report = evaluate_temporal_quality(
        _manifest([[_region()], [_region()], [_region()]]),
        parse_temporal_benchmark(payload),
    )

    absence = report["annotations"][1]
    assert absence["observed_coverage_ratio"] == 0.0
    assert absence["coverage_ratio"] == 1.0
    assert absence["coverage_passed"] is True
    assert absence["flicker_event_count"] == 0
    assert report["summary"]["gates_passed"] is True


def test_temporal_quality_exposes_spatial_drift_without_boundary_samples() -> None:
    manifest = _manifest([[_region()]])
    expected_box = {"x": 0.7, "y": 0.7, "width": 0.2, "height": 0.1}
    benchmark = parse_temporal_benchmark(
        _benchmark(1.0, bbox=expected_box, evaluate_boundaries=False)
    )

    report = evaluate_temporal_quality(manifest, benchmark)

    assert report["summary"]["boundary_sample_count"] == 0
    assert report["summary"]["boundary_error_ms_p95"] is None
    assert report["summary"]["bbox_iou_median"] == 0.0
    assert report["gate_results"]["boundary_error_ms_p95"] is True
    assert report["gate_results"]["bbox_iou_median"] is False
    assert report["summary"]["gates_passed"] is False


def test_temporal_quality_file_runner_writes_auditable_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tracks.json"
    benchmark_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(_manifest([[_region()]])), encoding="utf-8")
    benchmark_path.write_text(json.dumps(_benchmark(1.0)), encoding="utf-8")

    report = evaluate_temporal_quality_files(
        manifest_path,
        benchmark_path,
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["inputs"]["track_manifest"] == str(manifest_path.resolve())
    assert saved["summary"]["gates_passed"] is True


def test_temporal_benchmark_rejects_duplicate_annotation_ids() -> None:
    payload = _benchmark(1.0)
    payload["annotations"].append(dict(payload["annotations"][0]))

    with pytest.raises(ValueError, match="Duplicate annotation id"):
        parse_temporal_benchmark(payload)


def test_checked_in_zero_video_benchmark_is_valid() -> None:
    benchmark_path = (
        Path(__file__).parents[1] / "config" / "benchmarks" / ("0-temporal-quality.json")
    )

    benchmark = load_temporal_benchmark(benchmark_path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))

    assert benchmark.name == "0.mp4 temporal overlay baseline"
    assert payload["ground_truth"]["revision"] == 3
    assert len(benchmark.annotations) == 6
    assert sum(len(item.keyframes) for item in benchmark.annotations) == 18
    assert sum(not item.expect_visible for item in benchmark.annotations) == 1
