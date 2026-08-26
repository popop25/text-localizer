from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from text_localizer.domain import NormalizedBox, Point
from text_localizer.glossary import Glossary
from text_localizer.providers.base import OcrOutput, OcrRegion
from text_localizer.quality import AnalysisOptions
from text_localizer.region_tracks import (
    RegionTrackingOptions,
    build_region_track_manifest,
    localize_region_tracks,
    select_tracking_frames,
)
from text_localizer.track_editor import validate_track_manifest


def _region(text: str, y: float) -> dict[str, Any]:
    return {
        "bbox": {"x": 0.25, "y": y, "width": 0.2, "height": 0.04},
        "source_text": text,
        "translated_text": "LangChain 基礎",
        "ocr_confidence": 0.95,
        "action": "translate",
        "translation_quality_flags": [],
        "needs_review": False,
    }


def _base_manifest(video_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "video": {
            "path": str(video_path),
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": 10.0,
            "codec": "h264",
        },
        "sampling": {"interval_seconds": 1.0, "transition_precision_seconds": 0.1},
        "track_count": 2,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "regions": [_region("5 [실습] LangChain 기초", 0.5)],
            },
            {
                "track_id": "scene-0002",
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "regions": [_region("[실습] LangChain 기초", 0.4)],
            },
        ],
        "metrics": {"needs_review_count": 0},
    }


def _visual_manifest(frame_paths: list[Path]) -> dict[str, Any]:
    tiles = [
        {
            "column": column,
            "row": row,
            "bbox": {
                "x": column * 0.5,
                "y": row * 0.5,
                "width": 0.5,
                "height": 0.5,
            },
        }
        for row in range(2)
        for column in range(2)
    ]
    return {
        "schema_version": 1,
        "video": {"duration_seconds": 10.0, "fps": 30.0},
        "grid": {"columns": 2, "rows": 2, "tiles": tiles},
        "frames": [
            {
                "frame_index": index,
                "timestamp_seconds": index * 0.2,
                "path": str(path),
                "changed": index in {3, 4},
            }
            for index, path in enumerate(frame_paths)
        ],
        "events": [
            {
                "event_id": "visual-event-0001",
                "start_seconds": 0.4,
                "end_seconds": 0.8,
                "duration_seconds": 0.4,
                "first_frame_index": 2,
                "last_frame_index": 4,
                "peak_changed_ratio": 0.5,
                "changed_tiles": [{"column": 0, "row": 0}],
            }
        ],
    }


def test_select_tracking_frames_uses_pre_changed_and_post_frames(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame-{index}.png" for index in range(6)]

    selected = select_tracking_frames(_visual_manifest(paths), RegionTrackingOptions())

    assert [item.frame_index for item in selected] == [1, 3, 4, 5]
    assert [item.phase for item in selected] == ["pre", "changed", "changed", "post"]
    assert selected[0].roi == NormalizedBox(0.0, 0.0, 0.5, 0.5)
    assert all(item.event_start_seconds == 0.4 for item in selected)


def test_select_tracking_frames_supports_disjoint_include_windows(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame-{index}.png" for index in range(6)]
    visual_manifest = _visual_manifest(paths)

    excluded = select_tracking_frames(
        visual_manifest,
        RegionTrackingOptions(include_windows=((2.0, 3.0),)),
    )
    included = select_tracking_frames(
        visual_manifest,
        RegionTrackingOptions(include_windows=((0.3, 0.5), (2.0, 3.0))),
    )

    assert excluded == ()
    assert [item.frame_index for item in included] == [1, 3, 4, 5]


def test_region_tracking_options_reject_windows_with_contiguous_bounds() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        RegionTrackingOptions(start_seconds=1.0, include_windows=((2.0, 3.0),))


@pytest.mark.parametrize(
    "windows",
    [
        ((2.0, 3.0), (1.0, 1.5)),
        ((1.0, 2.0), (1.5, 3.0)),
        ((1.0, 2.0), (2.0, 3.0)),
    ],
)
def test_region_tracking_options_reject_unordered_or_overlapping_windows(
    windows: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        RegionTrackingOptions(include_windows=windows)


def test_build_region_tracks_merges_ocr_variants_and_injects_event_keyframes(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path / "video.mp4")
    observations = [
        {
            "event_id": "visual-event-0001",
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "phase": phase,
            "regions": [
                {
                    "source_text": "LangChain 기초",
                    "ocr_confidence": 0.9,
                    "bbox": {"x": 0.25, "y": y, "width": 0.2, "height": 0.04},
                }
            ],
        }
        for index, timestamp, phase, y in (
            (24, 4.8, "pre", 0.5),
            (25, 5.0, "changed", 0.45),
            (26, 5.2, "post", 0.4),
        )
    ]

    result = build_region_track_manifest(
        manifest,
        observations,
        visual_event_manifest_path=tmp_path / "visual-events.json",
        options=RegionTrackingOptions(),
    )

    validate_track_manifest(result)
    assert result["schema_version"] == 2
    assert result["region_track_count"] == 1
    track = result["region_tracks"][0]
    assert track["start_seconds"] == 0.0
    assert track["end_seconds"] == 10.0
    assert [item["timestamp_seconds"] for item in track["keyframes"]] == [
        0.0,
        4.8,
        5.0,
        5.2,
    ]
    assert [item["interpolation"] for item in track["keyframes"][-3:]] == [
        "linear",
        "linear",
        "hold",
    ]
    assert track["region"]["source_text"] == "[실습] LangChain 기초"
    assert track["text_consensus"]["observed_consensus_text"] == "LangChain 기초"
    assert track["text_consensus"]["observation_count"] == 4
    assert result["region_tracking"]["text_consensus_track_count"] == 1


def test_build_region_tracks_selects_consensus_source_and_matching_translation(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path / "video.mp4")
    correct = _region("Rest API 사용 경험 및 LLM 기본 개념 이해", 0.5)
    correct["translated_text"] = "Rest API の使用経験と LLM の基本概念の理解"
    correct["ocr_confidence"] = 0.88
    corrupt = _region("YReSt AP 사용 경험 및 LLM 기본 개념 이해", 0.4)
    corrupt["translated_text"] = "YReSt AP を使用する経験"
    corrupt["ocr_confidence"] = 0.99
    manifest["tracks"][0]["regions"] = [correct]
    manifest["tracks"][1]["regions"] = [corrupt]
    observations = [
        {
            "event_id": "visual-event-0001",
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "phase": phase,
            "regions": [
                {
                    "source_text": "Rest API 사용 경험 및 LLM 기본 개념 이해",
                    "ocr_confidence": 0.9,
                    "bbox": {"x": 0.25, "y": y, "width": 0.2, "height": 0.04},
                }
            ],
        }
        for index, timestamp, phase, y in (
            (9, 1.8, "pre", 0.5),
            (10, 2.0, "changed", 0.48),
            (11, 2.2, "post", 0.46),
        )
    ]

    result = build_region_track_manifest(
        manifest,
        observations,
        visual_event_manifest_path=tmp_path / "visual-events.json",
        options=RegionTrackingOptions(),
    )

    track = result["region_tracks"][0]
    assert track["region"]["source_text"].startswith("Rest API")
    assert track["region"]["translation_source_text"].startswith("Rest API")
    assert track["region"]["translated_text"] == correct["translated_text"]
    assert track["region"]["needs_review"] is False
    assert track["text_consensus"]["unique_text_count"] == 2
    assert result["region_tracking"]["text_consensus_changed_track_count"] == 1


def test_build_region_tracks_anchors_repeated_ocr_identifier_error_to_translation_source(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path / "video.mp4")
    correct = _region("Azure OpenAI & LangChain 실전", 0.5)
    correct["translated_text"] = "Azure OpenAI & LangChain 実践"
    manifest["tracks"][0]["regions"] = [correct]
    manifest["tracks"] = manifest["tracks"][:1]
    manifest["tracks"][0]["end_seconds"] = 2.0
    manifest["track_count"] = 1
    manifest["video"]["duration_seconds"] = 2.0
    observations = [
        {
            "event_id": "visual-event-0001",
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "phase": phase,
            "regions": [
                {
                    "source_text": "Azure OpenAl & LangChain 실전",
                    "ocr_confidence": 0.98,
                    "bbox": {"x": 0.25, "y": 0.5, "width": 0.2, "height": 0.04},
                }
            ],
        }
        for index, timestamp, phase in (
            (9, 0.8, "pre"),
            (10, 1.0, "changed"),
            (11, 1.2, "post"),
        )
    ]

    result = build_region_track_manifest(
        manifest,
        observations,
        visual_event_manifest_path=tmp_path / "visual-events.json",
        options=RegionTrackingOptions(),
    )

    track = result["region_tracks"][0]
    assert track["text_consensus"]["observed_consensus_text"] == ("Azure OpenAl & LangChain 실전")
    assert track["region"]["source_text"] == "Azure OpenAI & LangChain 실전"
    assert track["region"]["translation_source_text"] == "Azure OpenAI & LangChain 실전"
    assert track["region"]["translated_text"] == "Azure OpenAI & LangChain 実践"


def test_build_region_tracks_ends_after_trailing_event_misses(
    tmp_path: Path,
) -> None:
    manifest = _base_manifest(tmp_path / "video.mp4")
    manifest["video"]["duration_seconds"] = 3.0
    manifest["track_count"] = 3
    manifest["tracks"] = [
        {
            "track_id": "scene-0001",
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "regions": [_region("주식회사", 0.5)],
        },
        {
            "track_id": "scene-0002",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "regions": [_region("주식회사", 0.45)],
        },
        {
            "track_id": "scene-0003",
            "start_seconds": 2.0,
            "end_seconds": 3.0,
            "regions": [_region("주식회사", 0.4)],
        },
    ]
    observations = [
        {
            "event_id": "visual-event-0001",
            "frame_index": 4,
            "timestamp_seconds": 0.8,
            "phase": "pre",
            "regions": [
                {
                    "source_text": "주식회사",
                    "ocr_confidence": 0.95,
                    "bbox": {"x": 0.25, "y": 0.5, "width": 0.2, "height": 0.04},
                }
            ],
        },
        {
            "event_id": "visual-event-0001",
            "frame_index": 5,
            "timestamp_seconds": 1.0,
            "phase": "changed",
            "regions": [],
        },
        {
            "event_id": "visual-event-0001",
            "frame_index": 6,
            "timestamp_seconds": 1.2,
            "phase": "post",
            "regions": [],
        },
    ]

    result = build_region_track_manifest(
        manifest,
        observations,
        visual_event_manifest_path=tmp_path / "visual-events.json",
        options=RegionTrackingOptions(),
    )

    assert result["region_track_count"] == 1
    track = result["region_tracks"][0]
    assert track["start_seconds"] == 0.0
    assert track["end_seconds"] == 1.0
    assert [item["timestamp_seconds"] for item in track["keyframes"]] == [0.0, 0.8]
    assert result["region_tracking"]["visibility_adjusted_track_count"] == 1


class FakeOcrProvider:
    name = "fake-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, _image_path: Path) -> OcrOutput:
        self.calls += 1
        bbox = NormalizedBox(0.1, 0.1, 0.6, 0.2)
        return OcrOutput(
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.1),
                        Point(0.7, 0.1),
                        Point(0.7, 0.3),
                        Point(0.1, 0.3),
                    ),
                    bbox=bbox,
                    source_text="LangChain 기초",
                    confidence=0.9,
                ),
            ),
            latency_ms=12.0,
        )


def test_localize_region_tracks_runs_ocr_only_and_reuses_observations(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    frame_paths: list[Path] = []
    for index in range(6):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (100, 100), "white").save(path)
        frame_paths.append(path)
    base_path = tmp_path / "tracks.v1.json"
    visual_path = tmp_path / "visual-events.json"
    base_path.write_text(
        __import__("json").dumps(_base_manifest(video_path), ensure_ascii=False),
        encoding="utf-8",
    )
    visual_path.write_text(
        __import__("json").dumps(_visual_manifest(frame_paths), ensure_ascii=False),
        encoding="utf-8",
    )
    provider = FakeOcrProvider()
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"

    first = localize_region_tracks(
        base_path,
        visual_path,
        output_dir,
        work_dir,
        provider,
        glossary=Glossary(()),
        analysis_options=AnalysisOptions(group_adjacent=False),
        tracking_exclude_regions=(),
        options=RegionTrackingOptions(),
    )
    second = localize_region_tracks(
        base_path,
        visual_path,
        output_dir,
        work_dir,
        provider,
        glossary=Glossary(()),
        analysis_options=AnalysisOptions(group_adjacent=False),
        tracking_exclude_regions=(),
        options=RegionTrackingOptions(),
    )

    assert provider.calls == 4
    assert first["metrics"]["tracking_ocr_frame_count"] == 4
    assert second["metrics"]["tracking_ocr_frame_count"] == 0
    assert second["metrics"]["tracking_reused_observation_count"] == 4
    assert (output_dir / "tracks.json").is_file()
    assert "regionTrackBox" in (output_dir / "viewer.html").read_text(encoding="utf-8")
