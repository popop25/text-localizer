import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from text_localizer.glossary import Glossary
from text_localizer.translation_quality import (
    benchmark_translation_provider,
    evaluate_translation_quality,
    evaluate_translation_quality_files,
    load_translation_benchmark,
    parse_translation_benchmark,
)


class _FakeTranslationProvider:
    name = "fake-translator"

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        return tuple("Azure OpenAI & LangChain 実践" for _ in source_texts)


def _benchmark() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "synthetic translation quality",
        "gates": {
            "minimum_case_pass_rate": 1.0,
            "minimum_check_pass_rate": 1.0,
            "maximum_missing_cases": 0,
            "maximum_forbidden_hits": 0,
            "maximum_hangul_cases": 0,
        },
        "cases": [
            {
                "id": "course-title",
                "label": "Course title",
                "source_text": "Azure OpenAI & LangChain 실전",
                "reference_translation": "Azure OpenAI & LangChain 実践",
                "required_all": ["実践"],
                "required_any": [["講座", "コース", "実践"]],
                "forbidden": ["実行"],
                "preserved_terms": ["Azure OpenAI", "LangChain"],
            }
        ],
    }


def _manifest(translated_text: str) -> dict[str, Any]:
    region = {
        "action": "translate",
        "source_text": "Azure OpenAI & LangChain 실전",
        "translation_source_text": "Azure OpenAI & LangChain 실전",
        "translated_text": translated_text,
        "needs_review": False,
        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
    }
    return {
        "schema_version": 2,
        "video": {
            "path": "lecture.mp4",
            "width": 100,
            "height": 100,
            "fps": 30.0,
            "duration_seconds": 1.0,
            "codec": "h264",
        },
        "sampling": {"transition_precision_seconds": 0.1},
        "track_count": 1,
        "tracks": [
            {
                "track_id": "scene-0001",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "regions": [region],
            }
        ],
        "region_track_count": 1,
        "region_tracks": [
            {
                "region_track_id": "region-0001",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "region": region,
                "keyframes": [
                    {
                        "timestamp_seconds": 0.0,
                        "bbox": region["bbox"],
                        "interpolation": "hold",
                    }
                ],
            }
        ],
        "metrics": {"needs_review_count": 0},
        "translation_provider": "fake-translator",
    }


def test_translation_quality_passes_semantic_and_preservation_checks() -> None:
    benchmark = parse_translation_benchmark(_benchmark())

    report = evaluate_translation_quality(
        {"Azure OpenAI & LangChain 실전": "Azure OpenAI & LangChain 実践講座"},
        benchmark,
        provider="fake-translator",
    )

    assert report["summary"]["case_pass_rate"] == 1.0
    assert report["summary"]["check_pass_rate"] == 1.0
    assert report["summary"]["gates_passed"] is True
    assert report["cases"][0]["reference_similarity"] > 0.8


def test_translation_quality_exposes_forbidden_mistranslation() -> None:
    report = evaluate_translation_quality(
        {"Azure OpenAI & LangChain 실전": "Azure OpenAI & LangChain 実行"},
        parse_translation_benchmark(_benchmark()),
    )

    assert report["summary"]["passed_case_count"] == 0
    assert report["summary"]["forbidden_hit_count"] == 1
    assert report["gate_results"]["case_pass_rate"] is False
    assert report["gate_results"]["forbidden_hits"] is False


def test_translation_quality_exposes_missing_case_and_hangul() -> None:
    benchmark = parse_translation_benchmark(_benchmark())
    hangul_report = evaluate_translation_quality(
        {"Azure OpenAI & LangChain 실전": "Azure OpenAI LangChain 실전"},
        benchmark,
    )
    missing_report = evaluate_translation_quality({}, benchmark)

    assert hangul_report["summary"]["hangul_case_count"] == 1
    assert hangul_report["gate_results"]["hangul_cases"] is False
    assert missing_report["summary"]["missing_case_count"] == 1
    assert missing_report["gate_results"]["missing_cases"] is False


def test_translation_quality_file_runner_uses_schema_v2_region_tracks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tracks.json"
    benchmark_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "report.json"
    manifest_path.write_text(
        json.dumps(_manifest("Azure OpenAI & LangChain 実践")),
        encoding="utf-8",
    )
    benchmark_path.write_text(json.dumps(_benchmark()), encoding="utf-8")

    report = evaluate_translation_quality_files(
        manifest_path,
        benchmark_path,
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert saved["provider"] == "fake-translator"
    assert saved["summary"]["gates_passed"] is True


def test_translation_provider_benchmark_records_runtime_metadata() -> None:
    report = benchmark_translation_provider(
        _FakeTranslationProvider(),
        parse_translation_benchmark(_benchmark()),
        Glossary.empty(),
        batch_size=2,
    )

    assert report["summary"]["gates_passed"] is True
    assert report["run"]["batch_count"] == 1
    assert report["run"]["model_case_count"] == 1
    assert report["run"]["deterministic_case_count"] == 0


def test_translation_benchmark_rejects_duplicate_sources() -> None:
    payload = _benchmark()
    payload["cases"].append(dict(payload["cases"][0]))
    payload["cases"][1]["id"] = "duplicate"

    with pytest.raises(ValueError, match="Duplicate translation case source"):
        parse_translation_benchmark(payload)


def test_checked_in_zero_video_translation_benchmark_is_valid() -> None:
    benchmark_path = (
        Path(__file__).parents[1] / "config" / "benchmarks" / "0-translation-quality.json"
    )

    benchmark = load_translation_benchmark(benchmark_path)

    assert benchmark.name == "0.mp4 Korean to Japanese translation quality"
    assert len(benchmark.cases) == 14
