from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from text_localizer.glossary import Glossary
from text_localizer.providers.base import TranslationProvider
from text_localizer.track_editor import validate_track_manifest

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


@dataclass(frozen=True, slots=True)
class TranslationQualityGates:
    minimum_case_pass_rate: float = 0.9
    minimum_check_pass_rate: float = 0.95
    maximum_missing_cases: int = 0
    maximum_forbidden_hits: int = 0
    maximum_hangul_cases: int = 0


@dataclass(frozen=True, slots=True)
class TranslationCase:
    case_id: str
    label: str
    source_text: str
    reference_translation: str
    required_all: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    preserved_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranslationBenchmark:
    name: str
    gates: TranslationQualityGates
    cases: tuple[TranslationCase, ...]


def load_translation_benchmark(path: Path) -> TranslationBenchmark:
    if not path.is_file():
        raise FileNotFoundError(f"Translation benchmark does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Translation benchmark is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Translation benchmark root must be a mapping.")
    return parse_translation_benchmark(cast(dict[str, Any], raw))


def parse_translation_benchmark(payload: dict[str, Any]) -> TranslationBenchmark:
    if payload.get("schema_version") != 1:
        raise ValueError("Translation benchmark schema_version must be 1.")
    name = _nonblank_string(payload.get("name"), "name")
    gates = _parse_gates(payload.get("gates"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Translation benchmark requires at least one case.")

    cases: list[TranslationCase] = []
    case_ids: set[str] = set()
    normalized_sources: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _parse_case(raw_case, index)
        if case.case_id in case_ids:
            raise ValueError(f"Duplicate translation case id: {case.case_id}")
        normalized_source = _normalize_text(case.source_text)
        if normalized_source in normalized_sources:
            raise ValueError(f"Duplicate translation case source: {case.source_text}")
        case_ids.add(case.case_id)
        normalized_sources.add(normalized_source)
        cases.append(case)
    return TranslationBenchmark(name=name, gates=gates, cases=tuple(cases))


def evaluate_translation_quality(
    translations: dict[str, str],
    benchmark: TranslationBenchmark,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    normalized_translations = {
        _normalize_text(source): translated for source, translated in translations.items()
    }
    case_reports: list[dict[str, Any]] = []
    passed_case_count = 0
    passed_check_count = 0
    check_count = 0
    missing_case_count = 0
    forbidden_hit_count = 0
    hangul_case_count = 0
    reference_similarities: list[float] = []

    for case in benchmark.cases:
        translated = normalized_translations.get(_normalize_text(case.source_text))
        report = _evaluate_case(case, translated)
        case_reports.append(report)
        passed_case_count += int(report["passed"])
        passed_check_count += int(report["passed_check_count"])
        check_count += int(report["check_count"])
        missing_case_count += int(report["missing"])
        forbidden_hit_count += int(report["forbidden_hit_count"])
        hangul_case_count += int(report["hangul_detected"])
        similarity = report["reference_similarity"]
        if similarity is not None:
            reference_similarities.append(float(similarity))

    case_count = len(benchmark.cases)
    case_pass_rate = passed_case_count / case_count
    check_pass_rate = passed_check_count / check_count if check_count else 0.0
    mean_reference_similarity = (
        statistics.fmean(reference_similarities) if reference_similarities else None
    )
    gate_results = {
        "case_pass_rate": case_pass_rate >= benchmark.gates.minimum_case_pass_rate,
        "check_pass_rate": check_pass_rate >= benchmark.gates.minimum_check_pass_rate,
        "missing_cases": missing_case_count <= benchmark.gates.maximum_missing_cases,
        "forbidden_hits": forbidden_hit_count <= benchmark.gates.maximum_forbidden_hits,
        "hangul_cases": hangul_case_count <= benchmark.gates.maximum_hangul_cases,
    }

    return {
        "schema_version": 1,
        "benchmark": benchmark.name,
        "provider": provider,
        "summary": {
            "case_count": case_count,
            "evaluated_case_count": case_count - missing_case_count,
            "passed_case_count": passed_case_count,
            "case_pass_rate": case_pass_rate,
            "check_count": check_count,
            "passed_check_count": passed_check_count,
            "check_pass_rate": check_pass_rate,
            "missing_case_count": missing_case_count,
            "forbidden_hit_count": forbidden_hit_count,
            "hangul_case_count": hangul_case_count,
            "mean_reference_similarity": mean_reference_similarity,
            "gates_passed": all(gate_results.values()),
        },
        "gates": asdict(benchmark.gates),
        "gate_results": gate_results,
        "cases": case_reports,
    }


def evaluate_translation_quality_files(
    manifest_path: Path,
    benchmark_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Track manifest does not exist: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Track manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Track manifest root must be a mapping.")
    manifest = cast(dict[str, Any], raw)
    validate_track_manifest(manifest)
    provider = manifest.get("translation_provider")
    report = evaluate_translation_quality(
        _manifest_translations(manifest),
        load_translation_benchmark(benchmark_path),
        provider=provider if isinstance(provider, str) else None,
    )
    report["inputs"] = {
        "track_manifest": str(manifest_path.resolve()),
        "benchmark": str(benchmark_path.resolve()),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def benchmark_translation_provider(
    provider: TranslationProvider,
    benchmark: TranslationBenchmark,
    glossary: Glossary,
    *,
    batch_size: int = 4,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("Translation benchmark batch size must be at least 1.")
    translations: dict[str, str] = {}
    pending: list[tuple[TranslationCase, Any]] = []
    deterministic_case_count = 0
    restore_flag_count = 0

    for case in benchmark.cases:
        prepared = glossary.prepare(case.source_text)
        if prepared.requires_model_translation:
            pending.append((case, prepared))
            continue
        restored, restore_flags = glossary.restore(prepared.protected_text, prepared)
        translations[case.source_text] = restored
        deterministic_case_count += 1
        restore_flag_count += len(restore_flags)

    total_latency_ms = 0.0
    batch_count = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        started = perf_counter()
        outputs = provider.translate_many(tuple(prepared.protected_text for _, prepared in batch))
        total_latency_ms += (perf_counter() - started) * 1000
        batch_count += 1
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"Translation provider returned {len(outputs)} outputs for {len(batch)} inputs."
            )
        for (case, prepared), translated in zip(batch, outputs, strict=True):
            restored, restore_flags = glossary.restore(translated, prepared)
            translations[case.source_text] = restored
            restore_flag_count += len(restore_flags)

    report = evaluate_translation_quality(
        translations,
        benchmark,
        provider=provider.name,
    )
    report["run"] = {
        "batch_size": batch_size,
        "batch_count": batch_count,
        "model_case_count": len(pending),
        "deterministic_case_count": deterministic_case_count,
        "total_translation_latency_ms": total_latency_ms,
        "mean_model_case_latency_ms": total_latency_ms / len(pending) if pending else 0.0,
        "restore_flag_count": restore_flag_count,
        "glossary_fingerprint": glossary.fingerprint,
    }
    return report


def write_translation_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_translations(manifest: dict[str, Any]) -> dict[str, str]:
    translations: dict[str, str] = {}
    if manifest.get("schema_version") == 2:
        raw_region_tracks = manifest.get("region_tracks", [])
        if isinstance(raw_region_tracks, list):
            for raw_track in raw_region_tracks:
                if not isinstance(raw_track, dict):
                    continue
                region = raw_track.get("region")
                if isinstance(region, dict):
                    _add_region_translation(translations, region)
    if translations:
        return translations

    raw_tracks = manifest.get("tracks", [])
    if isinstance(raw_tracks, list):
        for raw_track in raw_tracks:
            if not isinstance(raw_track, dict):
                continue
            raw_regions = raw_track.get("regions", [])
            if not isinstance(raw_regions, list):
                continue
            for region in raw_regions:
                if isinstance(region, dict):
                    _add_region_translation(translations, region)
    return translations


def _add_region_translation(translations: dict[str, str], region: dict[str, Any]) -> None:
    if region.get("action") != "translate":
        return
    source = region.get("translation_source_text", region.get("source_text"))
    translated = region.get("translated_text")
    if isinstance(source, str) and source.strip() and isinstance(translated, str):
        translations.setdefault(source, translated)


def _evaluate_case(case: TranslationCase, translated: str | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if translated is None:
        return {
            "id": case.case_id,
            "label": case.label,
            "source_text": case.source_text,
            "translated_text": None,
            "reference_translation": case.reference_translation,
            "reference_similarity": None,
            "missing": True,
            "hangul_detected": False,
            "forbidden_hit_count": 0,
            "check_count": 0,
            "passed_check_count": 0,
            "passed": False,
            "checks": checks,
        }

    normalized_target = _normalize_text(translated)
    for term in case.required_all:
        checks.append(_check("required_all", term, _normalize_text(term) in normalized_target))
    for alternatives in case.required_any:
        checks.append(
            _check(
                "required_any",
                list(alternatives),
                any(_normalize_text(term) in normalized_target for term in alternatives),
            )
        )
    forbidden_hits = 0
    for term in case.forbidden:
        passed = _normalize_text(term) not in normalized_target
        forbidden_hits += int(not passed)
        checks.append(_check("forbidden", term, passed))
    for term in case.preserved_terms:
        checks.append(_check("preserved_term", term, term.casefold() in translated.casefold()))
    hangul_detected = bool(_HANGUL_RE.search(translated))
    checks.append(_check("no_hangul", None, not hangul_detected))

    passed_check_count = sum(bool(check["passed"]) for check in checks)
    return {
        "id": case.case_id,
        "label": case.label,
        "source_text": case.source_text,
        "translated_text": translated,
        "reference_translation": case.reference_translation,
        "reference_similarity": _character_ngram_f_score(
            case.reference_translation,
            translated,
        ),
        "missing": False,
        "hangul_detected": hangul_detected,
        "forbidden_hit_count": forbidden_hits,
        "check_count": len(checks),
        "passed_check_count": passed_check_count,
        "passed": passed_check_count == len(checks),
        "checks": checks,
    }


def _check(kind: str, value: str | list[str] | None, passed: bool) -> dict[str, Any]:
    return {"kind": kind, "value": value, "passed": passed}


def _character_ngram_f_score(reference: str, candidate: str, n: int = 2) -> float:
    reference_ngrams = _ngrams(_normalize_text(reference), n)
    candidate_ngrams = _ngrams(_normalize_text(candidate), n)
    if not reference_ngrams or not candidate_ngrams:
        return float(reference_ngrams == candidate_ngrams)
    overlap = sum(
        min(count, candidate_ngrams.get(fragment, 0))
        for fragment, count in reference_ngrams.items()
    )
    precision = overlap / sum(candidate_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ngrams(value: str, n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in range(max(0, len(value) - n + 1)):
        fragment = value[index : index + n]
        counts[fragment] = counts.get(fragment, 0) + 1
    return counts


def _parse_gates(raw: Any) -> TranslationQualityGates:
    values = {} if raw is None else _mapping(raw, "gates")
    return TranslationQualityGates(
        minimum_case_pass_rate=_ratio(
            values.get("minimum_case_pass_rate", 0.9),
            "gates.minimum_case_pass_rate",
        ),
        minimum_check_pass_rate=_ratio(
            values.get("minimum_check_pass_rate", 0.95),
            "gates.minimum_check_pass_rate",
        ),
        maximum_missing_cases=_nonnegative_integer(
            values.get("maximum_missing_cases", 0),
            "gates.maximum_missing_cases",
        ),
        maximum_forbidden_hits=_nonnegative_integer(
            values.get("maximum_forbidden_hits", 0),
            "gates.maximum_forbidden_hits",
        ),
        maximum_hangul_cases=_nonnegative_integer(
            values.get("maximum_hangul_cases", 0),
            "gates.maximum_hangul_cases",
        ),
    )


def _parse_case(raw: Any, index: int) -> TranslationCase:
    values = _mapping(raw, f"cases[{index}]")
    return TranslationCase(
        case_id=_nonblank_string(values.get("id"), f"cases[{index}].id"),
        label=_nonblank_string(values.get("label"), f"cases[{index}].label"),
        source_text=_nonblank_string(values.get("source_text"), f"cases[{index}].source_text"),
        reference_translation=_nonblank_string(
            values.get("reference_translation"),
            f"cases[{index}].reference_translation",
        ),
        required_all=_string_list(values.get("required_all", []), f"cases[{index}].required_all"),
        required_any=_string_groups(
            values.get("required_any", []),
            f"cases[{index}].required_any",
        ),
        forbidden=_string_list(values.get("forbidden", []), f"cases[{index}].forbidden"),
        preserved_terms=_string_list(
            values.get("preserved_terms", []),
            f"cases[{index}].preserved_terms",
        ),
    )


def _mapping(raw: Any, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a mapping.")
    return cast(dict[str, Any], raw)


def _nonblank_string(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return raw.strip()


def _string_list(raw: Any, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list.")
    return tuple(_nonblank_string(item, field) for item in raw)


def _string_groups(raw: Any, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list.")
    groups: list[tuple[str, ...]] = []
    for group in raw:
        values = _string_list(group, field)
        if not values:
            raise ValueError(f"{field} groups must not be empty.")
        groups.append(values)
    return tuple(groups)


def _ratio(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field} must be a number.")
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1.")
    return value


def _nonnegative_integer(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return cast(int, raw)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())
