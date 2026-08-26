from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from text_localizer.glossary import Glossary, PreparedText
from text_localizer.providers.base import TranslationProvider
from text_localizer.quality import evaluate_translation
from text_localizer.track_editor import load_track_manifest
from text_localizer.video_viewer import write_video_viewer

TRANSLATION_REVISION = "m36-v1"


@dataclass(frozen=True, slots=True)
class _TranslationResult:
    translated_text: str
    quality_flags: tuple[str, ...]
    latency_ms: float
    fallback_attempted: bool
    fallback_used: bool


def retranslate_region_tracks(
    manifest_path: Path,
    output_dir: Path,
    provider: TranslationProvider,
    glossary: Glossary,
    *,
    batch_size: int = 4,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("Translation batch size must be at least 1.")
    source_manifest = load_track_manifest(manifest_path)
    if source_manifest.get("schema_version") != 2:
        raise ValueError("Retranslating tracks requires a schema-v2 region track manifest.")
    manifest = copy.deepcopy(source_manifest)
    region_tracks = manifest.get("region_tracks")
    if not isinstance(region_tracks, list):
        raise ValueError("Schema-v2 manifest region_tracks must be a list.")

    source_regions: dict[str, list[dict[str, Any]]] = {}
    for raw_track in region_tracks:
        if not isinstance(raw_track, dict):
            continue
        region = raw_track.get("region")
        if not isinstance(region, dict) or region.get("action") != "translate":
            continue
        source = region.get("translation_source_text", region.get("source_text"))
        if not isinstance(source, str) or not source.strip():
            continue
        source_regions.setdefault(source, []).append(cast(dict[str, Any], region))

    prepared = {source: glossary.prepare(source) for source in source_regions}
    results, run_metrics = _translate_sources(provider, glossary, prepared, batch_size)
    for source, regions in source_regions.items():
        result = results[source]
        for region in regions:
            region["translation_source_text"] = source
            region["translated_text"] = result.translated_text
            region["translation_latency_ms"] = result.latency_ms
            region["translation_cache_hit"] = False
            region["translation_fallback_attempted"] = result.fallback_attempted
            region["translation_fallback_used"] = result.fallback_used
            region["translation_quality_flags"] = list(result.quality_flags)
            region["needs_review"] = bool(result.quality_flags)
            region["translation_reused"] = False

    needs_review_count = sum(
        bool(result.quality_flags) * len(source_regions[source])
        for source, result in results.items()
    )
    manifest["translation_provider"] = provider.name
    manifest["translation_revision"] = TRANSLATION_REVISION
    manifest["translation_glossary_fingerprint"] = glossary.fingerprint
    raw_metrics = manifest.get("metrics")
    metrics = cast(dict[str, Any], raw_metrics) if isinstance(raw_metrics, dict) else {}
    metrics.update(
        {
            "needs_review_count": needs_review_count,
            "retranslated_region_track_count": sum(len(items) for items in source_regions.values()),
            "retranslated_unique_source_count": len(source_regions),
            **run_metrics,
        }
    )
    manifest["metrics"] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.json"
    tracks_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    video = manifest.get("video")
    if not isinstance(video, dict) or not isinstance(video.get("path"), str):
        raise ValueError("Track manifest video.path is invalid.")
    viewer_path = output_dir / "viewer.html"
    write_video_viewer(viewer_path, Path(str(video["path"])), manifest)
    return {
        "schema_version": 1,
        "translation_provider": provider.name,
        "translation_revision": TRANSLATION_REVISION,
        "tracks": str(tracks_path.resolve()),
        "viewer": str(viewer_path.resolve()),
        "summary": {
            "region_track_count": sum(len(items) for items in source_regions.values()),
            "unique_source_count": len(source_regions),
            "needs_review_count": needs_review_count,
            **run_metrics,
        },
    }


def _translate_sources(
    provider: TranslationProvider,
    glossary: Glossary,
    prepared_by_source: dict[str, PreparedText],
    batch_size: int,
) -> tuple[dict[str, _TranslationResult], dict[str, int | float]]:
    results: dict[str, _TranslationResult] = {}
    pending: list[tuple[str, PreparedText]] = []
    for source, prepared in prepared_by_source.items():
        if prepared.requires_model_translation:
            pending.append((source, prepared))
            continue
        restored, restore_flags = glossary.restore(prepared.protected_text, prepared)
        flags = evaluate_translation(source, restored, prepared, restore_flags)
        results[source] = _TranslationResult(restored, flags, 0.0, False, False)

    total_latency_ms = 0.0
    batch_count = 0
    fallback_sources: list[tuple[str, PreparedText]] = []
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        started = perf_counter()
        translations = provider.translate_many(
            tuple(prepared.protected_text for _, prepared in batch)
        )
        latency_ms = (perf_counter() - started) * 1000
        if len(translations) != len(batch):
            raise RuntimeError(
                "Translation provider returned "
                f"{len(translations)} outputs for {len(batch)} inputs."
            )
        total_latency_ms += latency_ms
        batch_count += 1
        per_source_latency_ms = latency_ms / len(batch)
        for (source, prepared), translated in zip(batch, translations, strict=True):
            restored, restore_flags = glossary.restore(translated, prepared)
            flags = evaluate_translation(source, restored, prepared, restore_flags)
            attempted = _should_retry_without_placeholders(flags)
            results[source] = _TranslationResult(
                restored,
                flags,
                per_source_latency_ms,
                attempted,
                False,
            )
            if attempted:
                fallback_sources.append((source, prepared))

    fallback_accepted_count = 0
    for offset in range(0, len(fallback_sources), batch_size):
        batch = fallback_sources[offset : offset + batch_size]
        started = perf_counter()
        translations = provider.translate_many(tuple(source for source, _ in batch))
        latency_ms = (perf_counter() - started) * 1000
        if len(translations) != len(batch):
            raise RuntimeError(
                "Translation provider returned "
                f"{len(translations)} outputs for {len(batch)} inputs."
            )
        total_latency_ms += latency_ms
        batch_count += 1
        per_source_latency_ms = latency_ms / len(batch)
        for (source, prepared), translated in zip(batch, translations, strict=True):
            restored, restore_flags = glossary.restore(translated, prepared)
            flags = evaluate_translation(source, restored, prepared, restore_flags)
            previous = results[source]
            if not flags:
                results[source] = _TranslationResult(
                    restored,
                    (),
                    previous.latency_ms + per_source_latency_ms,
                    True,
                    True,
                )
                fallback_accepted_count += 1

    return results, {
        "translation_batch_count": batch_count,
        "translation_fallback_attempted_count": len(fallback_sources),
        "translation_fallback_accepted_count": fallback_accepted_count,
        "total_translation_latency_ms": total_latency_ms,
        "mean_model_source_latency_ms": total_latency_ms / len(pending) if pending else 0.0,
    }


def _should_retry_without_placeholders(flags: tuple[str, ...]) -> bool:
    return any(
        flag in {"hangul_in_translation", "repeated_fragment", "suspiciously_short"}
        or flag.startswith("missing_placeholder:")
        or flag.startswith("missing_glossary_target:")
        for flag in flags
    )
