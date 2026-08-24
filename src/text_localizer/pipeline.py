from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
from PIL import Image

from text_localizer.cache import TranslationCache
from text_localizer.classifier import classify_text_with_reason
from text_localizer.domain import RegionAction, TextRegion, TranslatedRegion
from text_localizer.glossary import Glossary
from text_localizer.providers.base import OcrProvider, TranslationProvider
from text_localizer.quality import (
    AnalysisOptions,
    PreparedRegion,
    box_to_dict,
    evaluate_translation,
    ocr_input_for_roi,
    prepare_regions,
    remap_ocr_output,
)
from text_localizer.rendering import render_preview
from text_localizer.viewer import write_viewer

_TRANSLATION_CACHE_REVISION = "m11-v1"


@dataclass(slots=True)
class _RegionWork:
    prepared: PreparedRegion
    region: TextRegion
    action_reason: str
    translated_text: str | None = None
    translation_latency_ms: float | None = None
    translation_cache_hit: bool = False
    translation_quality_flags: tuple[str, ...] = ()


def analyze_static_image(
    image_path: Path,
    output_dir: Path,
    ocr_provider: OcrProvider,
    translation_provider: TranslationProvider,
    cache: TranslationCache,
    *,
    glossary: Glossary | None = None,
    options: AnalysisOptions | None = None,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    active_glossary = glossary or Glossary.empty()
    active_options = options or AnalysisOptions()
    started = perf_counter()
    process = psutil.Process()
    output_dir.mkdir(parents=True, exist_ok=True)

    with ocr_input_for_roi(image_path, output_dir, active_options.roi) as (
        ocr_input_path,
        actual_roi,
    ):
        raw_ocr_output = ocr_provider.recognize(ocr_input_path)
    ocr_output = remap_ocr_output(raw_ocr_output, actual_roi)
    prepared_regions = prepare_regions(ocr_output.regions, active_glossary, active_options)
    work = [_build_region_work(item) for item in prepared_regions]

    cache_namespace = (
        f"{active_glossary.cache_namespace(translation_provider.name)}"
        f":pipeline:{_TRANSLATION_CACHE_REVISION}"
    )
    total_translation_latency_ms, translation_batch_count = _translate_regions(
        work,
        translation_provider,
        cache,
        cache_namespace,
        active_glossary,
        active_options.translation_batch_size,
    )

    translated_regions = tuple(
        TranslatedRegion(
            region=item.region,
            translated_text=item.translated_text,
            translation_latency_ms=item.translation_latency_ms,
        )
        for item in work
    )

    source_name = f"source{image_path.suffix.lower() or '.png'}"
    source_copy = output_dir / source_name
    shutil.copy2(image_path, source_copy)
    preview_path = output_dir / "preview.png"
    render_preview(image_path, translated_regions, preview_path)
    write_viewer(output_dir / "viewer.html", source_name, translated_regions)

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    memory_info = process.memory_info()
    peak_bytes = int(getattr(memory_info, "peak_wset", memory_info.rss))
    total_latency_ms = (perf_counter() - started) * 1000.0

    region_payload = [_region_payload(item) for item in work]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "input_image": str(image_path.resolve()),
        "image": {"width": image_width, "height": image_height},
        "ocr_provider": ocr_provider.name,
        "translation_provider": translation_provider.name,
        "translation_cache_revision": _TRANSLATION_CACHE_REVISION,
        "glossary_fingerprint": active_glossary.fingerprint,
        "analysis_options": {
            "roi": box_to_dict(actual_roi),
            "exclude_regions": [box_to_dict(box) for box in active_options.exclude_regions],
            "min_region_height": active_options.min_region_height,
            "group_adjacent": active_options.group_adjacent,
            "translation_batch_size": active_options.translation_batch_size,
        },
        "regions": region_payload,
        "metrics": {
            "ocr_latency_ms": ocr_output.latency_ms,
            "total_translation_latency_ms": total_translation_latency_ms,
            "translation_batch_count": translation_batch_count,
            "total_processing_latency_ms": total_latency_ms,
            "peak_process_memory_bytes": peak_bytes,
            "glossary_term_hits": sum(
                len(item.prepared.prepared_text.protected_terms) for item in work
            ),
            "needs_review_count": sum(bool(item.translation_quality_flags) for item in work),
        },
    }
    (output_dir / "regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _build_region_work(prepared: PreparedRegion) -> _RegionWork:
    if prepared.forced_action is not None:
        action = prepared.forced_action
        reason = prepared.forced_action_reason or "forced"
    else:
        action, reason = classify_text_with_reason(prepared.prepared_text.normalized_text)
    region = TextRegion(
        polygon=prepared.polygon,
        bbox=prepared.bbox,
        source_text=prepared.prepared_text.normalized_text,
        confidence=prepared.confidence,
        action=action,
    )
    return _RegionWork(prepared=prepared, region=region, action_reason=reason)


def _translate_regions(
    work: list[_RegionWork],
    provider: TranslationProvider,
    cache: TranslationCache,
    cache_namespace: str,
    glossary: Glossary,
    batch_size: int,
) -> tuple[float, int]:
    pending: list[_RegionWork] = []
    for item in work:
        if item.region.action is not RegionAction.TRANSLATE:
            continue
        cached = cache.get(cache_namespace, item.region.source_text)
        if cached is not None:
            item.translated_text = cached
            item.translation_latency_ms = 0.0
            item.translation_cache_hit = True
            item.translation_quality_flags = evaluate_translation(
                item.region.source_text,
                cached,
                item.prepared.prepared_text,
            )
            continue
        if item.prepared.prepared_text.glossary_only:
            restored, restore_flags = glossary.restore(
                item.prepared.prepared_text.protected_text,
                item.prepared.prepared_text,
            )
            item.translated_text = restored
            item.translation_latency_ms = 0.0
            item.translation_quality_flags = evaluate_translation(
                item.region.source_text,
                restored,
                item.prepared.prepared_text,
                restore_flags,
            )
            cache.put(cache_namespace, item.region.source_text, restored)
            continue
        pending.append(item)

    total_latency_ms = 0.0
    batch_count = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        source_texts = [item.prepared.prepared_text.protected_text for item in batch]
        batch_started = perf_counter()
        translations = provider.translate_many(source_texts)
        batch_latency_ms = (perf_counter() - batch_started) * 1000.0
        if len(translations) != len(batch):
            raise RuntimeError(
                "Translation provider returned a different number of results than inputs."
            )
        total_latency_ms += batch_latency_ms
        batch_count += 1
        amortized_latency_ms = batch_latency_ms / len(batch)
        for item, translated in zip(batch, translations, strict=True):
            restored, restore_flags = glossary.restore(
                translated,
                item.prepared.prepared_text,
            )
            item.translated_text = restored
            item.translation_latency_ms = amortized_latency_ms
            item.translation_quality_flags = evaluate_translation(
                item.region.source_text,
                restored,
                item.prepared.prepared_text,
                restore_flags,
            )
            cache.put(cache_namespace, item.region.source_text, restored)
    return total_latency_ms, batch_count


def _region_payload(item: _RegionWork) -> dict[str, Any]:
    region = item.region
    prepared_text = item.prepared.prepared_text
    return {
        "polygon": [asdict(point) for point in region.polygon],
        "bbox": asdict(region.bbox),
        "raw_source_text": item.prepared.raw_source_text,
        "source_text": region.source_text,
        "normalization_edits": list(prepared_text.normalization_edits),
        "group_member_count": item.prepared.member_count,
        "ocr_confidence": region.confidence,
        "action": region.action.value,
        "action_reason": item.action_reason,
        "translated_text": item.translated_text,
        "translation_latency_ms": item.translation_latency_ms,
        "translation_cache_hit": item.translation_cache_hit,
        "translation_quality_flags": list(item.translation_quality_flags),
        "needs_review": bool(item.translation_quality_flags),
    }
