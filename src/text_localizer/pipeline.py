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
from text_localizer.domain import NormalizedBox, Point, RegionAction, TextRegion, TranslatedRegion
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

TRANSLATION_CACHE_REVISION = "m15-v2"


@dataclass(slots=True)
class _RegionWork:
    prepared: PreparedRegion
    region: TextRegion
    action_reason: str
    translated_text: str | None = None
    translation_latency_ms: float | None = None
    translation_cache_hit: bool = False
    translation_quality_flags: tuple[str, ...] = ()
    translation_reused: bool = False


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
        f":pipeline:{TRANSLATION_CACHE_REVISION}"
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
        "translation_cache_revision": TRANSLATION_CACHE_REVISION,
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
            "source_ocr_latency_ms": ocr_output.latency_ms,
            "ocr_reused": False,
            "total_translation_latency_ms": total_translation_latency_ms,
            "translation_batch_count": translation_batch_count,
            "total_processing_latency_ms": total_latency_ms,
            "translation_reused_count": 0,
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


def relocalize_static_analysis(
    image_path: Path,
    output_dir: Path,
    prior_analysis: dict[str, Any],
    ocr_provider: OcrProvider,
    translation_provider: TranslationProvider,
    cache: TranslationCache,
    *,
    glossary: Glossary,
    options: AnalysisOptions,
) -> dict[str, Any]:
    """Re-run normalization and translation while preserving stored OCR geometry."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    started = perf_counter()
    process = psutil.Process()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_regions = _prepared_regions_from_payload(
        prior_analysis.get("regions"),
        glossary,
    )
    work = [_build_region_work(item) for item in prepared_regions]
    translation_reused_count = _reuse_prior_translations(
        work,
        prior_analysis.get("regions"),
        prior_translation_provider=prior_analysis.get("translation_provider"),
        current_translation_provider=translation_provider.name,
    )

    cache_namespace = (
        f"{glossary.cache_namespace(translation_provider.name)}"
        f":pipeline:{TRANSLATION_CACHE_REVISION}"
    )
    total_translation_latency_ms, translation_batch_count = _translate_regions(
        work,
        translation_provider,
        cache,
        cache_namespace,
        glossary,
        options.translation_batch_size,
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
    shutil.copy2(image_path, output_dir / source_name)
    render_preview(image_path, translated_regions, output_dir / "preview.png")
    write_viewer(output_dir / "viewer.html", source_name, translated_regions)

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    memory_info = process.memory_info()
    peak_bytes = int(getattr(memory_info, "peak_wset", memory_info.rss))
    total_latency_ms = (perf_counter() - started) * 1000.0
    prior_metrics = prior_analysis.get("metrics")
    if not isinstance(prior_metrics, dict):
        prior_metrics = {}
    source_ocr_latency_ms = float(
        prior_metrics.get(
            "source_ocr_latency_ms",
            prior_metrics.get("ocr_latency_ms", 0.0),
        )
    )
    prior_options = prior_analysis.get("analysis_options")
    roi_payload = (
        prior_options.get("roi") if isinstance(prior_options, dict) else box_to_dict(options.roi)
    )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "input_image": str(image_path.resolve()),
        "image": {"width": image_width, "height": image_height},
        "ocr_provider": ocr_provider.name,
        "translation_provider": translation_provider.name,
        "translation_cache_revision": TRANSLATION_CACHE_REVISION,
        "glossary_fingerprint": glossary.fingerprint,
        "analysis_options": {
            "roi": roi_payload,
            "exclude_regions": [box_to_dict(box) for box in options.exclude_regions],
            "min_region_height": options.min_region_height,
            "group_adjacent": options.group_adjacent,
            "translation_batch_size": options.translation_batch_size,
        },
        "regions": [_region_payload(item) for item in work],
        "metrics": {
            "ocr_latency_ms": 0.0,
            "source_ocr_latency_ms": source_ocr_latency_ms,
            "ocr_reused": True,
            "total_translation_latency_ms": total_translation_latency_ms,
            "translation_batch_count": translation_batch_count,
            "translation_reused_count": translation_reused_count,
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


def _prepared_regions_from_payload(
    raw_regions: Any,
    glossary: Glossary,
) -> tuple[PreparedRegion, ...]:
    if not isinstance(raw_regions, list):
        raise ValueError("Prior analysis regions must be a list.")

    prepared_regions: list[PreparedRegion] = []
    filter_reasons = {"below_min_region_height", "inside_excluded_region"}
    for raw_region in raw_regions:
        if not isinstance(raw_region, dict):
            raise ValueError("Prior analysis region must be a mapping.")
        raw_polygon = raw_region.get("polygon")
        raw_bbox = raw_region.get("bbox")
        raw_source_text = raw_region.get("raw_source_text")
        if not isinstance(raw_polygon, list) or not isinstance(raw_bbox, dict):
            raise ValueError("Prior analysis region geometry is invalid.")
        if not isinstance(raw_source_text, str) or not raw_source_text.strip():
            raise ValueError("Prior analysis raw source text is invalid.")

        polygon = tuple(
            Point(x=float(raw_point["x"]), y=float(raw_point["y"]))
            for raw_point in raw_polygon
            if isinstance(raw_point, dict)
        )
        if len(polygon) != len(raw_polygon):
            raise ValueError("Prior analysis polygon points are invalid.")
        bbox = NormalizedBox(
            x=float(raw_bbox["x"]),
            y=float(raw_bbox["y"]),
            width=float(raw_bbox["width"]),
            height=float(raw_bbox["height"]),
        )
        action_reason = raw_region.get("action_reason")
        forced_reason = (
            str(action_reason)
            if raw_region.get("action") == RegionAction.IGNORE.value
            and action_reason in filter_reasons
            else None
        )
        prepared_regions.append(
            PreparedRegion(
                polygon=polygon,
                bbox=bbox,
                raw_source_text=raw_source_text,
                prepared_text=glossary.prepare(raw_source_text),
                confidence=float(raw_region["ocr_confidence"]),
                member_count=int(raw_region.get("group_member_count", 1)),
                forced_action=RegionAction.IGNORE if forced_reason else None,
                forced_action_reason=forced_reason,
            )
        )
    return tuple(prepared_regions)


def _reuse_prior_translations(
    work: list[_RegionWork],
    raw_regions: Any,
    *,
    prior_translation_provider: Any,
    current_translation_provider: str,
) -> int:
    if prior_translation_provider != current_translation_provider:
        return 0
    if not isinstance(raw_regions, list) or len(raw_regions) != len(work):
        return 0

    reused_count = 0
    for item, raw_region in zip(work, raw_regions, strict=True):
        if item.region.action is not RegionAction.TRANSLATE:
            continue
        if not item.prepared.prepared_text.requires_model_translation:
            continue
        if not isinstance(raw_region, dict):
            continue
        if raw_region.get("source_text") != item.region.source_text:
            continue
        translated_text = raw_region.get("translated_text")
        if not isinstance(translated_text, str) or not translated_text.strip():
            continue
        quality_flags = evaluate_translation(
            item.region.source_text,
            translated_text,
            item.prepared.prepared_text,
        )
        if quality_flags:
            continue
        item.translated_text = translated_text
        item.translation_latency_ms = 0.0
        item.translation_quality_flags = ()
        item.translation_reused = True
        reused_count += 1
    return reused_count


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
        if item.translated_text is not None:
            continue
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
        if not item.prepared.prepared_text.requires_model_translation:
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
        "translation_reused": item.translation_reused,
    }
