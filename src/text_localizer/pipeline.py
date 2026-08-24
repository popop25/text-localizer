from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
from PIL import Image

from text_localizer.cache import TranslationCache
from text_localizer.classifier import classify_text
from text_localizer.domain import TextRegion, TranslatedRegion
from text_localizer.providers.base import OcrProvider, TranslationProvider
from text_localizer.rendering import render_preview
from text_localizer.viewer import write_viewer


def analyze_static_image(
    image_path: Path,
    output_dir: Path,
    ocr_provider: OcrProvider,
    translation_provider: TranslationProvider,
    cache: TranslationCache,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    started = perf_counter()
    process = psutil.Process()
    ocr_output = ocr_provider.recognize(image_path)
    translated_regions: list[TranslatedRegion] = []
    cache_hits: list[bool] = []

    for detected in ocr_output.regions:
        action = classify_text(detected.source_text)
        region = TextRegion(
            polygon=detected.polygon,
            bbox=detected.bbox,
            source_text=detected.source_text,
            confidence=detected.confidence,
            action=action,
        )
        translated_text: str | None = None
        translation_latency_ms: float | None = None
        cache_hit = False

        if action.value == "translate":
            translated_text = cache.get(translation_provider.name, region.source_text)
            cache_hit = translated_text is not None
            if translated_text is None:
                translation_started = perf_counter()
                translated_text = translation_provider.translate(region.source_text)
                translation_latency_ms = (perf_counter() - translation_started) * 1000.0
                cache.put(
                    translation_provider.name,
                    region.source_text,
                    translated_text,
                )
            else:
                translation_latency_ms = 0.0

        translated_regions.append(
            TranslatedRegion(
                region=region,
                translated_text=translated_text,
                translation_latency_ms=translation_latency_ms,
            )
        )
        cache_hits.append(cache_hit)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = f"source{image_path.suffix.lower() or '.png'}"
    source_copy = output_dir / source_name
    shutil.copy2(image_path, source_copy)
    preview_path = output_dir / "preview.png"
    region_tuple = tuple(translated_regions)
    render_preview(image_path, region_tuple, preview_path)
    write_viewer(output_dir / "viewer.html", source_name, region_tuple)

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    memory_info = process.memory_info()
    peak_bytes = int(getattr(memory_info, "peak_wset", memory_info.rss))
    total_latency_ms = (perf_counter() - started) * 1000.0

    region_payload = []
    for translated, cache_hit in zip(translated_regions, cache_hits, strict=True):
        region = translated.region
        region_payload.append(
            {
                "polygon": [asdict(point) for point in region.polygon],
                "bbox": asdict(region.bbox),
                "source_text": region.source_text,
                "ocr_confidence": region.confidence,
                "action": region.action.value,
                "translated_text": translated.translated_text,
                "translation_latency_ms": translated.translation_latency_ms,
                "translation_cache_hit": cache_hit,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "input_image": str(image_path.resolve()),
        "image": {"width": image_width, "height": image_height},
        "ocr_provider": ocr_provider.name,
        "translation_provider": translation_provider.name,
        "regions": region_payload,
        "metrics": {
            "ocr_latency_ms": ocr_output.latency_ms,
            "total_translation_latency_ms": sum(
                item.translation_latency_ms or 0.0 for item in translated_regions
            ),
            "total_processing_latency_ms": total_latency_ms,
            "peak_process_memory_bytes": peak_bytes,
        },
    }
    (output_dir / "regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
