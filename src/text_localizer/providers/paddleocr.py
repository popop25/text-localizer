from __future__ import annotations

import importlib
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from text_localizer.domain import NormalizedBox, Point
from text_localizer.providers.base import OcrOutput, OcrRegion


class PaddleOcrProvider:
    name = "paddleocr-ppocrv5-mobile-korean"

    def __init__(self, models_root: Path, confidence_threshold: float = 0.5) -> None:
        marker = models_root / "paddleocr" / ".ready.json"
        if not marker.is_file():
            raise RuntimeError(
                "PaddleOCR models are not prepared. Run "
                "'text-localizer setup-models --ocr' explicitly first."
            )

        self._models_root = models_root
        self._pipeline: Any = None
        self._confidence_threshold = confidence_threshold

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(self._models_root / "paddleocr"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            paddleocr_module = importlib.import_module("paddleocr")
            paddleocr_class: Any = paddleocr_module.PaddleOCR
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "PaddleOCR dependencies are unavailable. Install the 'ocr' extra."
            ) from exc

        self._pipeline = paddleocr_class(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            enable_mkldnn=False,
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return self._pipeline

    def recognize(self, image_path: Path) -> OcrOutput:
        with Image.open(image_path) as image:
            width, height = image.size

        started = perf_counter()
        prediction = self._get_pipeline().predict(str(image_path))
        latency_ms = (perf_counter() - started) * 1000.0
        regions: list[OcrRegion] = []

        for item in prediction:
            payload = getattr(item, "json", {})
            if callable(payload):
                payload = payload()
            if not isinstance(payload, dict):
                continue
            result = payload.get("res", payload)
            if not isinstance(result, dict):
                continue
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polygons = result.get("rec_polys", [])
            for text, score, polygon in zip(texts, scores, polygons, strict=False):
                source_text = str(text).strip()
                confidence = float(score)
                if not source_text or confidence < self._confidence_threshold:
                    continue
                points = _normalize_polygon(polygon, width, height)
                regions.append(
                    OcrRegion(
                        polygon=points,
                        bbox=_bbox_from_polygon(points),
                        source_text=source_text,
                        confidence=confidence,
                    )
                )

        return OcrOutput(regions=tuple(regions), latency_ms=latency_ms)


def _normalize_polygon(raw_polygon: Any, width: int, height: int) -> tuple[Point, ...]:
    points: list[Point] = []
    for raw_point in raw_polygon:
        x = min(1.0, max(0.0, float(raw_point[0]) / width))
        y = min(1.0, max(0.0, float(raw_point[1]) / height))
        points.append(Point(x=x, y=y))
    if len(points) < 4:
        raise RuntimeError("PaddleOCR returned a polygon with fewer than four points.")
    return tuple(points)


def _bbox_from_polygon(points: tuple[Point, ...]) -> NormalizedBox:
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    epsilon = 1e-9
    return NormalizedBox(
        x=min_x,
        y=min_y,
        width=max(max_x - min_x, epsilon),
        height=max(max_y - min_y, epsilon),
    )
