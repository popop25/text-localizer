from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from text_localizer.domain import NormalizedBox, Point


@dataclass(frozen=True, slots=True)
class OcrRegion:
    polygon: tuple[Point, ...]
    bbox: NormalizedBox
    source_text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class OcrOutput:
    regions: tuple[OcrRegion, ...]
    latency_ms: float


class OcrProvider(Protocol):
    @property
    def name(self) -> str: ...

    def recognize(self, image_path: Path) -> OcrOutput: ...


class TranslationProvider(Protocol):
    @property
    def name(self) -> str: ...

    def translate(self, source_text: str) -> str: ...

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]: ...
