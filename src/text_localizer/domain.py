from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RegionAction(StrEnum):
    TRANSLATE = "translate"
    PRESERVE = "preserve"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("Normalized point coordinates must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Normalized box values must be between 0 and 1")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Normalized box width and height must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("Normalized box must stay inside the image")


@dataclass(frozen=True, slots=True)
class TextRegion:
    polygon: tuple[Point, ...]
    bbox: NormalizedBox
    source_text: str
    confidence: float
    action: RegionAction

    def __post_init__(self) -> None:
        if len(self.polygon) < 4:
            raise ValueError("A text polygon must contain at least four points")
        if not self.source_text.strip():
            raise ValueError("Source text must not be blank")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranslatedRegion:
    region: TextRegion
    translated_text: str | None
    translation_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.region.action is RegionAction.TRANSLATE and not self.translated_text:
            raise ValueError("Translated text is required for translate regions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
