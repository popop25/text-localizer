from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from text_localizer.domain import NormalizedBox, Point, RegionAction
from text_localizer.glossary import Glossary, PreparedText
from text_localizer.providers.base import OcrOutput, OcrRegion

_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_REPEATED_FRAGMENT_RE = re.compile(r"([A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]{1,8}?)(?:\s*\1){2,}")


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    roi: NormalizedBox | None = None
    exclude_regions: tuple[NormalizedBox, ...] = ()
    min_region_height: float = 0.0
    group_adjacent: bool = True
    translation_batch_size: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_region_height <= 1.0:
            raise ValueError("Minimum region height must be between 0 and 1.")
        if self.translation_batch_size < 1:
            raise ValueError("Translation batch size must be at least 1.")


@dataclass(frozen=True, slots=True)
class PreparedRegion:
    polygon: tuple[Point, ...]
    bbox: NormalizedBox
    raw_source_text: str
    prepared_text: PreparedText
    confidence: float
    member_count: int
    forced_action: RegionAction | None = None
    forced_action_reason: str | None = None


@contextmanager
def ocr_input_for_roi(
    image_path: Path,
    output_dir: Path,
    roi: NormalizedBox | None,
) -> Iterator[tuple[Path, NormalizedBox | None]]:
    if roi is None:
        yield image_path, None
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            left = max(0, math.floor(roi.x * width))
            top = max(0, math.floor(roi.y * height))
            right = min(width, math.ceil((roi.x + roi.width) * width))
            bottom = min(height, math.ceil((roi.y + roi.height) * height))
            if right <= left or bottom <= top:
                raise ValueError("ROI resolves to an empty pixel area.")
            actual_roi = NormalizedBox(
                x=left / width,
                y=top / height,
                width=(right - left) / width,
                height=(bottom - top) / height,
            )
            with tempfile.NamedTemporaryFile(
                prefix=".text-localizer-roi-",
                suffix=".png",
                dir=output_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            image.crop((left, top, right, bottom)).save(temporary_path)
        yield temporary_path, actual_roi
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def remap_ocr_output(output: OcrOutput, roi: NormalizedBox | None) -> OcrOutput:
    if roi is None:
        return output
    regions = tuple(
        OcrRegion(
            polygon=tuple(_remap_point(point, roi) for point in region.polygon),
            bbox=_remap_box(region.bbox, roi),
            source_text=region.source_text,
            confidence=region.confidence,
        )
        for region in output.regions
    )
    return OcrOutput(regions=regions, latency_ms=output.latency_ms)


def prepare_regions(
    regions: tuple[OcrRegion, ...],
    glossary: Glossary,
    options: AnalysisOptions,
) -> tuple[PreparedRegion, ...]:
    filtered: list[PreparedRegion] = []
    candidates: list[OcrRegion] = []
    for region in regions:
        reason = _filter_reason(region.bbox, options)
        if reason is None:
            candidates.append(region)
            continue
        filtered.append(
            PreparedRegion(
                polygon=region.polygon,
                bbox=region.bbox,
                raw_source_text=region.source_text,
                prepared_text=glossary.prepare(region.source_text),
                confidence=region.confidence,
                member_count=1,
                forced_action=RegionAction.IGNORE,
                forced_action_reason=reason,
            )
        )

    groups = (
        _group_regions(candidates) if options.group_adjacent else [(item,) for item in candidates]
    )
    prepared = filtered + [_prepare_group(group, glossary) for group in groups]
    return tuple(sorted(prepared, key=lambda item: (item.bbox.y, item.bbox.x)))


def evaluate_translation(
    source_text: str,
    translated_text: str,
    prepared: PreparedText,
    restore_flags: tuple[str, ...] = (),
) -> tuple[str, ...]:
    flags = list(restore_flags)
    if _HANGUL_RE.search(translated_text):
        flags.append("hangul_in_translation")
    if _REPEATED_FRAGMENT_RE.search(translated_text):
        flags.append("repeated_fragment")

    compact_source = re.sub(r"\s+", "", source_text)
    compact_target = re.sub(r"\s+", "", translated_text)
    if len(compact_source) >= 8 and len(compact_target) < max(2, len(compact_source) * 0.25):
        flags.append("suspiciously_short")

    for term in prepared.protected_terms:
        if term.target not in translated_text:
            flags.append(f"missing_glossary_target:{term.target}")
    return tuple(dict.fromkeys(flags))


def box_to_dict(box: NormalizedBox | None) -> dict[str, float] | None:
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _remap_point(point: Point, roi: NormalizedBox) -> Point:
    return Point(
        x=min(1.0, max(0.0, roi.x + point.x * roi.width)),
        y=min(1.0, max(0.0, roi.y + point.y * roi.height)),
    )


def _remap_box(box: NormalizedBox, roi: NormalizedBox) -> NormalizedBox:
    left = min(1.0, max(0.0, roi.x + box.x * roi.width))
    top = min(1.0, max(0.0, roi.y + box.y * roi.height))
    right = min(1.0, max(left, roi.x + (box.x + box.width) * roi.width))
    bottom = min(1.0, max(top, roi.y + (box.y + box.height) * roi.height))
    return NormalizedBox(
        x=left,
        y=top,
        width=max(right - left, 1e-9),
        height=max(bottom - top, 1e-9),
    )


def _filter_reason(box: NormalizedBox, options: AnalysisOptions) -> str | None:
    if box.height < options.min_region_height:
        return "below_min_region_height"
    if any(_contains_center(excluded, box) for excluded in options.exclude_regions):
        return "inside_excluded_region"
    return None


def _contains_center(container: NormalizedBox, item: NormalizedBox) -> bool:
    center_x = item.x + item.width / 2
    center_y = item.y + item.height / 2
    return (
        container.x <= center_x <= container.x + container.width
        and container.y <= center_y <= container.y + container.height
    )


def _group_regions(regions: list[OcrRegion]) -> list[tuple[OcrRegion, ...]]:
    ordered = sorted(regions, key=lambda item: (item.bbox.y + item.bbox.height / 2, item.bbox.x))
    groups: list[list[OcrRegion]] = []
    for region in ordered:
        if groups and _can_merge(groups[-1][-1].bbox, region.bbox):
            groups[-1].append(region)
        else:
            groups.append([region])
    return [tuple(group) for group in groups]


def _can_merge(left: NormalizedBox, right: NormalizedBox) -> bool:
    overlap = min(left.y + left.height, right.y + right.height) - max(left.y, right.y)
    if overlap <= 0 or overlap / min(left.height, right.height) < 0.5:
        return False
    left_center = left.y + left.height / 2
    right_center = right.y + right.height / 2
    if abs(left_center - right_center) > max(left.height, right.height) * 0.5:
        return False
    gap = right.x - (left.x + left.width)
    return -0.005 <= gap <= max(0.012, max(left.height, right.height) * 1.5)


def _prepare_group(group: tuple[OcrRegion, ...], glossary: Glossary) -> PreparedRegion:
    raw_text = _join_text(tuple(item.source_text for item in group))
    bbox = _union_box(tuple(item.bbox for item in group))
    return PreparedRegion(
        polygon=_rectangle_polygon(bbox),
        bbox=bbox,
        raw_source_text=raw_text,
        prepared_text=glossary.prepare(raw_text),
        confidence=sum(item.confidence for item in group) / len(group),
        member_count=len(group),
    )


def _join_text(parts: tuple[str, ...]) -> str:
    result = ""
    for part in parts:
        stripped = part.strip()
        if not result or stripped[:1] in ",.;:!?)]}" or result[-1:] in "([{":
            result += stripped
        else:
            result += f" {stripped}"
    return result


def _union_box(boxes: tuple[NormalizedBox, ...]) -> NormalizedBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return NormalizedBox(x=left, y=top, width=right - left, height=bottom - top)


def _rectangle_polygon(box: NormalizedBox) -> tuple[Point, ...]:
    return (
        Point(box.x, box.y),
        Point(box.x + box.width, box.y),
        Point(box.x + box.width, box.y + box.height),
        Point(box.x, box.y + box.height),
    )
