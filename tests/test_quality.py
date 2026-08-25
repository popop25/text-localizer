from pathlib import Path

import pytest
from PIL import Image

from text_localizer.domain import NormalizedBox, Point, RegionAction
from text_localizer.glossary import Glossary
from text_localizer.providers.base import OcrOutput, OcrRegion
from text_localizer.quality import (
    AnalysisOptions,
    evaluate_translation,
    ocr_input_for_roi,
    prepare_regions,
    remap_ocr_output,
)


def _region(x: float, y: float, width: float, text: str) -> OcrRegion:
    box = NormalizedBox(x=x, y=y, width=width, height=0.05)
    return OcrRegion(
        polygon=(
            Point(box.x, box.y),
            Point(box.x + box.width, box.y),
            Point(box.x + box.width, box.y + box.height),
            Point(box.x, box.y + box.height),
        ),
        bbox=box,
        source_text=text,
        confidence=0.9,
    )


def test_ocr_coordinates_are_remapped_from_roi_to_original_image() -> None:
    output = OcrOutput(regions=(_region(0.1, 0.2, 0.3, "강의"),), latency_ms=5.0)
    roi = NormalizedBox(0.2, 0.25, 0.5, 0.5)

    remapped = remap_ocr_output(output, roi)

    box = remapped.regions[0].bbox
    assert (box.x, box.y, box.width, box.height) == pytest.approx((0.25, 0.35, 0.15, 0.025))


def test_roi_remapping_clamps_floating_point_boundary_drift() -> None:
    full_region = OcrRegion(
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
        bbox=NormalizedBox(0.0, 0.0, 1.0, 1.0),
        source_text="경계",
        confidence=0.9,
    )
    roi = NormalizedBox(0.22, 172 / 1080, 0.75, 908 / 1080)

    remapped = remap_ocr_output(OcrOutput((full_region,), 1.0), roi)

    box = remapped.regions[0].bbox
    assert box.x + box.width <= 1.0
    assert box.y + box.height <= 1.0


def test_roi_creates_a_cropped_ocr_input_and_removes_it(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    output_dir = tmp_path / "output"

    with ocr_input_for_roi(
        image_path,
        output_dir,
        NormalizedBox(0.25, 0.2, 0.5, 0.6),
    ) as (cropped_path, actual_roi):
        assert cropped_path.is_file()
        with Image.open(cropped_path) as cropped:
            assert cropped.size == (100, 60)
        assert actual_roi == NormalizedBox(0.25, 0.2, 0.5, 0.6)

    assert not cropped_path.exists()


def test_adjacent_regions_are_grouped_and_excluded_regions_remain_auditable() -> None:
    regions = (
        _region(0.1, 0.1, 0.1, "브라우저"),
        _region(0.3, 0.4, 0.1, "학습"),
        _region(0.405, 0.4, 0.1, "목표"),
    )
    options = AnalysisOptions(
        exclude_regions=(NormalizedBox(0.0, 0.0, 1.0, 0.2),),
        group_adjacent=True,
    )

    prepared = prepare_regions(regions, Glossary.empty(), options)

    assert len(prepared) == 2
    assert prepared[0].forced_action is RegionAction.IGNORE
    assert prepared[0].forced_action_reason == "inside_excluded_region"
    assert prepared[1].raw_source_text == "학습 목표"
    assert prepared[1].member_count == 2


def test_translation_quality_flags_hangul_repetition_and_short_output() -> None:
    prepared = Glossary.empty().prepare("기업용 서비스 개발 시 반드시 고려해야 할 중요한 점 이해")

    flags = evaluate_translation(prepared.normalized_text, "犬犬犬한", prepared)

    assert "hangul_in_translation" in flags
    assert "repeated_fragment" in flags
    assert "suspiciously_short" in flags


def test_translation_quality_does_not_flag_repeated_punctuation() -> None:
    prepared = Glossary.empty().prepare("채용 이력")

    flags = evaluate_translation(prepared.normalized_text, "採用履歴...", prepared)

    assert "repeated_fragment" not in flags
