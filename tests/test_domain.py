import pytest

from text_localizer.domain import NormalizedBox, Point, RegionAction, TextRegion


def test_text_region_accepts_normalized_coordinates() -> None:
    region = TextRegion(
        polygon=(Point(0.1, 0.2), Point(0.5, 0.2), Point(0.5, 0.3), Point(0.1, 0.3)),
        bbox=NormalizedBox(0.1, 0.2, 0.4, 0.1),
        source_text="안녕하세요",
        confidence=0.95,
        action=RegionAction.TRANSLATE,
    )
    assert region.bbox.width == pytest.approx(0.4)


def test_box_rejects_out_of_bounds_coordinates() -> None:
    with pytest.raises(ValueError, match="inside the image"):
        NormalizedBox(0.8, 0.2, 0.3, 0.1)
