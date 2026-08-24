import pytest

from text_localizer.classifier import classify_text, normalize_source_text
from text_localizer.domain import RegionAction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("확률적 경사 하강법", RegionAction.TRANSLATE),
        ("https://example.com/docs", RegionAction.PRESERVE),
        ("v3.12.9", RegionAction.PRESERVE),
        ("250 ms", RegionAction.PRESERVE),
        ("MODEL_NAME", RegionAction.PRESERVE),
        ("def train_model():", RegionAction.PRESERVE),
        ("---", RegionAction.IGNORE),
    ],
)
def test_classification_rules(text: str, expected: RegionAction) -> None:
    assert classify_text(text) is expected


def test_normalization_is_stable() -> None:
    assert normalize_source_text("  모델\n  학습  ") == "모델 학습"
