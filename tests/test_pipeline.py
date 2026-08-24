from pathlib import Path

from PIL import Image

from text_localizer.cache import TranslationCache
from text_localizer.domain import NormalizedBox, Point
from text_localizer.pipeline import analyze_static_image
from text_localizer.providers.base import OcrOutput, OcrRegion


class FakeOcrProvider:
    name = "fake-ocr"

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        return OcrOutput(
            latency_ms=12.5,
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.1),
                        Point(0.7, 0.1),
                        Point(0.7, 0.25),
                        Point(0.1, 0.25),
                    ),
                    bbox=NormalizedBox(0.1, 0.1, 0.6, 0.15),
                    source_text="확률적 경사 하강법",
                    confidence=0.96,
                ),
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.7),
                        Point(0.5, 0.7),
                        Point(0.5, 0.8),
                        Point(0.1, 0.8),
                    ),
                    bbox=NormalizedBox(0.1, 0.7, 0.4, 0.1),
                    source_text="https://example.com",
                    confidence=0.99,
                ),
            ),
        )


class FakeTranslationProvider:
    name = "fake-translator"

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, source_text: str) -> str:
        self.calls += 1
        assert source_text == "확률적 경사 하강법"
        return "確率的勾配降下法"


def test_static_pipeline_writes_outputs_and_reuses_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (800, 450), "white").save(image_path)
    output_dir = tmp_path / "output"
    translator = FakeTranslationProvider()
    cache = TranslationCache(tmp_path / "cache.sqlite3")

    first = analyze_static_image(
        image_path,
        output_dir,
        FakeOcrProvider(),
        translator,
        cache,
    )
    second = analyze_static_image(
        image_path,
        output_dir,
        FakeOcrProvider(),
        translator,
        cache,
    )

    assert translator.calls == 1
    assert first["regions"][0]["translated_text"] == "確率的勾配降下法"
    assert second["regions"][0]["translation_cache_hit"] is True
    for name in ("regions.json", "preview.png", "viewer.html", "source.png"):
        assert (output_dir / name).is_file()
