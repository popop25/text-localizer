from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from text_localizer.cache import TranslationCache
from text_localizer.domain import NormalizedBox, Point
from text_localizer.glossary import Glossary, GlossaryEntry
from text_localizer.pipeline import (
    TRANSLATION_CACHE_REVISION,
    analyze_static_image,
    relocalize_static_analysis,
)
from text_localizer.providers.base import OcrOutput, OcrRegion
from text_localizer.quality import AnalysisOptions


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
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        self.calls += 1
        assert tuple(source_texts) == ("확률적 경사 하강법",)
        return ("確率的勾配降下法",)


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


class BatchOcrProvider:
    name = "batch-ocr"

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        return OcrOutput(
            latency_ms=8.0,
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.2, 0.2),
                        Point(0.6, 0.2),
                        Point(0.6, 0.25),
                        Point(0.2, 0.25),
                    ),
                    bbox=NormalizedBox(0.2, 0.2, 0.4, 0.05),
                    source_text="OpenAl AP 기초",
                    confidence=0.92,
                ),
                OcrRegion(
                    polygon=(
                        Point(0.2, 0.4),
                        Point(0.7, 0.4),
                        Point(0.7, 0.45),
                        Point(0.2, 0.45),
                    ),
                    bbox=NormalizedBox(0.2, 0.4, 0.5, 0.05),
                    source_text="LangChain 기반 Agent 개발",
                    confidence=0.94,
                ),
            ),
        )


class BatchTranslationProvider:
    name = "batch-translator"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        sources = tuple(source_texts)
        self.calls.append(sources)
        assert sources == ("<TERM0> 기초", "<TERM0> 기반 <TERM1> 개발")
        return ("<TERM0> 基礎", "<TERM0>ベースの<TERM1>開発")


def test_pipeline_batches_translation_and_restores_glossary_terms(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (800, 450), "white").save(image_path)
    glossary = Glossary(
        (
            GlossaryEntry("OpenAI API", "OpenAI API", ("OpenAl AP",)),
            GlossaryEntry("LangChain", "LangChain"),
            GlossaryEntry("Agent", "Agent"),
        )
    )
    translator = BatchTranslationProvider()

    payload = analyze_static_image(
        image_path,
        tmp_path / "output",
        BatchOcrProvider(),
        translator,
        TranslationCache(tmp_path / "cache.sqlite3"),
        glossary=glossary,
        options=AnalysisOptions(group_adjacent=False, translation_batch_size=8),
    )

    assert len(translator.calls) == 1
    assert payload["schema_version"] == 2
    assert payload["metrics"]["translation_batch_count"] == 1
    assert payload["metrics"]["glossary_term_hits"] == 3
    assert [item["translated_text"] for item in payload["regions"]] == [
        "OpenAI API 基礎",
        "LangChainベースのAgent開発",
    ]
    assert all(not item["needs_review"] for item in payload["regions"])


class LabelOcrProvider:
    name = "label-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        self.calls += 1
        return OcrOutput(
            latency_ms=4.0,
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.1),
                        Point(0.8, 0.1),
                        Point(0.8, 0.25),
                        Point(0.1, 0.25),
                    ),
                    bbox=NormalizedBox(0.1, 0.1, 0.7, 0.15),
                    source_text="[\uc2e4]Tool Calling",
                    confidence=0.98,
                ),
            ),
        )


class UnexpectedTranslationProvider:
    name = "unexpected-translator"

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        raise AssertionError(f"unexpected model translation: {tuple(source_texts)}")


def test_relocalization_regenerates_deterministic_technical_labels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (800, 450), "white").save(image_path)
    glossary = Glossary((GlossaryEntry("[\uc2e4\uc2b5]", "[\u6f14\u7fd2]", ("[\uc2e4]",)),))
    ocr = LabelOcrProvider()
    translator = UnexpectedTranslationProvider()
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    prior = analyze_static_image(
        image_path,
        tmp_path / "old-output",
        ocr,
        translator,
        cache,
        glossary=glossary,
        options=AnalysisOptions(),
    )
    prior["regions"][0]["translated_text"] = "[\u6f14\u7fd2] \u30c4\u30fc\u30eb Calling"

    relocalized = relocalize_static_analysis(
        image_path,
        tmp_path / "new-output",
        prior,
        ocr,
        translator,
        cache,
        glossary=glossary,
        options=AnalysisOptions(),
    )

    assert ocr.calls == 1
    assert relocalized["regions"][0]["translated_text"] == ("[\u6f14\u7fd2] Tool Calling")
    assert relocalized["regions"][0]["translation_reused"] is False
    assert relocalized["metrics"]["translation_reused_count"] == 0


class ProtectedTermFallbackOcrProvider:
    name = "protected-term-fallback-ocr"

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        return OcrOutput(
            latency_ms=3.0,
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.1),
                        Point(0.8, 0.1),
                        Point(0.8, 0.25),
                        Point(0.1, 0.25),
                    ),
                    bbox=NormalizedBox(0.1, 0.1, 0.7, 0.15),
                    source_text="LangChain 기반 앱 개발",
                    confidence=0.98,
                ),
            ),
        )


class AcceptedProtectedTermFallbackTranslator:
    name = "accepted-protected-term-fallback-translator"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        sources = tuple(source_texts)
        self.calls.append(sources)
        if sources == ("<TERM0> 기반 앱 개발",):
            return ("アプリ開発",)
        if sources == ("LangChain 기반 앱 개발",):
            return ("LangChainベースのアプリ開発",)
        raise AssertionError(f"unexpected translation inputs: {sources}")


def test_pipeline_accepts_unprotected_retry_only_after_quality_passes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (800, 450), "white").save(image_path)
    translator = AcceptedProtectedTermFallbackTranslator()

    payload = analyze_static_image(
        image_path,
        tmp_path / "output",
        ProtectedTermFallbackOcrProvider(),
        translator,
        TranslationCache(tmp_path / "cache.sqlite3"),
        glossary=Glossary((GlossaryEntry("LangChain", "LangChain"),)),
        options=AnalysisOptions(),
    )

    assert translator.calls == [
        ("<TERM0> 기반 앱 개발",),
        ("LangChain 기반 앱 개발",),
    ]
    region = payload["regions"][0]
    assert region["translated_text"] == "LangChainベースのアプリ開発"
    assert region["translation_quality_flags"] == []
    assert region["translation_fallback_attempted"] is True
    assert region["translation_fallback_used"] is True
    assert region["needs_review"] is False
    assert payload["metrics"]["translation_batch_count"] == 2
    assert payload["metrics"]["translation_fallback_attempted_count"] == 1
    assert payload["metrics"]["translation_fallback_accepted_count"] == 1


def test_pipeline_retries_flagged_cached_translation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (800, 450), "white").save(image_path)
    glossary = Glossary((GlossaryEntry("LangChain", "LangChain"),))
    translator = AcceptedProtectedTermFallbackTranslator()
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    cache_namespace = (
        f"{glossary.cache_namespace(translator.name)}:pipeline:{TRANSLATION_CACHE_REVISION}"
    )
    cache.put(cache_namespace, "LangChain 기반 앱 개발", "アプリ開発")

    payload = analyze_static_image(
        image_path,
        tmp_path / "output",
        ProtectedTermFallbackOcrProvider(),
        translator,
        cache,
        glossary=glossary,
        options=AnalysisOptions(),
    )

    assert translator.calls == [("LangChain 기반 앱 개발",)]
    region = payload["regions"][0]
    assert region["translated_text"] == "LangChainベースのアプリ開発"
    assert region["translation_cache_hit"] is False
    assert region["translation_fallback_attempted"] is True
    assert region["translation_fallback_used"] is True
    assert payload["metrics"]["translation_batch_count"] == 1
    assert payload["metrics"]["translation_fallback_attempted_count"] == 1
    assert payload["metrics"]["translation_fallback_accepted_count"] == 1
