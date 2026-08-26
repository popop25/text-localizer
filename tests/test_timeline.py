import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from text_localizer.cache import TranslationCache
from text_localizer.domain import NormalizedBox, Point
from text_localizer.glossary import Glossary, GlossaryEntry
from text_localizer.profile import SceneDetectionOptions
from text_localizer.providers.base import OcrOutput, OcrRegion
from text_localizer.quality import AnalysisOptions
from text_localizer.timeline import (
    index_timestamped_frame_scenes,
    localize_scene_index,
)
from text_localizer.video import TimestampedFrame, VideoInfo


class CountingOcrProvider:
    name = "counting-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        self.calls += 1
        return OcrOutput(
            latency_ms=5.0,
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
            ),
        )


class CountingTranslationProvider:
    name = "counting-translator"

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        self.calls += 1
        return tuple("確率的勾配降下法" for _ in source_texts)


def test_timeline_analyzes_only_unique_scene_representatives(tmp_path: Path) -> None:
    frame_paths = [tmp_path / f"frame-{index:06d}.png" for index in range(3)]
    gradient = Image.linear_gradient("L").rotate(90, expand=True).resize((100, 100))
    gradient.save(frame_paths[0])
    gradient.copy().save(frame_paths[1])
    gradient.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(frame_paths[2])
    frames = tuple(
        TimestampedFrame(index=index, timestamp_seconds=float(index), path=str(path))
        for index, path in enumerate(frame_paths)
    )
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    video_info = VideoInfo(
        path=str(video_path.resolve()),
        width=100,
        height=100,
        fps=30.0,
        duration_seconds=3.0,
        codec="h264",
    )
    scene_index = index_timestamped_frame_scenes(
        frames,
        video_info,
        interval_seconds=1.0,
        profile_name="test",
        roi=None,
        exclude_regions=(),
        scene_options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )

    assert scene_index["unique_scenes"] == 2
    assert [(item["start_seconds"], item["end_seconds"]) for item in scene_index["scenes"]] == [
        (0.0, 2.0),
        (2.0, 3.0),
    ]

    ocr = CountingOcrProvider()
    translator = CountingTranslationProvider()
    output_dir = tmp_path / "output"
    manifest = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        TranslationCache(tmp_path / "translations.sqlite3"),
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )

    assert ocr.calls == 2
    assert translator.calls == 1
    assert manifest["track_count"] == 2
    assert manifest["metrics"]["sample_frame_count"] == 3
    assert manifest["metrics"]["unique_scene_count"] == 2
    assert manifest["metrics"]["total_scene_processing_latency_ms"] >= 0
    assert manifest["metrics"]["current_run_latency_ms"] >= 0
    assert manifest["tracks"][0]["regions"][0]["translated_text"] == "確率的勾配降下法"
    assert (output_dir / "tracks.json").is_file()
    assert (output_dir / "viewer.html").is_file()
    assert (output_dir / "editor.html").is_file()
    assert "maximum bracket width" in (output_dir / "viewer.html").read_text(encoding="utf-8")
    assert "region.needs_review" in (output_dir / "viewer.html").read_text(encoding="utf-8")
    viewer_html = (output_dir / "viewer.html").read_text(encoding="utf-8")
    assert "requestVideoFrameCallback" in viewer_html
    assert "metadata.mediaTime" in viewer_html
    assert "requestAnimationFrame" in viewer_html


def test_timeline_resume_reuses_existing_scene_analyses(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame-000000.png"
    Image.new("RGB", (100, 100), "white").save(frame_path)
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    scene_index = index_timestamped_frame_scenes(
        (TimestampedFrame(0, 0.0, str(frame_path)),),
        VideoInfo(str(video_path), 100, 100, 30.0, 1.0, "h264"),
        interval_seconds=1.0,
        profile_name=None,
        roi=None,
        exclude_regions=(),
        scene_options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )
    ocr = CountingOcrProvider()
    translator = CountingTranslationProvider()
    output_dir = tmp_path / "output"
    cache = TranslationCache(tmp_path / "translations.sqlite3")

    localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )
    resumed = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )

    assert ocr.calls == 1
    assert resumed["metrics"]["analysis_reused_count"] == 1

    regions_path = output_dir / "scenes" / "scene-0001" / "regions.json"
    stale = json.loads(regions_path.read_text(encoding="utf-8"))
    stale["translation_cache_revision"] = "stale"
    regions_path.write_text(
        json.dumps(stale, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    relocalized = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )

    assert ocr.calls == 1
    assert relocalized["metrics"]["analysis_relocalized_count"] == 1

    changed_options = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(min_region_height=0.2),
    )

    assert ocr.calls == 2
    assert changed_options["metrics"]["analysis_reused_count"] == 0


def test_timeline_resume_reanalyzes_changed_representative_content(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame-000000.png"
    Image.new("RGB", (100, 100), "white").save(frame_path)
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    scene_index = index_timestamped_frame_scenes(
        (TimestampedFrame(0, 0.0, str(frame_path)),),
        VideoInfo(str(video_path), 100, 100, 30.0, 1.0, "h264"),
        interval_seconds=1.0,
        profile_name=None,
        roi=None,
        exclude_regions=(),
        scene_options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )
    ocr = CountingOcrProvider()
    translator = CountingTranslationProvider()
    output_dir = tmp_path / "output"
    cache = TranslationCache(tmp_path / "translations.sqlite3")

    localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )
    Image.new("RGB", (100, 100), "black").save(frame_path)
    refreshed = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=Glossary.empty(),
        analysis_options=AnalysisOptions(),
    )

    assert ocr.calls == 2
    assert refreshed["metrics"]["analysis_reused_count"] == 0
    assert refreshed["metrics"]["analysis_relocalized_count"] == 0


class BracketOcrProvider:
    name = "bracket-ocr"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path) -> OcrOutput:
        del image_path
        self.calls += 1
        return OcrOutput(
            latency_ms=7.0,
            regions=(
                OcrRegion(
                    polygon=(
                        Point(0.1, 0.1),
                        Point(0.8, 0.1),
                        Point(0.8, 0.25),
                        Point(0.1, 0.25),
                    ),
                    bbox=NormalizedBox(0.1, 0.1, 0.7, 0.15),
                    source_text="[\uc2e4]LangGraph\uae30\ucd08",
                    confidence=0.98,
                ),
            ),
        )


class PlaceholderBreakingTranslator:
    name = "placeholder-breaking-translator"

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, source_text: str) -> str:
        return self.translate_many((source_text,))[0]

    def translate_many(self, source_texts: Sequence[str]) -> tuple[str, ...]:
        self.calls += 1
        return tuple("broken placeholder output" for _ in source_texts)


def test_timeline_relocalizes_stored_ocr_after_glossary_change(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame-000000.png"
    Image.new("RGB", (100, 100), "white").save(frame_path)
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"test-video-placeholder")
    scene_index = index_timestamped_frame_scenes(
        (TimestampedFrame(0, 0.0, str(frame_path)),),
        VideoInfo(str(video_path), 100, 100, 30.0, 1.0, "h264"),
        interval_seconds=1.0,
        profile_name=None,
        roi=None,
        exclude_regions=(),
        scene_options=SceneDetectionOptions(hash_size=8, hamming_threshold=0),
    )
    ocr = BracketOcrProvider()
    translator = PlaceholderBreakingTranslator()
    cache = TranslationCache(tmp_path / "translations.sqlite3")
    output_dir = tmp_path / "output"
    original_glossary = Glossary(
        (
            GlossaryEntry("LangGraph", "LangGraph"),
            GlossaryEntry("\uae30\ucd08", "\u57fa\u790e"),
        )
    )
    revised_glossary = Glossary(
        (
            GlossaryEntry("[\uc2e4\uc2b5]", "[\u6f14\u7fd2]", ("[\uc2e4]",)),
            GlossaryEntry("LangGraph", "LangGraph"),
            GlossaryEntry("\uae30\ucd08", "\u57fa\u790e"),
        )
    )

    original = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=original_glossary,
        analysis_options=AnalysisOptions(),
    )
    relocalized = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=revised_glossary,
        analysis_options=AnalysisOptions(),
    )
    resumed = localize_scene_index(
        scene_index,
        output_dir,
        ocr,
        translator,
        cache,
        glossary=revised_glossary,
        analysis_options=AnalysisOptions(),
    )

    assert original["metrics"]["needs_review_count"] == 1
    assert original["metrics"]["translation_fallback_attempted_count"] == 1
    assert original["metrics"]["translation_fallback_accepted_count"] == 0
    assert ocr.calls == 1
    assert translator.calls == 2
    assert relocalized["metrics"]["analysis_relocalized_count"] == 1
    assert relocalized["metrics"]["total_ocr_latency_ms"] == 0.0
    assert relocalized["metrics"]["needs_review_count"] == 0
    assert relocalized["tracks"][0]["regions"][0]["translated_text"] == (
        "[\u6f14\u7fd2] LangGraph \u57fa\u790e"
    )
    assert resumed["metrics"]["analysis_reused_count"] == 1
    assert resumed["metrics"]["analysis_relocalized_count"] == 0
