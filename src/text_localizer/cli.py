from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from text_localizer.cache import TranslationCache
from text_localizer.domain import NormalizedBox
from text_localizer.environment import run_doctor
from text_localizer.glossary import Glossary
from text_localizer.logging_config import configure_logging
from text_localizer.overlay_layout import OverlayLayoutOptions, layout_region_tracks
from text_localizer.overlay_style import OverlayStyleOptions, style_region_tracks
from text_localizer.pipeline import analyze_static_image
from text_localizer.profile import (
    LocalizationProfile,
    SceneDetectionOptions,
    load_profile,
    merge_analysis_options,
    merge_visual_event_options,
    resolve_glossary_path,
)
from text_localizer.providers.paddleocr import PaddleOcrProvider
from text_localizer.providers.translation import (
    GEMMA3_270M_PROVIDER,
    TRANSLATION_PROVIDER_CHOICES,
    create_translation_provider,
)
from text_localizer.region_motion import RegionMotionOptions, refine_region_motion
from text_localizer.region_tracks import RegionTrackingOptions, localize_region_tracks
from text_localizer.replacement_quality import evaluate_replacement_quality_files
from text_localizer.retranslate_tracks import retranslate_region_tracks
from text_localizer.scenes import index_frame_directory
from text_localizer.setup_models import setup_ocr_models, setup_translation_model
from text_localizer.temporal_quality import evaluate_temporal_quality_files
from text_localizer.timeline import index_video_timeline, localize_scene_index
from text_localizer.track_editor import load_track_manifest, write_track_editor
from text_localizer.translation_quality import (
    benchmark_translation_provider,
    evaluate_translation_quality_files,
    load_translation_benchmark,
    write_translation_quality_report,
)
from text_localizer.video import extract_representative_frames, inspect_video
from text_localizer.video_renderer import VideoRenderOptions, render_localized_video
from text_localizer.video_viewer import write_video_viewer
from text_localizer.visual_events import index_video_visual_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-localizer")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local runtime dependencies")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    inspect = subparsers.add_parser("inspect-video", help="Inspect an input video with FFprobe")
    inspect.add_argument("path", type=Path)

    extract = subparsers.add_parser("extract-frames", help="Extract representative video frames")
    extract.add_argument("path", type=Path)
    extract.add_argument("--output", type=Path, default=Path("data/work/frames"))
    extract.add_argument("--count", type=int, default=20)

    analyze = subparsers.add_parser("analyze-image", help="Analyze one static image")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--models-root", type=Path, default=Path("models"))
    analyze.add_argument(
        "--translation-provider",
        choices=TRANSLATION_PROVIDER_CHOICES,
        default=GEMMA3_270M_PROVIDER,
    )
    analyze.add_argument("--profile", type=Path)
    analyze.add_argument(
        "--cache",
        type=Path,
        default=Path("data/work/translations.sqlite3"),
    )
    analyze.add_argument(
        "--glossary",
        type=Path,
    )
    analyze.add_argument(
        "--roi",
        type=_parse_normalized_box,
        help="OCR crop as normalized x,y,width,height",
    )
    analyze.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
        help="Ignore regions centered inside normalized x,y,width,height (repeatable)",
    )
    analyze.add_argument("--min-region-height", type=float)
    analyze.add_argument("--translation-batch-size", type=int)
    analyze.add_argument(
        "--group-adjacent",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    deduplicate = subparsers.add_parser(
        "deduplicate-frames",
        help="Index consecutive frame scenes using an ROI-aware perceptual hash",
    )
    deduplicate.add_argument("path", type=Path, help="Directory containing frame images")
    deduplicate.add_argument("--profile", type=Path)
    deduplicate.add_argument("--pattern", default="frame-*.png")
    deduplicate.add_argument("--output", type=Path)
    deduplicate.add_argument("--roi", type=_parse_normalized_box)
    deduplicate.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
    )
    deduplicate.add_argument("--hash-size", type=int)
    deduplicate.add_argument("--hamming-threshold", type=int)

    index_video = subparsers.add_parser(
        "index-video",
        help="Extract timestamped samples and index scene time ranges",
    )
    index_video.add_argument("path", type=Path)
    index_video.add_argument("--profile", type=Path)
    index_video.add_argument("--work-dir", type=Path)
    index_video.add_argument("--sample-interval", type=float, default=1.0)
    index_video.add_argument("--roi", type=_parse_normalized_box)
    index_video.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
    )
    index_video.add_argument("--hash-size", type=int)
    index_video.add_argument("--hamming-threshold", type=int)
    index_video.add_argument("--boundary-refinement-iterations", type=int, default=0)

    localize_video = subparsers.add_parser(
        "localize-video",
        help="Analyze unique video scenes and emit synchronized overlay tracks",
    )
    localize_video.add_argument("path", type=Path)
    localize_video.add_argument("--profile", type=Path)
    localize_video.add_argument("--work-dir", type=Path)
    localize_video.add_argument("--output", type=Path)
    localize_video.add_argument("--sample-interval", type=float, default=1.0)
    localize_video.add_argument("--models-root", type=Path, default=Path("models"))
    localize_video.add_argument(
        "--translation-provider",
        choices=TRANSLATION_PROVIDER_CHOICES,
        default=GEMMA3_270M_PROVIDER,
    )
    localize_video.add_argument(
        "--cache",
        type=Path,
        default=Path("data/work/translations.sqlite3"),
    )
    localize_video.add_argument("--glossary", type=Path)
    localize_video.add_argument("--roi", type=_parse_normalized_box)
    localize_video.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
    )
    localize_video.add_argument("--min-region-height", type=float)
    localize_video.add_argument("--translation-batch-size", type=int)
    localize_video.add_argument(
        "--group-adjacent",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    localize_video.add_argument("--hash-size", type=int)
    localize_video.add_argument("--hamming-threshold", type=int)
    localize_video.add_argument("--boundary-refinement-iterations", type=int, default=0)

    index_visual_events = subparsers.add_parser(
        "index-visual-events",
        help="Index localized visual changes without loading OCR or translation models",
    )
    index_visual_events.add_argument("path", type=Path)
    index_visual_events.add_argument("--profile", type=Path)
    index_visual_events.add_argument("--work-dir", type=Path)
    index_visual_events.add_argument("--sample-interval", type=float)
    index_visual_events.add_argument("--roi", type=_parse_normalized_box)
    index_visual_events.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
    )
    index_visual_events.add_argument("--analysis-width", type=int)
    index_visual_events.add_argument("--tile-columns", type=int)
    index_visual_events.add_argument("--tile-rows", type=int)
    index_visual_events.add_argument("--tile-change-threshold", type=float)
    index_visual_events.add_argument("--minimum-changed-tiles", type=int)
    index_visual_events.add_argument("--global-change-ratio", type=float)

    track_regions = subparsers.add_parser(
        "track-regions",
        help="OCR visual-event keyframes and emit independent schema-v2 region tracks",
    )
    track_regions.add_argument("path", type=Path, help="Schema-v1 tracks JSON")
    track_regions.add_argument("visual_events", type=Path, help="Visual-events JSON")
    track_regions.add_argument("--profile", type=Path)
    track_regions.add_argument("--output", type=Path)
    track_regions.add_argument("--work-dir", type=Path)
    track_regions.add_argument("--models-root", type=Path, default=Path("models"))
    track_regions.add_argument("--glossary", type=Path)
    track_regions.add_argument("--minimum-event-ratio", type=float, default=0.25)
    track_regions.add_argument("--maximum-event-duration", type=float, default=1.5)
    track_regions.add_argument("--minimum-text-similarity", type=float, default=0.68)
    track_regions.add_argument("--tile-padding", type=int, default=0)
    track_regions.add_argument("--start-seconds", type=float)
    track_regions.add_argument("--end-seconds", type=float)
    track_regions.add_argument(
        "--include-window",
        type=_parse_time_window,
        action="append",
        default=[],
        help="OCR events overlapping START:END seconds (repeatable; exclusive with bounds)",
    )
    track_regions.add_argument(
        "--exclude-roi",
        type=_parse_normalized_box,
        action="append",
        default=[],
        help="Tracking-only OCR exclusion (repeatable)",
    )

    refine_motion = subparsers.add_parser(
        "refine-region-motion",
        help="Interpolate slow scene-anchor drift between sparse region keyframes",
    )
    refine_motion.add_argument("path", type=Path, help="Schema-v2 tracks JSON")
    refine_motion.add_argument("--output", type=Path)
    refine_motion.add_argument("--minimum-segment-seconds", type=float, default=1.0)
    refine_motion.add_argument(
        "--minimum-vertical-displacement-ratio", type=float, default=0.005
    )
    refine_motion.add_argument(
        "--maximum-vertical-speed-ratio-per-second", type=float, default=0.01
    )
    refine_motion.add_argument(
        "--maximum-horizontal-speed-ratio-per-second", type=float, default=0.002
    )
    refine_motion.add_argument("--maximum-size-change-ratio", type=float, default=0.4)

    style_regions = subparsers.add_parser(
        "style-regions",
        help="Infer source-matched colors and font scale for schema-v2 region tracks",
    )
    style_regions.add_argument("path", type=Path, help="Schema-v2 tracks JSON")
    style_regions.add_argument("--output", type=Path)
    style_regions.add_argument("--work-dir", type=Path)
    style_regions.add_argument("--background-padding-ratio", type=float, default=0.35)
    style_regions.add_argument("--minimum-contrast-ratio", type=float, default=3.0)
    style_regions.add_argument("--minimum-style-confidence", type=float, default=0.45)

    layout_regions = subparsers.add_parser(
        "layout-regions",
        help="Fit translated text to schema-v2 source regions without silent clipping",
    )
    layout_regions.add_argument("path", type=Path, help="Schema-v2 tracks JSON")
    layout_regions.add_argument("--output", type=Path)
    layout_regions.add_argument("--maximum-width-scale", type=float, default=1.8)
    layout_regions.add_argument("--maximum-frame-width-ratio", type=float, default=0.6)
    layout_regions.add_argument("--maximum-lines", type=int, default=2)
    layout_regions.add_argument("--minimum-font-scale", type=float, default=0.8)
    layout_regions.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a non-zero status when a layout quality gate fails",
    )

    retranslate_tracks = subparsers.add_parser(
        "retranslate-tracks",
        help="Retranslate schema-v2 region tracks without rerunning OCR or tracking",
    )
    retranslate_tracks.add_argument("path", type=Path, help="Schema-v2 tracks JSON")
    retranslate_tracks.add_argument("--output", type=Path)
    retranslate_tracks.add_argument("--models-root", type=Path, default=Path("models"))
    retranslate_tracks.add_argument(
        "--translation-provider",
        choices=TRANSLATION_PROVIDER_CHOICES,
        default=GEMMA3_270M_PROVIDER,
    )
    retranslate_tracks.add_argument(
        "--glossary",
        type=Path,
        default=Path("config/glossary.ko-ja.yaml"),
    )
    retranslate_tracks.add_argument("--batch-size", type=int, default=4)

    render_tracks = subparsers.add_parser(
        "render-tracks",
        help="Validate edited tracks and regenerate the viewer and editor",
    )
    render_tracks.add_argument("path", type=Path)
    render_tracks.add_argument("--output", type=Path)
    render_tracks.add_argument("--video", type=Path)
    evaluate_timeline = subparsers.add_parser(
        "evaluate-timeline",
        help="Measure temporal overlay quality against an annotated benchmark",
    )
    evaluate_timeline.add_argument("path", type=Path, help="Validated tracks JSON")
    evaluate_timeline.add_argument("benchmark", type=Path, help="Temporal benchmark JSON")
    evaluate_timeline.add_argument("--output", type=Path)
    evaluate_timeline.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a non-zero status when a quality gate fails",
    )
    evaluate_replacement = subparsers.add_parser(
        "evaluate-replacement",
        help="Measure in-place source-text replacement fidelity at audited frames",
    )
    evaluate_replacement.add_argument("path", type=Path, help="Schema-v2 tracks JSON")
    evaluate_replacement.add_argument("benchmark", type=Path, help="Replacement benchmark JSON")
    evaluate_replacement.add_argument("--output", type=Path)
    evaluate_replacement.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a non-zero status when a replacement quality gate fails",
    )
    evaluate_translations = subparsers.add_parser(
        "evaluate-translations",
        help="Measure stored Korean-to-Japanese translations against a benchmark",
    )
    evaluate_translations.add_argument("path", type=Path, help="Validated tracks JSON")
    evaluate_translations.add_argument("benchmark", type=Path, help="Translation benchmark JSON")
    evaluate_translations.add_argument("--output", type=Path)
    evaluate_translations.add_argument(
        "--require-pass",
        action="store_true",
        help="Return a non-zero status when a quality gate fails",
    )
    benchmark_translator = subparsers.add_parser(
        "benchmark-translator",
        help="Run a local translation provider against a Korean-to-Japanese benchmark",
    )
    benchmark_translator.add_argument("benchmark", type=Path)
    benchmark_translator.add_argument("--models-root", type=Path, default=Path("models"))
    benchmark_translator.add_argument(
        "--translation-provider",
        choices=TRANSLATION_PROVIDER_CHOICES,
        default=GEMMA3_270M_PROVIDER,
    )
    benchmark_translator.add_argument(
        "--glossary",
        type=Path,
        default=Path("config/glossary.ko-ja.yaml"),
    )
    benchmark_translator.add_argument("--batch-size", type=int, default=4)
    benchmark_translator.add_argument("--output", type=Path)
    benchmark_translator.add_argument("--require-pass", action="store_true")
    render_video = subparsers.add_parser(
        "render-video",
        help="Burn approved Japanese tracks into a new H.264 MP4",
    )
    render_video.add_argument("path", type=Path, help="Validated tracks JSON")
    render_video.add_argument("--output", type=Path)
    render_video.add_argument("--video", type=Path)
    render_video.add_argument("--crf", type=int, default=18)
    render_video.add_argument("--preset", default="medium")
    render_video.add_argument("--overwrite", action="store_true")

    setup = subparsers.add_parser("setup-models", help="Explicitly download model files")
    setup.add_argument("--ocr", action="store_true")
    setup.add_argument("--translation", action="store_true")
    setup.add_argument(
        "--translation-provider",
        choices=TRANSLATION_PROVIDER_CHOICES,
        default=GEMMA3_270M_PROVIDER,
    )
    setup.add_argument("--models-root", type=Path, default=Path("models"))
    return parser


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _parse_normalized_box(value: str) -> NormalizedBox:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError
        return NormalizedBox(*parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected normalized x,y,width,height values between 0 and 1."
        ) from exc


def _parse_time_window(value: str) -> tuple[float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(":")]
        if len(parts) != 2 or parts[0] < 0.0 or parts[1] <= parts[0]:
            raise ValueError
        return parts[0], parts[1]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a START:END window with 0 <= START < END."
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.command == "doctor":
            results = run_doctor()
            if args.as_json:
                print(json.dumps([result.to_dict() for result in results], indent=2))
            else:
                for result in results:
                    status = "OK" if result.ok else "FAIL"
                    print(f"[{status}] {result.name}: {result.detail}")
            return 0 if all(result.ok for result in results) else 1

        if args.command == "inspect-video":
            print(json.dumps(inspect_video(args.path).to_dict(), indent=2))
            return 0

        if args.command == "extract-frames":
            outputs = extract_representative_frames(args.path, args.output, args.count)
            print(json.dumps([str(path) for path in outputs], indent=2))
            return 0

        if args.command == "analyze-image":
            output = args.output or Path("data/output") / args.path.stem
            profile = load_profile(args.profile) if args.profile is not None else None
            ocr_provider = PaddleOcrProvider(args.models_root)
            translation_provider = create_translation_provider(
                args.translation_provider,
                args.models_root,
            )
            options = merge_analysis_options(
                profile,
                roi=args.roi,
                exclude_regions=tuple(args.exclude_roi),
                min_region_height=args.min_region_height,
                group_adjacent=args.group_adjacent,
                translation_batch_size=args.translation_batch_size,
            )
            payload = analyze_static_image(
                args.path,
                output,
                ocr_provider,
                translation_provider,
                TranslationCache(args.cache),
                glossary=Glossary.from_yaml(resolve_glossary_path(profile, args.glossary)),
                options=options,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "deduplicate-frames":
            profile = load_profile(args.profile) if args.profile is not None else None
            analysis_options = merge_analysis_options(
                profile,
                roi=args.roi,
                exclude_regions=tuple(args.exclude_roi),
            )
            base_scene_options = (
                profile.scene_detection if profile is not None else SceneDetectionOptions()
            )
            scene_options = SceneDetectionOptions(
                hash_size=(
                    args.hash_size if args.hash_size is not None else base_scene_options.hash_size
                ),
                hamming_threshold=(
                    args.hamming_threshold
                    if args.hamming_threshold is not None
                    else base_scene_options.hamming_threshold
                ),
            )
            output = args.output or args.path / "scene-index.json"
            payload = index_frame_directory(
                args.path,
                pattern=args.pattern,
                output_path=output,
                profile_name=profile.name if profile is not None else None,
                roi=analysis_options.roi,
                exclude_regions=analysis_options.exclude_regions,
                options=scene_options,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "render-tracks":
            manifest = load_track_manifest(args.path)

            video_path = args.video or Path(str(manifest["video"]["path"]))
            output = args.output or args.path.parent
            write_video_viewer(output / "viewer.html", video_path, manifest)
            editor_path: Path | None = None
            if manifest.get("schema_version") != 2:
                editor_path = output / "editor.html"
                write_track_editor(editor_path, video_path, manifest)
            print(
                json.dumps(
                    {
                        "viewer": str((output / "viewer.html").resolve()),
                        "editor": str(editor_path.resolve()) if editor_path is not None else None,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "refine-region-motion":
            output = args.output or args.path.with_name(f"{args.path.stem}-motion")
            report = refine_region_motion(
                args.path,
                output,
                options=RegionMotionOptions(
                    minimum_segment_seconds=args.minimum_segment_seconds,
                    minimum_vertical_displacement_ratio=(
                        args.minimum_vertical_displacement_ratio
                    ),
                    maximum_vertical_speed_ratio_per_second=(
                        args.maximum_vertical_speed_ratio_per_second
                    ),
                    maximum_horizontal_speed_ratio_per_second=(
                        args.maximum_horizontal_speed_ratio_per_second
                    ),
                    maximum_size_change_ratio=args.maximum_size_change_ratio,
                ),
            )
            print(json.dumps(report["region_motion"], ensure_ascii=False, indent=2))
            return 0

        if args.command == "render-video":
            output = args.output or args.path.with_name("localized.mp4")
            report = render_localized_video(
                args.path,
                output,
                video_path=args.video,
                options=VideoRenderOptions(
                    crf=args.crf,
                    preset=args.preset,
                    overwrite=args.overwrite,
                ),
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        if args.command == "evaluate-timeline":
            report = evaluate_temporal_quality_files(
                args.path,
                args.benchmark,
                output_path=args.output,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.require_pass and not report["summary"]["gates_passed"]:
                return 1
            return 0

        if args.command == "evaluate-replacement":
            report = evaluate_replacement_quality_files(
                args.path,
                args.benchmark,
                output_path=args.output,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.require_pass and not report["summary"]["gates_passed"]:
                return 1
            return 0

        if args.command == "evaluate-translations":
            report = evaluate_translation_quality_files(
                args.path,
                args.benchmark,
                output_path=args.output,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.require_pass and not report["summary"]["gates_passed"]:
                return 1
            return 0

        if args.command == "benchmark-translator":
            report = benchmark_translation_provider(
                create_translation_provider(args.translation_provider, args.models_root),
                load_translation_benchmark(args.benchmark),
                Glossary.from_yaml(args.glossary),
                batch_size=args.batch_size,
            )
            if args.output is not None:
                write_translation_quality_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.require_pass and not report["summary"]["gates_passed"]:
                return 1
            return 0

        if args.command == "index-visual-events":
            profile = load_profile(args.profile) if args.profile is not None else None
            analysis_options = merge_analysis_options(profile, roi=args.roi)
            visual_options = merge_visual_event_options(
                profile,
                sample_interval_seconds=args.sample_interval,
                analysis_width=args.analysis_width,
                tile_columns=args.tile_columns,
                tile_rows=args.tile_rows,
                tile_change_threshold=args.tile_change_threshold,
                minimum_changed_tiles=args.minimum_changed_tiles,
                global_change_ratio=args.global_change_ratio,
                exclude_regions=tuple(args.exclude_roi),
            )
            work_dir = args.work_dir or Path("data/work") / f"{args.path.stem}-visual-events"
            payload = index_video_visual_events(
                args.path,
                work_dir,
                profile_name=profile.name if profile is not None else None,
                roi=analysis_options.roi,
                options=visual_options,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.command == "track-regions":
            profile = load_profile(args.profile) if args.profile is not None else None
            analysis_options = merge_analysis_options(profile)
            tracking_exclusions = (
                profile.visual_events.exclude_regions if profile is not None else ()
            ) + tuple(args.exclude_roi)
            output = args.output or args.path.parent / "region-tracks"
            work_dir = args.work_dir or Path("data/work") / f"{output.name}-region-tracks"
            manifest = localize_region_tracks(
                args.path,
                args.visual_events,
                output,
                work_dir,
                PaddleOcrProvider(args.models_root),
                glossary=Glossary.from_yaml(resolve_glossary_path(profile, args.glossary)),
                analysis_options=analysis_options,
                tracking_exclude_regions=tracking_exclusions,
                options=RegionTrackingOptions(
                    minimum_event_ratio=args.minimum_event_ratio,
                    maximum_event_duration_seconds=args.maximum_event_duration,
                    minimum_text_similarity=args.minimum_text_similarity,
                    tile_padding=args.tile_padding,
                    start_seconds=args.start_seconds,
                    end_seconds=args.end_seconds,
                    include_windows=tuple(args.include_window),
                ),
            )
            print(
                json.dumps(
                    {
                        "tracks": str((output / "tracks.json").resolve()),
                        "viewer": str((output / "viewer.html").resolve()),
                        "region_track_count": manifest["region_track_count"],
                        "region_tracking": manifest["region_tracking"],
                        "metrics": {
                            key: value
                            for key, value in manifest["metrics"].items()
                            if str(key).startswith("tracking_")
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "style-regions":
            output = args.output or args.path.parent / "styled-region-tracks"
            work_dir = args.work_dir or Path("data/work") / f"{output.name}-styles"
            manifest = style_region_tracks(
                args.path,
                output,
                work_dir,
                options=OverlayStyleOptions(
                    background_padding_ratio=args.background_padding_ratio,
                    minimum_contrast_ratio=args.minimum_contrast_ratio,
                    minimum_style_confidence=args.minimum_style_confidence,
                ),
            )
            print(
                json.dumps(
                    {
                        "tracks": str((output / "tracks.json").resolve()),
                        "viewer": str((output / "viewer.html").resolve()),
                        "region_styling": manifest["region_styling"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "layout-regions":
            output = args.output or args.path.parent / "laid-out-region-tracks"
            manifest = layout_region_tracks(
                args.path,
                output,
                options=OverlayLayoutOptions(
                    maximum_width_scale=args.maximum_width_scale,
                    maximum_frame_width_ratio=args.maximum_frame_width_ratio,
                    maximum_lines=args.maximum_lines,
                    minimum_font_scale=args.minimum_font_scale,
                ),
            )
            summary = manifest["region_layout"]
            print(
                json.dumps(
                    {
                        "tracks": str((output / "tracks.json").resolve()),
                        "viewer": str((output / "viewer.html").resolve()),
                        "layout_quality": str((output / "layout-quality.json").resolve()),
                        "region_layout": summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if args.require_pass and not summary["gates_passed"]:
                return 1
            return 0

        if args.command == "retranslate-tracks":
            output = args.output or args.path.parent / "retranslated-region-tracks"
            report = retranslate_region_tracks(
                args.path,
                output,
                create_translation_provider(args.translation_provider, args.models_root),
                Glossary.from_yaml(args.glossary),
                batch_size=args.batch_size,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        if args.command in {"index-video", "localize-video"}:
            profile = load_profile(args.profile) if args.profile is not None else None
            analysis_options = merge_analysis_options(
                profile,
                roi=args.roi,
                exclude_regions=tuple(args.exclude_roi),
                min_region_height=(
                    args.min_region_height if args.command == "localize-video" else None
                ),
                group_adjacent=(args.group_adjacent if args.command == "localize-video" else None),
                translation_batch_size=(
                    args.translation_batch_size if args.command == "localize-video" else None
                ),
            )
            scene_options = _merge_scene_options(profile, args)
            work_dir = args.work_dir or Path("data/work") / f"{args.path.stem}-timeline"
            scene_index = index_video_timeline(
                args.path,
                work_dir,
                interval_seconds=args.sample_interval,
                profile_name=profile.name if profile is not None else None,
                roi=analysis_options.roi,
                exclude_regions=analysis_options.exclude_regions,
                scene_options=scene_options,
                boundary_refinement_iterations=args.boundary_refinement_iterations,
            )
            if args.command == "index-video":
                print(json.dumps(scene_index, ensure_ascii=False, indent=2))
                return 0

            output = args.output or Path("data/output") / f"{args.path.stem}-timeline"
            ocr_provider = PaddleOcrProvider(args.models_root)
            translation_provider = create_translation_provider(
                args.translation_provider,
                args.models_root,
            )
            manifest = localize_scene_index(
                scene_index,
                output,
                ocr_provider,
                translation_provider,
                TranslationCache(args.cache),
                glossary=Glossary.from_yaml(resolve_glossary_path(profile, args.glossary)),
                analysis_options=analysis_options,
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0

        if args.command == "setup-models":
            if not args.ocr and not args.translation:
                raise ValueError("Select --ocr, --translation, or both.")
            prepared: dict[str, str] = {}
            if args.ocr:
                prepared["ocr"] = str(setup_ocr_models(args.models_root))
            if args.translation:
                prepared["translation"] = str(
                    setup_translation_model(args.models_root, args.translation_provider)
                )
            print(json.dumps(prepared, indent=2))
            return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _merge_scene_options(
    profile: LocalizationProfile | None,
    args: argparse.Namespace,
) -> SceneDetectionOptions:
    base = profile.scene_detection if profile is not None else SceneDetectionOptions()
    return SceneDetectionOptions(
        hash_size=args.hash_size if args.hash_size is not None else base.hash_size,
        hamming_threshold=(
            args.hamming_threshold if args.hamming_threshold is not None else base.hamming_threshold
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
