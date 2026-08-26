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
from text_localizer.pipeline import analyze_static_image
from text_localizer.profile import (
    LocalizationProfile,
    SceneDetectionOptions,
    load_profile,
    merge_analysis_options,
    merge_visual_event_options,
    resolve_glossary_path,
)
from text_localizer.providers.m2m100 import M2M100TranslationProvider
from text_localizer.providers.paddleocr import PaddleOcrProvider
from text_localizer.scenes import index_frame_directory
from text_localizer.setup_models import setup_ocr_models, setup_translation_model
from text_localizer.temporal_quality import evaluate_temporal_quality_files
from text_localizer.timeline import index_video_timeline, localize_scene_index
from text_localizer.track_editor import load_track_manifest, write_track_editor
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
            translation_provider = M2M100TranslationProvider(args.models_root / "m2m100_418M")
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
            write_track_editor(output / "editor.html", video_path, manifest)
            print(
                json.dumps(
                    {
                        "viewer": str((output / "viewer.html").resolve()),
                        "editor": str((output / "editor.html").resolve()),
                    },
                    indent=2,
                )
            )
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
            translation_provider = M2M100TranslationProvider(args.models_root / "m2m100_418M")
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
                prepared["translation"] = str(setup_translation_model(args.models_root))
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
