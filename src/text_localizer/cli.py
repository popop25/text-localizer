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
from text_localizer.providers.m2m100 import M2M100TranslationProvider
from text_localizer.providers.paddleocr import PaddleOcrProvider
from text_localizer.quality import AnalysisOptions
from text_localizer.setup_models import setup_ocr_models, setup_translation_model
from text_localizer.video import extract_representative_frames, inspect_video


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
        "--cache",
        type=Path,
        default=Path("data/work/translations.sqlite3"),
    )
    analyze.add_argument(
        "--glossary",
        type=Path,
        default=Path("config/glossary.ko-ja.yaml"),
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
    analyze.add_argument("--min-region-height", type=float, default=0.0)
    analyze.add_argument("--translation-batch-size", type=int, default=4)
    analyze.add_argument("--no-group-adjacent", action="store_true")

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
            ocr_provider = PaddleOcrProvider(args.models_root)
            translation_provider = M2M100TranslationProvider(args.models_root / "m2m100_418M")
            options = AnalysisOptions(
                roi=args.roi,
                exclude_regions=tuple(args.exclude_roi),
                min_region_height=args.min_region_height,
                group_adjacent=not args.no_group_adjacent,
                translation_batch_size=args.translation_batch_size,
            )
            payload = analyze_static_image(
                args.path,
                output,
                ocr_provider,
                translation_provider,
                TranslationCache(args.cache),
                glossary=Glossary.from_yaml(args.glossary),
                options=options,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
