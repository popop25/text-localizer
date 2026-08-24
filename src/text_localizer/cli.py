from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from text_localizer.cache import TranslationCache
from text_localizer.environment import run_doctor
from text_localizer.logging_config import configure_logging
from text_localizer.pipeline import analyze_static_image
from text_localizer.providers.m2m100 import M2M100TranslationProvider
from text_localizer.providers.paddleocr import PaddleOcrProvider
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

    setup = subparsers.add_parser("setup-models", help="Explicitly download model files")
    setup.add_argument("--ocr", action="store_true")
    setup.add_argument("--translation", action="store_true")
    setup.add_argument("--models-root", type=Path, default=Path("models"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
            payload = analyze_static_image(
                args.path,
                output,
                ocr_provider,
                translation_provider,
                TranslationCache(args.cache),
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
