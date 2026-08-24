from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from text_localizer.environment import run_doctor
from text_localizer.logging_config import configure_logging
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
            if not args.path.is_file():
                raise FileNotFoundError(f"Input image does not exist: {args.path}")
            print(
                "analyze-image is registered; the M1 provider is not installed yet.",
                file=sys.stderr,
            )
            return 2
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
