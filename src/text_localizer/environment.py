from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

KNOWN_WINDOWS_BINARIES: dict[str, tuple[Path, ...]] = {
    "ffmpeg": (Path(r"D:\05_tools\ffmpeg\bin\ffmpeg.exe"),),
    "ffprobe": (Path(r"D:\05_tools\ffmpeg\bin\ffprobe.exe"),),
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_executable(
    name: str,
    *,
    environment_variable: str | None = None,
    known_paths: Iterable[Path] | None = None,
) -> Path | None:
    if environment_variable:
        configured = os.environ.get(environment_variable)
        if configured:
            path = Path(configured).expanduser()
            if path.is_file():
                return path.resolve()

    discovered = shutil.which(name)
    if discovered:
        return Path(discovered).resolve()

    candidates = tuple(known_paths or KNOWN_WINDOWS_BINARIES.get(name, ()))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def executable_version(path: Path) -> str:
    result = subprocess.run(
        [str(path), "-version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout or result.stderr
    return output.splitlines()[0].strip() if output else "version unavailable"


def run_doctor() -> list[CheckResult]:
    results = [
        CheckResult(
            name="python",
            ok=sys.version_info[:2] == (3, 12),
            detail=f"{sys.version.split()[0]} at {sys.executable}",
        )
    ]

    for name, variable in (("ffmpeg", "FFMPEG_BIN"), ("ffprobe", "FFPROBE_BIN")):
        path = resolve_executable(name, environment_variable=variable)
        if path is None:
            results.append(
                CheckResult(
                    name=name,
                    ok=False,
                    detail=f"Not found. Add it to PATH or set {variable}.",
                )
            )
        else:
            results.append(CheckResult(name=name, ok=True, detail=executable_version(path)))

    font_candidates = (
        Path(r"C:\Windows\Fonts\YuGothM.ttc"),
        Path(r"C:\Windows\Fonts\meiryo.ttc"),
    )
    font = next((path for path in font_candidates if path.is_file()), None)
    results.append(
        CheckResult(
            name="japanese-font",
            ok=font is not None,
            detail=str(font) if font else "Yu Gothic or Meiryo was not found.",
        )
    )
    return results
