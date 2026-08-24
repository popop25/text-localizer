from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from text_localizer.environment import resolve_executable


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    codec: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_frame_rate(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    divisor = float(denominator)
    return float(numerator) / divisor if divisor else 0.0


def evenly_spaced_timestamps(duration_seconds: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("Frame count must be at least 1")
    if duration_seconds <= 0.0:
        return [0.0]
    actual_count = min(count, max(1, int(duration_seconds) + 1))
    if actual_count == 1:
        return [duration_seconds / 2.0]
    edge_padding = min(0.5, duration_seconds * 0.02)
    start = edge_padding
    end = max(start, duration_seconds - edge_padding)
    step = (end - start) / (actual_count - 1)
    return [start + index * step for index in range(actual_count)]


def inspect_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    ffprobe = resolve_executable("ffprobe", environment_variable="FFPROBE_BIN")
    if ffprobe is None:
        raise RuntimeError("FFprobe was not found. Run doctor and set FFPROBE_BIN if needed.")

    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload: dict[str, Any] = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream was found in {path}")
    stream = streams[0]
    return VideoInfo(
        path=str(path.resolve()),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_frame_rate(str(stream.get("avg_frame_rate", "0"))),
        duration_seconds=float(payload.get("format", {}).get("duration", 0.0)),
        codec=str(stream.get("codec_name", "unknown")),
    )


def extract_representative_frames(video_path: Path, output_dir: Path, count: int) -> list[Path]:
    info = inspect_video(video_path)
    ffmpeg = resolve_executable("ffmpeg", environment_variable="FFMPEG_BIN")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Run doctor and set FFMPEG_BIN if needed.")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for index, timestamp in enumerate(
        evenly_spaced_timestamps(info.duration_seconds, count),
        start=1,
    ):
        output_path = output_dir / f"frame-{index:04d}.png"
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-y",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        outputs.append(output_path)
    return outputs
