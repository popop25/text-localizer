from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
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


@dataclass(frozen=True, slots=True)
class TimestampedFrame:
    index: int
    timestamp_seconds: float
    path: str

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


def interval_timestamps(duration_seconds: float, interval_seconds: float) -> list[float]:
    if interval_seconds <= 0.0:
        raise ValueError("Sample interval must be greater than zero")
    if duration_seconds <= 0.0:
        return [0.0]
    count = max(1, math.ceil(duration_seconds / interval_seconds))
    return [index * interval_seconds for index in range(count)]


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


def extract_frame_at_timestamp(
    video_path: Path,
    output_path: Path,
    timestamp_seconds: float,
) -> tuple[Path, bool]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if timestamp_seconds < 0.0:
        raise ValueError("Frame timestamp must not be negative.")
    if output_path.is_file():
        return output_path.resolve(), True
    ffmpeg = resolve_executable("ffmpeg", environment_variable="FFMPEG_BIN")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Run doctor and set FFMPEG_BIN if needed.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path.resolve(), False


def extract_timestamped_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
    *,
    video_info: VideoInfo | None = None,
) -> tuple[tuple[TimestampedFrame, ...], float, bool]:
    """Extract fixed-interval frames, reusing a matching complete manifest when possible."""

    info = video_info or inspect_video(video_path)
    interval_timestamps(info.duration_seconds, interval_seconds)
    manifest_path = output_dir / "samples.json"
    reused = _load_matching_sample_manifest(
        manifest_path,
        video_path=video_path,
        interval_seconds=interval_seconds,
    )
    if reused is not None:
        return reused, 0.0, True

    ffmpeg = resolve_executable("ffmpeg", environment_variable="FFMPEG_BIN")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found. Run doctor and set FFMPEG_BIN if needed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in output_dir.glob("frame-*.png"):
        stale_frame.unlink()
    output_pattern = output_dir / "frame-%06d.png"
    started = perf_counter()
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=fps=1/{interval_seconds:.9f}:start_time=0",
        "-start_number",
        "0",
        "-y",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    latency_ms = (perf_counter() - started) * 1000.0
    paths = sorted(output_dir.glob("frame-*.png"))
    if not paths:
        raise RuntimeError("FFmpeg did not extract any timestamped frames.")
    actual_timestamps = [index * interval_seconds for index in range(len(paths))]
    frames = tuple(
        TimestampedFrame(
            index=index,
            timestamp_seconds=timestamp,
            path=str(path.resolve()),
        )
        for index, (timestamp, path) in enumerate(zip(actual_timestamps, paths, strict=True))
    )
    manifest = {
        "schema_version": 1,
        "video_path": str(video_path.resolve()),
        "video_size_bytes": video_path.stat().st_size,
        "interval_seconds": interval_seconds,
        "video_modified_ns": video_path.stat().st_mtime_ns,
        "frame_count": len(frames),
        "frames": [frame.to_dict() for frame in frames],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frames, latency_ms, False


def _load_matching_sample_manifest(
    manifest_path: Path,
    *,
    video_path: Path,
    interval_seconds: float,
) -> tuple[TimestampedFrame, ...] | None:
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            return None
        if Path(str(payload["video_path"])).resolve() != video_path.resolve():
            return None
        if int(payload.get("video_size_bytes", -1)) != video_path.stat().st_size:
            return None
        if int(payload.get("video_modified_ns", -1)) != video_path.stat().st_mtime_ns:
            return None
        if not math.isclose(float(payload["interval_seconds"]), interval_seconds):
            return None
        raw_frames = payload["frames"]
        if not isinstance(raw_frames, list) or not raw_frames:
            return None
        if int(payload.get("frame_count", -1)) != len(raw_frames):
            return None
        frames = tuple(
            TimestampedFrame(
                index=int(item["index"]),
                timestamp_seconds=float(item["timestamp_seconds"]),
                path=str(Path(str(item["path"])).resolve()),
            )
            for item in raw_frames
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if any(not Path(frame.path).is_file() for frame in frames):
        return None
    return frames
