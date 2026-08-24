# text-localizer

`text-localizer` is a Windows-native, CPU-only pipeline for detecting Korean text in
lecture or screen-recording video frames and producing editable Japanese overlay data.
The original MP4 is never modified.

## Current milestone

M0 provides the repository foundation and model-independent video tooling:

- environment diagnostics;
- FFprobe video inspection;
- representative FFmpeg frame extraction;
- typed, normalized text-region objects;
- deterministic tests that do not download model weights.

M1 adds PaddleOCR and local `facebook/m2m100_418M` translation behind replaceable
interfaces. Model downloads are always explicit and model/video files remain outside Git.

## Requirements

- Windows 10 or 11, x64
- Python 3.12
- FFmpeg and FFprobe
- CPU inference; no Docker, WSL, hosted inference, or paid API

On the development machine, the tool auto-detects Python at
`C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe` and FFmpeg
under `D:\05_tools\ffmpeg\bin`. Other installations can use `PYTHON_BIN`, `FFMPEG_BIN`,
and `FFPROBE_BIN` environment variables.

## Bootstrap

```powershell
Set-Location D:\09_personal\text-localizer
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Commands

```powershell
text-localizer doctor --json
text-localizer inspect-video data\input\sample.mp4
text-localizer extract-frames data\input\sample.mp4 --count 20
text-localizer analyze-image data\work\frames\frame-0001.png
```

`analyze-image` is registered in M0 and completed by the M1 static-frame pipeline.

## Quality checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest
```

## Data policy

Place the user-supplied video at `data/input/sample.mp4`. Inputs, intermediate frames,
outputs, model weights, caches, `.env` files, and virtual environments are ignored by Git.
