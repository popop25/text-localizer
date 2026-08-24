# text-localizer

`text-localizer` is a Windows-native, CPU-only pipeline for detecting Korean text in
lecture or screen-recording video frames and producing editable Japanese overlay data.
The original MP4 is never modified.

## Current milestone

M0 provides the repository foundation and model-independent video tooling. M1 adds a
static-frame localization pipeline:

- Korean OCR through the CPU/mobile PaddleOCR models;
- translate, preserve, and ignore classification for detected text;
- local Korean-to-Japanese translation through `facebook/m2m100_418M`;
- a normalized SQLite translation cache;
- editable `regions.json`, a rendered preview, and a self-contained HTML viewer;
- OCR, translation, total processing, and peak-process-memory metrics;
- replaceable provider interfaces and deterministic tests without model downloads.

The source image and original MP4 are never modified. The static pipeline works on an extracted frame; temporal tracking and final video rendering are intentionally outside M1.

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

## Explicit model setup

Model packages and weights are not installed by bootstrap or tests. Download only the
component you want:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-models.ps1 -Model ocr
powershell -ExecutionPolicy Bypass -File .\scripts\setup-models.ps1 -Model translation
# Or prepare both:
powershell -ExecutionPolicy Bypass -File .\scripts\setup-models.ps1 -Model all
```

The application runs providers in offline/local-only mode after setup. PaddleOCR uses the
mobile detection model and Korean mobile recognition model on CPU. Translation loads only
from `models\m2m100_418M`. All downloaded content under `models\` is ignored by Git.

## Commands

```powershell
text-localizer doctor --json
text-localizer inspect-video data\input\sample.mp4
text-localizer extract-frames data\input\sample.mp4 --count 20
text-localizer analyze-image data\work\frames\frame-0001.png --output data\output\frame-0001
```

`analyze-image` writes:

- `source.<ext>`: an unchanged copy of the input frame;
- `regions.json`: normalized OCR, classification, translation, cache, and metric data;
- `preview.png`: Japanese overlay preview using Yu Gothic or Meiryo;
- `viewer.html`: a self-contained overlay viewer for visual review.

## Quality checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest
```

## Data policy


Without `data/input/sample.mp4`, repository, unit, CLI, and fake-provider pipeline
validation can still complete. Real OCR quality, Korean-to-Japanese output quality, and
CPU latency/memory measurements on the target video remain the single sample-dependent
validation step.
Place the user-supplied video at `data/input/sample.mp4`. Inputs, intermediate frames,
outputs, model weights, caches, `.env` files, and virtual environments are ignored by Git.
