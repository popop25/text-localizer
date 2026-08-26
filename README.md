# text-localizer

`text-localizer` is a Windows-native, CPU-only pipeline for detecting Korean text in
lecture or screen-recording video frames, producing editable Japanese overlay data, and
rendering an approved timeline into a separate localized MP4. The original MP4 is never
modified.

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

M1.1 hardens static quality with pre-OCR regions of interest, auditable OCR normalization,
protected glossary terms, conservative same-line grouping, batched translation, and
automatic review flags. See `docs/m1.1-quality-hardening.md` for the design and measured
sample results.

M1.2 adds reusable YAML localization profiles and ROI-aware frame scene deduplication before
OCR. See `docs/m1.2-scene-dedup-profiles.md` for the scene-index schema and measurements.

M1.3 adds fixed-interval timestamped sampling, scene time ranges, representative-only
OCR/translation, resumable per-scene analysis, `tracks.json`, and a browser video player with
synchronized Japanese overlays. See `docs/m1.3-timestamped-tracks.md` for the track schema and
full-video measurements.

M1.4 adds no-OCR scene-boundary bisection, a local browser track editor, edited-track
validation, and viewer regeneration. See `docs/m1.4-refined-boundaries-editor.md` for the
workflow and full-video measurements.

M1.5 hardens the translation review loop with safe glossary alias normalization,
deterministic glossary-only labels, stored-OCR relocalization, selective translation reuse,
analysis-revision invalidation, and lazy OCR/translation model loading. See
`docs/m1.5-review-roundtrip-hardening.md` for the model probes, design decisions, and
full-video measurements.

M2 renders a zero-review timeline through ASS/libass into a separate H.264 MP4, copies the
original audio streams without re-encoding, validates the finished media, and writes an audit
report. See `docs/m2-final-mp4-rendering.md` for safeguards and full-video results.

M2.1 replaces unreadable single-line shrinking with resolution-scaled minimum text, bounded
horizontal expansion, mixed Japanese/English wrapping, collision-aware placement, and layout
metrics. See `docs/m2.1-readable-overlays.md` for the policy and full-video validation.

M2.2 makes fixed-interval samples exact and auditable with an explicit FFmpeg rounding
policy, strict schema-v2 resume checks, and representative-image fingerprints that prevent
stale OCR reuse. It also restores command-level `render-tracks` execution. See
`docs/m2.2-exact-timestamp-sampling.md` for the diagnosis and full-video validation.

M2.3 adds a quality-gated raw-source retry for failed protected-term translations, records
fallback audit metrics, and makes the recurring LangChain course objective deterministic in
the saved profile. See `docs/m2.3-protected-term-fallback.md` for the failure probes and
zero-review automatic full-video validation.

M3.0 adds an annotated temporal-quality benchmark and a model-free evaluator for visible
interval coverage, keyframe recall, boundary error, spatial IoU, and flicker. The first
0.mp4 baseline proves that hard scene transitions are already precise while scrolling text
needs independent region tracking. See docs/m3.0-temporal-quality-benchmark.md.

M3.1 drives the generated viewer and editor preview from decoded video-frame media time,
with an animation-frame fallback for older browsers. It removes playback-clock lag while
leaving the failing region-tracking benchmark visible. See docs/m3.1-frame-synced-viewer.md.

The source image and original MP4 are never modified. Intermediate data, review artifacts, and
the final rendered video stay separate from the source.

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
text-localizer deduplicate-frames data\work\frames --output data\work\scene-index.json
text-localizer index-video data\input\sample.mp4 --sample-interval 1
text-localizer localize-video data\input\sample.mp4 --profile config\profiles\browser-lecture.yaml
text-localizer render-tracks data\output\sample-timeline\tracks.edited.json
text-localizer evaluate-timeline data\output\0-m23-auto\tracks.json config\benchmarks\0-temporal-quality.json
text-localizer render-video data\output\sample-timeline\tracks.edited.json --output data\output\sample-final\sample.ja.mp4
```

For a screen recording, restrict OCR to the content area and optionally exclude content
inside that area. Coordinates are normalized `x,y,width,height` values against the full
image:

```powershell
text-localizer analyze-image data\work\frames\frame-0001.png `
  --output data\output\frame-0001 `
  --roi 0.22,0.16,0.75,0.84 `
  --exclude-roi 0.22,0.37,0.75,0.15 `
  --min-region-height 0.015
```

`--exclude-roi` is repeatable. Translation batches default to four items and can be changed
with `--translation-batch-size`. Use `--no-group-adjacent` to disable same-line grouping.

The tested browser layout can be reused through a saved profile:

```powershell
text-localizer analyze-image data\work\frames\frame-0001.png `
  --profile config\profiles\browser-lecture.yaml

text-localizer deduplicate-frames data\work\frames `
  --profile config\profiles\browser-lecture.yaml `
  --output data\work\scene-index.json
```

Scene indexing hashes only the ROI, masks configured exclusions, keeps every source frame,
and writes auditable representative and distance decisions. It does not load OCR or
translation models.

Timestamp and localize a complete video in two inspectable phases:

```powershell
text-localizer index-video data\input\sample.mp4 `
  --profile config\profiles\browser-lecture.yaml `
  --work-dir data\work\sample-timeline `
  --sample-interval 1 `
  --boundary-refinement-iterations 3

text-localizer localize-video data\input\sample.mp4 `
  --profile config\profiles\browser-lecture.yaml `
  --work-dir data\work\sample-timeline `
  --output data\output\sample-timeline `
  --sample-interval 1 `
  --boundary-refinement-iterations 3
```

`index-video` loads neither OCR nor translation. Boundary refinement hashes midpoint frames
inside detected transition brackets and does not add OCR calls. `localize-video` reuses the
timestamped samples, analyzes only scene representatives, and resumes from matching per-scene
`regions.json` files. It writes `tracks.json`, `viewer.html`, and `editor.html`; the viewer
references the unchanged source MP4 and suppresses regions carrying automatic review flags.
If the glossary or analysis revision changes while OCR geometry and analysis options still
match, `localize-video` re-runs normalization and translation from stored OCR instead of
loading PaddleOCR again. Prior model translations are reused only when the provider and
normalized source match and current quality checks still pass.
After editing and exporting JSON in the browser, use `render-tracks` to validate the complete
timeline and regenerate both HTML artifacts.

Once every review flag is resolved, burn the approved overlays into a separate MP4:

```powershell
text-localizer render-video data\output\sample-timeline\tracks.edited.json `
  --output data\output\sample-final\sample.ja.mp4
```

`render-video` refuses to overwrite the source, requires zero review flags, validates the
source against the track manifest, renders H.264/yuv420p video with libx264, and stream-copies
audio. It writes the MP4, the generated ASS timeline, and a `.render.json` audit report.
Existing output artifacts require an explicit `--overwrite`.
The default `readable-spatial-v1` layout keeps text at least 16 pixels high on 1080p video,
uses bounded expansion and multi-line wrapping only when needed, and records layout metrics in
the schema-v2 audit report.


`analyze-image` writes:

- `source.<ext>`: an unchanged copy of the input frame;
- `regions.json`: normalized OCR, classification, translation, cache, and metric data;
- `preview.png`: Japanese overlay preview using Yu Gothic or Meiryo;
- `viewer.html`: a self-contained overlay viewer for visual review.

Schema v2 also records raw and normalized source text, normalization edits, grouping and
classification reasons, glossary-aware cache metadata, and per-region `needs_review` flags.

## Quality checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest
```

## Data policy


Without `data/input/sample.mp4`, repository, unit, CLI, and fake-provider pipeline
validation can still complete. Real OCR, Korean-to-Japanese output, visual overlay quality,
and final-render latency remain sample-dependent validation steps.
Place the user-supplied video at `data/input/sample.mp4`. Inputs, intermediate frames,
outputs, model weights, caches, `.env` files, and virtual environments are ignored by Git.
