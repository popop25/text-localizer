# M2 final MP4 rendering

M2 turns an approved temporal track manifest into a separate, portable MP4. Japanese overlays
are burned into the video with ASS/libass; the original audio streams are copied. The source
MP4 is read-only throughout the operation.

## Rendering design

The renderer applies the same normalized geometry and timestamps used by the browser review
player:

1. Load and validate the complete track manifest.
2. Refuse rendering while any region still has `needs_review`.
3. Verify that the selected source matches the manifest dimensions, frame rate, and duration.
4. Convert approved `translate` regions to positioned ASS events. Each region has an
   almost-opaque background rectangle and a clipped Yu Gothic or Meiryo text event.
5. Fit the Japanese text to its OCR box and quantize track boundaries to ASS centiseconds.
6. Render video with FFmpeg `libx264`, CRF 18, the medium preset, and `yuv420p`; copy audio
   streams without re-encoding.
7. Probe the temporary MP4 and require H.264, matching dimensions, frame rate and duration, and
   identical input/output audio codec lists.
8. Confirm that the source size and modification time did not change, then atomically publish
   the video, ASS file, and JSON audit report.

The command refuses to target the source path. It also refuses existing MP4, ASS, or report
artifacts unless `--overwrite` is explicitly provided. A failed encode or validation leaves
the prior published artifacts untouched.

## Command

```powershell
text-localizer render-video data\output\0-m13-timeline\tracks.json `
  --output data\output\0-m2-final\0.ja.mp4
```

Optional controls:

- `--video`: select a source other than the path stored in the manifest; it must still match.
- `--crf`: libx264 quality from 0 to 51; default 18.
- `--preset`: a supported libx264 preset; default `medium`.
- `--overwrite`: replace all three output artifacts after a successful render.

The command produces:

- `0.ja.mp4`: burned-in Japanese video with copied audio;
- `0.ja.ass`: inspectable subtitle/overlay event source;
- `0.ja.render.json`: source fingerprint, media properties, encoding parameters, counts,
  latency, and artifact paths.

## Full-video validation

Measured on the user-supplied 175.275102-second Korean lecture `0.mp4`, Windows native,
Python 3.12.9, CPU only:

| Measurement | Result |
| --- | ---: |
| Source / output video | H.264, 1920x1080, 30000/1001 fps |
| Source / output video frames | 5253 / 5253 |
| Source / output duration | 175.275102 s / 175.275102 s |
| Source / output audio codec | AAC / AAC (stream copy) |
| Approved overlay regions | 207 |
| ASS dialogue events | 414 |
| Remaining review flags | 0 |
| Encoder | libx264, medium, CRF 18, yuv420p |
| Render latency | 59.514 s |
| Source / output size | 59,335,743 / 14,549,330 bytes |
| Full output decode | PASS |
| Source fingerprint unchanged | PASS |

Visual samples at 8, 21, 92, and 140 seconds confirmed that the overlays follow the reviewed
timeline, use the reviewed coordinates, exclude configured non-content regions, and cover the
original Korean text with near-opaque backgrounds.

## Known limitations

This milestone performs spatial overlay replacement, not generative inpainting. Long Japanese
translations can become small when constrained to a short Korean OCR box. Burned-in overlays
cannot be toggled after rendering. M2 translates on-screen text only; it does not translate,
dub, or synthesize narration, and configured excluded regions remain unchanged.

All generated media and audit artifacts remain ignored by Git.
