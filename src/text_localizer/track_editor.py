from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, cast

TIME_TOLERANCE_SECONDS = 1e-6


def load_track_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Track manifest does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Track manifest root must be a mapping.")
    manifest = cast(dict[str, Any], raw)
    validate_track_manifest(manifest)
    return manifest


def validate_track_manifest(manifest: dict[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("Track manifest schema_version must be 1 or 2.")
    video = manifest.get("video")
    tracks = manifest.get("tracks")
    sampling = manifest.get("sampling")
    if not isinstance(video, dict):
        raise ValueError("Track manifest requires video metadata.")
    if not isinstance(video.get("path"), str):
        raise ValueError("Track manifest video.path must be a string.")
    duration = _positive_number(video.get("duration_seconds"), "video.duration_seconds")
    if not isinstance(sampling, dict):
        raise ValueError("Track manifest requires sampling metadata.")
    _positive_number(
        sampling.get("transition_precision_seconds"), "sampling.transition_precision_seconds"
    )
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("Track manifest requires at least one track.")
    if manifest.get("track_count") != len(tracks):
        raise ValueError("Track manifest track_count does not match tracks.")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Track manifest requires metrics.")
    review_count = metrics.get("needs_review_count")
    if isinstance(review_count, bool) or not isinstance(review_count, int) or review_count < 0:
        raise ValueError("Track manifest metrics.needs_review_count must be non-negative.")

    previous_end: float | None = None
    track_ids: set[str] = set()
    actual_review_count = 0
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise ValueError(f"Track {index} must be a mapping.")
        track_id = track.get("track_id")
        if not isinstance(track_id, str) or not track_id or track_id in track_ids:
            raise ValueError(f"Track {index} has an invalid or duplicate track_id.")
        track_ids.add(track_id)
        start = _number(track.get("start_seconds"), f"tracks[{index}].start_seconds")
        end = _number(track.get("end_seconds"), f"tracks[{index}].end_seconds")
        if start < 0.0 or end <= start or end > duration + TIME_TOLERANCE_SECONDS:
            raise ValueError(f"Track {track_id} has an invalid time range.")
        if index == 0 and abs(start) > TIME_TOLERANCE_SECONDS:
            raise ValueError("The first track must start at zero.")
        if previous_end is not None and abs(start - previous_end) > TIME_TOLERANCE_SECONDS:
            raise ValueError(f"Track {track_id} is not contiguous with the previous track.")
        previous_end = end
        regions = track.get("regions")
        if not isinstance(regions, list):
            raise ValueError(f"Track {track_id} regions must be a list.")
        for region_index, region in enumerate(regions):
            _validate_region(region, track_id, region_index)
            actual_review_count += int(region["needs_review"])
    if previous_end is None or abs(previous_end - duration) > TIME_TOLERANCE_SECONDS:
        raise ValueError("The final track must end at the video duration.")
    if schema_version == 2:
        actual_review_count = _validate_region_tracks(manifest, duration)
    if review_count != actual_review_count:
        raise ValueError("Track manifest needs_review_count does not match its regions.")


def _validate_region_tracks(manifest: dict[str, Any], duration: float) -> int:
    raw_tracks = manifest.get("region_tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("Schema-v2 track manifest requires region_tracks.")
    if manifest.get("region_track_count") != len(raw_tracks):
        raise ValueError("Track manifest region_track_count does not match region_tracks.")

    track_ids: set[str] = set()
    review_count = 0
    for track_index, raw_track in enumerate(raw_tracks):
        if not isinstance(raw_track, dict):
            raise ValueError(f"Region track {track_index} must be a mapping.")
        track_id = raw_track.get("region_track_id")
        if not isinstance(track_id, str) or not track_id or track_id in track_ids:
            raise ValueError(f"Region track {track_index} has an invalid or duplicate id.")
        track_ids.add(track_id)
        start = _number(raw_track.get("start_seconds"), f"region_tracks[{track_index}].start")
        end = _number(raw_track.get("end_seconds"), f"region_tracks[{track_index}].end")
        if start < 0.0 or end <= start or end > duration + TIME_TOLERANCE_SECONDS:
            raise ValueError(f"Region track {track_id} has an invalid time range.")
        region = raw_track.get("region")
        _validate_region(region, track_id, 0)
        assert isinstance(region, dict)
        review_count += int(region["needs_review"])
        if "overlay_style" in raw_track:
            _validate_overlay_style(raw_track.get("overlay_style"), track_id)
        if "overlay_layout" in raw_track:
            _validate_overlay_layout(raw_track.get("overlay_layout"), track_id, region)
        if "overlay_replacement" in raw_track:
            _validate_overlay_replacement(raw_track.get("overlay_replacement"), track_id)

        keyframes = raw_track.get("keyframes")
        if not isinstance(keyframes, list) or not keyframes:
            raise ValueError(f"Region track {track_id} requires keyframes.")
        previous_timestamp: float | None = None
        for keyframe_index, keyframe in enumerate(keyframes):
            if not isinstance(keyframe, dict):
                raise ValueError(
                    f"Region track {track_id} keyframe {keyframe_index} must be a mapping."
                )
            timestamp = _number(
                keyframe.get("timestamp_seconds"),
                f"region_tracks[{track_index}].keyframes[{keyframe_index}].timestamp",
            )
            if timestamp < start - TIME_TOLERANCE_SECONDS or timestamp > end:
                raise ValueError(f"Region track {track_id} keyframe is outside its time range.")
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f"Region track {track_id} keyframes must be strictly ordered.")
            previous_timestamp = timestamp
            _validate_bbox(
                keyframe.get("bbox"),
                f"Region track {track_id} keyframe {keyframe_index}",
            )
            if keyframe.get("interpolation") not in {"hold", "linear"}:
                raise ValueError(
                    f"Region track {track_id} keyframe interpolation must be hold or linear."
                )
        if abs(float(keyframes[0]["timestamp_seconds"]) - start) > TIME_TOLERANCE_SECONDS:
            raise ValueError(f"Region track {track_id} first keyframe must match its start time.")
    return review_count


def write_track_editor(
    output_path: Path,
    video_path: Path,
    manifest: dict[str, Any],
) -> None:
    validate_track_manifest(manifest)
    if manifest.get("schema_version") == 2:
        raise ValueError("Schema-v2 region keyframes are read-only; use viewer.html for review.")
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    video_json = json.dumps(video_path.resolve().as_uri()).replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>text-localizer track editor</title>
  <style>
    :root {{ color-scheme: dark; font-family: "Yu Gothic", "Meiryo", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0c0e12; color: #f3f5f8; }}
    button, input, select, textarea {{ font: inherit; }}
    button {{ cursor: pointer; }}
    .topbar {{ display: flex; gap: .75rem; align-items: center; padding: .75rem 1rem;
               background: #171a20; position: sticky; top: 0; z-index: 20; }}
    .topbar .status {{ margin-left: auto; color: #aeb7c4; }}
    .workspace {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(340px, 1fr);
                  min-height: calc(100vh - 58px); }}
    .preview {{ padding: 1rem; }}
    .stage {{ position: sticky; top: 74px; background: black; width: 100%; overflow: hidden; }}
    video {{ display: block; width: 100%; height: auto; }}
    .overlay-layer {{ position: absolute; inset: 0; pointer-events: none; }}
    .overlay {{ position: absolute; display: flex; align-items: center; overflow: hidden;
                padding: 2px 4px; background: rgba(11, 13, 17, .88); border: 1px solid transparent;
                line-height: 1.15; }}
    .overlay.review {{ border-color: #ffca6b; background: rgba(74, 51, 14, .9); }}
    .panel {{ padding: 1rem; background: #12151a; border-left: 1px solid #292e37; overflow: auto; }}
    .track-nav, .times {{ display: grid; gap: .6rem; margin-bottom: 1rem; }}
    .track-nav {{ grid-template-columns: auto 1fr auto auto; }}
    .times {{ grid-template-columns: 1fr 1fr; }}
    label {{ display: grid; gap: .3rem; color: #c7ced8; }}
    input, select, textarea, button {{
      border: 1px solid #39414e; border-radius: 5px; background: #1a1f27;
      color: #f3f5f8; padding: .45rem .55rem;
    }}
    textarea {{ width: 100%; min-height: 4rem; resize: vertical; }}
    .region {{
      border: 1px solid #303743; border-radius: 7px; padding: .7rem; margin-bottom: .7rem;
    }}
    .region.review {{ border-color: #9a6a22; }}
    .source {{ color: #aeb7c4; white-space: pre-wrap; margin: 0 0 .55rem; }}
    .meta {{ display: flex; justify-content: space-between; gap: .5rem; margin-top: .45rem;
             color: #8f9aaa; font-size: .82rem; }}
    .approve {{ display: flex; grid-template-columns: auto 1fr; align-items: center; gap: .45rem; }}
    .empty {{ color: #8f9aaa; }}
    @media (max-width: 900px) {{
      .workspace {{ grid-template-columns: 1fr; }}
      .stage {{ position: relative; top: auto; }}
      .panel {{ border-left: 0; border-top: 1px solid #292e37; }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <button id="export">Export edited JSON</button>
    <label>Import JSON <input id="import" type="file" accept="application/json"></label>
    <label><input id="overlay-toggle" type="checkbox" checked> Preview overlays</label>
    <span id="status" class="status">Ready</span>
  </div>
  <main class="workspace">
    <section class="preview">
      <div class="stage">
        <video id="video" controls preload="metadata"></video>
        <div id="overlays" class="overlay-layer"></div>
      </div>
    </section>
    <aside class="panel">
      <div class="track-nav">
        <button id="previous" aria-label="Previous track">←</button>
        <select id="track-select" aria-label="Track"></select>
        <button id="seek">Seek</button>
        <button id="next" aria-label="Next track">→</button>
      </div>
      <div class="times">
        <label>Start seconds <input id="start" type="number" min="0" step="0.001"></label>
        <label>End seconds <input id="end" type="number" min="0" step="0.001"></label>
      </div>
      <div id="regions"></div>
    </aside>
  </main>
  <script>
    let manifest = {manifest_json};
    const videoSource = {video_json};
    const video = document.getElementById("video");
    const layer = document.getElementById("overlays");
    const select = document.getElementById("track-select");
    const startInput = document.getElementById("start");
    const endInput = document.getElementById("end");
    const regionsNode = document.getElementById("regions");
    const status = document.getElementById("status");
    const overlayToggle = document.getElementById("overlay-toggle");
    let selectedIndex = 0;
    let activePreviewTrackId = null;
    let fallbackFrameRequest = null;
    let videoFrameCallbackActive = false;

    function selectedTrack() {{ return manifest.tracks[selectedIndex]; }}
    function clamp(value, minimum, maximum) {{
      return Math.min(maximum, Math.max(minimum, value));
    }}
    function setStatus(message) {{ status.textContent = message; }}

    function findTrack(time) {{
      return manifest.tracks.find((item, index) =>
        time >= item.start_seconds &&
        (time < item.end_seconds ||
          (index === manifest.tracks.length - 1 && time <= item.end_seconds))
      );
    }}

    function rebuildTrackOptions() {{
      select.replaceChildren();
      manifest.tracks.forEach((track, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        const start = track.start_seconds.toFixed(3);
        const end = track.end_seconds.toFixed(3);
        option.textContent = `${{track.track_id}} · ${{start}}–${{end}}`;
        select.appendChild(option);
      }});
      select.value = String(selectedIndex);
    }}

    function renderOverlays(track) {{
      layer.replaceChildren();
      if (!track || !overlayToggle.checked) return;
      track.regions.forEach(region => {{
        if (region.action !== "translate" || !region.translated_text) return;
        const node = document.createElement("div");
        node.className = "overlay" + (region.needs_review ? " review" : "");
        node.textContent = region.translated_text;
        node.style.left = (region.bbox.x * 100) + "%";
        node.style.top = (region.bbox.y * 100) + "%";
        node.style.width = (region.bbox.width * 100) + "%";
        node.style.height = (region.bbox.height * 100) + "%";
        node.style.fontSize = Math.max(10, region.bbox.height * video.clientHeight * .8) + "px";
        layer.appendChild(node);
      }});
    }}

    function renderPreview(mediaTime = video.currentTime, force = false) {{
      const track = findTrack(mediaTime) || selectedTrack();
      const nextTrackId = track ? track.track_id : null;
      if (!force && nextTrackId === activePreviewTrackId) return;
      activePreviewTrackId = nextTrackId;
      renderOverlays(track);
    }}

    function startVideoFrameClock() {{
      if (videoFrameCallbackActive ||
          typeof video.requestVideoFrameCallback !== "function") return;
      videoFrameCallbackActive = true;
      const onVideoFrame = (_now, metadata) => {{
        renderPreview(metadata.mediaTime);
        video.requestVideoFrameCallback(onVideoFrame);
      }};
      video.requestVideoFrameCallback(onVideoFrame);
    }}

    function startFallbackFrameClock() {{
      if (typeof video.requestVideoFrameCallback === "function" ||
          fallbackFrameRequest !== null) return;
      const onAnimationFrame = () => {{
        renderPreview(video.currentTime);
        if (video.paused || video.ended) {{
          fallbackFrameRequest = null;
          return;
        }}
        fallbackFrameRequest = requestAnimationFrame(onAnimationFrame);
      }};
      fallbackFrameRequest = requestAnimationFrame(onAnimationFrame);
    }}

    function renderRegions() {{
      const track = selectedTrack();
      regionsNode.replaceChildren();
      const editable = track.regions.filter(region => region.action === "translate");
      if (!editable.length) {{
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "No translated regions in this scene.";
        regionsNode.appendChild(empty);
        return;
      }}
      editable.forEach((region, index) => {{
        const card = document.createElement("section");
        card.className = "region" + (region.needs_review ? " review" : "");
        const source = document.createElement("p");
        source.className = "source";
        source.textContent = region.source_text;
        const target = document.createElement("textarea");
        target.value = region.translated_text || "";
        target.setAttribute("aria-label", `Translation ${{index + 1}}`);
        target.addEventListener("input", () => {{
          region.translated_text = target.value;
          region.human_edited = true;
          renderPreview(video.currentTime, true);
          setStatus("Unsaved edits");
        }});
        const approve = document.createElement("label");
        approve.className = "approve";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = !region.needs_review;
        const approveText = document.createElement("span");
        approveText.textContent = "Approved for overlay";
        checkbox.addEventListener("change", () => {{
          region.needs_review = !checkbox.checked;
          region.human_reviewed = checkbox.checked;
          card.classList.toggle("review", region.needs_review);
          renderPreview(video.currentTime, true);
          setStatus("Unsaved review change");
        }});
        approve.append(checkbox, approveText);
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = region.translation_quality_flags?.length
          ? region.translation_quality_flags.join(", ")
          : `OCR ${{(region.ocr_confidence * 100).toFixed(1)}}%`;
        card.append(source, target, approve, meta);
        regionsNode.appendChild(card);
      }});
    }}

    function renderSelected(seek = false) {{
      const track = selectedTrack();
      rebuildTrackOptions();
      startInput.value = track.start_seconds.toFixed(3);
      endInput.value = track.end_seconds.toFixed(3);
      startInput.disabled = selectedIndex === 0;
      endInput.disabled = selectedIndex === manifest.tracks.length - 1;
      renderRegions();
      if (seek) video.currentTime = track.start_seconds;
      renderPreview(video.currentTime, true);
    }}

    function updateTimes() {{
      const track = selectedTrack();
      const requestedStart = Number(startInput.value);
      const requestedEnd = Number(endInput.value);
      if (!Number.isFinite(requestedStart) || !Number.isFinite(requestedEnd)) {{
        setStatus("Invalid time value");
        renderSelected(false);
        return;
      }}
      if (selectedIndex > 0) {{
        const previous = manifest.tracks[selectedIndex - 1];
        track.start_seconds = clamp(requestedStart, previous.start_seconds + .001,
                                    track.end_seconds - .001);
        previous.end_seconds = track.start_seconds;
      }}
      if (selectedIndex < manifest.tracks.length - 1) {{
        const following = manifest.tracks[selectedIndex + 1];
        track.end_seconds = clamp(requestedEnd, track.start_seconds + .001,
                                  following.end_seconds - .001);
        following.start_seconds = track.end_seconds;
      }}
      setStatus("Unsaved time change");
      renderSelected(false);
    }}

    function reviewCount() {{
      return manifest.tracks.reduce((total, track) =>
        total + track.regions.filter(region => region.needs_review).length, 0);
    }}

    function exportManifest() {{
      manifest.track_count = manifest.tracks.length;
      manifest.metrics.needs_review_count = reviewCount();
      manifest.editor = {{ schema_version: 1, edited_at: new Date().toISOString() }};
      const blob = new Blob([JSON.stringify(manifest, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "tracks.edited.json";
      link.click();
      URL.revokeObjectURL(url);
      setStatus("Exported tracks.edited.json");
    }}

    select.addEventListener("change", () => {{
      selectedIndex = Number(select.value);
      renderSelected(true);
    }});
    document.getElementById("previous").addEventListener("click", () => {{
      selectedIndex = Math.max(0, selectedIndex - 1); renderSelected(true);
    }});
    document.getElementById("next").addEventListener("click", () => {{
      selectedIndex = Math.min(manifest.tracks.length - 1, selectedIndex + 1); renderSelected(true);
    }});
    document.getElementById("seek").addEventListener("click", () => {{
      video.currentTime = selectedTrack().start_seconds;
    }});
    startInput.addEventListener("change", updateTimes);
    endInput.addEventListener("change", updateTimes);
    overlayToggle.addEventListener("change", () => renderPreview(video.currentTime, true));
    document.getElementById("export").addEventListener("click", exportManifest);
    document.getElementById("import").addEventListener("change", async event => {{
      const file = event.target.files?.[0];
      if (!file) return;
      try {{
        const imported = JSON.parse(await file.text());
        if (imported.schema_version !== 1 || !Array.isArray(imported.tracks)) throw new Error();
        manifest = imported;
        selectedIndex = 0;
        renderSelected(true);
        setStatus(`Imported ${{file.name}}`);
      }} catch {{
        setStatus("Import failed: invalid track JSON");
      }}
    }});
    video.addEventListener("loadedmetadata", () => {{
      renderPreview();
      startVideoFrameClock();
    }});
    video.addEventListener("play", startFallbackFrameClock);
    video.addEventListener("timeupdate", () => {{
      if (typeof video.requestVideoFrameCallback !== "function") renderPreview();
    }});
    video.addEventListener("seeked", () => renderPreview(video.currentTime, true));
    video.addEventListener("pause", () => renderPreview());
    window.addEventListener("resize", () => renderPreview(video.currentTime, true));
    video.src = videoSource;
    renderSelected(false);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _validate_region(raw: Any, track_id: str, region_index: int) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Track {track_id} region {region_index} must be a mapping.")
    _validate_bbox(raw.get("bbox"), f"Track {track_id} region {region_index}")
    translated = raw.get("translated_text")
    if translated is not None and not isinstance(translated, str):
        raise ValueError(f"Track {track_id} region {region_index} translation is invalid.")
    if not isinstance(raw.get("needs_review"), bool):
        raise ValueError(f"Track {track_id} region {region_index} needs_review must be boolean.")


def _validate_overlay_style(raw: Any, track_id: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Region track {track_id} overlay_style must be a mapping.")
    if raw.get("schema_version") != 1 or raw.get("algorithm") not in {
        "source-palette-v1",
        "source-palette-v2",
    }:
        raise ValueError(f"Region track {track_id} overlay_style has an unsupported schema.")
    for field in ("foreground_color", "background_color"):
        value = raw.get(field)
        if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
            raise ValueError(f"Region track {track_id} overlay_style {field} is invalid.")
    for field, minimum, maximum in (
        ("background_opacity", 0.0, 1.0),
        ("font_size_ratio", 0.0, 1.0),
        ("line_height", 0.8, 2.0),
        ("border_radius_px", 0.0, 100.0),
        ("confidence", 0.0, 1.0),
        ("background_uniformity", 0.0, 1.0),
        ("foreground_pixel_ratio", 0.0, 1.0),
        ("contrast_ratio", 1.0, 21.0),
        ("sample_timestamp_seconds", 0.0, float("inf")),
    ):
        value = _number(raw.get(field), f"Region track {track_id} overlay_style {field}")
        if value < minimum or value > maximum or (field == "font_size_ratio" and value == 0.0):
            raise ValueError(f"Region track {track_id} overlay_style {field} is out of range.")
    font_weight = raw.get("font_weight")
    if (
        isinstance(font_weight, bool)
        or not isinstance(font_weight, int)
        or font_weight < 100
        or font_weight > 900
    ):
        raise ValueError(f"Region track {track_id} overlay_style font_weight is invalid.")
    if raw.get("text_shadow") not in {"none", "0 1px 1px #000000"}:
        raise ValueError(f"Region track {track_id} overlay_style text_shadow is invalid.")
    for field in ("foreground_adjusted", "fallback_used"):
        if not isinstance(raw.get(field), bool):
            raise ValueError(f"Region track {track_id} overlay_style {field} must be boolean.")
    fallback_reason = raw.get("fallback_reason")
    if fallback_reason not in {None, "no-distinct-foreground", "low-style-confidence"}:
        raise ValueError(f"Region track {track_id} overlay_style fallback_reason is invalid.")
    if (fallback_reason is None) == (raw.get("fallback_used") is True):
        raise ValueError(
            f"Region track {track_id} overlay_style fallback reason does not match its state."
        )
    if raw.get("algorithm") == "source-palette-v2" and not raw.get("fallback_used"):
        glyph_height_ratio = _number(
            raw.get("glyph_height_ratio"),
            f"Region track {track_id} overlay_style glyph_height_ratio",
        )
        if not 0.0 < glyph_height_ratio <= 1.0:
            raise ValueError(
                f"Region track {track_id} overlay_style glyph_height_ratio is out of range."
            )
        if raw.get("glyph_height_method") != "column-span-p80-v1":
            raise ValueError(
                f"Region track {track_id} overlay_style glyph_height_method is invalid."
            )


def _validate_overlay_layout(raw: Any, track_id: str, region: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Region track {track_id} overlay_layout must be a mapping.")
    if raw.get("schema_version") != 1 or raw.get("algorithm") != "adaptive-fit-v1":
        raise ValueError(f"Region track {track_id} overlay_layout has an unsupported schema.")
    translated = region.get("translated_text")
    expected_text = (
        translated.replace("\r\n", "\n").replace("\r", "\n").strip()
        if isinstance(translated, str)
        else None
    )
    if not expected_text or raw.get("text") != expected_text:
        raise ValueError(f"Region track {track_id} overlay_layout text is stale or invalid.")
    digest = raw.get("text_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Region track {track_id} overlay_layout text_sha256 is invalid.")
    lines = raw.get("lines")
    if (
        not isinstance(lines, list)
        or not lines
        or any(not isinstance(line, str) for line in lines)
        or re.sub(r"\s+", "", "".join(lines)) != re.sub(r"\s+", "", expected_text)
    ):
        raise ValueError(f"Region track {track_id} overlay_layout lines are invalid.")
    line_count = raw.get("line_count")
    if isinstance(line_count, bool) or not isinstance(line_count, int) or line_count != len(lines):
        raise ValueError(f"Region track {track_id} overlay_layout line_count is invalid.")
    if raw.get("fit_strategy") not in {"source-box", "expand", "wrap", "shrink", "review"}:
        raise ValueError(f"Region track {track_id} overlay_layout fit_strategy is invalid.")
    for field, minimum, maximum in (
        ("source_font_size_ratio", 0.0, 1.0),
        ("resolved_font_size_ratio", 0.0, 1.0),
        ("font_scale", 0.1, 1.0),
        ("line_height", 0.8, 2.0),
        ("padding_x_ratio", 0.0, 0.1),
        ("padding_y_ratio", 0.0, 0.1),
        ("left_extension_ratio", 0.0, 1.0),
        ("right_extension_ratio", 0.0, 1.0),
        ("top_extension_ratio", 0.0, 1.0),
        ("bottom_extension_ratio", 0.0, 1.0),
        ("target_width_ratio", 0.0, 1.0),
        ("target_height_ratio", 0.0, 1.0),
    ):
        value = _number(raw.get(field), f"Region track {track_id} overlay_layout {field}")
        positive_fields = {
            "source_font_size_ratio",
            "resolved_font_size_ratio",
            "target_width_ratio",
            "target_height_ratio",
        }
        if value < minimum or value > maximum or (field in positive_fields and value == 0.0):
            raise ValueError(f"Region track {track_id} overlay_layout {field} is out of range.")
    for field in ("content_fits", "layout_needs_review"):
        if not isinstance(raw.get(field), bool):
            raise ValueError(f"Region track {track_id} overlay_layout {field} must be boolean.")


def _validate_overlay_replacement(raw: Any, track_id: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Region track {track_id} overlay_replacement must be a mapping.")
    if raw.get("schema_version") != 1 or raw.get("algorithm") != "padded-source-box-v1":
        raise ValueError(f"Region track {track_id} overlay_replacement has an unsupported schema.")
    if raw.get("composite_order") != ["erase-source", "draw-translation"]:
        raise ValueError(f"Region track {track_id} overlay_replacement order is invalid.")
    for field in (
        "erase_padding_left_ratio",
        "erase_padding_right_ratio",
        "erase_padding_top_ratio",
        "erase_padding_bottom_ratio",
    ):
        value = _number(raw.get(field), f"Region track {track_id} overlay_replacement {field}")
        if value < 0.0 or value > 0.1:
            raise ValueError(
                f"Region track {track_id} overlay_replacement {field} is out of range."
            )


def _validate_bbox(raw: Any, label: str) -> None:
    bbox = raw
    if not isinstance(bbox, dict):
        raise ValueError(f"{label} requires bbox.")
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} bbox is invalid.") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError(f"{label} bbox is invalid.")
    if min(x, y, width, height) < 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError(f"{label} bbox is invalid.")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError(f"{label} bbox exceeds the image.")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result
