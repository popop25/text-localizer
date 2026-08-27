from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_video_viewer(
    output_path: Path,
    video_path: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    video_json = json.dumps(video_path.resolve().as_uri()).replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>text-localizer timeline preview</title>
  <style>
    :root {{ color-scheme: dark; font-family: "Yu Gothic", "Meiryo", sans-serif; }}
    body {{ margin: 0; background: #0d0f12; color: #f5f7fa; }}
    main {{ width: min(1280px, 100vw); margin: 0 auto; }}
    .toolbar {{ display: flex; gap: 1rem; align-items: center; padding: .75rem 1rem;
                background: #171a20; position: sticky; top: 0; z-index: 10; }}
    .toolbar label {{ display: flex; gap: .45rem; align-items: center; }}
    .status {{ margin-left: auto; color: #b8c0cc; font-variant-numeric: tabular-nums; }}
    .stage {{ position: relative; width: 100%; background: black; overflow: hidden; }}
    video {{ display: block; width: 100%; height: auto; }}
    .overlay-layer {{ position: absolute; inset: 0; pointer-events: none; }}
    .overlay {{ position: absolute; box-sizing: border-box; display: flex;
                align-items: center; overflow: hidden; padding: 2px 4px; white-space: pre;
                background: rgba(11, 13, 17, .88); color: white; line-height: 1.15;
                border-radius: 2px; text-shadow: 0 1px 1px black; }}
    .overlay.layout-review {{ outline: 2px solid #ffcb6b; overflow: visible; }}
    .notice {{ padding: .7rem 1rem; color: #aeb7c4; font-size: .9rem; }}
    .warning {{ color: #ffcb6b; }}
  </style>
</head>
<body>
  <main>
    <div class="toolbar">
      <label><input id="toggle" type="checkbox" checked> Japanese overlay</label>
      <span id="review" class="warning"></span>
      <span id="status" class="status">00:00.000 · no scene</span>
    </div>
    <div class="stage" id="stage">
      <video id="video" controls preload="metadata"></video>
      <div id="overlays" class="overlay-layer" aria-live="off"></div>
    </div>
    <p id="precision" class="notice"></p>
  </main>
  <script>
    const manifest = {manifest_json};
    const videoSource = {video_json};
    const video = document.getElementById("video");
    const layer = document.getElementById("overlays");
    const toggle = document.getElementById("toggle");
    const status = document.getElementById("status");
    const precision = document.getElementById("precision");
    const review = document.getElementById("review");
    let activeTrackId = null;
    let fallbackFrameRequest = null;
    let videoFrameCallbackActive = false;

    function formatTime(value) {{
      const minutes = Math.floor(value / 60).toString().padStart(2, "0");
      const seconds = (value % 60).toFixed(3).padStart(6, "0");
      return `${{minutes}}:${{seconds}}`;
    }}

    function findTrack(time) {{
      return manifest.tracks.find((item, index) =>
        time >= item.start_seconds &&
        (time < item.end_seconds ||
          (index === manifest.tracks.length - 1 && time <= item.end_seconds))
      );
    }}

    function interpolateBox(left, right, fraction) {{
      const value = key => left[key] + (right[key] - left[key]) * fraction;
      return {{ x: value("x"), y: value("y"), width: value("width"),
                height: value("height") }};
    }}

    function regionTrackBox(track, time) {{
      const keyframes = track.keyframes;
      const followingIndex = keyframes.findIndex(item => item.timestamp_seconds > time);
      let index = followingIndex < 0 ? keyframes.length - 1 : followingIndex - 1;
      if (index < 0) index = 0;
      const current = keyframes[index];
      const following = keyframes[index + 1];
      if (!following || current.interpolation !== "linear") return current.bbox;
      const span = following.timestamp_seconds - current.timestamp_seconds;
      if (span <= 0) return current.bbox;
      const fraction = Math.min(1, Math.max(0,
        (time - current.timestamp_seconds) / span));
      return interpolateBox(current.bbox, following.bbox, fraction);
    }}

    function activeRegions(time) {{
      if (manifest.schema_version === 2) {{
        return manifest.region_tracks
          .filter(track => time >= track.start_seconds && time < track.end_seconds)
          .map(track => ({{ id: track.region_track_id, region: track.region,
                           bbox: regionTrackBox(track, time),
                           style: track.overlay_style || null,
                           layout: track.overlay_layout || null }}));
      }}
      const track = findTrack(time);
      return track ? track.regions.map((region, index) => ({{
        id: `${{track.track_id}}-${{index}}`, region, bbox: region.bbox, style: null
      }})) : [];
    }}

    function hexToRgba(hex, opacity) {{
      const value = Number.parseInt(hex.slice(1), 16);
      const red = (value >> 16) & 255;
      const green = (value >> 8) & 255;
      const blue = value & 255;
      return `rgba(${{red}}, ${{green}}, ${{blue}}, ${{opacity}})`;
    }}

    function applyOverlayStyle(node, style, bbox) {{
      if (!style) {{
        node.style.fontSize = Math.max(10, bbox.height * video.clientHeight * .8) + "px";
        return;
      }}
      node.style.color = style.foreground_color;
      node.style.backgroundColor = hexToRgba(
        style.background_color, style.background_opacity);
      node.style.fontSize = Math.max(10, style.font_size_ratio * video.clientHeight) + "px";
      node.style.fontWeight = String(style.font_weight);
      node.style.lineHeight = String(style.line_height);
      node.style.borderRadius = style.border_radius_px + "px";
      node.style.textShadow = style.text_shadow;
      node.dataset.styleConfidence = style.confidence.toFixed(3);
    }}

    function clamp(value, minimum, maximum) {{
      return Math.min(maximum, Math.max(minimum, value));
    }}

    function applyOverlayLayout(node, layout, bbox) {{
      const width = Math.min(1, bbox.width + layout.left_extension_ratio +
        layout.right_extension_ratio);
      const height = Math.min(1, bbox.height + layout.top_extension_ratio +
        layout.bottom_extension_ratio);
      const x = clamp(bbox.x - layout.left_extension_ratio, 0, Math.max(0, 1 - width));
      const y = clamp(bbox.y - layout.top_extension_ratio, 0, Math.max(0, 1 - height));
      node.style.fontSize = Math.max(
        10, layout.resolved_font_size_ratio * video.clientHeight) + "px";
      node.style.lineHeight = String(layout.line_height);
      node.style.padding = `${{Math.max(1, layout.padding_y_ratio * video.clientHeight)}}px ` +
        `${{Math.max(2, layout.padding_x_ratio * video.clientWidth)}}px`;
      node.dataset.layoutStrategy = layout.fit_strategy;
      node.dataset.layoutLines = String(layout.line_count);
      return {{ x, y, width, height }};
    }}

    function overlayFits(node) {{
      return node.scrollWidth <= node.clientWidth + 1 &&
        node.scrollHeight <= node.clientHeight + 1;
    }}

    function fitOverlayText(node, minimumSize) {{
      const preferred = Number.parseFloat(node.style.fontSize);
      if (!Number.isFinite(preferred)) return false;
      if (overlayFits(node)) return true;
      const minimum = Math.min(preferred, Math.max(10, minimumSize));
      node.style.fontSize = minimum + "px";
      if (!overlayFits(node)) return false;
      let low = minimum;
      let high = preferred;
      for (let iteration = 0; iteration < 9; iteration += 1) {{
        const candidate = (low + high) / 2;
        node.style.fontSize = candidate + "px";
        if (overlayFits(node)) low = candidate;
        else high = candidate;
      }}
      node.style.fontSize = low + "px";
      node.dataset.runtimeFit = "binary-shrink";
      return true;
    }}

    function updateReviewWarning(layoutReviewCount) {{
      const messages = [];
      if (manifest.metrics.needs_review_count)
        messages.push(`${{manifest.metrics.needs_review_count}} translation review`);
      if (layoutReviewCount)
        messages.push(`${{layoutReviewCount}} layout review`);
      review.textContent = messages.join(" · ");
    }}

    function render(time) {{
      layer.replaceChildren();
      if (!toggle.checked) {{
        updateReviewWarning(0);
        return;
      }}
      let layoutReviewCount = 0;
      for (const item of activeRegions(time)) {{
        const region = item.region;
        const bbox = item.bbox;
        if (region.action !== "translate" ||
            !region.translated_text || region.needs_review) continue;
        const node = document.createElement("div");
        node.className = "overlay";
        const normalizedText = region.translated_text.trim();
        const layout = item.layout && item.layout.text === normalizedText
          ? item.layout : null;
        node.textContent = layout ? layout.lines.join("\n") : normalizedText;
        applyOverlayStyle(node, item.style, bbox);
        const resolvedBox = layout ? applyOverlayLayout(node, layout, bbox) : bbox;
        node.style.left = (resolvedBox.x * 100) + "%";
        node.style.top = (resolvedBox.y * 100) + "%";
        node.style.width = (resolvedBox.width * 100) + "%";
        node.style.height = (resolvedBox.height * 100) + "%";
        layer.appendChild(node);
        const preferred = Number.parseFloat(node.style.fontSize);
        const sourceSize = layout
          ? layout.source_font_size_ratio * video.clientHeight : preferred;
        const fits = fitOverlayText(node, Math.max(10, sourceSize * .8));
        if (!fits || (layout && layout.layout_needs_review)) {{
          node.classList.add("layout-review");
          layoutReviewCount += 1;
        }}
      }}
      updateReviewWarning(layoutReviewCount);
    }}

    function update(mediaTime = video.currentTime) {{
      const track = findTrack(mediaTime);
      const nextTrackId = track ? track.track_id : null;
      if (manifest.schema_version === 2 || nextTrackId !== activeTrackId) {{
        activeTrackId = nextTrackId;
        render(mediaTime);
      }}
      const label = manifest.schema_version === 2
        ? `${{activeRegions(mediaTime).length}} region track(s)`
        : (nextTrackId || "no scene");
      status.textContent = `${{formatTime(mediaTime)}} · ${{label}}`;
    }}

    function startVideoFrameClock() {{
      if (videoFrameCallbackActive ||
          typeof video.requestVideoFrameCallback !== "function") return;
      videoFrameCallbackActive = true;
      const onVideoFrame = (_now, metadata) => {{
        update(metadata.mediaTime);
        video.requestVideoFrameCallback(onVideoFrame);
      }};
      video.requestVideoFrameCallback(onVideoFrame);
    }}

    function startFallbackFrameClock() {{
      if (typeof video.requestVideoFrameCallback === "function" ||
          fallbackFrameRequest !== null) return;
      const onAnimationFrame = () => {{
        update(video.currentTime);
        if (video.paused || video.ended) {{
          fallbackFrameRequest = null;
          return;
        }}
        fallbackFrameRequest = requestAnimationFrame(onAnimationFrame);
      }};
      fallbackFrameRequest = requestAnimationFrame(onAnimationFrame);
    }}

    toggle.addEventListener("change", () => render(video.currentTime));
    video.addEventListener("loadedmetadata", () => {{
      update();
      startVideoFrameClock();
    }});
    video.addEventListener("play", startFallbackFrameClock);
    video.addEventListener("timeupdate", () => {{
      if (typeof video.requestVideoFrameCallback !== "function") update();
    }});
    video.addEventListener("seeked", () => update());
    video.addEventListener("pause", () => update());
    window.addEventListener("resize", () => render(video.currentTime));
    updateReviewWarning(0);
    precision.textContent = manifest.schema_version === 2
      ? `Region geometry uses ${{manifest.region_tracking.algorithm}} keyframes; ` +
        (manifest.region_styling
          ? `overlay palette uses ${{manifest.region_styling.algorithm}}; `
          : "overlay palette uses the fixed fallback; ") +
        (manifest.region_layout
          ? `text layout uses ${{manifest.region_layout.algorithm}}; `
          : "text layout uses runtime box fitting; ") +
        `base scene bracket width is ${{manifest.sampling.transition_precision_seconds
          .toFixed(3)}} seconds.`
      : `Scene boundary estimate: maximum bracket width ` +
        `${{manifest.sampling.transition_precision_seconds.toFixed(3)}} seconds.`;

    video.src = videoSource;
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
