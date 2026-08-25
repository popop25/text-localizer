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
                align-items: center; overflow: hidden; padding: 2px 4px;
                background: rgba(11, 13, 17, .88); color: white; line-height: 1.15;
                border-radius: 2px; text-shadow: 0 1px 1px black; }}
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

    function render(track) {{
      layer.replaceChildren();
      if (!track || !toggle.checked) return;
      for (const region of track.regions) {{
        if (region.action !== "translate" ||
            !region.translated_text || region.needs_review) continue;
        const node = document.createElement("div");
        node.className = "overlay";
        node.textContent = region.translated_text;
        node.style.left = (region.bbox.x * 100) + "%";
        node.style.top = (region.bbox.y * 100) + "%";
        node.style.width = (region.bbox.width * 100) + "%";
        node.style.height = (region.bbox.height * 100) + "%";
        node.style.fontSize = Math.max(10, region.bbox.height * video.clientHeight * .8) + "px";
        layer.appendChild(node);
      }}
    }}

    function update() {{
      const track = findTrack(video.currentTime);
      const nextTrackId = track ? track.track_id : null;
      if (nextTrackId !== activeTrackId) {{
        activeTrackId = nextTrackId;
        render(track);
      }}
      status.textContent = `${{formatTime(video.currentTime)}} · ${{nextTrackId || "no scene"}}`;
    }}

    toggle.addEventListener("change", () => render(findTrack(video.currentTime)));
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    window.addEventListener("resize", () => render(findTrack(video.currentTime)));
    review.textContent = manifest.metrics.needs_review_count
      ? `${{manifest.metrics.needs_review_count}} review flag(s)`
      : "";
    precision.textContent = `Scene boundary estimate: maximum bracket width ` +
      `${{manifest.sampling.transition_precision_seconds.toFixed(3)}} seconds.`;

    video.src = videoSource;
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
