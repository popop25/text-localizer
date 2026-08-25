from __future__ import annotations

import json
from pathlib import Path

from text_localizer.domain import RegionAction, TranslatedRegion


def write_viewer(
    output_path: Path,
    image_file_name: str,
    regions: tuple[TranslatedRegion, ...],
) -> None:
    overlays = [
        {
            "x": item.region.bbox.x,
            "y": item.region.bbox.y,
            "width": item.region.bbox.width,
            "height": item.region.bbox.height,
            "text": item.translated_text,
        }
        for item in regions
        if item.region.action is RegionAction.TRANSLATE and item.translated_text
    ]
    overlay_json = json.dumps(overlays, ensure_ascii=False).replace("<", "\\u003c")
    image_json = json.dumps(image_file_name).replace("<", "\\u003c")
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>text-localizer static preview</title>
  <style>
    body {{ margin: 0; background: #111; color: white; font-family: "Yu Gothic", sans-serif; }}
    .stage {{ position: relative; display: inline-block; max-width: 100vw; }}
    .stage img {{ display: block; max-width: 100vw; height: auto; }}
    .overlay {{ position: absolute; box-sizing: border-box; background: rgba(18,18,18,.86);
                color: white; overflow: hidden; padding: 2px; display: flex;
                align-items: center; line-height: 1.15; }}
  </style>
</head>
<body>
  <main class="stage" id="stage"><img id="source" alt="source frame"></main>
  <script>
    const imageFile = {image_json};
    const overlays = {overlay_json};
    const image = document.getElementById("source");
    const stage = document.getElementById("stage");
    image.addEventListener("load", () => {{ for (const item of overlays) {{
      const node = document.createElement("div");
      node.className = "overlay";
      node.textContent = item.text;
      node.style.left = (item.x * 100) + "%";
      node.style.top = (item.y * 100) + "%";
      node.style.width = (item.width * 100) + "%";
      node.style.height = (item.height * 100) + "%";
      node.style.fontSize = Math.max(10, item.height * image.clientHeight * 0.8) + "px";
      stage.appendChild(node);
    }} }}, {{ once: true }});
    image.src = imageFile;
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
