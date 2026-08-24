from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from text_localizer.domain import RegionAction, TranslatedRegion

_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
)


def find_japanese_font() -> Path:
    font = next((path for path in _FONT_CANDIDATES if path.is_file()), None)
    if font is None:
        raise RuntimeError("A Japanese-capable font was not found. Install Yu Gothic or Meiryo.")
    return font


def render_preview(
    image_path: Path,
    regions: tuple[TranslatedRegion, ...],
    output_path: Path,
) -> None:
    font_path = find_japanese_font()
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for translated in regions:
        region = translated.region
        if region.action is not RegionAction.TRANSLATE or not translated.translated_text:
            continue
        box = region.bbox
        left = round(box.x * width)
        top = round(box.y * height)
        right = round((box.x + box.width) * width)
        bottom = round((box.y + box.height) * height)
        draw.rectangle((left, top, right, bottom), fill=(18, 18, 18, 220))
        font = _fit_font(
            draw,
            translated.translated_text,
            font_path,
            max(1, right - left),
            max(1, bottom - top),
        )
        draw.text(
            (left + 2, top),
            translated.translated_text,
            font=font,
            fill=(255, 255, 255, 255),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    box_width: int,
    box_height: int,
) -> ImageFont.FreeTypeFont:
    start_size = max(10, min(72, int(box_height * 0.9)))
    for size in range(start_size, 9, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= box_width - 4 and bottom - top <= box_height:
            return font
    return ImageFont.truetype(str(font_path), size=10)
