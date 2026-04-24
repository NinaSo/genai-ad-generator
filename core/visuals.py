from io import BytesIO
import re
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _clean_text(text: str) -> str:
    stripped = re.sub(r"\[\s*end\s*\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", stripped).strip()


def _sanitize_cta(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.!\n\t")

    if not cleaned:
        return "Learn More"

    words = cleaned.split()
    if len(words) > 10:
        cleaned = " ".join(words[:10])

    if len(cleaned) > 58:
        cleaned = cleaned[:58].rsplit(" ", 1)[0].strip()

    return cleaned or "Learn More"


def _mix(c1: Tuple[int, int, int], c2: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
    return tuple(int((1.0 - ratio) * a + ratio * b) for a, b in zip(c1, c2))


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), "Ag", font=font)
    return max(1, box[3] - box[1])


def _load_font(size: int, bold: bool) -> ImageFont.ImageFont:
    preferred = (
        ["Avenir Next Bold.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["Avenir Next.ttc", "Arial.ttf", "DejaVuSans.ttf"]
    )
    for font_name in preferred:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> List[str]:
    words = _clean_text(text).split()
    if not words:
        return []

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) <= max_lines:
        return lines

    clipped = lines[:max_lines]
    tail = clipped[-1]
    while tail:
        candidate = f"{tail}..."
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            clipped[-1] = candidate
            break
        tail = " ".join(tail.split()[:-1])

    if not tail:
        clipped[-1] = "..."
    return clipped


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    max_lines: int,
    start_size: int,
    min_size: int,
    bold: bool,
    gap_scale: float = 1.0,
) -> Tuple[ImageFont.ImageFont, List[str], int]:
    for size in range(start_size, min_size - 1, -2):
        font = _load_font(size, bold=bold)
        lines = _wrap_text(draw, text, font, max_width=max_width, max_lines=max_lines)
        gap = max(8, int(size * 0.18 * gap_scale))
        total_height = len(lines) * (_line_height(draw, font) + gap)
        if total_height <= max_height:
            return font, lines, gap

    font = _load_font(min_size, bold=bold)
    lines = _wrap_text(draw, text, font, max_width=max_width, max_lines=max_lines)
    return font, lines, max(6, int(min_size * 0.15 * gap_scale))


def _layout_config(layout_mode: str) -> Dict[str, float]:
    mode = layout_mode.lower().strip()
    configs = {
        "spacious": {
            "margin_ratio": 0.062,
            "content_pad_ratio": 0.068,
            "chip_size": 26,
            "headline_start": 92,
            "headline_min": 46,
            "headline_lines": 3,
            "sub_start": 46,
            "sub_min": 28,
            "sub_lines": 2,
            "body_start": 38,
            "body_min": 24,
            "body_lines": 4,
            "cta_start": 40,
            "cta_min": 26,
            "tag_start": 24,
            "tag_min": 18,
            "tag_lines": 1,
            "gap_scale": 1.28,
            "section_gap_1": 0.020,
            "section_gap_2": 0.024,
            "section_gap_3": 0.024,
            "tag_zone_ratio": 0.085,
            "cta_zone_ratio": 0.175,
        },
        "balanced": {
            "margin_ratio": 0.055,
            "content_pad_ratio": 0.060,
            "chip_size": 30,
            "headline_start": 102,
            "headline_min": 54,
            "headline_lines": 3,
            "sub_start": 56,
            "sub_min": 32,
            "sub_lines": 3,
            "body_start": 46,
            "body_min": 28,
            "body_lines": 5,
            "cta_start": 50,
            "cta_min": 30,
            "tag_start": 28,
            "tag_min": 20,
            "tag_lines": 2,
            "gap_scale": 1.12,
            "section_gap_1": 0.015,
            "section_gap_2": 0.018,
            "section_gap_3": 0.020,
            "tag_zone_ratio": 0.090,
            "cta_zone_ratio": 0.170,
        },
        "compact": {
            "margin_ratio": 0.045,
            "content_pad_ratio": 0.052,
            "chip_size": 32,
            "headline_start": 112,
            "headline_min": 58,
            "headline_lines": 3,
            "sub_start": 62,
            "sub_min": 34,
            "sub_lines": 3,
            "body_start": 52,
            "body_min": 30,
            "body_lines": 6,
            "cta_start": 54,
            "cta_min": 32,
            "tag_start": 30,
            "tag_min": 22,
            "tag_lines": 2,
            "gap_scale": 1.0,
            "section_gap_1": 0.012,
            "section_gap_2": 0.014,
            "section_gap_3": 0.016,
            "tag_zone_ratio": 0.095,
            "cta_zone_ratio": 0.165,
        },
    }
    return configs.get(mode, configs["spacious"])


def _theme_style(theme: str, tone: str) -> Dict[str, str]:
    text = f"{theme} {tone}".lower()

    if any(k in text for k in ["eco", "green", "nature", "sustain", "earth", "organic"]):
        return {
            "bg_top": "#0d2b1f",
            "bg_bottom": "#2f5f3f",
            "card_bg": "#10281d",
            "card_outline": "#8ed1a4",
            "accent": "#8fe388",
            "secondary": "#2ec4b6",
            "title": "#f3fff5",
            "subtitle": "#d6f7dc",
            "body": "#c6e8cf",
            "tag": "#a1d9b1",
            "accent_text": "#09230f",
            "motif": "eco",
        }
    if any(k in text for k in ["luxury", "premium", "elegant", "high-end", "exclusive"]):
        return {
            "bg_top": "#1e1a16",
            "bg_bottom": "#3d2d1f",
            "card_bg": "#231d16",
            "card_outline": "#f4c272",
            "accent": "#f2bc65",
            "secondary": "#c28a38",
            "title": "#fff8ec",
            "subtitle": "#f5dfb7",
            "body": "#e8d2ab",
            "tag": "#d8be90",
            "accent_text": "#231404",
            "motif": "luxury",
        }
    if any(k in text for k in ["tech", "digital", "ai", "cyber", "app", "software", "saas", "futur"]):
        return {
            "bg_top": "#07162b",
            "bg_bottom": "#12395f",
            "card_bg": "#0a2038",
            "card_outline": "#54b8ff",
            "accent": "#23d2ff",
            "secondary": "#4d73ff",
            "title": "#ebf7ff",
            "subtitle": "#cfeaff",
            "body": "#b7dbfb",
            "tag": "#8dc9f4",
            "accent_text": "#03172a",
            "motif": "tech",
        }
    if any(k in text for k in ["sport", "fitness", "energy", "performance", "active"]):
        return {
            "bg_top": "#2b0d12",
            "bg_bottom": "#7c1a2c",
            "card_bg": "#351019",
            "card_outline": "#ffb17a",
            "accent": "#ff7a45",
            "secondary": "#ff4d6d",
            "title": "#fff3f0",
            "subtitle": "#ffd7c8",
            "body": "#f8bfaa",
            "tag": "#f4a894",
            "accent_text": "#2f0f08",
            "motif": "energy",
        }
    if any(k in text for k in ["beauty", "wellness", "fashion", "lifestyle", "self-care"]):
        return {
            "bg_top": "#2b1536",
            "bg_bottom": "#6d336b",
            "card_bg": "#311c41",
            "card_outline": "#ff9ec8",
            "accent": "#ff7eb6",
            "secondary": "#ffb176",
            "title": "#fff1f8",
            "subtitle": "#ffd8ea",
            "body": "#efbfd6",
            "tag": "#e09fc0",
            "accent_text": "#34111f",
            "motif": "lifestyle",
        }

    return {
        "bg_top": "#101a34",
        "bg_bottom": "#264274",
        "card_bg": "#132240",
        "card_outline": "#83b8ff",
        "accent": "#ffd166",
        "secondary": "#70d6ff",
        "title": "#f4f8ff",
        "subtitle": "#d6e6ff",
        "body": "#c4d8f5",
        "tag": "#a7c0e8",
        "accent_text": "#1f1402",
        "motif": "clean",
    }


def _draw_background(width: int, height: int, style: Dict[str, str]) -> Image.Image:
    img = Image.new("RGB", (width, height), style["bg_top"])
    draw = ImageDraw.Draw(img)
    top = _hex_to_rgb(style["bg_top"])
    bottom = _hex_to_rgb(style["bg_bottom"])

    for y in range(height):
        ratio = y / max(1, height - 1)
        draw.line([(0, y), (width, y)], fill=_mix(top, bottom, ratio))

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    accent = _hex_to_rgb(style["accent"])
    secondary = _hex_to_rgb(style["secondary"])
    motif = style["motif"]

    if motif == "eco":
        odraw.ellipse([width - 320, -140, width + 120, 260], fill=(*accent, 55))
        odraw.ellipse([-180, height - 420, 260, height + 120], fill=(*secondary, 55))
        odraw.polygon([(80, height - 120), (280, height - 350), (420, height - 80)], fill=(*accent, 35))
    elif motif == "tech":
        odraw.rectangle([width - 280, -40, width + 40, 280], fill=(*secondary, 40))
        odraw.polygon([(0, 220), (260, 120), (340, 300), (0, 430)], fill=(*accent, 35))
        odraw.rectangle([width - 450, height - 330, width + 10, height + 20], fill=(*accent, 35))
    elif motif == "luxury":
        odraw.ellipse([width - 250, -80, width + 100, 280], fill=(*accent, 45))
        odraw.ellipse([-180, height - 280, 220, height + 140], fill=(*secondary, 35))
        odraw.rectangle([0, height - 150, width, height - 90], fill=(*accent, 30))
    elif motif == "energy":
        odraw.polygon([(0, 200), (320, 120), (190, 420)], fill=(*accent, 40))
        odraw.polygon([(width, 150), (width - 260, 260), (width, 420)], fill=(*secondary, 45))
        odraw.rectangle([width - 520, height - 260, width + 20, height + 20], fill=(*accent, 35))
    elif motif == "lifestyle":
        odraw.ellipse([width - 330, -100, width + 110, 290], fill=(*accent, 50))
        odraw.ellipse([-190, height - 320, 230, height + 100], fill=(*secondary, 45))
        odraw.polygon([(120, 0), (320, 140), (40, 260)], fill=(*secondary, 30))
    else:
        odraw.ellipse([width - 280, -100, width + 120, 260], fill=(*secondary, 45))
        odraw.ellipse([-150, height - 260, 220, height + 120], fill=(*accent, 35))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def render_png_ad(
    sections: Dict[str, str],
    size: Tuple[int, int] = (1080, 1350),
    theme: str = "",
    tone: str = "friendly",
    layout_mode: str = "spacious",
    font_scale: float = 1.0,
) -> Image.Image:
    width, height = size
    style = _theme_style(theme=theme, tone=tone)
    layout = _layout_config(layout_mode=layout_mode)
    img = _draw_background(width=width, height=height, style=style)

    scale = min(width, height) / 1080.0
    font_mult = max(0.75, min(1.45, font_scale)) * scale

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    card_margin = int(min(width, height) * layout["margin_ratio"])
    card = [card_margin, card_margin, width - card_margin, height - card_margin]
    card_bg = _hex_to_rgb(style["card_bg"])
    card_outline = _hex_to_rgb(style["card_outline"])
    odraw.rounded_rectangle(card, radius=int(44 * scale), fill=(*card_bg, 216), outline=(*card_outline, 165), width=max(2, int(3 * scale)))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    headline = _clean_text(sections.get("headline", ""))
    subheadline = _clean_text(sections.get("subheadline", ""))
    body = _clean_text(sections.get("body", ""))
    cta = _sanitize_cta(sections.get("cta", ""))
    hashtags = _clean_text(sections.get("hashtags", ""))

    content_pad = int(min(width, height) * layout["content_pad_ratio"])
    content_x = card_margin + content_pad
    content_w = width - (content_x * 2)
    y = card_margin + int(54 * scale)

    chip_text = _clean_text(theme) if _clean_text(theme) else "Campaign"
    chip_font = _load_font(max(16, int(layout["chip_size"] * font_mult)), bold=True)
    chip_box = draw.textbbox((0, 0), chip_text, font=chip_font)
    chip_w = chip_box[2] + int(52 * scale)
    chip_h = chip_box[3] + int(24 * scale)
    accent_rgb = _hex_to_rgb(style["accent"])
    draw.rounded_rectangle([content_x, y, content_x + chip_w, y + chip_h], radius=chip_h // 2, fill=style["accent"])
    draw.text((content_x + int(26 * scale), y + int(10 * scale)), chip_text, font=chip_font, fill=style["accent_text"])
    y += chip_h + int(height * layout["section_gap_1"])

    headline_font, headline_lines, headline_gap = _fit_text(
        draw,
        headline,
        max_width=content_w,
        max_height=int(height * 0.24),
        max_lines=int(layout["headline_lines"]),
        start_size=max(32, int(layout["headline_start"] * font_mult)),
        min_size=max(24, int(layout["headline_min"] * font_mult)),
        bold=True,
        gap_scale=layout["gap_scale"],
    )
    for line in headline_lines:
        draw.text((content_x, y), line, font=headline_font, fill=style["title"])
        y += _line_height(draw, headline_font) + headline_gap
    y += int(height * layout["section_gap_2"])

    sub_font, sub_lines, sub_gap = _fit_text(
        draw,
        subheadline,
        max_width=content_w,
        max_height=int(height * 0.14),
        max_lines=int(layout["sub_lines"]),
        start_size=max(26, int(layout["sub_start"] * font_mult)),
        min_size=max(20, int(layout["sub_min"] * font_mult)),
        bold=False,
        gap_scale=layout["gap_scale"],
    )
    for line in sub_lines:
        draw.text((content_x, y), line, font=sub_font, fill=style["subtitle"])
        y += _line_height(draw, sub_font) + sub_gap
    y += int(height * layout["section_gap_3"])

    tag_zone = int(height * layout["tag_zone_ratio"])
    cta_zone = int(height * layout["cta_zone_ratio"])
    cta_top = height - card_margin - tag_zone - cta_zone
    body_max_height = max(int(86 * scale), cta_top - y - int(20 * scale))

    body_font, body_lines, body_gap = _fit_text(
        draw,
        body,
        max_width=content_w,
        max_height=body_max_height,
        max_lines=int(layout["body_lines"]),
        start_size=max(22, int(layout["body_start"] * font_mult)),
        min_size=max(18, int(layout["body_min"] * font_mult)),
        bold=False,
        gap_scale=layout["gap_scale"],
    )
    for line in body_lines:
        draw.text((content_x, y), line, font=body_font, fill=style["body"])
        y += _line_height(draw, body_font) + body_gap

    cta_font, cta_lines, cta_gap = _fit_text(
        draw,
        cta,
        max_width=content_w - int(120 * scale),
        max_height=int(cta_zone * 0.62),
        max_lines=2,
        start_size=max(24, int(layout["cta_start"] * font_mult)),
        min_size=max(18, int(layout["cta_min"] * font_mult)),
        bold=True,
        gap_scale=1.0,
    )
    if not cta_lines:
        cta_lines = ["Learn More"]

    line_h = _line_height(draw, cta_font)
    cta_line_gap = max(int(6 * scale), cta_gap // 2)
    cta_text_h = len(cta_lines) * line_h + max(0, len(cta_lines) - 1) * cta_line_gap
    cta_text_w = max(draw.textbbox((0, 0), line, font=cta_font)[2] for line in cta_lines)

    btn_h = max(int(88 * scale), int(cta_zone * 0.56), cta_text_h + int(42 * scale))
    btn_w = min(content_w, max(int(360 * scale), cta_text_w + int(120 * scale)))
    btn_x = content_x + (content_w - btn_w) // 2
    btn_y = cta_top + int((cta_zone - btn_h) * 0.42)
    draw.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], radius=max(24, int(32 * scale)), fill=style["accent"])

    text_y = btn_y + (btn_h - cta_text_h) // 2
    for line in cta_lines:
        line_w = draw.textbbox((0, 0), line, font=cta_font)[2]
        draw.text(
            (btn_x + (btn_w - line_w) // 2, text_y),
            line,
            font=cta_font,
            fill=style["accent_text"],
        )
        text_y += line_h + cta_line_gap

    tag_font = _load_font(max(16, int(layout["tag_start"] * font_mult)), bold=False)
    tag_lines = _wrap_text(draw, hashtags, tag_font, max_width=content_w, max_lines=int(layout["tag_lines"]))
    tag_y = height - card_margin - tag_zone + int(18 * scale)
    for index, line in enumerate(tag_lines):
        draw.text(
            (content_x, tag_y + index * (_line_height(draw, tag_font) + int(8 * scale))),
            line,
            font=tag_font,
            fill=style["tag"],
        )

    draw.ellipse(
        [content_x + content_w - int(26 * scale), card_margin + int(36 * scale), content_x + content_w, card_margin + int(62 * scale)],
        fill=accent_rgb,
    )
    return img


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
