"""Compose the RescueAgent demo MP4 from title cards + captured frames.

Run after frames exist in docs/demo-frames/:

    python scripts/make_demo_video.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "docs" / "demo-frames"
OUT = ROOT / "docs" / "rescueagent-demo.mp4"
POSTER = ROOT / "docs" / "demo-poster.png"
# Cursor-generated title cards land here when the image tool is used.
CURSOR_ASSETS = Path(
    r"C:\Users\bhart\.cursor\projects"
    r"\c-Users-bhart-Downloads-RescueAgent-main-RescueAgent-main-rescueagent"
    r"\assets"
)
W, H = 1280, 720
BRAND = (255, 122, 47)
BG = (14, 16, 19)
TEXT = (233, 235, 238)
MUTED = (155, 163, 174)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for n in names:
        if os.path.exists(n):
            try:
                return ImageFont.truetype(n, size)
            except OSError:
                continue
    return ImageFont.load_default()


def card(lines: list[tuple[str, int, tuple, bool]], hold: float = 3.5):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 8, H), fill=BRAND)
    y = 160
    for text, size, colour, bold in lines:
        font = _font(size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, y), text, font=font, fill=colour)
        y += int(size * 1.35)
    return img, hold


def fit_frame(path: Path, caption: str = "") -> "Image.Image":
    from PIL import Image, ImageDraw

    src = Image.open(path).convert("RGB")
    src.thumbnail((W, H - (72 if caption else 0)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), BG)
    x = (W - src.width) // 2
    y = (H - src.height - (48 if caption else 0)) // 2
    canvas.paste(src, (x, y))
    if caption:
        draw = ImageDraw.Draw(canvas)
        font = _font(22, bold=False)
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, H - 42), caption, font=font, fill=MUTED)
    return canvas


def title_cards():
    return [
        card([
            ("RescueAgent", 72, TEXT, True),
            ("Edinburgh Food Rescue", 36, BRAND, False),
            ("Strands Agents SDK  ·  Amazon Bedrock Qwen 3 235B", 24, MUTED, False),
        ], 4.0),
        card([
            ("The problem", 28, BRAND, True),
            ("Edible food is binned at closing time", 36, TEXT, True),
            ("while a shelter a mile away runs short.", 36, TEXT, True),
            ("The blocker is coordination, not goodwill.", 24, MUTED, False),
        ], 4.5),
        card([
            ("How it works", 28, BRAND, True),
            ("A kitchen types one sentence.", 32, TEXT, True),
            ("Strands classifies it against UK FSA rules,", 28, TEXT, False),
            ("matches a shelter, and offers the job", 28, TEXT, False),
            ("to the three nearest capable volunteers.", 28, TEXT, False),
        ], 5.0),
    ]


def end_card():
    return card([
        ("RescueAgent", 56, TEXT, True),
        ("Strands Agents  ·  AWS Bedrock Qwen 3 235B", 26, BRAND, False),
        ("github.com/Bharth2003/RescueAgent", 24, MUTED, False),
    ], 4.0)


def collect_generated_slides():
    named = [
        (CURSOR_ASSETS / "demo-poster.png", 4.2),
        (CURSOR_ASSETS / "demo-problem.png", 4.6),
        (CURSOR_ASSETS / "demo-how.png", 5.2),
    ]
    out = []
    for path, hold in named:
        if path.exists():
            out.append((fit_frame(path), hold))
    return out


def collect_app_frames():
    if not FRAMES.exists():
        return []
    items = []
    captions = {
        "01": "Login — Strands Agents on Amazon Bedrock",
        "02": "Live console — manager and driver, same map",
        "03": "Kitchen + surplus food",
        "04": "Agent classifies, matches, broadcasts",
        "05": "Driver accepts the offer",
        "06": "Live tracking along real Edinburgh roads",
        "07": "Handover and delivery complete",
    }
    for p in sorted(FRAMES.glob("*.png")):
        key = p.stem.split("-")[0]
        items.append((fit_frame(p, captions.get(key, "")), 3.2))
    return items


def write_mp4(clips):
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError:
        sys.exit("Install: pip install pillow imageio imageio-ffmpeg numpy")

    fps = 8
    writer = imageio.get_writer(
        OUT, fps=fps, codec="libx264", quality=7,
        pixelformat="yuv420p", macro_block_size=None,
    )
    try:
        for img, hold in clips:
            frame = np.asarray(img.convert("RGB"))
            n = max(1, int(hold * fps))
            for _ in range(n):
                writer.append_data(frame)
    finally:
        writer.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


def main():
    from PIL import Image  # noqa: F401 — fail fast if Pillow is missing

    FRAMES.mkdir(parents=True, exist_ok=True)
    clips = collect_generated_slides() or title_cards()
    clips += collect_app_frames()
    end = CURSOR_ASSETS / "demo-end.png"
    clips.append((fit_frame(end), 4.2) if end.exists() else end_card())
    first = clips[0][0]
    first.save(POSTER)
    print(f"wrote {POSTER}")
    write_mp4(clips)


if __name__ == "__main__":
    main()
