"""Small, dependency-light image helpers shared by the try-on and recommendation modules."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps


def to_rgb(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Convert any PIL mode to RGB. Transparent pixels become `background` (white by default),
    which matches the white padding CatVTON uses for garment images."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Phone photos store rotation in EXIF instead of rotating pixels. Without this, a portrait
    photo can reach the model sideways."""
    return ImageOps.exif_transpose(image) or image


def center_crop_to_aspect(image: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop to the width:height aspect ratio (same arithmetic as CatVTON's resize_and_crop,
    without the resize)."""
    w, h = image.size
    if w / h < width / height:
        new_w, new_h = w, w * height // width
    else:
        new_w, new_h = h * width // height, h
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def pad_to_aspect(image: Image.Image, width: int, height: int, fill=(255, 255, 255)) -> Image.Image:
    """Fit inside a width:height canvas without cropping, padding with `fill`
    (same idea as CatVTON's resize_and_padding)."""
    w, h = image.size
    if w / h < width / height:
        canvas_w, canvas_h = h * width // height, h
    else:
        canvas_w, canvas_h = w, w * height // width
    canvas = Image.new("RGB", (canvas_w, canvas_h), fill)
    canvas.paste(image, ((canvas_w - w) // 2, (canvas_h - h) // 2))
    return canvas


def limit_long_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale (never upscale) so the longest side is at most max_side."""
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def image_to_bytes(image: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


def b64_png_to_image(data: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    image.load()
    return image


def side_by_side(images: list[Image.Image], height: int = 1024, gap: int = 8, background=(255, 255, 255)) -> Image.Image:
    """Concatenate images horizontally at a common height (for comparison sheets and the report)."""
    resized = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS) for im in images]
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    sheet = Image.new("RGB", (total_w, height), background)
    x = 0
    for im in resized:
        sheet.paste(to_rgb(im), (x, 0))
        x += im.width + gap
    return sheet


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 64, 64), alpha: float = 0.45) -> Image.Image:
    """Tint the masked region so users can see which area the model repainted."""
    base = np.asarray(to_rgb(image), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(image.size), dtype=np.float32)[..., None] / 255.0
    tinted = base * (1 - alpha * m) + np.array(color, dtype=np.float32) * alpha * m
    return Image.fromarray(tinted.clip(0, 255).astype(np.uint8))
