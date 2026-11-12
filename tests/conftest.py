"""Shared test fixtures: synthetic images (no personal photos are stored in the repo)."""

import io

import numpy as np
import pytest
from PIL import Image


def make_image(width=600, height=800, color=(180, 120, 90), mode="RGB", noise=True) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    if noise:  # texture so the blur check sees edges
        arr = np.clip(arr.astype(int) + rng.integers(-40, 40, arr.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    return image.convert(mode) if mode != "RGB" else image


def to_bytes(image: Image.Image, fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def person_image():
    return make_image(600, 800)


@pytest.fixture
def garment_image():
    # garment-like: coloured rectangle on a white background
    image = Image.new("RGB", (500, 600), (255, 255, 255))
    image.paste(make_image(300, 400, (30, 60, 150)), (100, 100))
    return image


@pytest.fixture
def jpeg_bytes(person_image):
    return to_bytes(person_image, "JPEG")
