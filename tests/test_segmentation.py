"""Tests for garment background removal (virtual_tryon/garment_segmentation.py)."""

import builtins

import numpy as np
import pytest
from PIL import Image, ImageDraw

from virtual_tryon import garment_segmentation as seg


def garment_on_clutter() -> Image.Image:
    """A dark T-shirt shape on a striped, coloured 'bedsheet'."""
    image = Image.new("RGB", (400, 500))
    draw = ImageDraw.Draw(image)
    for y in range(0, 500, 20):
        draw.rectangle([0, y, 400, y + 10], fill=(200, 40, 40))
        draw.rectangle([0, y + 10, 400, y + 20], fill=(230, 200, 60))
    draw.polygon([(120, 80), (280, 80), (360, 160), (310, 200), (290, 170), (290, 440), (110, 440), (110, 170),
                  (90, 200), (40, 160)], fill=(25, 30, 60))
    return image


def test_falls_back_when_rembg_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rembg":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    image = garment_on_clutter()
    result = seg.remove_background(image)
    assert not result.applied and result.image.size == image.size and "unavailable" in result.message


@pytest.mark.slow
def test_removes_cluttered_background():
    """Runs the real U^2-Net model (downloads ~176 MB on first use)."""
    pytest.importorskip("rembg")
    result = seg.remove_background(garment_on_clutter())
    if not result.applied and "unavailable" in result.message:
        pytest.skip(result.message)      # e.g. no internet to download the model
    assert result.applied and 0.2 < result.foreground_fraction < 0.8
    arr = np.asarray(result.image)
    corners = np.concatenate([arr[:5, :5].reshape(-1, 3), arr[-5:, -5:].reshape(-1, 3)])
    assert corners.mean() > 240          # background is now white
    centre = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    assert centre.mean() < 80            # garment kept
