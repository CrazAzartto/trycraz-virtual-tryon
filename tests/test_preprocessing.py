"""Unit tests for virtual_tryon.preprocessing and common.image_utils."""

import pytest
from PIL import Image

from conftest import make_image
from common.image_utils import center_crop_to_aspect, limit_long_side, overlay_mask, pad_to_aspect, side_by_side
from virtual_tryon.preprocessing import prepare_pair


@pytest.mark.parametrize("size", [(600, 800), (1200, 800), (500, 1000), (3000, 4000)])
def test_center_crop_gives_3_to_4(size):
    out = center_crop_to_aspect(make_image(*size, noise=False), 768, 1024)
    assert abs(out.width / out.height - 0.75) < 0.01


def test_center_crop_matches_catvton_arithmetic():
    # CatVTON resize_and_crop on a 1200x800 landscape image keeps h=800, new_w = 800*768//1024 = 600
    out = center_crop_to_aspect(make_image(1200, 800, noise=False), 768, 1024)
    assert out.size == (600, 800)


@pytest.mark.parametrize("size", [(500, 600), (800, 400), (300, 900)])
def test_pad_keeps_whole_garment(size):
    image = make_image(*size, color=(10, 20, 30), noise=False)
    out = pad_to_aspect(image, 768, 1024)
    assert abs(out.width / out.height - 0.75) < 0.01
    assert out.width >= image.width and out.height >= image.height   # nothing cropped
    assert out.getpixel((0, 0)) in [(255, 255, 255), (10, 20, 30)]


def test_limit_long_side_never_upscales():
    assert limit_long_side(make_image(300, 400), 1536).size == (300, 400)
    assert max(limit_long_side(make_image(3000, 4000), 1536).size) == 1536


def test_prepare_pair(person_image, garment_image):
    wide_person = make_image(1600, 900)
    pair = prepare_pair(wide_person, garment_image)
    assert pair.person_was_cropped
    assert pair.person_preview.size == (768, 1024)
    assert abs(pair.garment.width / pair.garment.height - 0.75) < 0.01
    assert not prepare_pair(make_image(768, 1024), garment_image).person_was_cropped


def test_overlay_and_sheet(person_image):
    mask = Image.new("L", person_image.size, 0)
    mask.paste(255, (100, 100, 300, 300))
    tinted = overlay_mask(person_image, mask)
    assert tinted.getpixel((10, 10)) == person_image.getpixel((10, 10))   # unmasked unchanged
    assert tinted.getpixel((200, 200)) != person_image.getpixel((200, 200))
    sheet = side_by_side([person_image, tinted], height=256, gap=4)
    assert sheet.height == 256 and sheet.width == 192 * 2 + 4
