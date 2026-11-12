"""Unit tests for common.validation (error handling required by brief section 4.3)."""

import io

import pytest
from PIL import Image

from conftest import make_image, to_bytes
from common.validation import (
    ImageValidationError, validate_garment_type, validate_generation_params, validate_image_bytes,
)


def code_of(callable_, *args):
    with pytest.raises(ImageValidationError) as err:
        callable_(*args)
    return err.value.code


def test_valid_jpeg_passes(jpeg_bytes):
    out = validate_image_bytes(jpeg_bytes, "me.jpg", "person")
    assert out.image.mode == "RGB" and out.image.size == (600, 800) and out.original_format == "JPEG"


@pytest.mark.parametrize("data", [None, b""])
def test_missing_image(data):
    assert code_of(validate_image_bytes, data, "x.jpg", "person") == "missing"


def test_unsupported_extension(jpeg_bytes):
    assert code_of(validate_image_bytes, jpeg_bytes, "photo.gif", "person") == "extension"


def test_not_an_image():
    assert code_of(validate_image_bytes, b"this is plain text, not pixels", "fake.jpg", "garment") == "corrupt"


def test_truncated_jpeg(jpeg_bytes):
    assert code_of(validate_image_bytes, jpeg_bytes[: len(jpeg_bytes) // 3], "cut.jpg", "person") == "corrupt"


def test_unsupported_real_format():
    data = to_bytes(make_image(400, 400), "BMP")
    assert code_of(validate_image_bytes, data, "photo.png", "person") == "format"


def test_too_small():
    assert code_of(validate_image_bytes, to_bytes(make_image(200, 300)), "s.jpg", "person") == "too_small"


def test_extreme_aspect_ratio():
    assert code_of(validate_image_bytes, to_bytes(make_image(300, 1200)), "tall.jpg", "person") == "aspect"


def test_file_too_large(monkeypatch, jpeg_bytes):
    from config import settings
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0.001)
    assert code_of(validate_image_bytes, jpeg_bytes, "big.jpg", "person") == "too_large_file"


def test_rgba_png_becomes_rgb_on_white():
    rgba = Image.new("RGBA", (400, 400), (0, 0, 0, 0))   # fully transparent
    out = validate_image_bytes(to_bytes(rgba, "PNG"), "g.png", "garment")
    assert out.image.mode == "RGB" and out.image.getpixel((10, 10)) == (255, 255, 255)


def test_exif_rotation_is_applied():
    image = make_image(600, 400)                       # stored landscape
    exif = Image.Exif()
    exif[0x0112] = 6                                   # orientation: rotate 90 degrees CW to display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    out = validate_image_bytes(buffer.getvalue(), "phone.jpg", "person")
    assert out.image.size == (400, 600)                # displayed portrait


def test_garment_type_and_params():
    assert validate_garment_type("upper") == "upper"
    assert code_of(validate_garment_type, "hat") == "garment_type"
    validate_generation_params(30, 2.5, 42)
    assert code_of(validate_generation_params, 1, 2.5, 42) == "steps"
    assert code_of(validate_generation_params, 30, 9.0, 42) == "guidance"
    assert code_of(validate_generation_params, 30, 2.5, -5) == "seed"
