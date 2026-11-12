"""Validation of user-supplied images.

Every check raises ImageValidationError with a message written for the end user, not a
stack trace. The UI shows `error.user_message` directly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from config import settings
from common.image_utils import apply_exif_orientation, to_rgb

# Decompression-bomb protection: refuse absurd pixel counts before decoding.
Image.MAX_IMAGE_PIXELS = settings.MAX_SIDE_PX * settings.MAX_SIDE_PX


class ImageValidationError(ValueError):
    def __init__(self, user_message: str, code: str):
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code   # stable machine-readable code, used by tests


@dataclass
class ValidatedImage:
    image: Image.Image          # RGB, EXIF orientation applied
    original_format: str
    original_size: tuple[int, int]
    file_size_bytes: int


def validate_image_bytes(data: Optional[bytes], filename: Optional[str], role: str) -> ValidatedImage:
    """Validate raw upload bytes. `role` is "person" or "garment" (used in messages)."""
    if data is None or len(data) == 0:
        raise ImageValidationError(f"Please upload a {role} image.", "missing")

    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
            raise ImageValidationError(
                f"The {role} file '{filename}' has an unsupported type ({ext or 'no extension'}). Use {allowed}.",
                "extension",
            )

    size_mb = len(data) / 2**20
    if size_mb > settings.MAX_UPLOAD_MB:
        raise ImageValidationError(
            f"The {role} image is {size_mb:.1f} MB; the limit is {settings.MAX_UPLOAD_MB} MB.", "too_large_file"
        )

    try:
        Image.open(io.BytesIO(data)).verify()       # structural check: catches truncated/corrupted files
        image = Image.open(io.BytesIO(data))        # verify() invalidates the object; reopen to decode
        image.load()
    except Image.DecompressionBombError:
        raise ImageValidationError(f"The {role} image has too many pixels to process safely.", "too_many_pixels")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError(
            f"The {role} file could not be read as an image. It may be corrupted or not really an image.", "corrupt"
        )

    fmt = image.format or "UNKNOWN"
    if fmt not in settings.ALLOWED_PIL_FORMATS:
        raise ImageValidationError(f"The {role} image format {fmt} is not supported. Use JPEG, PNG or WEBP.", "format")

    image = apply_exif_orientation(image)
    w, h = image.size
    if min(w, h) < settings.MIN_SIDE_PX:
        raise ImageValidationError(
            f"The {role} image is only {w}x{h} pixels. Please use at least "
            f"{settings.MIN_SIDE_PX} pixels on the shorter side.", "too_small"
        )
    if max(w, h) > settings.MAX_SIDE_PX:
        raise ImageValidationError(
            f"The {role} image is {w}x{h} pixels; the maximum is {settings.MAX_SIDE_PX} on the longer side.", "too_big"
        )
    if max(w, h) / min(w, h) > settings.MAX_ASPECT_RATIO:
        raise ImageValidationError(
            f"The {role} image is very long or very wide ({w}x{h}). Please crop it closer to a portrait shape.",
            "aspect",
        )

    return ValidatedImage(image=to_rgb(image), original_format=fmt, original_size=(w, h), file_size_bytes=len(data))


def validate_garment_type(garment_type: str) -> str:
    if garment_type not in settings.GARMENT_TYPES:
        raise ImageValidationError(
            f"Unknown garment type '{garment_type}'. Choose one of: {', '.join(settings.GARMENT_TYPES)}.", "garment_type"
        )
    return garment_type


def validate_generation_params(steps: int, guidance: float, seed: int) -> None:
    if not 5 <= steps <= settings.MAX_STEPS:
        raise ImageValidationError(f"Steps must be between 5 and {settings.MAX_STEPS}.", "steps")
    if not 0.0 <= guidance <= 7.5:
        raise ImageValidationError("Guidance must be between 0 and 7.5.", "guidance")
    if not -1 <= seed <= 2**31 - 1:
        raise ImageValidationError("Seed must be -1 (random) or a non-negative integer.", "seed")
