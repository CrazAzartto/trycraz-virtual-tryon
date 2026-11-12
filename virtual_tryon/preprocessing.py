"""Client-side preprocessing before an image is sent to the GPU server.

The server applies CatVTON's own preprocessing (resize_and_crop for the person,
resize_and_padding for the garment). We do the *same geometric crop* on the laptop first so:
  1. the user sees exactly which part of the photo the model will use (crop preview), and
  2. the original and generated images are pixel-aligned in the comparison slider.
We also downscale before upload: a 12 MP phone photo becomes ~1.5 MP, cutting upload time
through the tunnel without losing anything (the model works at 768x1024 anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from config import settings
from common.image_utils import center_crop_to_aspect, limit_long_side, pad_to_aspect

UPLOAD_LONG_SIDE = 1536   # 1.5x the model resolution: headroom for the server's LANCZOS resize


@dataclass
class PreparedPair:
    person: Image.Image            # 3:4, ready to upload
    garment: Image.Image           # 3:4, white-padded, ready to upload
    person_preview: Image.Image    # 768x1024: what the model sees; used for comparison
    person_was_cropped: bool       # True if the centre crop removed part of the photo


def prepare_person(image: Image.Image) -> tuple[Image.Image, bool]:
    cropped = center_crop_to_aspect(image, settings.MODEL_WIDTH, settings.MODEL_HEIGHT)
    was_cropped = cropped.size != image.size
    return limit_long_side(cropped, UPLOAD_LONG_SIDE), was_cropped


def prepare_garment(image: Image.Image) -> Image.Image:
    padded = pad_to_aspect(image, settings.MODEL_WIDTH, settings.MODEL_HEIGHT)
    return limit_long_side(padded, UPLOAD_LONG_SIDE)


def prepare_pair(person: Image.Image, garment: Image.Image) -> PreparedPair:
    person_ready, was_cropped = prepare_person(person)
    preview = person_ready.resize((settings.MODEL_WIDTH, settings.MODEL_HEIGHT), Image.LANCZOS)
    return PreparedPair(
        person=person_ready,
        garment=prepare_garment(garment),
        person_preview=preview,
        person_was_cropped=was_cropped,
    )
