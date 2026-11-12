"""Garment background removal (our preprocessing step before try-on).

Why: CatVTON was trained on product photos with plain white backgrounds (VITON-HD, DressCode).
Home photos of garments on doors, bedsheets or floors confuse it: in our first catalogue test, a
black blazer photographed on a hanger in front of a door and a pink wall came out as a grey
hooded top. Putting the garment on white first removes that distribution shift.

How: salient-object segmentation with U^2-Net (Qin et al., Pattern Recognition 2020), via the
`rembg` package (MIT licence; model weights downloaded once, ~176 MB). U^2-Net is a nested
U-shaped encoder-decoder that predicts a per-pixel foreground probability; we composite the
foreground onto white using that alpha matte and crop to the garment's bounding box.

If rembg is not installed or fails, the original image is returned and `applied` is False,
so try-on still works.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_MODEL = "u2net"
MIN_FOREGROUND = 0.03      # less than this: segmentation probably failed, keep original
MAX_FOREGROUND = 0.98      # more than this: nothing was removed
CROP_MARGIN = 0.05

_sessions: dict = {}
_lock = threading.Lock()


@dataclass
class SegmentationResult:
    image: Image.Image                 # garment on white (or the original if not applied)
    mask: Optional[Image.Image]        # L-mode alpha matte at the original size
    foreground_fraction: float
    applied: bool
    message: str = ""


def _session(model: str):
    with _lock:
        if model not in _sessions:
            from rembg import new_session
            _sessions[model] = new_session(model)
        return _sessions[model]


def remove_background(image: Image.Image, model: str = DEFAULT_MODEL, crop: bool = True) -> SegmentationResult:
    image = image.convert("RGB")
    try:
        from rembg import remove
        rgba = remove(image, session=_session(model), post_process_mask=True)
    except Exception as exc:   # missing package, model download failure, ...
        log.warning("Background removal unavailable: %s", exc)
        return SegmentationResult(image, None, 1.0, False, f"background removal unavailable ({type(exc).__name__})")

    alpha = rgba.split()[3]
    fraction = float((np.asarray(alpha) > 127).mean())
    if not MIN_FOREGROUND <= fraction <= MAX_FOREGROUND:
        return SegmentationResult(image, alpha, fraction, False,
                                  f"segmentation looked unreliable (foreground {fraction:.0%}); original kept")

    on_white = Image.new("RGB", image.size, (255, 255, 255))
    on_white.paste(image, mask=alpha)
    if crop:
        box = alpha.point(lambda v: 255 if v > 127 else 0).getbbox()
        if box:
            mx, my = int((box[2] - box[0]) * CROP_MARGIN), int((box[3] - box[1]) * CROP_MARGIN)
            box = (max(0, box[0] - mx), max(0, box[1] - my), min(image.width, box[2] + mx), min(image.height, box[3] + my))
            on_white = on_white.crop(box)
    return SegmentationResult(on_white, alpha, fraction, True, "background removed")
