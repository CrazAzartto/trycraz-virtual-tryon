"""Automatic analysis of a garment photo (user uploads, and pre-filling catalogue annotations).

Two techniques:
  1. Dominant colours (classical CV, no learning):
       - estimate the background colour as the median of the image border,
       - foreground = pixels whose CIELAB distance from the background exceeds a threshold
         (CIELAB is used because Euclidean distance there roughly matches perceived colour difference),
       - k-means (k=3) on the foreground pixels in CIELAB,
       - name each large cluster by its nearest palette colour (also in CIELAB).
  2. Garment type: CLIP zero-shot classification over the knowledge base's garment types.
The type then supplies default formality / warmth / styles from config/recommendation.yaml.

Everything returned here is an ESTIMATE and is labelled so in the UI. Catalogue annotations
pre-filled this way are marked annotation=auto until a human verifies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .catalogue import Garment, garment_from_type
from .knowledge import Knowledge, default_knowledge

BACKGROUND_DELTA_E = 18.0     # Lab distance from background to count as garment pixel
MIN_CLUSTER_SHARE = 0.18      # clusters smaller than this are ignored when naming colours
ANALYSIS_SIZE = 192           # pixels on the long side; enough for colour statistics


@dataclass
class ColourAnalysis:
    colours: list                 # [(name, share)] largest first, share of garment pixels
    foreground_fraction: float
    background_rgb: tuple


@dataclass
class GarmentAnalysis:
    garment: Garment
    colour: ColourAnalysis
    type_candidates: list = field(default_factory=list)   # [(garment_type, prob)] top 3
    type_source: str = "clip"                              # "clip" | "user"


def _to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIELAB float (L in 0-100, a/b roughly -128..127) via OpenCV."""
    flat = rgb_uint8.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).reshape(rgb_uint8.shape[:-1] + (3,))


def dominant_colours(image: Image.Image, kb: Optional[Knowledge] = None, k: int = 3) -> ColourAnalysis:
    kb = kb or default_knowledge()
    small = image.convert("RGB")
    small.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    rgb = np.asarray(small)
    lab = _to_lab(rgb)

    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=-1)
    mask = distance > BACKGROUND_DELTA_E
    fg_fraction = float(mask.mean())
    pixels = lab[mask] if mask.sum() >= 50 else lab.reshape(-1, 3)   # fall back to whole image

    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)   # deterministic clustering
    _, labels, centres = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)

    names = list(kb.colours)
    palette_lab = _to_lab(np.array([kb.colours[n] for n in names], dtype=np.uint8)[None, :, :])[0]
    found: dict[str, float] = {}
    for centre, share in sorted(zip(centres, shares), key=lambda cs: -cs[1]):
        if share < MIN_CLUSTER_SHARE:
            continue
        nearest = names[int(np.argmin(np.linalg.norm(palette_lab - centre, axis=1)))]
        found[nearest] = found.get(nearest, 0.0) + float(share)
    colours = sorted(found.items(), key=lambda kv: -kv[1])
    bg_rgb = tuple(int(v) for v in np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]), axis=0))
    return ColourAnalysis(colours=[(n, round(s, 2)) for n, s in colours], foreground_fraction=round(fg_fraction, 2),
                          background_rgb=bg_rgb)


def analyse_garment(image: Image.Image, clip_scorer=None, garment_type: Optional[str] = None,
                    kb: Optional[Knowledge] = None, name: str = "Your garment") -> GarmentAnalysis:
    """Estimate attributes of a garment photo. Pass `garment_type` to skip CLIP type detection
    (e.g. when the user picked it, or CLIP is unavailable)."""
    kb = kb or default_knowledge()
    colour = dominant_colours(image, kb)
    candidates, source = [], "user"
    if garment_type is None:
        if clip_scorer is None:
            raise ValueError("garment_type is required when CLIP is unavailable")
        probs = clip_scorer.garment_type_probs(clip_scorer.embed(image))
        candidates = list(probs.items())[:3]
        garment_type, source = candidates[0][0], "clip"
    pattern = "solid" if colour.colours and colour.colours[0][1] >= 0.75 else "printed"
    garment = garment_from_type(
        garment_type, kb, id="uploaded", name=name, colours=tuple(n for n, _ in colour.colours[:2]),
        pattern=pattern, estimated=True, annotation="auto",
    )
    return GarmentAnalysis(garment=garment, colour=colour, type_candidates=candidates, type_source=source)
