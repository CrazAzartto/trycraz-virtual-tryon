"""Loads config/recommendation.yaml into typed objects and validates it.

Keeping the knowledge base typed means a typo in the YAML (e.g. an unknown style) fails at
start-up with a clear message instead of silently producing odd rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config import settings

FACTORS = ("occasion", "formality", "weather", "colour", "style", "clip")


@dataclass(frozen=True)
class Occasion:
    key: str
    label: str
    formality: float
    preferred_styles: frozenset
    discouraged_types: frozenset
    excluded_types: frozenset
    palette: tuple
    clip_prompt: str


@dataclass(frozen=True)
class Weather:
    key: str
    label: str
    target_warmth_outdoor: float
    target_warmth_indoor: float
    want_breathable: bool
    want_water_resistant: bool
    clip_prompt: str


@dataclass(frozen=True)
class GarmentType:
    key: str
    label: str
    tryon_type: str
    formality: float
    warmth: float
    breathable: bool
    water_resistant: bool
    styles: tuple
    clip_label: str


@dataclass(frozen=True)
class Knowledge:
    weights: dict
    strong: float
    weak: float
    recommend: float
    possible: float
    occasions: dict
    settings: dict
    weather: dict
    colours: dict          # name -> (r, g, b)
    styles: tuple
    garment_types: dict


class KnowledgeError(ValueError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise KnowledgeError(message)


def load_knowledge(path: Path = settings.RECOMMENDATION_CONFIG) -> Knowledge:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    styles = tuple(raw["styles"])
    colours = {name: tuple(rgb) for name, rgb in raw["colours"].items()}

    weights = {k: float(raw["weights"].get(k, 0.0)) for k in FACTORS}
    _check(all(w >= 0 for w in weights.values()) and sum(weights.values()) > 0, "weights must be >= 0 and not all zero")
    total = sum(weights.values())
    weights = {k: w / total for k, w in weights.items()}

    garment_types = {}
    for key, g in raw["garment_types"].items():
        _check(g["tryon_type"] in settings.GARMENT_TYPES, f"garment type {key}: bad tryon_type {g['tryon_type']}")
        _check(set(g["styles"]) <= set(styles), f"garment type {key}: unknown style in {g['styles']}")
        _check(1 <= g["formality"] <= 5 and 1 <= g["warmth"] <= 5, f"garment type {key}: formality/warmth out of 1-5")
        garment_types[key] = GarmentType(key, g["label"], g["tryon_type"], float(g["formality"]), float(g["warmth"]),
                                         bool(g["breathable"]), bool(g["water_resistant"]), tuple(g["styles"]), g["clip_label"])

    occasions = {}
    for key, o in raw["occasions"].items():
        for field in ("discouraged_types", "excluded_types"):
            unknown = set(o[field]) - set(garment_types)
            _check(not unknown, f"occasion {key}: unknown garment types in {field}: {unknown}")
        _check(set(o["preferred_styles"]) <= set(styles), f"occasion {key}: unknown style")
        _check(set(o["palette"]) <= set(colours), f"occasion {key}: unknown colour in palette")
        occasions[key] = Occasion(key, o["label"], float(o["formality"]), frozenset(o["preferred_styles"]),
                                  frozenset(o["discouraged_types"]), frozenset(o["excluded_types"]),
                                  tuple(o["palette"]), o["clip_prompt"])

    weather = {}
    for key, w in raw["weather"].items():
        outdoor, indoor = w["target_warmth"]
        weather[key] = Weather(key, w["label"], float(outdoor), float(indoor), bool(w["want_breathable"]),
                               bool(w["want_water_resistant"]), w["clip_prompt"])

    return Knowledge(
        weights=weights, strong=float(raw["thresholds"]["strong"]), weak=float(raw["thresholds"]["weak"]),
        recommend=float(raw["thresholds"]["recommend"]), possible=float(raw["thresholds"]["possible"]),
        occasions=occasions, settings={k: v["label"] for k, v in raw["settings"].items()}, weather=weather,
        colours=colours, styles=styles, garment_types=garment_types,
    )


@lru_cache(maxsize=1)
def default_knowledge() -> Knowledge:
    return load_knowledge()
