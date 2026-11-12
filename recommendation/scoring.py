"""Factor functions. Each returns a value in [0, 1] plus the facts needed to explain it.

    S(g, c) = sum_k  w_k * f_k(g, c)        k in {occasion, formality, weather, colour, style, clip}

Every function is pure (no I/O, no model calls) so it can be unit-tested and explained in
isolation. The CLIP factor receives precomputed probabilities from clip_encoder.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .catalogue import Garment
from .context_processor import TargetProfile

NEUTRAL = 0.5   # value used when a factor has no information (e.g. no colour preference)
PLURAL_LABELS = ("jeans", "trousers", "chinos", "shorts", "pants")


def type_phrase(p: "TargetProfile", garment_type: str) -> str:
    """'a hoodie is' / 'jeans are' - grammatical subject for explanation sentences."""
    label = p.type_labels.get(garment_type, garment_type.replace("_", " ")).lower()
    if label.endswith(PLURAL_LABELS):
        return f"{label} are"
    article = "an" if label[0] in "aeiou" else "a"
    return f"{article} {label} is"


@dataclass
class FactorScore:
    """A factor value plus two phrases. explanations.py uses `positive` only if the value is high
    and `negative` only if it is low (same thresholds as the ranking), so a sentence can never
    praise a garment for something that scored badly."""

    value: float
    positive: str = ""        # e.g. "formality 4/5 is close to the 4.5 expected"
    negative: str = ""        # e.g. "much more casual (1.5/5) than expected (4.5)"
    informative: bool = True  # False when the factor had nothing to judge (value = NEUTRAL)
    data: dict = field(default_factory=dict)


def occasion_fit(g: Garment, p: TargetProfile) -> FactorScore:
    """Catalogue tags if present (our human annotation), otherwise style overlap with the occasion."""
    occ = p.occasion
    name = occ.label.lower()
    if g.garment_type in occ.discouraged_types:
        return FactorScore(0.0, negative=f"{type_phrase(p, g.garment_type)} usually not worn for {name}")
    if g.has_occasion_tags:
        if occ.key in g.occasions_primary:
            return FactorScore(1.0, positive=f"it is a good choice for {name}")
        if occ.key in g.occasions_secondary:
            return FactorScore(0.6, positive=f"it is acceptable for {name}", negative=f"it is only a fallback for {name}")
        return FactorScore(0.2, negative=f"it is not usually chosen for {name}")
    overlap = set(g.styles) & occ.preferred_styles
    if overlap:
        return FactorScore(0.8, positive=f"its {', '.join(sorted(overlap)).replace('_', ' ')} style suits {name}")
    return FactorScore(0.3, negative=f"its style is not typical for {name}")


def formality_fit(g: Garment, p: TargetProfile) -> FactorScore:
    """Linear in the distance between garment formality and the target (both on a 1-5 scale)."""
    distance = abs(g.formality - p.target_formality)
    value = max(0.0, 1.0 - distance / 4.0)
    direction = "more casual" if g.formality < p.target_formality else "more formal"
    return FactorScore(
        value,
        positive=f"its formality ({g.formality:g}/5) is close to the {p.target_formality:.1f} expected",
        negative=f"it is much {direction} ({g.formality:g}/5) than expected ({p.target_formality:.1f})",
        data={"distance": distance},
    )


def weather_fit(g: Garment, p: TargetProfile) -> FactorScore:
    """70% warmth match + 30% weather-specific property (breathability / water resistance)."""
    distance = abs(g.warmth - p.target_warmth)
    warmth_value = max(0.0, 1.0 - distance / 3.0)
    special, special_reason = 1.0, ""
    if p.want_breathable:
        special = 1.0 if g.breathable else 0.0
        special_reason = "breathable" if g.breathable else "not very breathable"
    elif p.want_water_resistant:
        special = 1.0 if g.water_resistant else 0.3
        special_reason = "water resistant" if g.water_resistant else "not water resistant"
    value = 0.7 * warmth_value + 0.3 * special

    weather_word = p.weather.label.lower() + (" (indoors)" if p.context.setting == "indoor" else "")
    too = "too light" if g.warmth < p.target_warmth else "too warm"
    if special < 1.0:   # right warmth but missing the property this weather calls for
        positive = f"its warmth suits {weather_word} weather"
    else:
        positive = f"it suits {weather_word} weather" + (" (breathable)" if p.want_breathable else "") \
            + (" (water resistant)" if p.want_water_resistant else "")
    negative = f"it is {too} for {weather_word} weather" + (f" and {special_reason}" if special < 1.0 else "")
    return FactorScore(value, positive=positive, negative=negative, data={"warmth_distance": distance, "special": special})


def colour_fit(g: Garment, p: TargetProfile) -> FactorScore:
    """User colours if given; otherwise the occasion's colour conventions; otherwise neutral."""
    colours = set(g.colours)
    if p.user_colours:
        match = colours & p.user_colours
        if match:
            return FactorScore(1.0, positive=f"it comes in your preferred colour ({', '.join(sorted(match)).replace('_', ' ')})")
        return FactorScore(0.0, negative="it is not in your preferred colours")
    if p.palette and colours:
        match = colours & p.palette
        if match:
            return FactorScore(1.0, positive=f"{', '.join(sorted(match)).replace('_', ' ')} is a classic colour for {p.occasion.label.lower()}")
        return FactorScore(0.25, negative=f"its colour is bolder than usual for {p.occasion.label.lower()}")
    return FactorScore(NEUTRAL, informative=False)


def style_fit(g: Garment, p: TargetProfile) -> FactorScore:
    if not p.user_styles:
        return FactorScore(NEUTRAL, informative=False)
    match = set(g.styles) & p.user_styles
    if match:
        return FactorScore(1.0, positive=f"it matches your {', '.join(sorted(match)).replace('_', ' ')} style preference")
    return FactorScore(0.0, negative="it is not in your preferred styles")


def clip_fit(clip_probs: Optional[dict], p: TargetProfile) -> FactorScore:
    """CLIP zero-shot evidence from the garment IMAGE.

    clip_probs = {"occasion": {key: prob over all occasions}, "weather": {key: prob over all weather}}
    For each, the value is p(selected) / max p, i.e. 1.0 if CLIP thinks this image is most typical of
    the selected occasion/weather among all options. The final value averages occasion and weather.
    """
    if not clip_probs:
        return FactorScore(NEUTRAL, informative=False)
    occ_probs, wx_probs = clip_probs["occasion"], clip_probs["weather"]
    occ_rel = occ_probs[p.occasion.key] / max(occ_probs.values())
    wx_rel = wx_probs[p.weather.key] / max(wx_probs.values())
    value = 0.5 * occ_rel + 0.5 * wx_rel
    top_occasion = max(occ_probs, key=occ_probs.get)
    return FactorScore(
        value,
        positive="the photo itself looks like typical wear for this occasion and weather (CLIP)",
        negative=f"visually it looks more like {top_occasion.replace('_', ' ')} wear (CLIP)",
        data={"occasion_relative": occ_rel, "weather_relative": wx_rel, "top_occasion": top_occasion},
    )


def hard_exclusion(g: Garment, p: TargetProfile) -> Optional[str]:
    """Reason to remove a garment before ranking, or None."""
    ctx = p.context
    if g.tryon_type not in ctx.categories:
        return "not in the selected garment categories"
    if ctx.wear_line and g.wear_line not in (ctx.wear_line, "unisex"):
        return f"listed as {g.wear_line}"
    if g.garment_type in p.occasion.excluded_types:
        return f"{type_phrase(p, g.garment_type)} not appropriate for {p.occasion.label.lower()}"
    return None
