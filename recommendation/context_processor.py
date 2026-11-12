"""Turns what the user selected (occasion, setting, weather, preferences) into a TargetProfile:
the concrete numbers and sets the scoring functions compare each garment against.

Analogy: the user says "interview, indoors, it's cold, I like navy". The context processor
translates that into "formality about 4.5, warmth about 3, formal/business styles, avoid hoodies,
navy is a plus" - a description of the ideal garment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import settings

from .knowledge import Knowledge, Occasion, Weather, default_knowledge


@dataclass
class Context:
    occasion: str
    setting: str = "indoor"                  # indoor | outdoor
    weather: str = "mild"                    # hot | mild | cold | rainy
    colours: tuple = ()                      # preferred colour names (optional)
    styles: tuple = ()                       # preferred styles (optional)
    formality_shift: float = 0.0             # user nudge: -1 (more casual) .. +1 (more formal)
    wear_line: Optional[str] = None          # optional filter: menswear | womenswear (unisex always allowed)
    categories: tuple = settings.GARMENT_TYPES  # which try-on types to include: upper / lower / overall


class ContextError(ValueError):
    pass


@dataclass
class TargetProfile:
    context: Context
    occasion: Occasion
    weather: Weather
    target_formality: float
    target_warmth: float
    want_breathable: bool
    want_water_resistant: bool
    preferred_styles: frozenset              # occasion's conventional styles
    user_styles: frozenset
    user_colours: frozenset
    palette: frozenset                       # occasion colour conventions (used when no user colours)
    clip_occasion_prompt: str
    clip_weather_prompt: str
    notes: list = field(default_factory=list)  # human-readable summary of the profile (shown in the UI)
    type_labels: dict = field(default_factory=dict)  # garment_type -> readable label, for explanations


def validate_context(ctx: Context, kb: Knowledge) -> None:
    if ctx.occasion not in kb.occasions:
        raise ContextError(f"Unknown occasion '{ctx.occasion}'. Choose one of {sorted(kb.occasions)}.")
    if ctx.setting not in kb.settings:
        raise ContextError(f"Unknown setting '{ctx.setting}'.")
    if ctx.weather not in kb.weather:
        raise ContextError(f"Unknown weather '{ctx.weather}'.")
    if set(ctx.colours) - set(kb.colours):
        raise ContextError(f"Unknown colours: {sorted(set(ctx.colours) - set(kb.colours))}")
    if set(ctx.styles) - set(kb.styles):
        raise ContextError(f"Unknown styles: {sorted(set(ctx.styles) - set(kb.styles))}")
    if not -1.0 <= ctx.formality_shift <= 1.0:
        raise ContextError("formality_shift must be between -1 and 1.")
    if ctx.wear_line not in (None, "menswear", "womenswear"):
        raise ContextError("wear_line must be menswear, womenswear or empty.")
    if not ctx.categories or set(ctx.categories) - set(settings.GARMENT_TYPES):
        raise ContextError(f"categories must be a non-empty subset of {settings.GARMENT_TYPES}.")


def build_profile(ctx: Context, kb: Optional[Knowledge] = None) -> TargetProfile:
    kb = kb or default_knowledge()
    validate_context(ctx, kb)
    occasion, weather = kb.occasions[ctx.occasion], kb.weather[ctx.weather]
    outdoor = ctx.setting == "outdoor"

    # The user's nudge moves the occasion's conventional formality by up to one level.
    target_formality = min(5.0, max(1.0, occasion.formality + ctx.formality_shift))
    target_warmth = weather.target_warmth_outdoor if outdoor else weather.target_warmth_indoor
    want_water = weather.want_water_resistant and outdoor   # rain does not matter indoors

    notes = [
        f"Target formality {target_formality:.1f}/5 ({occasion.label}"
        + (f", adjusted {ctx.formality_shift:+.1f} by your preference)" if ctx.formality_shift else ")"),
        f"Target warmth {target_warmth:.1f}/5 ({weather.label.lower()}, {ctx.setting})",
    ]
    if weather.want_breathable:
        notes.append("Breathable fabrics preferred")
    if want_water:
        notes.append("Water resistance preferred (rain, outdoors)")
    if occasion.excluded_types:
        labels = sorted(kb.garment_types[t].label for t in occasion.excluded_types)
        notes.append("Not suitable for this occasion: " + ", ".join(labels))

    return TargetProfile(
        context=ctx, occasion=occasion, weather=weather,
        target_formality=target_formality, target_warmth=target_warmth,
        want_breathable=weather.want_breathable, want_water_resistant=want_water,
        preferred_styles=occasion.preferred_styles, user_styles=frozenset(ctx.styles),
        user_colours=frozenset(ctx.colours), palette=frozenset(occasion.palette),
        clip_occasion_prompt=occasion.clip_prompt, clip_weather_prompt=weather.clip_prompt, notes=notes,
        type_labels={k: t.label for k, t in kb.garment_types.items()},
    )
