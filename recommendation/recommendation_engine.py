"""Context-aware recommendation engine: filter -> score -> rank -> explain.

Independent of the try-on module (brief section 5.3): it needs only the catalogue, the
knowledge base and, optionally, a CLIP scorer. Without CLIP the CLIP weight is redistributed
over the other factors, and the result says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .catalogue import Garment
from .context_processor import Context, TargetProfile, build_profile
from .explanations import Explanation, explain, verdict_for
from .knowledge import FACTORS, Knowledge, default_knowledge
from .scoring import (
    FactorScore, clip_fit, colour_fit, formality_fit, hard_exclusion, occasion_fit, style_fit, weather_fit,
)


@dataclass
class Recommendation:
    garment: Garment
    score: float
    factors: dict                    # factor name -> FactorScore
    explanation: Explanation
    rank: int = 0

    @property
    def contributions(self) -> dict:
        return {k: round(v, 4) for k, v in self.extra.get("contributions", {}).items()}

    extra: dict = field(default_factory=dict)


@dataclass
class RecommendationResult:
    profile: TargetProfile
    items: list                      # ranked Recommendations (top_k)
    all_ranked: list                 # every eligible garment, ranked
    excluded: list                   # (Garment, reason)
    weights: dict                    # weights actually used (after redistribution)
    clip_used: bool


def normalise_weights(weights: dict, drop: tuple = ()) -> dict:
    kept = {k: (0.0 if k in drop else float(weights.get(k, 0.0))) for k in FACTORS}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("At least one factor must have a positive weight.")
    return {k: v / total for k, v in kept.items()}


class RecommendationEngine:
    def __init__(self, garments: list[Garment], kb: Optional[Knowledge] = None, clip_scorer=None,
                 weights: Optional[dict] = None):
        self.garments = garments
        self.kb = kb or default_knowledge()
        self.clip_scorer = clip_scorer
        self.base_weights = normalise_weights(weights or self.kb.weights)
        self._clip_cache: dict[str, dict] = {}

    # -- CLIP ---------------------------------------------------------------
    def _clip_probs(self, garment: Garment) -> Optional[dict]:
        if self.clip_scorer is None:
            return None
        if garment.id not in self._clip_cache:
            self._clip_cache[garment.id] = self.clip_scorer.garment_probs_for(garment)
        return self._clip_cache[garment.id]

    # -- scoring ------------------------------------------------------------
    def score(self, garment: Garment, profile: TargetProfile, weights: dict,
              clip_probs: Optional[dict] = None) -> Recommendation:
        factors: dict[str, FactorScore] = {
            "occasion": occasion_fit(garment, profile),
            "formality": formality_fit(garment, profile),
            "weather": weather_fit(garment, profile),
            "colour": colour_fit(garment, profile),
            "style": style_fit(garment, profile),
            "clip": clip_fit(clip_probs, profile),
        }
        contributions = {k: weights[k] * f.value for k, f in factors.items()}
        score = sum(contributions.values())
        verdict = verdict_for(score, self.kb.recommend, self.kb.possible)
        explanation = explain(factors, weights, self.kb.strong, self.kb.weak, verdict)
        return Recommendation(garment, score, factors, explanation, extra={"contributions": contributions})

    def recommend(self, ctx: Context, top_k: Optional[int] = None, use_clip: bool = True,
                  weights: Optional[dict] = None) -> RecommendationResult:
        profile = build_profile(ctx, self.kb)
        clip_used = use_clip and self.clip_scorer is not None
        w = normalise_weights(weights or self.base_weights, drop=() if clip_used else ("clip",))

        eligible, excluded = [], []
        for g in self.garments:
            reason = hard_exclusion(g, profile)
            if reason:
                excluded.append((g, reason))
            else:
                eligible.append(g)

        ranked = [self.score(g, profile, w, self._clip_probs(g) if clip_used else None) for g in eligible]
        ranked.sort(key=lambda r: (-r.score, r.garment.id))   # id breaks ties deterministically
        for i, rec in enumerate(ranked, start=1):
            rec.rank = i
        k = len(ranked) if top_k is None else top_k
        return RecommendationResult(profile, ranked[:k], ranked, excluded, w, clip_used)

    def assess(self, garment: Garment, ctx: Context, clip_probs: Optional[dict] = None) -> tuple[Recommendation, Optional[str]]:
        """Suitability of a single (e.g. user-uploaded) garment for a context.

        Returns the scored recommendation and, if the garment would be excluded, the reason.
        """
        profile = build_profile(ctx, self.kb)
        # Category and wear-line filters describe what the user wants to browse; they don't
        # make an uploaded garment unsuitable, so only the occasion exclusion applies here.
        exclusion = None
        if garment.garment_type in profile.occasion.excluded_types:
            exclusion = hard_exclusion(garment, build_profile(
                Context(occasion=ctx.occasion, setting=ctx.setting, weather=ctx.weather), self.kb))
        w = normalise_weights(self.base_weights, drop=() if clip_probs else ("clip",))
        return self.score(garment, profile, w, clip_probs), exclusion
