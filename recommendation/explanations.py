"""Builds explanations from factor scores - never from separate templates.

A reason is stated only if that factor scored >= `strong`; a caution only if it scored <= `weak`.
Reasons are ordered by their weighted contribution to the final score, so the first reason is
the one that mattered most. Because explanations are derived from the same numbers as the
ranking, they cannot drift from it (tests/test_recommendations.py checks this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scoring import FactorScore


@dataclass
class Explanation:
    reasons: list = field(default_factory=list)    # (factor, phrase)
    cautions: list = field(default_factory=list)   # (factor, phrase)
    summary: str = ""
    verdict: str = ""                              # "Recommended" | "A reasonable option" | "Weak match"

    @property
    def reason_factors(self) -> set:
        return {f for f, _ in self.reasons}

    @property
    def caution_factors(self) -> set:
        return {f for f, _ in self.cautions}


def verdict_for(score: float, recommend: float, possible: float) -> str:
    if score >= recommend:
        return "Recommended"
    if score >= possible:
        return "A reasonable option"
    return "Weak match"


def explain(factors: dict[str, FactorScore], weights: dict[str, float], strong: float, weak: float,
            verdict: str = "Recommended", max_reasons: int = 3) -> Explanation:
    ranked = sorted(factors.items(), key=lambda kv: weights.get(kv[0], 0.0) * kv[1].value, reverse=True)
    reasons = [(name, f.positive) for name, f in ranked
               if f.informative and f.positive and f.value >= strong and weights.get(name, 0) > 0][:max_reasons]
    cautions = [(name, f.negative) for name, f in sorted(factors.items(), key=lambda kv: kv[1].value)
                if f.informative and f.negative and f.value <= weak and weights.get(name, 0) > 0]

    if reasons:
        summary = f"{verdict}: " + _join([phrase for _, phrase in reasons]) + "."
    else:
        summary = f"{verdict}: no single factor fits strongly."
    if cautions:
        summary += " However, " + _join([phrase for _, phrase in cautions[:2]]) + "."
    return Explanation(reasons=reasons, cautions=cautions, summary=summary, verdict=verdict)


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]
