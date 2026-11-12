"""Offline evaluation of the recommender (brief section 10.2).

Inputs
    - a catalogue CSV
    - a scenarios YAML: each scenario = a context + graded relevance labels for catalogue ids
      (2 = good choice, 1 = acceptable, 0 = unsuitable; unlabelled ids count as 0).
      IMPORTANT: labels must be written before running the engine on that catalogue, so the
      evaluation is not tuned to its own output. Record who labelled and when in the YAML.

Metrics
    - Precision@k: fraction of the top-k with relevance >= 1.
    - nDCG@k: rewards putting the most relevant items first (graded relevance, log2 discount),
      normalised by the best possible ordering of the labels.
    - Ablations: the same scenarios with factor subsets (attributes only, CLIP only, hybrid).
    - Context sensitivity: Kendall's tau between the full rankings of every pair of scenarios
      (low tau = the ranking really changes when the context changes).

    .venv\\Scripts\\python -m evaluation.rec_eval --catalogue data/catalogue/clothing_catalogue.csv ^
        --scenarios evaluation/rec_scenarios.yaml [--clip] [--out evaluation/results/rec_eval.md]
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Optional

import yaml

from recommendation.catalogue import load_catalogue
from recommendation.context_processor import Context
from recommendation.recommendation_engine import RecommendationEngine

ABLATIONS = {
    "attributes only": {"occasion": 0.30, "formality": 0.20, "weather": 0.20, "colour": 0.10, "style": 0.05, "clip": 0.0},
    "CLIP only": {"occasion": 0, "formality": 0, "weather": 0, "colour": 0, "style": 0, "clip": 1.0},
    "hybrid (default)": None,
}


def precision_at_k(ranked_ids: list, relevance: dict, k: int) -> float:
    top = ranked_ids[:k]
    return sum(1 for i in top if relevance.get(i, 0) >= 1) / max(1, len(top))


def ndcg_at_k(ranked_ids: list, relevance: dict, k: int) -> float:
    def dcg(gains):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    actual = dcg([relevance.get(i, 0) for i in ranked_ids[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def kendall_tau(order_a: list, order_b: list, min_common: int = 5) -> Optional[float]:
    """Kendall's tau-a over the items both rankings contain (1 = identical order, -1 = reversed).
    Returns None when fewer than `min_common` items are shared (e.g. a tops-only vs a bottoms-only
    scenario), because the comparison would be meaningless."""
    common = [i for i in order_a if i in set(order_b)]
    if len(common) < min_common:
        return None
    pos_b = {item: idx for idx, item in enumerate(order_b)}
    concordant = discordant = 0
    for x, y in itertools.combinations(common, 2):   # x before y in order_a
        if pos_b[x] < pos_b[y]:
            concordant += 1
        else:
            discordant += 1
    pairs = concordant + discordant
    return (concordant - discordant) / pairs if pairs else 1.0


def load_scenarios(path: Path) -> tuple[list[dict], dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["scenarios"], data.get("meta", {})


def evaluate(engine: RecommendationEngine, scenarios: list[dict], k: int, weights: Optional[dict], use_clip: bool) -> list[dict]:
    rows = []
    for sc in scenarios:
        ctx = Context(**sc["context"])
        result = engine.recommend(ctx, use_clip=use_clip, weights=weights)
        ranked = [r.garment.id for r in result.all_ranked]
        rel = {str(i): int(v) for i, v in sc["relevance"].items()}
        rows.append({"name": sc["name"], "ranked": ranked, f"P@{k}": precision_at_k(ranked, rel, k),
                     f"nDCG@{k}": ndcg_at_k(ranked, rel, k), "top": ranked[:k]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--clip", action="store_true", help="load CLIP (needed for the CLIP and hybrid rows)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    garments = load_catalogue(Path(args.catalogue))
    scorer = None
    if args.clip:
        from recommendation.clip_encoder import ClipEncoder, ClipScorer
        scorer = ClipScorer(ClipEncoder())
        scorer.embed_catalogue(garments)
    engine = RecommendationEngine(garments, clip_scorer=scorer)
    scenarios, meta = load_scenarios(Path(args.scenarios))
    k = args.k

    lines = [f"# Recommender evaluation\n", f"Catalogue: {args.catalogue} ({len(garments)} garments)  ",
             f"Scenarios: {args.scenarios} ({len(scenarios)})  ", f"Labels: {meta}\n"]
    summary = []
    for label, weights in ABLATIONS.items():
        needs_clip = label != "attributes only"
        if needs_clip and scorer is None:
            continue
        rows = evaluate(engine, scenarios, k, weights, use_clip=needs_clip)
        mean_p = sum(r[f"P@{k}"] for r in rows) / len(rows)
        mean_n = sum(r[f"nDCG@{k}"] for r in rows) / len(rows)
        summary.append((label, mean_p, mean_n))
        lines.append(f"\n## {label}\n\n| Scenario | P@{k} | nDCG@{k} | top-{k} |\n|---|---|---|---|")
        lines += [f"| {r['name']} | {r[f'P@{k}']:.2f} | {r[f'nDCG@{k}']:.2f} | {', '.join(r['top'])} |" for r in rows]
        if label.startswith("hybrid") or scorer is None:
            taus = [t for a, b in itertools.combinations(rows, 2) if (t := kendall_tau(a["ranked"], b["ranked"])) is not None]
            if taus:
                lines.append(f"\nContext sensitivity: mean Kendall tau over {len(taus)} scenario pairs sharing >= 5 garments = "
                             f"{sum(taus) / len(taus):.2f} (min {min(taus):.2f}, max {max(taus):.2f}). "
                             "1 would mean the ranking ignores context.")
    lines.append(f"\n## Summary\n\n| Configuration | mean P@{k} | mean nDCG@{k} |\n|---|---|---|")
    lines += [f"| {label} | {p:.3f} | {n:.3f} |" for label, p, n in summary]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
