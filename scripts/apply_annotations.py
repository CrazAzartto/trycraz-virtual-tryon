"""Apply hand-written annotations (data/catalogue/manual_annotations.csv) to catalogue rows that
build_catalogue.py pre-filled automatically, and record how often the automatic guesses agreed.

    .venv\\Scripts\\python scripts\\apply_annotations.py

Matching is by the original source file name stored in each row's notes ("from <file>; ...").
Blank fields in the manual file keep the garment-type defaults. Occasion tags are re-derived
from the final attributes with the same rule build_catalogue.py uses. The automatic guesses are
kept in the notes and summarised in evaluation/results/auto_annotation_agreement.md.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from recommendation.catalogue import load_catalogue, save_catalogue  # noqa: E402
from recommendation.knowledge import default_knowledge  # noqa: E402
from scripts.build_catalogue import derive_occasions  # noqa: E402

MANUAL = settings.CATALOGUE_DIR / "manual_annotations.csv"
REPORT = settings.PROJECT_ROOT / "evaluation" / "results" / "auto_annotation_agreement.md"


def main() -> int:
    kb = default_knowledge()
    manual = {r["source_file"]: r for r in csv.DictReader(open(MANUAL, encoding="utf-8"))}
    garments = load_catalogue()
    comparisons = []
    for g in garments:
        match = re.search(r"from ([^;]+);", g.notes)
        if not match or match.group(1) not in manual:
            continue
        m = manual[match.group(1)]
        auto_type, auto_colours = g.garment_type, g.colours
        t = kb.garment_types[m["garment_type"]]
        g.garment_type, g.tryon_type = t.key, t.tryon_type
        g.formality = float(m["formality"]) if m["formality"] else t.formality
        g.warmth = float(m["warmth"]) if m["warmth"] else t.warmth
        g.breathable, g.water_resistant = t.breathable, t.water_resistant
        g.styles = tuple(m["styles"].split(";")) if m["styles"] else t.styles
        g.colours = tuple(m["colours"].split(";"))
        g.pattern = m["pattern"] or "solid"
        g.name, g.wear_line = m["name"], m["wear_line"]
        g.occasions_primary, g.occasions_secondary = derive_occasions(g, kb)
        g.annotation = "manual"
        if "auto guess:" not in g.notes:
            g.notes += f"; auto guess: type={auto_type}, colours={'+'.join(auto_colours)}"
        comparisons.append((g.id, m["dataset_label"], g.garment_type, auto_type, g.colours, auto_colours))
    save_catalogue(garments)

    n = len(comparisons)
    if not n:
        print("No rows matched.")
        return 1
    type_ok = sum(c[2] == c[3] for c in comparisons)
    colour_ok = sum(bool(c[5]) and c[5][0] == c[4][0] for c in comparisons)
    colour_any = sum(bool(set(c[4]) & set(c[5])) for c in comparisons)
    lines = [
        "# Automatic vs manual annotation agreement\n",
        f"Garments compared: {n} (Clothing Dataset images; manual labels by visual inspection).\n",
        "| Measure | Agreement |", "|---|---|",
        f"| Garment type, CLIP zero-shot top-1 vs manual (26 classes) | {type_ok}/{n} = {type_ok / n:.0%} |",
        f"| Main colour, k-means name vs manual main colour | {colour_ok}/{n} = {colour_ok / n:.0%} |",
        f"| Any colour overlap | {colour_any}/{n} = {colour_any / n:.0%} |",
        "\nCaveats: small sample; one annotator; several garment types are genuinely ambiguous "
        "(polo vs t-shirt, jacket vs blazer); colour names are on a 16-colour palette.\n",
        "| id | dataset label | manual type | CLIP type | manual colours | auto colours |", "|---|---|---|---|---|---|",
    ]
    lines += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]}{'' if c[2] == c[3] else ' ✗'} | {', '.join(c[4])} | {', '.join(c[5])} |"
              for c in comparisons]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:7]))
    print(f"Applied {n} manual annotations. Report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
