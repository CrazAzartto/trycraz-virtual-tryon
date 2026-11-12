"""Clothing catalogue: loading, validation and the Garment record.

The catalogue is a CSV (data/catalogue/clothing_catalogue.csv) so it can be reviewed and edited
in Excel. List-valued columns use ';' as separator. Every row is validated against the
knowledge base; all problems are reported together with their row numbers.

Columns
    id, name, image_file, garment_type, tryon_type, formality (1-5), warmth (1-5),
    breathable (yes/no), water_resistant (yes/no), colours, pattern, styles,
    occasions_primary, occasions_secondary, wear_line (menswear|womenswear|unisex),
    source, license, annotation (auto|manual|verified), tryon_tested (pending|ok|failed), notes

annotation:  auto     = pre-filled automatically (CLIP type, k-means colours, type defaults)
             manual   = set by hand from visual inspection by the developer (during development
                        this was done by the AI assistant); not yet confirmed by the project owner
             verified = confirmed by the project owner
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from config import settings

from .knowledge import Knowledge, default_knowledge

COLUMNS = [
    "id", "name", "image_file", "garment_type", "tryon_type", "formality", "warmth", "breathable",
    "water_resistant", "colours", "pattern", "styles", "occasions_primary", "occasions_secondary",
    "wear_line", "source", "license", "annotation", "tryon_tested", "notes",
]
WEAR_LINES = ("menswear", "womenswear", "unisex")
PATTERNS = ("solid", "striped", "checked", "printed", "graphic", "textured")


@dataclass
class Garment:
    id: str
    name: str
    garment_type: str
    tryon_type: str
    formality: float
    warmth: float
    breathable: bool
    water_resistant: bool
    colours: tuple = ()
    pattern: str = "solid"
    styles: tuple = ()
    occasions_primary: tuple = ()
    occasions_secondary: tuple = ()
    wear_line: str = "unisex"
    image_file: str = ""
    source: str = ""
    license: str = ""
    annotation: str = "auto"
    tryon_tested: str = "pending"
    notes: str = ""
    estimated: bool = False            # True for user-uploaded garments analysed automatically
    extra: dict = field(default_factory=dict)

    @property
    def image_path(self) -> Optional[Path]:
        return settings.CATALOGUE_IMAGE_DIR / self.image_file if self.image_file else None

    @property
    def has_occasion_tags(self) -> bool:
        return bool(self.occasions_primary or self.occasions_secondary)


class CatalogueError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Catalogue has problems:\n  " + "\n  ".join(problems))


def _split(value: str) -> tuple:
    return tuple(v.strip() for v in (value or "").split(";") if v.strip())


def _bool(value: str) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1", "y")


def parse_row(row: dict, line: int, kb: Knowledge, problems: list[str], check_image: bool = True) -> Optional[Garment]:
    where = f"row {line} (id={row.get('id', '?')})"
    try:
        g = Garment(
            id=row["id"].strip(), name=row["name"].strip(), garment_type=row["garment_type"].strip(),
            tryon_type=row["tryon_type"].strip(), formality=float(row["formality"]), warmth=float(row["warmth"]),
            breathable=_bool(row["breathable"]), water_resistant=_bool(row["water_resistant"]),
            colours=_split(row["colours"]), pattern=(row.get("pattern") or "solid").strip(), styles=_split(row["styles"]),
            occasions_primary=_split(row["occasions_primary"]), occasions_secondary=_split(row["occasions_secondary"]),
            wear_line=(row.get("wear_line") or "unisex").strip(), image_file=(row.get("image_file") or "").strip(),
            source=(row.get("source") or "").strip(), license=(row.get("license") or "").strip(),
            annotation=(row.get("annotation") or "auto").strip(), tryon_tested=(row.get("tryon_tested") or "pending").strip(),
            notes=(row.get("notes") or "").strip(),
        )
    except (KeyError, ValueError, AttributeError) as exc:
        problems.append(f"{where}: cannot parse ({exc})")
        return None

    checks = [
        (bool(g.id), "empty id"),
        (g.garment_type in kb.garment_types, f"unknown garment_type '{g.garment_type}'"),
        (g.tryon_type in settings.GARMENT_TYPES, f"unknown tryon_type '{g.tryon_type}'"),
        (1 <= g.formality <= 5, "formality must be 1-5"),
        (1 <= g.warmth <= 5, "warmth must be 1-5"),
        (set(g.colours) <= set(kb.colours), f"unknown colour in {g.colours}"),
        (g.pattern in PATTERNS, f"unknown pattern '{g.pattern}'"),
        (set(g.styles) <= set(kb.styles), f"unknown style in {g.styles}"),
        (set(g.occasions_primary) | set(g.occasions_secondary) <= set(kb.occasions), "unknown occasion"),
        (not set(g.occasions_primary) & set(g.occasions_secondary), "occasion listed as both primary and secondary"),
        (g.wear_line in WEAR_LINES, f"wear_line must be one of {WEAR_LINES}"),
        (g.annotation in ("auto", "manual", "verified"), "annotation must be auto, manual or verified"),
        (g.tryon_tested in ("pending", "ok", "failed"), "tryon_tested must be pending, ok or failed"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if check_image and g.image_file and not (settings.CATALOGUE_IMAGE_DIR / g.image_file).exists():
        bad.append(f"image file not found: {g.image_file}")
    if bad:
        problems.extend(f"{where}: {msg}" for msg in bad)
        return None
    return g


def load_catalogue(path: Path = settings.CATALOGUE_CSV, kb: Optional[Knowledge] = None,
                   require_images: bool = True) -> list[Garment]:
    kb = kb or default_knowledge()
    problems: list[str] = []
    garments: list[Garment] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:   # utf-8-sig: Excel adds a BOM
        reader = csv.DictReader(fh)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise CatalogueError([f"missing columns: {sorted(missing)}"])
        for line, row in enumerate(reader, start=2):
            g = parse_row(row, line, kb, problems, check_image=require_images)
            if g:
                garments.append(g)
    ids = [g.id for g in garments]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        problems.append(f"duplicate ids: {sorted(duplicates)}")
    if problems:
        raise CatalogueError(problems)
    return garments


def save_catalogue(garments: list[Garment], path: Path = settings.CATALOGUE_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for g in garments:
            writer.writerow({
                "id": g.id, "name": g.name, "image_file": g.image_file, "garment_type": g.garment_type,
                "tryon_type": g.tryon_type, "formality": g.formality, "warmth": g.warmth,
                "breathable": "yes" if g.breathable else "no", "water_resistant": "yes" if g.water_resistant else "no",
                "colours": ";".join(g.colours), "pattern": g.pattern, "styles": ";".join(g.styles),
                "occasions_primary": ";".join(g.occasions_primary), "occasions_secondary": ";".join(g.occasions_secondary),
                "wear_line": g.wear_line, "source": g.source, "license": g.license, "annotation": g.annotation,
                "tryon_tested": g.tryon_tested, "notes": g.notes,
            })


def garment_from_type(garment_type: str, kb: Optional[Knowledge] = None, **overrides) -> Garment:
    """A Garment pre-filled with the knowledge-base defaults for its type."""
    kb = kb or default_knowledge()
    t = kb.garment_types[garment_type]
    base = Garment(id=overrides.pop("id", garment_type), name=overrides.pop("name", t.label), garment_type=t.key,
                   tryon_type=t.tryon_type, formality=t.formality, warmth=t.warmth, breathable=t.breathable,
                   water_resistant=t.water_resistant, styles=t.styles)
    return replace(base, **overrides)
