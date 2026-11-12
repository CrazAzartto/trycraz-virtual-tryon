"""Add garment photos to the demo catalogue with automatic, reviewable annotations.

    .venv\\Scripts\\python scripts\\build_catalogue.py --input user_photos\\garments ^
        --source "own photo" --license "own work" [--wear-line menswear] [--no-clip]

For each image not yet in the catalogue:
  1. validate + fix EXIF orientation,
  2. remove the background with U^2-Net (garment on white, cropped); if that is unavailable or
     unreliable, fall back to a colour-distance crop (border colour = background),
  3. resize (long side 1024) and save as data/catalogue/images/<id>.jpg,
  4. pre-fill attributes: garment type by CLIP zero-shot, colours by k-means, the rest from the
     garment-type defaults in config/recommendation.yaml; occasions derived from the knowledge base,
  5. append a row with annotation=auto.
Finally writes outputs/catalogue_review.jpg (contact sheet) for human review. Rows become
annotation=verified only after a person has checked them (edit the CSV).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.validation import ImageValidationError, validate_image_bytes  # noqa: E402
from config import settings  # noqa: E402
from recommendation.catalogue import COLUMNS, Garment, load_catalogue, save_catalogue  # noqa: E402
from recommendation.garment_analyzer import _to_lab, analyse_garment  # noqa: E402
from recommendation.knowledge import Knowledge, default_knowledge  # noqa: E402
from virtual_tryon.garment_segmentation import remove_background  # noqa: E402

CROP_MARGIN = 0.06
LONG_SIDE = 1024


def crop_to_garment(image: Image.Image, delta_e: float = 18.0) -> Image.Image:
    """Bounding box of non-background pixels, with a margin. Returns the image unchanged if the
    foreground is implausible (almost nothing, or almost everything)."""
    small = image.copy()
    small.thumbnail((256, 256))
    lab = _to_lab(np.asarray(small.convert("RGB")))
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    mask = np.linalg.norm(lab - np.median(border, axis=0), axis=-1) > delta_e
    if not 0.05 < mask.mean() < 0.95:
        return image
    ys, xs = np.where(mask)
    # robust box: ignore the outermost 0.5% of foreground pixels (specks, shadows)
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    sx, sy = image.width / small.width, image.height / small.height
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    box = (max(0, int((x0 - mx) * sx)), max(0, int((y0 - my) * sy)),
           min(image.width, int((x1 + mx) * sx)), min(image.height, int((y1 + my) * sy)))
    return image.crop(box)


def derive_occasions(garment: Garment, kb: Knowledge) -> tuple[tuple, tuple]:
    """Default occasion tags from the knowledge base (reviewed by a human afterwards)."""
    primary, secondary = [], []
    for key, occ in kb.occasions.items():
        if garment.garment_type in occ.discouraged_types or garment.garment_type in occ.excluded_types:
            continue
        gap = abs(garment.formality - occ.formality)
        if set(garment.styles) & occ.preferred_styles and gap <= 1.0:
            primary.append(key)
        elif gap <= 1.5:
            secondary.append(key)
    return tuple(primary), tuple(secondary)


def contact_sheet(garments: list[Garment], path: Path, thumb: int = 220) -> None:
    cols = 5
    rows = max(1, -(-len(garments) // cols))
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 60)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, g in enumerate(garments):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 60)
        img = Image.open(g.image_path).convert("RGB")
        img.thumbnail((thumb - 10, thumb - 10))
        sheet.paste(img, (x + (thumb - img.width) // 2, y + 5))
        draw.text((x + 5, y + thumb), f"{g.id} [{g.annotation}]", fill="black")
        draw.text((x + 5, y + thumb + 15), f"{g.garment_type} f{g.formality:g} w{g.warmth:g}", fill="black")
        draw.text((x + 5, y + thumb + 30), ",".join(g.colours), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--prefix", default="own")
    parser.add_argument("--wear-line", default="unisex", choices=["menswear", "womenswear", "unisex"])
    parser.add_argument("--no-clip", action="store_true", help="skip CLIP; garment type must then be edited by hand")
    args = parser.parse_args()

    kb = default_knowledge()
    settings.CATALOGUE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_catalogue() if settings.CATALOGUE_CSV.exists() else []
    known_hashes = {g.notes.split("sha1=")[-1][:12] for g in existing if "sha1=" in g.notes}
    next_number = 1 + max([int(g.id.split("_")[-1]) for g in existing if g.id.startswith(args.prefix + "_")] or [0])

    scorer = None
    if not args.no_clip:
        from recommendation.clip_encoder import ClipEncoder, ClipScorer
        scorer = ClipScorer(ClipEncoder(), kb)

    added = []
    for path in sorted(Path(args.input).iterdir()):
        if path.suffix.lower() not in settings.ALLOWED_EXTENSIONS:
            if path.is_file():
                print(f"skip {path.name}: unsupported type (convert HEIC photos to JPG first)")
            continue
        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()[:12]
        if digest in known_hashes:
            continue
        try:
            image = validate_image_bytes(data, path.name, "garment").image
        except ImageValidationError as exc:
            print(f"skip {path.name}: {exc.user_message}")
            continue
        segmentation = remove_background(image)
        image = segmentation.image if segmentation.applied else crop_to_garment(image)
        image.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
        garment_id = f"{args.prefix}_{next_number:03d}"
        next_number += 1
        image.save(settings.CATALOGUE_IMAGE_DIR / f"{garment_id}.jpg", quality=92)

        if scorer is not None:
            analysis = analyse_garment(image, clip_scorer=scorer, kb=kb)
            candidates = ", ".join(f"{t} {p:.2f}" for t, p in analysis.type_candidates)
        else:
            analysis = analyse_garment(image, garment_type="t_shirt", kb=kb)
            candidates = "no CLIP: type defaulted to t_shirt, EDIT"
        g = analysis.garment
        primary, secondary = derive_occasions(g, kb)
        colour_word = g.colours[0].replace("_", " ") if g.colours else ""
        g.id, g.image_file = garment_id, f"{garment_id}.jpg"
        g.name = f"{colour_word.capitalize()} {kb.garment_types[g.garment_type].label.lower()}".strip()
        g.occasions_primary, g.occasions_secondary = primary, secondary
        g.wear_line, g.source, g.license = args.wear_line, args.source, args.license
        g.annotation, g.estimated = "auto", False
        g.notes = (f"from {path.name}; background: {segmentation.message}; "
                   f"CLIP type candidates: {candidates}; sha1={digest}")
        added.append(g)
        print(f"{garment_id}  {path.name:<28} -> {g.garment_type:<14} colours={g.colours}  ({candidates})")

    if added:
        save_catalogue(existing + added)
    all_garments = existing + added
    if all_garments:
        sheet = settings.OUTPUT_DIR / "catalogue_review.jpg"
        contact_sheet(all_garments, sheet)
        print(f"\n{len(added)} added, {len(all_garments)} total. Review sheet: {sheet}")
        print("Check every 'auto' row in the CSV, correct it, then set annotation=verified.")
    else:
        print("Nothing to add.")
    assert COLUMNS  # keep import used
    return 0


if __name__ == "__main__":
    sys.exit(main())
