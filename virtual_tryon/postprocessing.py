"""Saving and presenting try-on results.

Every real result is saved with its inputs and a JSON record of parameters, timings and the
engine name. This gives (a) reproducibility for the report, (b) material for the human
rating study, and (c) the clearly-labelled "saved results" gallery used if Colab is down
during a demo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from config import settings
from common.image_utils import overlay_mask, side_by_side

from .backends.base import TryOnResult


@dataclass
class SavedRun:
    folder: Path
    record: dict


def comparison_sheet(person_preview: Image.Image, garment: Image.Image, result: TryOnResult) -> Image.Image:
    """Person | garment | mask overlay | result - used in the report and the download."""
    return side_by_side([person_preview, garment, overlay_mask(person_preview, result.mask), result.image], height=512)


def save_run(
    person_preview: Image.Image,
    garment: Image.Image,
    result: TryOnResult,
    garment_label: str,
    garment_id: Optional[str] = None,
    output_dir: Path = settings.OUTPUT_DIR,
) -> SavedRun:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(output_dir) / "runs" / f"{stamp}_{result.params.get('garment_type', 'x')}"
    suffix = 1
    while folder.exists():
        folder = folder.with_name(f"{folder.name}-{suffix}")
        suffix += 1
    folder.mkdir(parents=True)
    person_preview.save(folder / "person.jpg", quality=95)
    garment.save(folder / "garment.jpg", quality=95)
    result.image.save(folder / "result.png")
    result.mask.save(folder / "mask.png")
    record = {
        "timestamp": stamp,
        "engine": result.engine,
        "garment_label": garment_label,
        "garment_id": garment_id,
        "params": result.params,
        "seed_used": result.seed_used,
        "timings": result.timings,
        "round_trip_s": result.round_trip_s,
        "peak_gpu_memory_gb": result.peak_gpu_memory_gb,
    }
    (folder / "record.json").write_text(json.dumps(record, indent=2))
    return SavedRun(folder, record)


def list_saved_runs(output_dir: Path = settings.OUTPUT_DIR) -> list[SavedRun]:
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for folder in sorted(runs_dir.iterdir(), reverse=True):
        record_file = folder / "record.json"
        if record_file.exists() and (folder / "result.png").exists():
            try:
                runs.append(SavedRun(folder, json.loads(record_file.read_text())))
            except json.JSONDecodeError:
                continue
    return runs
