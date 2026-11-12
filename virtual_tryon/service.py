"""Try-on service: the one function the UI calls.

    validate -> (quality report) -> preprocess -> backend.run -> save

Kept separate from Streamlit so it can be unit-tested and reused by the evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from common.validation import validate_garment_type, validate_generation_params

from .backends.base import ProgressCallback, TryOnBackend, TryOnRequest, TryOnResult
from .postprocessing import SavedRun, save_run
from .preprocessing import PreparedPair, prepare_pair


@dataclass
class TryOnOutcome:
    prepared: PreparedPair
    result: TryOnResult
    saved: Optional[SavedRun]


class TryOnService:
    def __init__(self, backend: TryOnBackend, save_results: bool = True):
        self.backend = backend
        self.save_results = save_results

    def run(
        self,
        person: Image.Image,
        garment: Image.Image,
        garment_type: str,
        steps: int,
        guidance: float,
        seed: int,
        repaint: bool = True,
        garment_label: str = "uploaded garment",
        garment_id: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> TryOnOutcome:
        """`person` and `garment` must already have passed utils.validation.validate_image_bytes."""
        validate_garment_type(garment_type)
        validate_generation_params(steps, guidance, seed)
        prepared = prepare_pair(person, garment)
        request = TryOnRequest(
            person=prepared.person, garment=prepared.garment, garment_type=garment_type,
            steps=steps, guidance=guidance, seed=seed, repaint=repaint,
        )
        result = self.backend.run(request, progress)
        result.image = result.image.resize(prepared.person_preview.size)   # guard: keep pixel alignment
        saved = None
        # Never put test-double output into the results gallery.
        if self.save_results and "MOCK" not in result.engine:
            saved = save_run(prepared.person_preview, prepared.garment, result, garment_label, garment_id)
        return TryOnOutcome(prepared, result, saved)
