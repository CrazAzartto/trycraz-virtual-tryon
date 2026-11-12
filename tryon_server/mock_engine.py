"""TEST DOUBLE - not a try-on method.

A stand-in engine used only by automated tests and for developing the client/UI without a GPU.
It returns a crude 50/50 blend so tests can check the HTTP contract, image round-trips, error
handling and progress reporting. It is never presented as a try-on result: the server reports
engine="MOCK (test only)" in /health and in every response, and the app shows a warning banner.
"""

from __future__ import annotations

import time

from PIL import Image

from .catvton_engine import GARMENT_TYPES, InvalidRequestError, TryOnOutput, TryOnParams


class MockEngine:
    name = "MOCK (test only)"

    def __init__(self, width: int = 768, height: int = 1024, step_delay_s: float = 0.0, fail_with: Exception | None = None):
        self.width, self.height = width, height
        self.step_delay_s = step_delay_s
        self.fail_with = fail_with

    def info(self) -> dict:
        return {"engine": self.name, "resolution": [self.width, self.height], "garment_types": list(GARMENT_TYPES)}

    def generate(self, person, garment, params: TryOnParams, user_mask=None, progress=None) -> TryOnOutput:
        params.validate()
        progress = progress or (lambda *a: None)
        if self.fail_with:
            raise self.fail_with
        size = (self.width, self.height)
        person = person.convert("RGB").resize(size)
        garment = garment.convert("RGB").resize(size)
        for i in range(1, params.steps + 1):
            if self.step_delay_s:
                time.sleep(self.step_delay_s)
            progress("diffusion", i, params.steps)
        mask = Image.new("L", size, 0)
        return TryOnOutput(
            result=Image.blend(person, garment, 0.5), mask=mask,
            person_input=person, garment_input=garment,
            timings={"total_s": 0.0}, seed_used=params.seed,
        )


__all__ = ["MockEngine", "InvalidRequestError"]
