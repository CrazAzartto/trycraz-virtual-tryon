"""In-process backend: runs the engine in the same Python process as the app.

Intended for a machine with a Linux NVIDIA GPU (>= 8 GB). Not usable on the project laptop
(4 GB VRAM, Windows) - see docs/phase1_feasibility.md section 7. Also used by tests with the
MockEngine, which exercises exactly the same code path without a GPU.
"""

from __future__ import annotations

import time
from typing import Optional

from tryon_server.catvton_engine import EngineError, GPUOutOfMemoryError, InvalidRequestError, TryOnParams

from .base import (
    BackendStatus, GenerationFailed, GPUOutOfMemory, InvalidRequest, ProgressCallback, TryOnBackend, TryOnRequest,
    TryOnResult,
)


class LocalBackend(TryOnBackend):
    name = "local"

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def with_catvton(cls, catvton_repo: str, **config_kwargs) -> "LocalBackend":
        from pathlib import Path
        from tryon_server.catvton_engine import CatVTONEngine, EngineConfig
        return cls(CatVTONEngine(EngineConfig(catvton_repo=Path(catvton_repo), **config_kwargs)))

    def health(self) -> BackendStatus:
        info = self.engine.info()
        return BackendStatus(True, f"Local engine: {info.get('engine')}", info)

    def run(self, request: TryOnRequest, progress: Optional[ProgressCallback] = None) -> TryOnResult:
        params = TryOnParams(request.garment_type, request.steps, request.guidance, request.seed, request.repaint)
        start = time.perf_counter()
        try:
            out = self.engine.generate(request.person, request.garment, params, request.mask, progress)
        except InvalidRequestError as exc:
            raise InvalidRequest(str(exc))
        except GPUOutOfMemoryError as exc:
            raise GPUOutOfMemory(detail=str(exc))
        except EngineError as exc:
            raise GenerationFailed(detail=str(exc))
        return TryOnResult(
            image=out.result, mask=out.mask, engine=self.engine.info().get("engine", "local"),
            timings=out.timings, params=vars(params), seed_used=out.seed_used,
            peak_gpu_memory_gb=out.peak_gpu_memory_gb, round_trip_s=round(time.perf_counter() - start, 1),
        )
