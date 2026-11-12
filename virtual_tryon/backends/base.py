"""Backend interface: the UI and service layer depend only on this, never on CatVTON directly.

Swapping Colab for Kaggle, a local GPU, or a different model (e.g. Leffa) means writing one
new subclass; the recommender and UI do not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

ProgressCallback = Callable[[str, int, int], None]   # (stage, step, total)


# ---------------------------------------------------------------------------
# Errors. Each carries a message that can be shown to the user as-is.
# ---------------------------------------------------------------------------
class TryOnError(Exception):
    user_message = "The try-on failed."

    def __init__(self, user_message: Optional[str] = None, detail: Optional[str] = None):
        if user_message:
            self.user_message = user_message
        self.detail = detail
        super().__init__(self.user_message + (f" ({detail})" if detail else ""))


class BackendNotConfigured(TryOnError):
    user_message = "No try-on server is set. Start the Colab notebook and paste its URL and key in the sidebar."


class BackendUnavailable(TryOnError):
    user_message = "Cannot reach the try-on server. The Colab session may have stopped or the URL changed."


class AuthenticationFailed(TryOnError):
    user_message = "The try-on server rejected the API key. Copy the key printed by the Colab notebook."


class InvalidRequest(TryOnError):
    user_message = "The try-on server rejected the request."


class GPUOutOfMemory(TryOnError):
    user_message = "The GPU ran out of memory. Try fewer steps, or restart the Colab runtime."


class GenerationFailed(TryOnError):
    user_message = "The model could not generate an image for these inputs."


class ServerBusy(TryOnError):
    user_message = "The try-on server is busy. Please try again in a minute."


class TryOnTimeout(TryOnError):
    user_message = "The try-on took too long and was abandoned."


# ---------------------------------------------------------------------------
# Data passed across the interface
# ---------------------------------------------------------------------------
@dataclass
class TryOnRequest:
    person: Image.Image
    garment: Image.Image
    garment_type: str = "upper"
    steps: int = 30
    guidance: float = 2.5
    seed: int = 42
    repaint: bool = True
    mask: Optional[Image.Image] = None


@dataclass
class TryOnResult:
    image: Image.Image
    mask: Image.Image
    engine: str
    timings: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    seed_used: Optional[int] = None
    peak_gpu_memory_gb: Optional[float] = None
    round_trip_s: Optional[float] = None   # measured by the client (includes upload + queue)


@dataclass
class BackendStatus:
    reachable: bool
    message: str
    info: dict = field(default_factory=dict)

    @property
    def is_mock(self) -> bool:
        return "MOCK" in str(self.info.get("engine", ""))


class TryOnBackend(ABC):
    name: str = "backend"

    @abstractmethod
    def health(self) -> BackendStatus:
        """Cheap check that the backend can accept work."""

    @abstractmethod
    def run(self, request: TryOnRequest, progress: Optional[ProgressCallback] = None) -> TryOnResult:
        """Run one try-on. Raises a TryOnError subclass on failure."""
