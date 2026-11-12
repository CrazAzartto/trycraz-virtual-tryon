"""CatVTON inference engine.

This module is the only place that touches the pretrained CatVTON code. It runs on the
GPU machine (Google Colab), never on the laptop.

What CatVTON does (pretrained, not ours):
    1. AutoMasker: DensePose (body-part map) + SCHP-LIP + SCHP-ATR (human parsing) are fused
       by hand-written rules into a "clothing-agnostic" mask: the region where the old garment
       is and the new one should go, while face, hands, hair and other garments are protected.
    2. The masked person image and the garment image are encoded by the SD VAE into latents
       and concatenated side by side (along the height axis).
    3. A Stable Diffusion 1.5 inpainting UNet (cross-attention removed, self-attention
       fine-tuned by the CatVTON authors) denoises the concatenated latent for N DDIM steps.
       Self-attention lets the person half attend to the garment half, which replaces the
       explicit warping stage of older try-on methods.
    4. The person half of the latent is decoded back to an image.

What this file adds (ours):
    - Reproducible, explicit loading with a pinned attention checkpoint and dtype selection
      (bf16 on Ampere+, fp16 on T4).
    - Downloads only the weights that are actually used.
    - Optional repainting of unmasked pixels from the original photo (identity/background
      preservation), per-stage timings, peak-GPU-memory measurement, step-level progress
      reporting, and typed errors.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("tryon.engine")

GARMENT_TYPES = ("upper", "lower", "overall")
ATTN_VERSIONS = {"mix": "mix-48k-1024", "vitonhd": "vitonhd-16k-512", "dresscode": "dresscode-16k-512"}

ProgressCallback = Callable[[str, int, int], None]  # (stage, step, total_steps)


# ---------------------------------------------------------------------------
# Errors (mapped to HTTP status codes by server.py)
# ---------------------------------------------------------------------------
class EngineError(Exception):
    """Base class for errors the engine reports to the client."""

    http_status = 500


class InvalidRequestError(EngineError):
    http_status = 422


class GPUOutOfMemoryError(EngineError):
    http_status = 507


class GenerationError(EngineError):
    http_status = 500


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    catvton_repo: Path                         # local clone of github.com/Zheng-Chong/CatVTON
    base_model: str = "booksforcharlie/stable-diffusion-inpainting"  # mirror used by CatVTON's own app.py
    attn_repo: str = "zhengchong/CatVTON"
    attn_version: str = "mix"                  # "mix" = trained on VITON-HD + DressCode + DeepFashion at 1024
    precision: str = "auto"                    # "auto" | "fp16" | "bf16" | "fp32"
    width: int = 768
    height: int = 1024
    skip_safety_check: bool = False
    device: str = "cuda"


@dataclass
class TryOnParams:
    garment_type: str = "upper"
    steps: int = 30
    guidance: float = 2.5
    seed: int = 42
    repaint: bool = True       # copy original pixels back outside the (blurred) mask

    def validate(self) -> None:
        if self.garment_type not in GARMENT_TYPES:
            raise InvalidRequestError(f"garment_type must be one of {GARMENT_TYPES}, got '{self.garment_type}'")
        if not 5 <= self.steps <= 100:
            raise InvalidRequestError(f"steps must be between 5 and 100, got {self.steps}")
        if not 0.0 <= self.guidance <= 7.5:
            raise InvalidRequestError(f"guidance must be between 0 and 7.5, got {self.guidance}")
        if not -1 <= self.seed <= 2**31 - 1:
            raise InvalidRequestError("seed must be -1 (random) or a non-negative 32-bit integer")


@dataclass
class TryOnOutput:
    result: Image.Image
    mask: Image.Image
    person_input: Image.Image          # the exact 768x1024 person image the model saw
    garment_input: Image.Image         # the exact padded garment image the model saw
    timings: dict = field(default_factory=dict)
    peak_gpu_memory_gb: Optional[float] = None
    seed_used: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def trusted_torch_load():
    """Temporarily make torch.load default to weights_only=False.

    PyTorch >= 2.6 changed the default of torch.load to weights_only=True. CatVTON's SCHP loader
    calls torch.load(path) without the argument, and the official SCHP checkpoints may contain
    non-tensor Python objects. We only load the official checkpoints downloaded from the
    authors' Hugging Face repository, so disabling the restriction inside this block is safe.
    """
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def select_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "fp32" or device == "cpu":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto: bf16 needs compute capability >= 8.0 (Ampere). A Colab T4 is 7.5 -> fp16.
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class CatVTONEngine:
    name = "CatVTON"

    def __init__(self, config: EngineConfig):
        self.config = config
        self._lock = threading.Lock()   # one GPU, one generation at a time
        self.dtype = select_dtype(config.precision, config.device)

        repo = Path(config.catvton_repo).resolve()
        if not (repo / "model" / "pipeline.py").exists():
            raise FileNotFoundError(f"CatVTON repository not found at {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))   # CatVTON uses top-level imports: model.*, utils, detectron2, densepose
        # CatVTON's code does `from utils import ...` meaning ITS utils.py. If some other module named
        # `utils` was imported earlier in this process, Python would reuse it and CatVTON would crash
        # with a confusing ImportError. (This is why our own helper package is called `common`.)
        for name in ("utils", "model"):
            loaded = sys.modules.get(name)
            if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(str(repo)):
                raise ImportError(
                    f"A different module named '{name}' ({getattr(loaded, '__file__', '?')}) is already imported; "
                    "it shadows CatVTON's own module. Rename it or load the engine in a fresh process."
                )

        # Imported here, not at module top, because they only exist after the repo is on sys.path.
        from diffusers.image_processor import VaeImageProcessor
        from huggingface_hub import snapshot_download
        from model.cloth_masker import AutoMasker
        from model.pipeline import CatVTONPipeline
        from utils import repaint_result, resize_and_crop, resize_and_padding

        self._resize_and_crop = resize_and_crop
        self._resize_and_padding = resize_and_padding
        self._repaint_result = repaint_result

        t0 = time.perf_counter()
        sub = ATTN_VERSIONS[config.attn_version]
        log.info("Downloading CatVTON weights (%s, DensePose, SCHP) ...", sub)
        attn_path = snapshot_download(
            repo_id=config.attn_repo,
            allow_patterns=[f"{sub}/*", "DensePose/*", "SCHP/*"],   # skip the two unused attention variants
        )
        log.info("Loading CatVTON pipeline (dtype=%s) ...", self.dtype)
        with trusted_torch_load():
            self.pipeline = CatVTONPipeline(
                base_ckpt=config.base_model,
                attn_ckpt=attn_path,
                attn_ckpt_version=config.attn_version,
                weight_dtype=self.dtype,
                device=config.device,
                skip_safety_check=config.skip_safety_check,
            )
            self.automasker = AutoMasker(
                densepose_ckpt=str(Path(attn_path) / "DensePose"),
                schp_ckpt=str(Path(attn_path) / "SCHP"),
                device=config.device,
            )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.load_seconds = time.perf_counter() - t0
        log.info("CatVTON ready in %.1f s", self.load_seconds)

    # -- public API ---------------------------------------------------------
    def info(self) -> dict:
        info = {
            "engine": self.name,
            "attn_version": self.config.attn_version,
            "base_model": self.config.base_model,
            "dtype": str(self.dtype).replace("torch.", ""),
            "resolution": [self.config.width, self.config.height],
            "garment_types": list(GARMENT_TYPES),
            "safety_checker": not self.config.skip_safety_check,
            "load_seconds": round(self.load_seconds, 1),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        return info

    def generate(
        self,
        person: Image.Image,
        garment: Image.Image,
        params: TryOnParams,
        user_mask: Optional[Image.Image] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> TryOnOutput:
        params.validate()
        progress = progress or (lambda stage, step, total: None)
        with self._lock:
            try:
                return self._generate_locked(person, garment, params, user_mask, progress)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise GPUOutOfMemoryError(
                    "The GPU ran out of memory. Try fewer steps, or restart the Colab runtime."
                ) from exc
            except EngineError:
                raise
            except Exception as exc:  # anything else from CatVTON / detectron2
                log.exception("Generation failed")
                raise GenerationError(f"Generation failed: {type(exc).__name__}: {exc}") from exc

    # -- internals ----------------------------------------------------------
    def _generate_locked(self, person, garment, params, user_mask, progress) -> TryOnOutput:
        cfg = self.config
        size = (cfg.width, cfg.height)
        timings = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 1. Preprocessing: identical to CatVTON's app.py
        t = time.perf_counter()
        person = self._resize_and_crop(person.convert("RGB"), size)       # centre-crop to 3:4, resize
        garment = self._resize_and_padding(garment.convert("RGB"), size)  # fit inside 3:4, pad white
        timings["preprocess_s"] = time.perf_counter() - t

        # 2. Clothing-agnostic mask (DensePose + SCHP), or a user-drawn mask
        progress("masking", 0, params.steps)
        t = time.perf_counter()
        if user_mask is not None:
            mask = self._resize_and_crop(user_mask.convert("L"), size)
            mask = Image.fromarray(np.where(np.array(mask) > 0, 255, 0).astype(np.uint8))
        else:
            mask = self.automasker(person, params.garment_type)["mask"]
        if np.array(mask).max() == 0:
            raise GenerationError(
                "No clothing region was found for the selected garment type. "
                "Check that the person is clearly visible and the garment type is correct."
            )
        blurred_mask = self.mask_processor.blur(mask, blur_factor=9)
        timings["mask_s"] = time.perf_counter() - t

        # 3. Diffusion with step-level progress
        seed = params.seed if params.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator(device=cfg.device).manual_seed(seed)
        t = time.perf_counter()
        with self._count_steps(params.steps, progress):
            result = self.pipeline(
                image=person,
                condition_image=garment,
                mask=blurred_mask,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["diffusion_s"] = time.perf_counter() - t

        # 4. Post-processing
        if params.repaint:
            result = self._repaint_result(result, person, blurred_mask)
        timings["total_s"] = sum(timings.values())
        timings = {k: round(v, 2) for k, v in timings.items()}

        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        progress("done", params.steps, params.steps)
        return TryOnOutput(
            result=result, mask=mask, person_input=person, garment_input=garment,
            timings=timings, peak_gpu_memory_gb=round(peak, 2) if peak else None, seed_used=seed,
        )

    @contextlib.contextmanager
    def _count_steps(self, total: int, progress: ProgressCallback):
        """Report denoising progress by wrapping the scheduler's step() method.

        CatVTONPipeline has no callback argument; every denoising iteration calls
        noise_scheduler.step() exactly once, so counting those calls gives the step number.
        Safe because generate() holds a lock (one job at a time).
        """
        scheduler = self.pipeline.noise_scheduler
        original_step = scheduler.step
        counter = {"n": 0}

        def counting_step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            counter["n"] += 1
            progress("diffusion", counter["n"], total)
            return out

        scheduler.step = counting_step
        try:
            yield
        finally:
            scheduler.step = original_step
