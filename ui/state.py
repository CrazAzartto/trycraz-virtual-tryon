"""Cached resources and session state shared by all pages.

Streamlit re-runs a page script on every interaction. Heavy objects (catalogue, CLIP) are
created once per server process with st.cache_resource; per-user values (server URL, chosen
garment, last result) live in st.session_state.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from PIL import Image

from config import settings
from recommendation.catalogue import CatalogueError, Garment, load_catalogue
from recommendation.recommendation_engine import RecommendationEngine
from virtual_tryon.backends.remote import RemoteBackend

# session_state keys
SERVER_URL = "vto_server_url"
API_KEY = "vto_api_key"
SERVER_STATUS = "vto_server_status"
USE_CLIP = "vto_use_clip"
CLIP_ERROR = "vto_clip_error"                  # set once if CLIP failed to load (not retried every rerun)
TRYON_GARMENT_ID = "vto_tryon_garment_id"      # set by "Try this on" on the recommendation page
CONTEXT = "vto_context"                        # last Context used on the recommendation page
LAST_OUTCOME = "vto_last_outcome"
LAST_RECOMMENDATIONS = "vto_last_recommendations"
AUDIENCE = "vto_audience"                      # "menswear" | "womenswear" | None (chosen on the pages)


def init_session() -> None:
    defaults = {
        SERVER_URL: settings.TRYON_SERVER_URL, API_KEY: settings.TRYON_API_KEY, SERVER_STATUS: None,
        USE_CLIP: True, CLIP_ERROR: None, TRYON_GARMENT_ID: None, CONTEXT: None, LAST_OUTCOME: None, LAST_RECOMMENDATIONS: None,
        AUDIENCE: None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the clothing catalogue...")
def catalogue() -> list[Garment]:
    return load_catalogue()


def catalogue_or_error() -> tuple[list[Garment], Optional[str]]:
    try:
        return catalogue(), None
    except FileNotFoundError:
        return [], "The catalogue file data/catalogue/clothing_catalogue.csv does not exist yet."
    except CatalogueError as exc:
        return [], "The catalogue has errors:\n\n" + "\n".join(f"- {p}" for p in exc.problems[:10])


@st.cache_resource(show_spinner="Loading CLIP (first time on this laptop: 1-2 minutes)...")
def clip_scorer():
    from recommendation.clip_encoder import ClipEncoder, ClipScorer
    scorer = ClipScorer(ClipEncoder())
    scorer.embed_catalogue(catalogue())
    return scorer


def clip_or_none():
    """CLIP scorer if enabled and loadable; otherwise None (the recommender still works)."""
    if not st.session_state.get(USE_CLIP) or st.session_state.get(CLIP_ERROR):
        return None
    try:
        return clip_scorer()
    except Exception as exc:   # no internet on first run, out of memory, ...
        # The toggle widget's own key cannot be changed after it is drawn, so remember the failure separately.
        st.session_state[CLIP_ERROR] = f"{type(exc).__name__}: {str(exc)[:120]}"
        st.warning("CLIP could not be loaded, so recommendations continue without it (see sidebar).")
        return None


@st.cache_resource
def _engine(with_clip: bool) -> RecommendationEngine:
    return RecommendationEngine(catalogue(), clip_scorer=clip_scorer() if with_clip else None)


def engine() -> RecommendationEngine:
    return _engine(clip_or_none() is not None)


def garment_by_id(garment_id: Optional[str]) -> Optional[Garment]:
    garments, _ = catalogue_or_error()
    return next((g for g in garments if g.id == garment_id), None)


# ---------------------------------------------------------------------------
# Try-on backend
# ---------------------------------------------------------------------------
def backend() -> RemoteBackend:
    return RemoteBackend(st.session_state[SERVER_URL], st.session_state[API_KEY])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def image_digest(image: Image.Image) -> str:
    return hashlib.sha1(image.tobytes() + str(image.size).encode()).hexdigest()


def to_bytes(image: Image.Image, fmt: str = "PNG", quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


@dataclass
class StoredOutcome:
    """What the try-on page keeps between reruns (plain bytes, cheap to hold in session)."""
    person_png: bytes
    result_png: bytes
    mask_png: bytes
    garment_png: bytes
    garment_label: str
    garment_id: Optional[str]
    engine: str
    timings: dict
    round_trip_s: Optional[float]
    peak_gpu_memory_gb: Optional[float]
    params: dict
    saved_folder: Optional[str]
