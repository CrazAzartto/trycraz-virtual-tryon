"""Central configuration for the laptop application.

All tunable values live here (or in recommendation.yaml) so application logic never
hard-codes paths, URLs or model parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CATALOGUE_DIR = DATA_DIR / "catalogue"
CATALOGUE_CSV = CATALOGUE_DIR / "clothing_catalogue.csv"
CATALOGUE_IMAGE_DIR = CATALOGUE_DIR / "images"
EMBEDDING_CACHE = CATALOGUE_DIR / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"            # small local models (git-ignored)
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLE_PEOPLE_DIR = ASSETS_DIR / "sample_people"
DOCS_DIR = PROJECT_ROOT / "docs"
RECOMMENDATION_CONFIG = PROJECT_ROOT / "config" / "recommendation.yaml"

# ---------------------------------------------------------------------------
# Try-on server (GPU backend)
# ---------------------------------------------------------------------------
# The Colab notebook prints a public URL. It can be set here through the
# TRYON_SERVER_URL environment variable or typed into the app's sidebar.
TRYON_SERVER_URL = os.environ.get("TRYON_SERVER_URL", "").strip()
TRYON_API_KEY = os.environ.get("TRYON_API_KEY", "").strip()
TRYON_BACKEND = os.environ.get("TRYON_BACKEND", "remote")  # "remote" | "local"
HEALTH_TIMEOUT_S = 10
TRYON_TIMEOUT_S = 300          # first request on a cold server can be slow

# ---------------------------------------------------------------------------
# Try-on model parameters (CatVTON)
# ---------------------------------------------------------------------------
MODEL_WIDTH = 768
MODEL_HEIGHT = 1024            # CatVTON's native 3:4 resolution
GARMENT_TYPES = ("upper", "lower", "overall")
DEFAULT_STEPS = 30             # CatVTON default is 50; 30 is our "fast" default
MAX_STEPS = 100
DEFAULT_GUIDANCE = 2.5         # CatVTON default classifier-free guidance
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ALLOWED_PIL_FORMATS = ("JPEG", "PNG", "WEBP", "MPO")  # MPO = JPEG from some phone cameras
MAX_UPLOAD_MB = 15
MIN_SIDE_PX = 256              # smaller images carry too little detail for a 1024 px output
MAX_SIDE_PX = 8000
MAX_ASPECT_RATIO = 3.0         # long side / short side

# ---------------------------------------------------------------------------
# Recommendation / CLIP
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEFAULT_TOP_K = 6

# ---------------------------------------------------------------------------
# Quality gate (MediaPipe Pose)
# ---------------------------------------------------------------------------
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
