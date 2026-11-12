"""CLIP image-text model for the recommender (runs on the laptop CPU).

CLIP (Radford et al., 2021) embeds images and sentences into the same vector space, trained so
that a photo and its caption point in similar directions. We use it for zero-shot
classification: embed the garment photo, embed one sentence per class ("formal professional
outfit for a job interview", ...), take cosine similarities, scale by CLIP's learned
temperature and apply softmax. The result is a probability over classes without any training.

Used for:
  1. the `clip` ranking factor (which occasion / weather does the image look most typical of),
  2. analysing uploaded garments (which garment type is it?).

Prompt ensembling: each class sentence is embedded with several templates ("a photo of X",
"a product photo of X", ...) and the normalised embeddings are averaged - a standard trick
from the CLIP paper that makes zero-shot predictions less sensitive to wording.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from config import settings

from .catalogue import Garment
from .knowledge import Knowledge, default_knowledge

log = logging.getLogger(__name__)

TEMPLATES = ("a photo of {}.", "a product photo of {}.", "a photo of a garment: {}.")
GARMENT_TEMPLATES = ("a photo of {}.", "a product photo of {}.", "a flat-lay photo of {}.")


def _features(output) -> "torch.Tensor":
    """transformers returns a tensor (v4) or a model output object (some v5 paths); accept both."""
    import torch
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    raise TypeError(f"Unexpected CLIP output type {type(output)}")


class ClipEncoder:
    def __init__(self, model_name: str = settings.CLIP_MODEL_NAME, device: Optional[str] = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.logit_scale = float(self.model.logit_scale.exp().item())   # learned temperature (~100)
        self._lock = threading.Lock()

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        import torch
        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = [im.convert("RGB") for im in images[i:i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out.append(_features(self.model.get_image_features(**inputs)).float().cpu().numpy())
        return _l2(np.concatenate(out))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        with self._lock, torch.inference_mode():
            inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
            return _l2(_features(self.model.get_text_features(**inputs)).float().cpu().numpy())

    def encode_classes(self, phrases: Sequence[str], templates: Sequence[str] = TEMPLATES) -> np.ndarray:
        """One prompt-ensembled embedding per phrase."""
        per_template = [self.encode_texts([t.format(p) for p in phrases]) for t in templates]
        return _l2(np.mean(per_template, axis=0))

    def zero_shot(self, image_embeddings: np.ndarray, class_embeddings: np.ndarray) -> np.ndarray:
        """Softmax over classes of scaled cosine similarity. Shapes: (N, D) x (C, D) -> (N, C)."""
        logits = self.logit_scale * image_embeddings @ class_embeddings.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


class EmbeddingCache:
    """Image embeddings on disk, keyed by model name + SHA-1 of the image bytes.
    Editing or replacing a catalogue photo automatically invalidates its entry."""

    def __init__(self, model_name: str, folder: Path = settings.EMBEDDING_CACHE):
        self.path = Path(folder) / (model_name.replace("/", "__") + ".npz")
        self._data: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as npz:
                self._data = {k: npz[k] for k in npz.files}

    @staticmethod
    def key(image_bytes: bytes) -> str:
        return hashlib.sha1(image_bytes).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._data.get(key)

    def put(self, key: str, embedding: np.ndarray) -> None:
        self._data[key] = embedding

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self._data)


class ClipScorer:
    """Precomputes class embeddings once; scores garments (catalogue or uploaded)."""

    def __init__(self, encoder: ClipEncoder, kb: Optional[Knowledge] = None, cache: Optional[EmbeddingCache] = None):
        self.encoder = encoder
        self.kb = kb or default_knowledge()
        self.cache = cache if cache is not None else EmbeddingCache(encoder.model_name)
        self.occasion_keys = list(self.kb.occasions)
        self.weather_keys = list(self.kb.weather)
        self.type_keys = list(self.kb.garment_types)
        self.occasion_emb = encoder.encode_classes([self.kb.occasions[k].clip_prompt for k in self.occasion_keys])
        self.weather_emb = encoder.encode_classes([self.kb.weather[k].clip_prompt for k in self.weather_keys])
        self.type_emb = encoder.encode_classes([self.kb.garment_types[k].clip_label for k in self.type_keys],
                                               GARMENT_TEMPLATES)

    # -- embeddings ---------------------------------------------------------
    def embed(self, image: Image.Image, image_bytes: Optional[bytes] = None) -> np.ndarray:
        if image_bytes is None:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        key = EmbeddingCache.key(image_bytes)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        embedding = self.encoder.encode_images([image])[0]
        self.cache.put(key, embedding)
        return embedding

    def embed_catalogue(self, garments: list[Garment]) -> int:
        """Embed all catalogue images not yet cached; returns how many were computed."""
        computed = 0
        for g in garments:
            if g.image_path and g.image_path.exists():
                data = g.image_path.read_bytes()
                if self.cache.get(EmbeddingCache.key(data)) is None:
                    self.embed(Image.open(io.BytesIO(data)), data)
                    computed += 1
        if computed:
            self.cache.save()
        return computed

    # -- probabilities ------------------------------------------------------
    def probs_from_embedding(self, embedding: np.ndarray) -> dict:
        e = embedding[None, :]
        occ = self.encoder.zero_shot(e, self.occasion_emb)[0]
        wx = self.encoder.zero_shot(e, self.weather_emb)[0]
        return {"occasion": dict(zip(self.occasion_keys, occ.tolist())),
                "weather": dict(zip(self.weather_keys, wx.tolist()))}

    def garment_probs_for(self, garment: Garment) -> Optional[dict]:
        """Used by the engine for catalogue garments (cached by image hash)."""
        if not (garment.image_path and garment.image_path.exists()):
            return None
        data = garment.image_path.read_bytes()
        return self.probs_from_embedding(self.embed(Image.open(io.BytesIO(data)), data))

    def garment_type_probs(self, embedding: np.ndarray) -> dict:
        probs = self.encoder.zero_shot(embedding[None, :], self.type_emb)[0]
        return dict(sorted(zip(self.type_keys, probs.tolist()), key=lambda kv: -kv[1]))
