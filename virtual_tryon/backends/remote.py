"""HTTP client for tryon_server (running on Colab behind a Cloudflare tunnel)."""

from __future__ import annotations

import time
from typing import Optional

import requests

from config import settings
from common.image_utils import b64_png_to_image, image_to_bytes

from .base import (
    AuthenticationFailed, BackendNotConfigured, BackendStatus, BackendUnavailable, GenerationFailed,
    GPUOutOfMemory, InvalidRequest, ProgressCallback, ServerBusy, TryOnBackend, TryOnError, TryOnRequest,
    TryOnResult, TryOnTimeout,
)

POLL_INTERVAL_S = 1.0


def normalise_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


class RemoteBackend(TryOnBackend):
    name = "remote"

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout_s: float = settings.TRYON_TIMEOUT_S,
                 session: Optional[requests.Session] = None):
        self.base_url = normalise_url(base_url)
        self.api_key = (api_key or "").strip() or None
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    # -- helpers ------------------------------------------------------------
    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(self, method: str, path: str, timeout: float, **kwargs) -> requests.Response:
        if not self.base_url:
            raise BackendNotConfigured()
        try:
            return self.session.request(method, self.base_url + path, headers=self._headers(), timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise BackendUnavailable("The try-on server did not answer in time.", detail=str(exc))
        except requests.exceptions.RequestException as exc:
            raise BackendUnavailable(detail=type(exc).__name__)

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:200]
        code = response.status_code
        if code == 401:
            raise AuthenticationFailed(detail=detail)
        if code in (400, 413, 422):
            raise InvalidRequest(str(detail), detail=f"HTTP {code}")
        if code == 503:
            raise ServerBusy(detail=detail)
        if code in (502, 504, 521, 522, 523, 524, 530):   # Cloudflare: origin down / tunnel gone
            raise BackendUnavailable(detail=f"HTTP {code}")
        raise TryOnError(f"The try-on server returned an error (HTTP {code}).", detail=str(detail))

    # -- interface ----------------------------------------------------------
    def health(self) -> BackendStatus:
        try:
            response = self._request("GET", "/health", timeout=settings.HEALTH_TIMEOUT_S)
            self._raise_for_status(response)
            info = response.json()
        except TryOnError as exc:
            return BackendStatus(False, exc.user_message)
        except ValueError:
            return BackendStatus(False, "The URL answered, but it is not a try-on server.")
        if info.get("auth_required"):
            # Verify the key too, so a wrong key is reported now rather than after an upload:
            # an unknown job id gives 404 with a valid key and 401 with a wrong one.
            try:
                probe = self._request("GET", "/jobs/key-check", timeout=settings.HEALTH_TIMEOUT_S)
            except TryOnError as exc:
                return BackendStatus(False, exc.user_message, info)
            if probe.status_code == 401:
                return BackendStatus(False, AuthenticationFailed.user_message, info)
        gpu = info.get("gpu", "no GPU reported")
        return BackendStatus(True, f"Connected: {info.get('engine')} on {gpu}", info)

    def run(self, request: TryOnRequest, progress: Optional[ProgressCallback] = None) -> TryOnResult:
        progress = progress or (lambda *a: None)
        start = time.perf_counter()
        files = {
            "person": ("person.jpg", image_to_bytes(request.person, "JPEG", 95), "image/jpeg"),
            "garment": ("garment.jpg", image_to_bytes(request.garment, "JPEG", 95), "image/jpeg"),
        }
        if request.mask is not None:
            files["mask"] = ("mask.png", image_to_bytes(request.mask.convert("L"), "PNG"), "image/png")
        form = {
            "garment_type": request.garment_type, "steps": str(request.steps), "guidance": str(request.guidance),
            "seed": str(request.seed), "repaint": "true" if request.repaint else "false",
        }
        progress("uploading", 0, request.steps)
        response = self._request("POST", "/jobs", timeout=60, files=files, data=form)
        self._raise_for_status(response)
        job_id = response.json()["job_id"]

        deadline = start + self.timeout_s
        while True:
            if time.perf_counter() > deadline:
                raise TryOnTimeout(detail=f"job {job_id}")
            time.sleep(POLL_INTERVAL_S)
            response = self._request("GET", f"/jobs/{job_id}", timeout=30)
            self._raise_for_status(response)
            status = response.json()
            if status["status"] == "queued":
                progress("queued", 0, request.steps)
            elif status["status"] == "running":
                progress(status.get("stage", "running"), status.get("step", 0), status.get("total_steps") or request.steps)
            elif status["status"] == "error":
                raise self._job_error(status)
            elif status["status"] == "done":
                return self._parse_result(status["result"], time.perf_counter() - start)

    @staticmethod
    def _job_error(status: dict) -> TryOnError:
        message, code = status.get("error", "unknown error"), status.get("error_status", 500)
        if code == 507:
            return GPUOutOfMemory(detail=message)
        if code == 422:
            return InvalidRequest(message)
        return GenerationFailed(message if "No clothing region" in message else None, detail=message)

    @staticmethod
    def _parse_result(result: dict, round_trip_s: float) -> TryOnResult:
        return TryOnResult(
            image=b64_png_to_image(result["result_png"]).convert("RGB"),
            mask=b64_png_to_image(result["mask_png"]).convert("L"),
            engine=result.get("engine", "unknown"),
            timings=result.get("timings", {}),
            params=result.get("params", {}),
            seed_used=result.get("seed_used"),
            peak_gpu_memory_gb=result.get("peak_gpu_memory_gb"),
            round_trip_s=round(round_trip_s, 1),
        )
