"""HTTP API around the try-on engine (runs on the GPU machine).

Endpoints
    GET  /health          -> engine info, GPU, readiness (no API key needed)
    POST /jobs            -> submit a try-on job (multipart form), returns {job_id}
    GET  /jobs/{job_id}   -> status + progress; includes the result images when done

Images in responses are base64-encoded PNG. Every request except /health must carry the
header `X-API-Key` if the server was started with an API key (the Colab notebook always
generates one, because the tunnel URL is public).

Run (on Colab, from the notebook):
    python -m tryon_server.server --catvton-repo /content/third_party/CatVTON --api-key <key>
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import secrets
import time
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .catvton_engine import EngineError, TryOnParams
from .jobs import Job, JobManager, QueueFullError

log = logging.getLogger("tryon.server")
API_VERSION = "1.0"


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_upload(data: bytes, field_name: str, max_bytes: int) -> Image.Image:
    if not data:
        raise HTTPException(400, f"The {field_name} image is empty.")
    if len(data) > max_bytes:
        raise HTTPException(413, f"The {field_name} image is larger than {max_bytes // 2**20} MB.")
    try:
        Image.open(io.BytesIO(data)).verify()   # detects truncated/corrupted files
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(400, f"The {field_name} file is not a valid image ({exc}).")
    return image


def create_app(engine, api_key: Optional[str] = None, max_upload_mb: int = 15) -> FastAPI:
    app = FastAPI(title="Virtual Try-On Server", version=API_VERSION)
    started = time.time()
    max_bytes = max_upload_mb * 2**20

    def run_job(job: Job):
        p = job.payload
        output = engine.generate(p["person"], p["garment"], p["params"], p.get("mask"), progress=job.progress)
        return {
            "engine": engine.info().get("engine"),
            "result_png": encode_png(output.result),
            "mask_png": encode_png(output.mask),
            "timings": output.timings,
            "peak_gpu_memory_gb": output.peak_gpu_memory_gb,
            "seed_used": output.seed_used,
            "params": vars(p["params"]),
        }

    jobs = JobManager(run_job)

    def require_key(x_api_key: Optional[str] = Header(default=None)):
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(401, "Missing or wrong API key.")

    @app.exception_handler(QueueFullError)
    async def _queue_full(_, exc: QueueFullError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "api_version": API_VERSION,
            "uptime_s": round(time.time() - started),
            "auth_required": bool(api_key),
            **engine.info(),
        }

    @app.post("/jobs", status_code=202, dependencies=[Depends(require_key)])
    async def submit(
        person: UploadFile = File(...),
        garment: UploadFile = File(...),
        garment_type: str = Form("upper"),
        steps: int = Form(30),
        guidance: float = Form(2.5),
        seed: int = Form(42),
        repaint: bool = Form(True),
        mask: Optional[UploadFile] = File(None),
    ):
        params = TryOnParams(garment_type=garment_type, steps=steps, guidance=guidance, seed=seed, repaint=repaint)
        try:
            params.validate()
        except EngineError as exc:
            raise HTTPException(exc.http_status, str(exc))
        payload = {
            "person": decode_upload(await person.read(), "person", max_bytes),
            "garment": decode_upload(await garment.read(), "garment", max_bytes),
            "params": params,
        }
        if mask is not None:
            mask_bytes = await mask.read()
            if mask_bytes:
                payload["mask"] = decode_upload(mask_bytes, "mask", max_bytes)
        job = jobs.submit(payload)
        return {"job_id": job.id, "status": job.status}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id (the server may have restarted).")
        body = job.public_status(jobs.queue_position(job))
        if job.status == "done":
            body["result"] = job.result
        return body

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual try-on inference server")
    parser.add_argument("--engine", choices=["catvton", "mock"], default="catvton")
    parser.add_argument("--catvton-repo", default="third_party/CatVTON")
    parser.add_argument("--attn-version", default="mix", choices=["mix", "vitonhd", "dresscode"])
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--skip-safety-check", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.engine == "mock":
        from .mock_engine import MockEngine
        engine = MockEngine(step_delay_s=0.05)
        log.warning("Running the MOCK engine: outputs are NOT try-on results.")
    else:
        from pathlib import Path
        from .catvton_engine import CatVTONEngine, EngineConfig
        engine = CatVTONEngine(EngineConfig(
            catvton_repo=Path(args.catvton_repo), attn_version=args.attn_version,
            precision=args.precision, skip_safety_check=args.skip_safety_check,
        ))

    import uvicorn
    uvicorn.run(create_app(engine, api_key=args.api_key), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
