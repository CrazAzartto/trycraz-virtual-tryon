"""HTTP-contract tests for tryon_server with the MockEngine test double (no GPU needed).

These verify the API the laptop client depends on: auth, job lifecycle, progress, error
mapping and image round-trips. They say nothing about try-on quality.
"""

import time

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, to_bytes
from tryon_server.catvton_engine import GenerationError, GPUOutOfMemoryError
from tryon_server.mock_engine import MockEngine
from tryon_server.server import create_app
from common.image_utils import b64_png_to_image

KEY = "test-key"


def client_for(engine, api_key=KEY):
    return TestClient(create_app(engine, api_key=api_key))


def files(person=None, garment=None):
    return {
        "person": ("p.jpg", person if person is not None else to_bytes(make_image(600, 800)), "image/jpeg"),
        "garment": ("g.jpg", garment if garment is not None else to_bytes(make_image(500, 600)), "image/jpeg"),
    }


def wait_for(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers={"X-API-Key": KEY}).json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_reports_engine_without_key():
    body = client_for(MockEngine()).get("/health").json()
    assert body["status"] == "ok" and body["auth_required"] is True and "MOCK" in body["engine"]


def test_jobs_require_api_key():
    client = client_for(MockEngine())
    assert client.post("/jobs", files=files()).status_code == 401
    assert client.post("/jobs", files=files(), headers={"X-API-Key": "wrong"}).status_code == 401


def test_full_job_lifecycle():
    client = client_for(MockEngine(step_delay_s=0.01))
    response = client.post("/jobs", files=files(), data={"garment_type": "upper", "steps": "10"}, headers={"X-API-Key": KEY})
    assert response.status_code == 202
    body = wait_for(client, response.json()["job_id"])
    assert body["status"] == "done" and body["step"] == 10 and body["total_steps"] == 10
    result = body["result"]
    assert b64_png_to_image(result["result_png"]).size == (768, 1024)
    assert b64_png_to_image(result["mask_png"]).mode == "L"
    assert result["params"]["garment_type"] == "upper" and result["params"]["steps"] == 10


@pytest.mark.parametrize("form,status", [
    ({"garment_type": "hat"}, 422),
    ({"steps": "1"}, 422),
    ({"guidance": "20"}, 422),
])
def test_invalid_parameters_rejected(form, status):
    response = client_for(MockEngine()).post("/jobs", files=files(), data=form, headers={"X-API-Key": KEY})
    assert response.status_code == status


def test_corrupt_upload_rejected():
    response = client_for(MockEngine()).post("/jobs", files=files(person=b"not an image"), headers={"X-API-Key": KEY})
    assert response.status_code == 400 and "person" in response.json()["detail"]


def test_empty_upload_rejected():
    response = client_for(MockEngine()).post("/jobs", files=files(garment=b""), headers={"X-API-Key": KEY})
    assert response.status_code == 400


@pytest.mark.parametrize("error,code", [(GPUOutOfMemoryError("oom"), 507), (GenerationError("boom"), 500)])
def test_engine_errors_reported_in_job(error, code):
    client = client_for(MockEngine(fail_with=error))
    job_id = client.post("/jobs", files=files(), headers={"X-API-Key": KEY}).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "error" and body["error_status"] == code


def test_unknown_job_is_404():
    assert client_for(MockEngine()).get("/jobs/nope", headers={"X-API-Key": KEY}).status_code == 404
