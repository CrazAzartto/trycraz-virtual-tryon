"""End-to-end test of the laptop client against a real HTTP server (MockEngine, localhost).

Exercises exactly the network path used with Colab: multipart upload -> job id -> polling
with progress -> base64 PNG decode -> error mapping.
"""

import socket
import threading
import time

import pytest
import uvicorn

from conftest import make_image
from tryon_server.catvton_engine import GPUOutOfMemoryError
from tryon_server.mock_engine import MockEngine
from tryon_server.server import create_app
from virtual_tryon.backends.base import (
    AuthenticationFailed, BackendNotConfigured, BackendUnavailable, GPUOutOfMemory, InvalidRequest, TryOnRequest,
)
from virtual_tryon.backends.remote import RemoteBackend, normalise_url

KEY = "k3y"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(engine):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine, api_key=KEY), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture
def live_server():
    server, url = start_server(MockEngine(step_delay_s=0.01))
    yield url
    server.should_exit = True


def request():
    return TryOnRequest(person=make_image(600, 800), garment=make_image(500, 600), garment_type="upper", steps=8)


def test_normalise_url():
    assert normalise_url(" abc.trycloudflare.com/ ") == "https://abc.trycloudflare.com"
    assert normalise_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"


def test_health_and_run(live_server):
    backend = RemoteBackend(live_server, KEY)
    status = backend.health()
    assert status.reachable and status.is_mock
    seen = []
    result = backend.run(request(), progress=lambda stage, step, total: seen.append((stage, step, total)))
    assert result.image.size == (768, 1024) and result.mask.mode == "L"
    assert "MOCK" in result.engine and result.round_trip_s is not None
    assert seen[0][0] == "uploading"


def test_wrong_key(live_server):
    status = RemoteBackend(live_server, "bad").health()
    assert not status.reachable and "API key" in status.message     # reported by the connection check
    with pytest.raises(AuthenticationFailed):
        RemoteBackend(live_server, "bad").run(request())


def test_invalid_request(live_server):
    bad = request()
    bad.garment_type = "socks"
    with pytest.raises(InvalidRequest):
        RemoteBackend(live_server, KEY).run(bad)


def test_unreachable_server():
    backend = RemoteBackend(f"http://127.0.0.1:{free_port()}", KEY)
    assert not backend.health().reachable
    with pytest.raises(BackendUnavailable):
        backend.run(request())


def test_not_configured():
    with pytest.raises(BackendNotConfigured):
        RemoteBackend("", KEY).run(request())


def test_gpu_oom_is_mapped():
    server, url = start_server(MockEngine(fail_with=GPUOutOfMemoryError("oom")))
    try:
        with pytest.raises(GPUOutOfMemory):
            RemoteBackend(url, KEY).run(request())
    finally:
        server.should_exit = True
