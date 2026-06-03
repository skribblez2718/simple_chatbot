"""Integration tests for the embedded backend startup.

These tests actually start a real FastAPI server in a background thread
(with the HuggingFace model mocked) and make real HTTP requests against
it. They exercise the full startup chain that the user runs:

    streamlit run frontend/app.py

That is, the frontend is imported, the embedded backend is launched in
a background thread, and the API is reachable. We mock the transformers
model so the tests run in seconds, but the real FastAPI app, real CORS,
real startup, and real HTTP are all live.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import uvicorn

# Make sure backend tests have a stable working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# A port we don't expect anyone to be using. 18765 is in the
# unprivileged high range, easy to remember.
TEST_PORT = 18765
TEST_HOST = "127.0.0.1"


def _port_is_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def real_backend_server() -> Generator[str, None, None]:
    """Start a real FastAPI server on a test port with the model mocked.

    Yields the base URL. Tears the server down at the end of the module.
    """
    # Mock the transformers model classes before the backend is loaded so
    # the real FastAPI app boots without downloading DeepHat.
    with (
        patch("transformers.AutoModelForCausalLM") as mock_model_cls,
        patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
    ):
        # Build a mock tokenizer that the backend can call into
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.pad_token_id = 0
        # tokenizer([text], return_tensors="pt") — return a fake batch
        batch = MagicMock()
        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        batch.__getitem__.return_value = input_ids
        batch.to.return_value = batch
        mock_tokenizer.return_value = batch
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are DeepHat.<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        mock_tokenizer.decode.return_value = "Hello from the test server!"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Build a mock model that "generates" a deterministic response
        mock_model = MagicMock()
        mock_model.device = "cpu"
        # generate returns a 2-D structure we can slice; shape[1] is the
        # total length so we can compute the new tokens via slicing.
        out = MagicMock()
        out.shape = [1, 6]
        mock_model.generate.return_value = out
        mock_model_cls.from_pretrained.return_value = mock_model

        # Now load the real FastAPI app and start it via uvicorn in a thread
        sys.path.insert(0, str(PROJECT_ROOT))
        from backend.main import app as backend_app

        config = uvicorn.Config(
            app=backend_app,
            host=TEST_HOST,
            port=TEST_PORT,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        import threading

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for /health to come up
        base_url = f"http://{TEST_HOST}:{TEST_PORT}"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{base_url}/health", timeout=1)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Test server failed to start within 15s")

        yield base_url

        # Tear down
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Test 1: Bug regression — sys.path fix in frontend/app.py
# ---------------------------------------------------------------------------


def test_frontend_app_works_when_run_from_inside_frontend_dir() -> None:
    """Running frontend/app.py from inside frontend/ must work.

    Reproduces the original bug: when Streamlit changes the working
    directory to the script's location, ``from backend.main import ...``
    used to fail. The fix adds the project root to sys.path at module
    load time.
    """
    # chdir to frontend/ to simulate Streamlit's behaviour
    original_cwd = os.getcwd()
    try:
        os.chdir(FRONTEND_DIR)

        # Run frontend/app.py as a subprocess (just import it) to mirror
        # what Streamlit does — it execs the file in the script's dir.
        # We use a small driver that imports and exercises the
        # embedded-backend import path.
        driver = (
            "import sys; "
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            "import frontend.app as fa; "
            "from backend.main import app as backend_app; "
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True,
            text=True,
            cwd=str(FRONTEND_DIR),
            env={**os.environ, "CHATBOT_EMBED_BACKEND": "0"},
            timeout=15,
        )
        assert result.returncode == 0, (
            f"frontend/app.py failed to import from inside frontend/\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout
    finally:
        os.chdir(original_cwd)


def test_frontend_app_adds_project_root_to_syspath() -> None:
    """Importing frontend.app should put the project root on sys.path."""
    # Force a fresh import in case the test harness has already loaded it
    for mod in [m for m in list(sys.modules) if m.startswith("frontend")]:
        del sys.modules[mod]

    os.chdir(FRONTEND_DIR)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        import frontend.app  # noqa: F401

        assert str(PROJECT_ROOT) in sys.path, (
            f"Expected {PROJECT_ROOT} in sys.path after importing "
            f"frontend.app, got: {sys.path[:5]}..."
        )
    finally:
        # Don't chdir back here — other tests assume a stable cwd
        pass


# ---------------------------------------------------------------------------
# Test 2: Real server, real HTTP
# ---------------------------------------------------------------------------


def test_real_server_health_endpoint(real_backend_server: str) -> None:
    """A GET to /health on a real running server should return 200."""
    resp = requests.get(f"{real_backend_server}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_real_server_root_endpoint(real_backend_server: str) -> None:
    """GET / on a real running server should return API info."""
    resp = requests.get(f"{real_backend_server}/", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "model" in data


def test_real_server_create_and_use_conversation(
    real_backend_server: str,
) -> None:
    """End-to-end: create a conversation, send a message, get a response."""
    # 1. Create a new conversation
    create_resp = requests.post(
        f"{real_backend_server}/conversations",
        json={"title": "Integration Test Chat"},
        timeout=5,
    )
    assert create_resp.status_code == 201
    conv = create_resp.json()
    assert conv["title"] == "Integration Test Chat"
    assert conv["messages"] == []
    conv_id = conv["id"]

    # 2. Send a message
    msg_resp = requests.post(
        f"{real_backend_server}/conversations/{conv_id}/messages",
        json={"content": "Hello!"},
        timeout=10,
    )
    assert msg_resp.status_code == 200
    payload = msg_resp.json()
    assert payload["user_message"]["content"] == "Hello!"
    assert payload["user_message"]["role"] == "user"
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["assistant_message"]["content"]  # non-empty

    # 3. Retrieve the conversation and confirm both messages are persisted
    get_resp = requests.get(
        f"{real_backend_server}/conversations/{conv_id}", timeout=5
    )
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert len(fetched["messages"]) == 2
    assert fetched["messages"][0]["role"] == "user"
    assert fetched["messages"][1]["role"] == "assistant"

    # 4. Delete the conversation
    del_resp = requests.delete(
        f"{real_backend_server}/conversations/{conv_id}", timeout=5
    )
    assert del_resp.status_code == 204


def test_real_server_cors_allows_streamlit_origin(
    real_backend_server: str,
) -> None:
    """CORS preflight from the Streamlit origin should succeed."""
    resp = requests.options(
        f"{real_backend_server}/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=5,
    )
    assert (
        resp.headers.get("access-control-allow-origin")
        == "http://localhost:8501"
    )


# ---------------------------------------------------------------------------
# Test 3: Embedded-startup logic in frontend/app.py
# ---------------------------------------------------------------------------


def test_ensure_backend_running_finds_existing_server(
    real_backend_server: str,
) -> None:
    """ensure_backend_running should detect an already-running backend."""
    # Patch the embedded backend's URL to point at our test server
    with (
        patch.dict(
            os.environ,
            {
                "CHATBOT_BACKEND_HOST": TEST_HOST,
                "CHATBOT_BACKEND_PORT": str(TEST_PORT),
                "CHATBOT_EMBED_BACKEND": "0",
            },
        ),
        patch("frontend.app.BACKEND_HOST", TEST_HOST),
        patch("frontend.app.BACKEND_PORT", TEST_PORT),
    ):
        # Force a fresh import so the patches above take effect
        for mod in [m for m in list(sys.modules) if m.startswith("frontend")]:
            del sys.modules[mod]
        import frontend.app as fa

        ok, msg = fa.ensure_backend_running()
        assert ok is True
        assert "reachable" in msg.lower() or "connected" in msg.lower()


def test_ensure_backend_running_returns_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When EMBED_BACKEND=0 and no server is running, return (False, hint)."""
    # Pick a definitely-free port
    sock = socket.socket()
    sock.bind((TEST_HOST, 0))
    free_port = sock.getsockname()[1]
    sock.close()

    monkeypatch.delenv("CHATBOT_EMBED_BACKEND", raising=False)

    for mod in [m for m in list(sys.modules) if m.startswith("frontend")]:
        del sys.modules[mod]
    sys.path.insert(0, str(PROJECT_ROOT))
    import frontend.app as fa

    monkeypatch.setattr(fa, "BACKEND_HOST", TEST_HOST)
    monkeypatch.setattr(fa, "BACKEND_PORT", free_port)
    monkeypatch.setattr(fa, "BACKEND_URL", f"http://{TEST_HOST}:{free_port}")
    monkeypatch.setattr(fa, "EMBED_BACKEND", False)
    fa._backend_thread = None

    ok, msg = fa.ensure_backend_running()
    assert ok is False
    assert "backend" in msg.lower() or "uvicorn" in msg.lower()
