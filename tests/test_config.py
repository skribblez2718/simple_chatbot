"""Tests for backend configuration and device detection logic."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


def test_model_name_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_NAME should be sourced from CHATBOT_MODEL_NAME env var."""
    # Unload any previously-imported backend modules so the new env takes effect
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]

    monkeypatch.setenv("CHATBOT_MODEL_NAME", "test/model-from-env")

    import backend.main as backend_module

    assert backend_module.MODEL_NAME == "test/model-from-env"


def test_model_name_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_NAME should fall back to the default if env var unset."""
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]

    monkeypatch.delenv("CHATBOT_MODEL_NAME", raising=False)

    import backend.main as backend_module

    assert backend_module.MODEL_NAME == "DeepHat/DeepHat-V1-7B"


def test_detect_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """_detect_device should return 'cuda' when torch sees a GPU."""
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]
    import backend.main as backend_module

    # Reload to make sure the patched torch is used
    importlib.reload(backend_module)

    fake_torch = backend_module.torch
    with (
        patch.object(fake_torch.cuda, "is_available", return_value=True),
        patch.object(
            fake_torch.cuda, "get_device_name", return_value="Fake GPU 9000"
        ),
    ):
        assert backend_module._detect_device() == "cuda"


def test_detect_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """_detect_device should return 'cpu' when no GPU is available."""
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]
    import backend.main as backend_module
    importlib.reload(backend_module)

    fake_torch = backend_module.torch
    with (
        patch.object(fake_torch.cuda, "is_available", return_value=False),
        patch.object(
            fake_torch.backends, "mps", create=True
        ) as fake_mps,
    ):
        # Disable MPS
        type(fake_mps).is_available = lambda self: False
        assert backend_module._detect_device() == "cpu"


def test_detect_device_prefers_amd_rocam(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMD GPUs report CUDA available via ROCm — should still return 'cuda'."""
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]
    import backend.main as backend_module
    importlib.reload(backend_module)

    fake_torch = backend_module.torch
    fake_version = type("V", (), {"hip": "5.7.0"})()

    with (
        patch.object(fake_torch.cuda, "is_available", return_value=True),
        patch.object(
            fake_torch.cuda, "get_device_name", return_value="AMD Radeon RX 7900"
        ),
        patch.object(fake_torch, "version", fake_version),
    ):
        assert backend_module._detect_device() == "cuda"


def test_allowed_origins_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """ALLOWED_ORIGINS should not contain duplicate entries."""
    for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
        del sys.modules[mod]

    # Use defaults that match the static fallbacks to test dedup
    monkeypatch.setenv("CHATBOT_FRONTEND_HOST", "localhost")
    monkeypatch.setenv("CHATBOT_FRONTEND_PORT", "8501")

    import backend.main as backend_module
    importlib.reload(backend_module)

    assert len(backend_module.ALLOWED_ORIGINS) == len(
        set(backend_module.ALLOWED_ORIGINS)
    )
