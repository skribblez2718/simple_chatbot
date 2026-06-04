"""Test fixtures for the simple chatbot application."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def _make_tokenizer_call_result() -> MagicMock:
    """Create the return value for tokenizer() call — mimics BatchEncoding."""
    result = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = [1, 5]
    result.__getitem__.return_value = input_ids
    # .to() returns self (chainable)
    result.to.return_value = result
    return result


@pytest.fixture
def mock_model() -> Generator[MagicMock, None, None]:
    """Mock the transformers model and tokenizer to avoid downloading."""
    with (
        patch("transformers.AutoModelForCausalLM") as mock_model_cls,
        patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
    ):
        mock_model_obj = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_obj.generate.return_value = [
            MagicMock()  # full output (input + generated)
        ]
        # Set the shape so we can slice
        mock_model_obj.generate.return_value[0].shape = [1, 10]

        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = (
            "Hello! I am a mock model. How can I help you today?"
        )
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.side_effect = lambda *args, **kwargs: (
            _make_tokenizer_call_result()
        )
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are DeepHat...<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        mock_tokenizer.to.return_value = _make_tokenizer_call_result()

        mock_model_cls.from_pretrained.return_value = mock_model_obj
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        yield mock_model_obj


@pytest.fixture
async def app(mock_model: MagicMock) -> FastAPI:
    """Create the FastAPI app with mocked model and temp-file database."""
    import tempfile
    from pathlib import Path

    import backend.main as backend_module
    from backend.main import app as fastapi_app

    # Use a temp file database (NOT :memory: — each connection to
    # :memory: creates a separate, isolated database).
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    backend_module.DB_PATH = tmp.name
    tmp.close()
    backend_module._init_db()

    # The tokenizer was set on the mock, need to extract it
    backend_module._model = mock_model
    # Get the tokenizer from the patch
    backend_module._tokenizer = MagicMock()
    backend_module._tokenizer.decode.return_value = (
        "Hello! I am a mock model. How can I help you today?"
    )
    backend_module._tokenizer.pad_token_id = 0
    backend_module._tokenizer.eos_token_id = 0
    backend_module._tokenizer.side_effect = lambda *args, **kwargs: (
        _make_tokenizer_call_result()
    )
    backend_module._tokenizer.apply_chat_template.return_value = (
        "<|im_start|>assistant\n"
    )

    yield fastapi_app

    # Cleanup
    backend_module._model = None
    backend_module._tokenizer = None
    try:
        Path(tmp.name).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
