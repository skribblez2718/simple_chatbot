"""Simple Chatbot Backend — FastAPI server for DeepHat-V1-7B inference."""
# ruff: noqa: E501

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("CHATBOT_MODEL_NAME", "DeepHat/DeepHat-V1-7B")
DB_PATH = Path(__file__).resolve().parent.parent / "chatbot.db"
ALLOWED_ORIGINS = sorted(
    {
        f"http://{os.environ.get('CHATBOT_FRONTEND_HOST', 'localhost')}:"
        f"{os.environ.get('CHATBOT_FRONTEND_PORT', '8501')}",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    }
)

logger = logging.getLogger("simple-chatbot")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Detect the best available compute device for model inference.

    Priority:  1. NVIDIA CUDA or AMD ROCm (both via ``torch.cuda``).
               2. Apple Silicon MPS.
               3. CPU.
    """
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if getattr(torch.version, "hip", None):
                logger.info("Detected AMD GPU via ROCm (%s): %s", torch.version.hip, gpu_name)
            else:
                logger.info("Detected NVIDIA GPU via CUDA: %s", gpu_name)
            return "cuda"
    except Exception as exc:
        logger.warning("CUDA probe failed: %s", exc)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon via MPS")
            return "mps"
    except Exception as exc:
        logger.warning("MPS probe failed: %s", exc)
    logger.info("No GPU detected — using CPU (model will run in system RAM)")
    return "cpu"


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_model: Any = None
_tokenizer: Any = None
_device: str | None = None


def load_model() -> bool:
    global _model, _tokenizer, _device
    if _model is not None:
        return True

    from transformers import AutoModelForCausalLM, AutoTokenizer

    visible = _visible_cuda_devices()
    _device = _detect_device()
    logger.info("Loading model %s (visible CUDA: %s, primary: %s) ...", MODEL_NAME, visible, _device)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if len(visible) > 1:
        try:
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
            _log_device_map(_model)
            return True
        except Exception as exc:
            logger.warning("Multi-GPU sharding failed (%s); falling back", exc)

    device_candidates: list[str] = []
    if _device and _device != "cpu":
        device_candidates.append(_device)
    if "cuda" not in device_candidates:
        device_candidates.append("cuda")
    device_candidates.append("cpu")

    seen: set[str] = set()
    for device in device_candidates:
        if device in seen:
            continue
        seen.add(device)
        try:
            logger.info("Attempting load on %s ...", device)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map={"": device})
            logger.info("Model loaded successfully on %s", _model.device)
            return True
        except Exception as exc:
            logger.warning("Load on %s failed: %s", device, exc)

    logger.error("Failed to load model on any device (tried: %s)", seen)
    return False


def _visible_cuda_devices() -> list[str]:
    try:
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _log_device_map(model: Any) -> None:
    try:
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            placement: dict[str, list[str]] = {}
            for layer, dev in device_map.items():
                placement.setdefault(str(dev), []).append(str(layer))
            summary = ", ".join(
                f"{d}={len(ly)} layers" for d, ly in sorted(placement.items())
            )
            logger.info("Model sharded: %s", summary)
            return
    except Exception:
        pass
    logger.info("Model loaded on %s", getattr(model, "device", "?"))


def generate_response(messages: list[dict[str, str]]) -> str:
    """Generate a complete response (synchronous, used by /chat)."""
    chat_messages = _build_chat_messages(messages)
    model_inputs = _prepare_inputs(chat_messages)

    generated_ids = _model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=3,
        repetition_penalty=1.18,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
    )
    response_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:]
    return (
        _tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        or "No response generated."
    )


def stream_response(messages: list[dict[str, str]]) -> Generator[str, None, None]:
    """Yield tokens one at a time for SSE streaming.

    Uses ``transformers`` streaming via ``TextStreamer`` — each yielded
    string is a decoded token fragment.  The frontend consumes these
    as SSE events and displays them incrementally.
    """
    from transformers import TextStreamer

    chat_messages = _build_chat_messages(messages)
    model_inputs = _prepare_inputs(chat_messages)

    class _TokenizerStreamer(TextStreamer):
        """Thin wrapper that yields decoded tokens from the streamer queue."""

        def __init__(self, tokenizer: Any, **kwargs: Any):
            super().__init__(tokenizer, **kwargs)

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            pass  # handled via the iterator

    # Thread-based streaming: TextStreamer runs in a background thread
    # and puts tokens on a queue we iterate over.
    from queue import Empty, Queue
    from threading import Thread

    queue: Queue = Queue()

    class _QueueStreamer(TextStreamer):
        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            queue.put((text, stream_end))

    streamer = _QueueStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate() -> None:
        _model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3,
            repetition_penalty=1.18,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
            streamer=streamer,
        )
        queue.put(("", True))

    thread = Thread(target=_generate, daemon=True)
    thread.start()

    while True:
        try:
            text, stream_end = queue.get(timeout=120)
            if text:
                yield text
            if stream_end:
                break
        except Empty:
            break


def _build_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the full chat message list with system prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are DeepHat, a helpful AI assistant for cybersecurity "
                "and DevOps. Give thorough, well-structured answers with "
                "code examples when relevant. After answering, stop."
            ),
        },
    ] + messages[-20:]


def _prepare_inputs(chat_messages: list[dict[str, str]]) -> Any:
    """Tokenize and move inputs to the correct device."""
    text = _tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    if _device and _device.startswith("cuda"):
        target = _device
    else:
        target = _device or "cpu"
    return _tokenizer([text], return_tensors="pt").to(target)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_db() -> sqlite3.Connection:
    """Return a new connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    conn = _get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Conversation',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                position        INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, position);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _conv_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a conversation row (with messages joined or loaded separately)."""
    conv = {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [],
    }
    # If messages columns are present (from a JOIN query), extract them
    if "msg_role" in row.keys() and row["msg_role"] is not None:
        conv["messages"] = [
            {
                "role": row["msg_role"],
                "content": row["msg_content"],
                "timestamp": row["msg_timestamp"],
            }
        ]
    return conv


def _load_messages(conv_id: str, conn: sqlite3.Connection) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY position",
        (conv_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def _list_conversations() -> list[dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for row in rows:
            conv = _conv_row_to_dict(row)
            conv["messages"] = _load_messages(row["id"], conn)
            result.append(conv)
        return result
    finally:
        conn.close()


def _get_conversation_by_id(conv_id: str) -> dict[str, Any]:
    conn = _get_db()
    try:
        row = conn.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        conv = _conv_row_to_dict(row)
        conv["messages"] = _load_messages(conv_id, conn)
        return conv
    finally:
        conn.close()


def _create_conversation(title: str | None) -> dict[str, Any]:
    conn = _get_db()
    try:
        conv_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        title = title or "New Conversation"
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        conn.commit()
        return {"id": conv_id, "title": title, "messages": [], "created_at": now, "updated_at": now}
    finally:
        conn.close()


def _delete_conversation(conv_id: str) -> None:
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        conn.commit()
    finally:
        conn.close()


def _add_message(conv_id: str, body: Any) -> dict[str, Any]:
    conn = _get_db()
    try:
        # Verify conversation exists
        row = conn.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

        # Determine next position
        max_pos = conn.execute(
            "SELECT COALESCE(MAX(position), -1) AS mp FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()["mp"]

        now = datetime.now(UTC).isoformat()

        # Insert user message
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp, position) VALUES (?, ?, ?, ?, ?)",
            (conv_id, "user", body.content, now, max_pos + 1),
        )

        # Load all messages for chat context
        all_msgs = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY position",
            (conv_id,),
        ).fetchall()

        # Generate AI response
        chat_messages: list[dict[str, str]] = [{"role": m["role"], "content": m["content"]} for m in all_msgs]

        if not load_model():
            assistant_response = "The model is still loading. Please try again shortly."
        else:
            try:
                assistant_response = generate_response(chat_messages)
            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)
                assistant_response = "An unexpected error occurred during inference."

        # Insert assistant message
        ai_now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp, position) VALUES (?, ?, ?, ?, ?)",
            (conv_id, "assistant", assistant_response, ai_now, max_pos + 2),
        )

        # Auto-title from first user message if still default
        current_title = row["title"]
        if current_title == "New Conversation":
            title_text = body.content[:50]
            if len(body.content) > 50:
                title_text += "…"
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title_text, conv_id))

        # Update timestamp
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ai_now, conv_id))
        conn.commit()

        user_message = {"role": "user", "content": body.content, "timestamp": now}
        assistant_message = {"role": "assistant", "content": assistant_response, "timestamp": ai_now}

        return {"user_message": user_message, "assistant_message": assistant_message}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    response: str


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str


class ConversationOut(BaseModel):
    id: str
    title: str
    messages: list[MessageOut]
    created_at: str
    updated_at: str


class AddMessageResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Simple Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting up — initializing database and loading model...")
    _init_db()
    load_model()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Simple Chatbot API", "version": "0.1.0", "model": MODEL_NAME, "status": "running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> dict[str, str]:
    if not load_model():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is still loading.")
    try:
        messages: list[dict[str, str]] = [{"role": "user", "content": request.message}]
        return {"response": generate_response(messages)}
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference error.") from exc


@app.get("/conversations", response_model=list[ConversationOut])
async def list_conversations() -> list[dict[str, Any]]:
    return _list_conversations()


@app.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(body: ConversationCreate) -> dict[str, Any]:
    return _create_conversation(body.title)


@app.get("/conversations/{conv_id}", response_model=ConversationOut)
async def get_conversation(conv_id: str) -> dict[str, Any]:
    return _get_conversation_by_id(conv_id)


@app.delete("/conversations/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conv_id: str) -> None:
    _delete_conversation(conv_id)


@app.post("/conversations/{conv_id}/messages", response_model=AddMessageResponse)
async def add_message(conv_id: str, body: MessageCreate) -> dict[str, Any]:
    return _add_message(conv_id, body)


@app.post("/conversations/{conv_id}/messages/stream")
async def add_message_stream(conv_id: str, body: MessageCreate) -> StreamingResponse:
    """Add a user message and stream the AI response back as SSE.

    The response is ``text/event-stream`` with ``data: {{"token": "..."}}``
    events for each generated token fragment, then a final
    ``data: {{"done": true}}``.  The frontend reassembles them into a
    complete message and displays it incrementally.
    """
    conn = _get_db()
    try:
        # Verify conversation exists and insert the user message.
        row = conn.execute(
            "SELECT id, title FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        max_pos = conn.execute(
            "SELECT COALESCE(MAX(position), -1) AS mp "
            "FROM messages WHERE conversation_id = ?",
            (conv_id,),
        ).fetchone()["mp"]

        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO messages "
            "(conversation_id, role, content, timestamp, position) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, "user", body.content, now, max_pos + 1),
        )

        # Auto-title from first user message.
        if row["title"] == "New Conversation":
            title_text = body.content[:50]
            if len(body.content) > 50:
                title_text += "\u2026"
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title_text, conv_id),
            )

        # Build message history.
        all_msgs = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conv_id,),
        ).fetchall()
        chat_messages: list[dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in all_msgs
        ]

        conn.commit()
    finally:
        conn.close()

    if not load_model():
        # No model yet — send a single error token.
        def _error_stream() -> Generator[str, None, None]:
            yield f"data: {json.dumps({'token': 'Model is loading. Try again shortly.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            _error_stream(), media_type="text/event-stream"
        )

    def _generate_sse() -> Generator[str, None, None]:
        full = ""
        try:
            for token in stream_response(chat_messages):
                full += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Persist the full assistant response.
            ai_now = datetime.now(UTC).isoformat()
            conn2 = _get_db()
            try:
                max_p = conn2.execute(
                    "SELECT COALESCE(MAX(position), -1) FROM messages "
                    "WHERE conversation_id = ?",
                    (conv_id,),
                ).fetchone()[0]
                conn2.execute(
                    "INSERT INTO messages "
                    "(conversation_id, role, content, timestamp, position) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conv_id, "assistant", full, ai_now, max_p + 1),
                )
                conn2.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (ai_now, conv_id),
                )
                conn2.commit()
            finally:
                conn2.close()

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            yield (
                f"data: {json.dumps({'token': 'Error during generation.'})}\n\n"
            )
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(_generate_sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
