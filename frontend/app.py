"""Simple Chatbot Frontend — Streamlit chat UI with conversation sidebar.

Design follows ChatGPT / Claude / v0 patterns:
  - User messages: right-aligned with subtle blue tint
  - AI messages: left-aligned, no background
  - Sidebar: conversations + theme toggle at top, always visible
  - Streamlit handles base theme (light/dark) via st._config
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure the project root is on sys.path so `from backend.main import ...`
# works when Streamlit runs this file directly from inside the frontend/
# directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()

logger = logging.getLogger("chatbot-frontend")

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit command.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepHat Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ---------------------------------------------------------------------------
# Theme — base set via .streamlit/config.toml.
#   The toggle handler calls st._config.set_option + st.rerun to switch.
# ---------------------------------------------------------------------------

if "theme_initialized" not in st.session_state:
    st.session_state.theme_initialized = True
    st.session_state.theme_preference = "dark"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_HOST = os.environ.get("CHATBOT_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("CHATBOT_BACKEND_PORT", "8000"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
EMBED_BACKEND = os.environ.get("CHATBOT_EMBED_BACKEND", "1") == "1"
_RAW_MODEL_NAME = os.environ.get("CHATBOT_MODEL_NAME", "DeepHat/DeepHat-V1-7B")
# Display name defaults to the model name with the "org/" prefix stripped
# and "/" replaced with "-". Override via CHATBOT_MODEL_DISPLAY_NAME.
MODEL_DISPLAY_NAME = os.environ.get(
    "CHATBOT_MODEL_DISPLAY_NAME",
    _RAW_MODEL_NAME.split("/")[-1],
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def check_backend() -> bool:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_conversations() -> list[dict[str, Any]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/conversations", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def create_conversation_api(
    title: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = {"title": title} if title else {}
        resp = requests.post(
            f"{BACKEND_URL}/conversations", json=payload, timeout=5
        )
        if resp.status_code == 201:
            return resp.json()
        return None
    except requests.RequestException:
        return None


def delete_conversation_api(conv_id: str) -> bool:
    try:
        resp = requests.delete(
            f"{BACKEND_URL}/conversations/{conv_id}", timeout=5
        )
        return resp.status_code == 204
    except requests.RequestException:
        return False


def send_message_api(
    conv_id: str, content: str
) -> dict[str, Any] | None:
    """Send a message synchronously (used by tests and non-streaming paths)."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/conversations/{conv_id}/messages",
            json={"content": content},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def send_message_stream(
    conv_id: str, content: str
) -> Generator[str, None, None] | None:
    """Send a message and yield tokens from the SSE response stream.

    Returns None if the backend is unreachable, otherwise yields each
    token fragment as it arrives.  Callers should iterate until
    exhaustion to get the complete response.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/conversations/{conv_id}/messages/stream",
            json={"content": content},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if data.get("done"):
                return
            if "token" in data:
                yield data["token"]
    except requests.RequestException as exc:
        logger.error("Stream request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Embedded backend
# ---------------------------------------------------------------------------

_backend_thread: threading.Thread | None = None
_backend_error: BaseException | None = None
_backend_lock = threading.Lock()


def _port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_port_owner(host: str, port: int) -> str | None:
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                if "LISTEN" not in line:
                    continue
                match = re.search(
                    r'users:\(\("(?P<cmd>[^"]+)",pid=(?P<pid>\d+),[^)]*\)',
                    line,
                )
                if match:
                    return f"PID {match.group('pid')} ({match.group('cmd')})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["lsof", "-iTCP", f"{port}", "-sTCP:LISTEN", "-nP"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout:
            lines = [ln for ln in result.stdout.splitlines() if "LISTEN" in ln]
            if lines:
                parts = lines[0].split()
                if len(parts) >= 2:
                    return f"PID {parts[1]} ({parts[0]})"
    except (FileNotFoundError, subprocess.TimeoutExpired, AttributeError):
        pass
    return None


def _wait_for_backend(timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check_backend():
            return True
        time.sleep(0.5)
    return False


def _start_embedded_backend() -> bool:
    global _backend_thread, _backend_error
    with _backend_lock:
        if _backend_thread is not None and _backend_thread.is_alive():
            return _wait_for_backend()
        if _port_is_open(BACKEND_HOST, BACKEND_PORT):
            return True
        try:
            from backend.main import app as backend_app  # noqa: I001
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the FastAPI backend. "
                f"Make sure the virtual env is activated. Error: {exc}"
            ) from exc
        _backend_error = None
        config = uvicorn.Config(
            app=backend_app,
            host=BACKEND_HOST, port=BACKEND_PORT,
            log_level="info", access_log=False,
        )
        server = uvicorn.Server(config)

        def _run_server() -> None:
            try:
                server.run()
            except OSError as exc:
                global _backend_error
                _backend_error = exc

        thread = threading.Thread(
            target=_run_server, name="chatbot-backend", daemon=True
        )
        thread.start()
        _backend_thread = thread

    if _wait_for_backend():
        return True
    if _backend_error is not None:
        owner = _find_port_owner(BACKEND_HOST, BACKEND_PORT) or "an unknown process"
        raise RuntimeError(
            f"Could not start backend on {BACKEND_HOST}:{BACKEND_PORT}: "
            f"{_backend_error}. Port held by {owner}. "
            f"Free it or set CHATBOT_BACKEND_PORT in .env."
        )
    return False


def ensure_backend_running() -> tuple[bool, str]:
    if check_backend():
        return True, "Backend reachable."
    if not EMBED_BACKEND:
        return False, (
            "Backend not reachable. Start manually:\n\n"
            "```\nuvicorn backend.main:app --port 8000\n```"
        )
    try:
        started = _start_embedded_backend()
    except RuntimeError as exc:
        return False, str(exc)
    if started:
        return True, "Backend started in background."
    return False, "Backend failed to start within 60s."


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_session() -> None:
    defaults = {
        "conversations": [],
        "current_conv_id": None,
        "backend_ok": None,
        "backend_msg": "",
        "generating": False,
        "pending_prompt": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if not st.session_state.conversations:
        st.session_state.conversations = get_conversations()


# ---------------------------------------------------------------------------
# CSS — layout only.  Streamlit's built-in light/dark themes handle all
# colours (background, text, sidebar, buttons).  We only add chat-specific
# layout rules (message bubbles, code blocks, input).
# ---------------------------------------------------------------------------


CHAT_CSS = """
<style>
/* === Hide Streamlit chrome ===================================== */
.stApp > header, .stApp > footer, #MainMenu,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] { display: none !important; }

/* === Sidebar =================================================== */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.15);
}
/* Collapsed sidebar: Streamlit shifts content -300px off-screen via
   CSS transform. We zero the shift so the expand button stays visible
   within a thin strip. Hide conversation content — only the header
   with the expand/collapse arrow shows. */
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 44px !important;
    max-width: 44px !important;
    transform: none !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] {
    display: none !important;
}
/* The expand button — make it clearly visible in the thin strip.
   Larger size, background, and centered within the 44px strip. */
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarHeader"] {
    justify-content: center !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] button[kind="headerNoPadding"] {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    background: rgba(128, 128, 128, 0.12) !important;
    transition: background 150ms ease;
}
section[data-testid="stSidebar"][aria-expanded="false"] button[kind="headerNoPadding"]:hover {
    background: rgba(128, 128, 128, 0.22) !important;
}
/* The expand button inside the collapsed strip — Streamlit hides
   the parent with visibility:hidden but renders the icon via a
   child. Ensure the control itself can receive clicks. */
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}
/* Collapse button (>> when open, << when collapsed). Streamlit
   sets visibility:hidden but the icon child is visible. Make the
   parent visible too so Playwright/automation can click it. */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}

/* Sidebar buttons — clean, minimal, no border until hover.
   Streamlit generates unique key classes on grandparent divs
   (e.g. st-key-sel_UUID, st-key-del_UUID).  Use attribute
   selectors to match them. */
[class*="st-key-sel_"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    padding: 0.4rem 0.55rem !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    transition: background-color 120ms ease;
}
[class*="st-key-sel_"] button:hover {
    background-color: rgba(128, 128, 128, 0.08) !important;
    border-color: rgba(128, 128, 128, 0.15) !important;
}

/* Active conversation — the button with type="primary" */
[class*="st-key-sel_"] button[kind="primary"] {
    background-color: rgba(128, 128, 128, 0.12) !important;
    font-weight: 500 !important;
}

/* Delete button — subtle until hover */
[class*="st-key-del_"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    font-size: 0.85rem !important;
    padding: 0.2rem 0.3rem !important;
    border-radius: 4px !important;
    opacity: 0;
    transition: opacity 120ms;
}
/* Show delete on row hover — the parent column hover reveals it */
[class*="st-key-sel_"]:hover + div button,
div:hover > div > [class*="st-key-del_"] button,
[data-testid="stHorizontalBlock"]:hover [class*="st-key-del_"] button {
    opacity: 0.45;
}
[class*="st-key-del_"] button:hover {
    opacity: 1 !important;
    background-color: rgba(220, 38, 38, 0.10) !important;
    color: #dc2626 !important;
}

/* New Chat — primary button */
.st-key-new-chat button {
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* Status pills */
.st-key-status-ok div[data-testid="stAlert"] {
    border: none !important;
    padding: 0.45rem 0.65rem !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    background-color: rgba(34, 197, 94, 0.10) !important;
    color: #16a34a !important;
}
.st-key-status-err div[data-testid="stAlert"] {
    border: none !important;
    padding: 0.45rem 0.65rem !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    background-color: rgba(239, 68, 68, 0.10) !important;
    color: #dc2626 !important;
}

/* === Main chat area ============================================ */
.main .block-container {
    max-width: 48rem;
    padding: 1rem 2rem 7rem 2rem;
}

/* === Message bubbles =========================================== */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 1.2rem 0 !important;
    gap: 0.65rem;
}

/* USER — right-aligned, blue-tinted bubble.
   Custom emoji avatars (🧑/🤖) do NOT get data-testid attributes;
   use aria-label on the content div instead. */
.stChatMessage:has([aria-label="Chat message from user"]) {
    flex-direction: row-reverse;
    align-items: center !important;
}
/* USER message bubble - force minimum height and use flexbox for vertical centering.
   The inner stVerticalBlock has flex: 0 0 auto which can collapse it.
   We set min-height on the bubble and overflow: visible everywhere. */
.stChatMessage:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
    background-color: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 14px 14px 8px 14px;
    padding: 0.7rem 1rem;
    max-width: 75%;
    margin-left: auto !important;
    margin-right: 0 !important;
    text-align: left;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    min-height: 2.8rem !important;
    overflow: visible !important;
}
.stChatMessage:has([aria-label="Chat message from user"]) \
    [data-testid="stChatMessageContent"] * {
    overflow: visible !important;
}
/* Force the inner vertical block to match its text content height
   so that the parent's equal padding produces visual centering. */
.stChatMessage:has([aria-label="Chat message from user"]) \
    [data-testid="stChatMessageContent"] \
    [data-testid="stVerticalBlock"] {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    flex: 0 0 auto !important;
    overflow: visible !important;
}
.stChatMessage:has([aria-label="Chat message from user"]) \
    [data-testid="stChatMessageContent"] \
    [data-testid="stVerticalBlock"] > div {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    flex: 0 0 auto !important;
}
/* Force the stMarkdown element to size to its content (it has flex: 0 1 auto
   by default which prevents growth - this was causing the text to overflow). */
.stChatMessage:has([aria-label="Chat message from user"]) \
    [data-testid="stChatMessageContent"] \
    [data-testid="stMarkdown"] {
    height: auto !important;
    min-height: 1.5em !important;
    max-height: none !important;
    flex: 1 1 auto !important;
    overflow: visible !important;
    display: block !important;
}

/* ASSISTANT — left-aligned, no background */
.stChatMessage:has(
    [aria-label="Chat message from assistant"]
) [data-testid="stChatMessageContent"] {
    background-color: transparent;
    padding: 0.45rem 0.25rem;
    max-width: 100%;
    border: none;
    box-shadow: none;
}

/* Avatar styling — the div containing the emoji */

/* Markdown inside messages */
[data-testid="stChatMessageContent"] p {
    line-height: 1.6;
    margin: 0.35em 0;
    font-size: 0.95rem;
}
/* USER message <p> - override general p styling to fit the centered bubble */
.stChatMessage:has([aria-label="Chat message from user"]) \
    [data-testid="stChatMessageContent"] p {
    text-align: left;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.4;
}
[data-testid="stChatMessageContent"] p:first-child { margin-top: 0; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"] code {
    background-color: rgba(128, 128, 128, 0.12);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.88em;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
[data-testid="stChatMessageContent"] pre {
    background-color: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    overflow-x: auto;
    font-size: 0.85em;
}
[data-testid="stChatMessageContent"] pre code {
    background: transparent;
    padding: 0;
}
[data-testid="stChatMessageContent"] a {
    text-underline-offset: 2px;
}
[data-testid="stChatMessageContent"] blockquote {
    border-left: 3px solid rgba(128, 128, 128, 0.3);
    margin: 0.4em 0;
    padding: 0.15em 0.85em;
    opacity: 0.85;
}
[data-testid="stChatMessageContent"] table {
    border-collapse: collapse;
    margin: 0.5em 0;
    font-size: 0.9em;
}
[data-testid="stChatMessageContent"] th,
[data-testid="stChatMessageContent"] td {
    border: 1px solid rgba(128, 128, 128, 0.15);
    padding: 0.35em 0.6em;
    text-align: left;
}

/* === Chat input ================================================ */
div[data-testid="stChatInput"] {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
div[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important;
}

/* === Welcome / empty state ===================================== */
.welcome-hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem 1rem;
    max-width: 36rem;
    margin: 0 auto;
}
.welcome-hero h1 {
    font-size: 1.7rem;
    font-weight: 600;
    margin: 0 0 0.4rem 0;
}
.welcome-hero p {
    opacity: 0.65;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}
.welcome-model {
    font-size: 0.78rem;
    text-align: center;
    margin-top: 1.5rem;
    opacity: 0.5;
    letter-spacing: 0.02em;
}

/* === Header / model badge ====================================== */
.model-header {
    text-align: center;
    padding: 0.25rem 0 0.5rem 0;
    margin-bottom: 0.5rem;
}
.model-header h2 {
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0;
}
.model-header .subtitle {
    font-size: 0.78rem;
    opacity: 0.55;
    margin-top: 0.15rem;
}

/* === Chip buttons (welcome prompt suggestions) ================= */
/* Allow text wrapping so long prompt labels don't clip. */
.st-key-chip_0 button,
.st-key-chip_1 button,
.st-key-chip_2 button,
.st-key-chip_3 button {
    background-color: rgba(128, 128, 128, 0.06) !important;
    border: 1px solid rgba(128, 128, 128, 0.15) !important;
    padding: 0.5rem 0.85rem !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    white-space: normal !important;
    text-align: center !important;
    line-height: 1.4 !important;
    height: auto !important;
    min-height: auto !important;
    color: inherit !important;
    transition: background-color 120ms ease, border-color 120ms ease, transform 80ms ease;
}
.st-key-chip_0 button:hover,
.st-key-chip_1 button:hover,
.st-key-chip_2 button:hover,
.st-key-chip_3 button:hover {
    background-color: rgba(128, 128, 128, 0.12) !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
    transform: translateY(-1px);
}
.st-key-chip_0 button:active,
.st-key-chip_1 button:active,
.st-key-chip_2 button:active,
.st-key-chip_3 button:active {
    transform: translateY(0);
}
/* Let chip containers grow when text wraps */
div[data-testid="stButton"]:has(.st-key-chip_0),
div[data-testid="stButton"]:has(.st-key-chip_1),
div[data-testid="stButton"]:has(.st-key-chip_2),
div[data-testid="stButton"]:has(.st-key-chip_3) {
    height: auto !important;
}
div[data-testid="stElementContainer"]:has(.st-key-chip_0),
div[data-testid="stElementContainer"]:has(.st-key-chip_1),
div[data-testid="stElementContainer"]:has(.st-key-chip_2),
div[data-testid="stElementContainer"]:has(.st-key-chip_3) {
    height: auto !important;
}

/* === Timestamps below messages ================================== */
/* Align with bubble position: right for user, left for assistant */
div[data-testid="stCaptionContainer"] {
    font-size: 0.72rem;
    opacity: 0.5;
    padding-top: 0.15rem;
    padding-bottom: 0.25rem;
}
/* User message timestamps — right-aligned */
.stChatMessage:has([aria-label="Chat message from user"]) + div [data-testid="stCaptionContainer"] {
    text-align: right;
    padding-right: 0.5rem;
}
/* Assistant message timestamps — left-aligned */
.stChatMessage:has([aria-label="Chat message from assistant"]) + div \
    [data-testid="stCaptionContainer"] {
    text-align: left;
    padding-left: 0.5rem;
}
.st-key-theme-toggle button {
    background: transparent !important;
    border: 1px solid rgba(128, 128, 128, 0.15) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.6rem !important;
    color: inherit !important;
    transition: background-color 120ms ease;
}
.st-key-theme-toggle button:hover {
    background-color: rgba(128, 128, 128, 0.08) !important;
}
.st-key-theme-light button,
.st-key-theme-dark button {
    font-size: 1.1rem !important;
    padding: 0.35rem 0.45rem !important;
    border-radius: 8px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    transition: background-color 120ms;
}
.st-key-theme-light button:hover,
.st-key-theme-dark button:hover {
    background-color: rgba(128, 128, 128, 0.08) !important;
}

/* === Code blocks — prevent clipping ============================= */
pre, code, [data-testid="stCodeBlock"], [data-testid="stCode"] {
    max-width: 100% !important;
    overflow-x: auto !important;
}
/* Inline code — allow wrapping on narrow screens so expressions
   like `cursor_position = ((term_width / 2) + ...)` don't clip. */
[data-testid="stMarkdown"] code,
[data-testid="stChatMessageContent"] code {
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    white-space: pre-wrap !important;
}

/* === "powered by" — always below chips, not inline ============= */
.welcome-model {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    margin-top: 1rem !important;
}

/* === Tablet sidebar (640-1024px) — narrower sidebar ============= */
@media (min-width: 641px) and (max-width: 1024px) {
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 200px !important;
    }
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* === Mobile responsiveness (<768px) ============================== */
@media (max-width: 768px) {
    /* Main content area — remove side padding on phone screens */
    .main .block-container {
        padding: 0.5rem 0.75rem 6rem 0.75rem !important;
        max-width: 100% !important;
    }

    /* User message bubbles — wider on mobile, less wasted space */
    .stChatMessage:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
        max-width: 88% !important;
        padding: 0.55rem 0.75rem 0.65rem 0.75rem !important;
        border-radius: 12px 12px 6px 12px !important;
    }

    /* Assistant bubbles — tighter padding */
    .stChatMessage:has([aria-label="Chat message from assistant"]) \
        [data-testid="stChatMessageContent"] {
        padding: 0.3rem 0.15rem !important;
    }

    /* Welcome hero — less vertical padding */
    .welcome-hero { padding: 2rem 0.5rem 1rem 0.5rem !important; }
    .welcome-hero h1 { font-size: 1.35rem !important; }
    .welcome-hero p { font-size: 0.88rem !important; }

    /* Chips — stack vertically on narrow screens */
    .st-key-chip_0 button,
    .st-key-chip_1 button,
    .st-key-chip_2 button,
    .st-key-chip_3 button {
        font-size: 0.82rem !important;
        padding: 0.45rem 0.8rem !important;
    }

    /* Code blocks — prevent clipping with horizontal scroll */
    pre, code, [data-testid="stCodeBlock"] {
        max-width: 100% !important;
        overflow-x: auto !important;
        white-space: pre !important;
    }

    /* Chat input — larger touch target */
    div[data-testid="stChatInput"] textarea {
        font-size: 1rem !important;
        min-height: 44px !important;
    }

    /* Timestamps — even smaller */
    [data-testid="stCaptionContainer"] {
        font-size: 0.7rem !important;
    }

    /* Header / model badge */
    .model-header h2 { font-size: 0.95rem !important; }

    /* "powered by" — ensure it's below chips, not inline */
    .welcome-model {
        display: block !important;
        width: 100% !important;
        text-align: center !important;
        margin-top: 1rem !important;
    }
}

    /* Chat input — larger touch target */
    div[data-testid="stChatInput"] textarea {
        font-size: 1rem !important;
        min-height: 44px !important;
    }

    /* Timestamps — even smaller */
    [data-testid="stChatMessageContent"] small {
        font-size: 0.7rem !important;
    }

    /* Header / model badge */
    .model-header h2 { font-size: 0.95rem !important; }
}

/* === Very small screens (<400px) ================================= */
@media (max-width: 400px) {
    .main .block-container {
        padding: 0.3rem 0.4rem 5.5rem 0.4rem !important;
    }

    .stChatMessage:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
        max-width: 94% !important;
    }
}
</style>
"""


def apply_theme() -> None:
    """Inject chat-specific layout CSS once per rerun."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

SUGGESTED_PROMPTS = [
    "Tell me about DeepHat's architecture",
    "Write a secure Python HTTPS client",
    "Explain the OWASP Top 10 for web apps",
    "Generate a Dockerfile for a FastAPI app",
]


def render_sidebar() -> None:
    with st.sidebar:
        # ---- Connection status (read-only; main() handles reconnects) ----
        if st.session_state.get("backend_ok"):
            st.markdown(
                "<div class='st-key-status-ok' data-testid='stAlert'>"
                "🟢 Backend connected</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='st-key-status-err' data-testid='stAlert'>"
                "🔴 Backend unreachable</div>",
                unsafe_allow_html=True,
            )
            if st.button("Retry", key="retry_conn", use_container_width=True):
                ok, msg = ensure_backend_running()
                st.session_state.backend_ok = ok
                st.session_state.backend_msg = msg
                st.rerun()

        # ---- New Chat ----
        if st.button(
            "➕  New Chat", key="new-chat", use_container_width=True,
            type="primary",
        ):
            conv = create_conversation_api()
            if conv:
                st.session_state.conversations.insert(0, conv)
                st.session_state.current_conv_id = conv["id"]
                st.rerun()

        # ---- Theme (top, always visible) ----
        cur = st.session_state.theme_preference
        use_light = cur == "light"
        if st.button(
            "🌙 Dark" if use_light else "☀️ Light",
            key="theme-toggle",
            use_container_width=True,
        ):
            new = "dark" if use_light else "light"
            st.session_state.theme_preference = new
            st._config.set_option("theme.base", new)
            st.rerun()

        # ---- Conversations label ----
        st.markdown(
            "<div style='margin: 0.75rem 0 0.25rem 0; font-size: 0.75rem; "
            "opacity: 0.5; text-transform: uppercase; letter-spacing: 0.04em;'>"
            "Conversations</div>",
            unsafe_allow_html=True,
        )

        conversations = st.session_state.get("conversations", [])
        if not conversations:
            st.caption("No conversations yet. Start a new chat!")

        # Search / filter when > 5 conversations
        if len(conversations) > 5:
            filter_text = st.text_input(
                "Search",
                key="conv_search",
                placeholder="Filter conversations\u2026",
                label_visibility="collapsed",
            )
            if filter_text:
                filter_lower = filter_text.lower()
                conversations = [
                    c for c in conversations
                    if filter_lower in (c.get("title", "") or "").lower()
                ]

        for conv in conversations:
            _render_conv_row(conv)


def _render_conv_row(conv: dict[str, Any]) -> None:
    """Render a single conversation row with select + delete buttons.

    The delete button uses a two-click confirmation pattern: first
    click shows "✕", second click shows "🗑" to confirm.
    """
    cid = conv["id"]
    title = conv.get("title", "Untitled")
    display = title[:32] + "\u2026" if len(title) > 32 else title
    is_active = cid == st.session_state.current_conv_id
    confirm_key = f"confirm_del_{cid}"

    cols = st.columns([5, 1])
    if cols[0].button(
        f"{'● ' if is_active else '○ '}{display}",
        key=f"sel_{cid}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.session_state.current_conv_id = cid
        if confirm_key in st.session_state:
            del st.session_state[confirm_key]
        st.rerun()

    # Two-click delete: first click sets confirm_key, second deletes.
    if st.session_state.get(confirm_key):
        if cols[1].button(
            "🗑", key=f"del_confirm_{cid}",
            help="Click again to confirm deletion",
        ):
            if delete_conversation_api(cid):
                st.session_state.conversations = [
                    c for c in st.session_state.conversations
                    if c["id"] != cid
                ]
                if st.session_state.current_conv_id == cid:
                    st.session_state.current_conv_id = None
                del st.session_state[confirm_key]
                st.rerun()
    else:
        if cols[1].button(
            "✕", key=f"del_{cid}",
            help=f"Delete '{title[:30]}'",
        ):
            st.session_state[confirm_key] = True
            st.rerun()


# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------


def render_welcome() -> None:
    """Render the welcome screen with wrapping prompt suggestion chips."""
    st.html(
        f"""<div class="welcome-hero">
            <h1>How can I help you today?</h1>
            <p>I'm {MODEL_DISPLAY_NAME}, an AI assistant for cybersecurity
            and DevOps. Pick a starter, or type anything below.</p>
        </div>"""
    )

    # Chips as buttons in equal columns for desktop. On mobile,
    # media queries override the column layout to stack vertically.
    chips_cols = st.columns(len(SUGGESTED_PROMPTS))
    for i, prompt_text in enumerate(SUGGESTED_PROMPTS):
        with chips_cols[i]:
            if st.button(
                prompt_text,
                key=f"chip_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

    st.html(
        f"""<div class="welcome-model">
            powered by {MODEL_DISPLAY_NAME}
        </div>"""
    )


def render_chat() -> None:
    conv_id = st.session_state.current_conv_id
    title = "Chat"
    if conv_id:
        for c in st.session_state.conversations:
            if c["id"] == conv_id:
                title = c.get("title", "DeepHat Chat")
                break

    st.html(
        f"""<div class="model-header">
            <h2>{title}</h2>
            <div class="subtitle">{MODEL_DISPLAY_NAME}</div>
        </div>"""
    )

    messages = get_messages()
    if not messages:
        render_welcome()
    else:
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "assistant"
            with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
                st.markdown(msg.get("content", ""))
            # Timestamps go below the bubble, outside st.chat_message,
            # so they don't inflate short user bubbles.
            ts = msg.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    ago = _relative_time(dt)
                    st.caption(
                        ago,
                        help=ts.replace("T", " ").split(".")[0],
                    )
                except (ValueError, TypeError):
                    pass

    # --- Input handling ---
    # Two sources: the chat_input bar (manual typing) AND the
    # pending_prompt set by the welcome chip buttons.

    prompt = st.chat_input(
        f"Message {MODEL_DISPLAY_NAME}…",
        disabled=st.session_state.generating,
    )

    # Check for chip-triggered prompt FIRST so it always wins.
    chip_prompt = st.session_state.pop("pending_prompt", None)
    effective = chip_prompt or (prompt.strip() if prompt else None)

    if effective:
        _handle_prompt(effective, conv_id)


def get_messages() -> list[dict[str, Any]]:
    if st.session_state.current_conv_id is None:
        return []
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.current_conv_id:
            return c.get("messages", [])
    return []


def _handle_prompt(prompt: str, conv_id: str | None) -> None:
    """Process a user prompt and display the streaming AI response."""
    st.session_state.generating = True

    if conv_id is None:
        conv = create_conversation_api()
        if conv:
            conv_id = conv["id"]
            st.session_state.current_conv_id = conv_id
            st.session_state.conversations.insert(0, conv)
        else:
            st.error("Failed to create conversation. Is the backend running?")
            st.session_state.generating = False
            return

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Update local state with the user message.
    for i, conv in enumerate(st.session_state.conversations):
        if conv["id"] == conv_id:
            conv["messages"].append(
                {"role": "user", "content": prompt, "timestamp": ""}
            )
            st.session_state.conversations[i] = conv
            break

    with st.chat_message("assistant", avatar="🤖"):
        stream = send_message_stream(conv_id, prompt)
        if stream is None:
            st.markdown("⚠️  Unable to reach the backend.")
            st.session_state.generating = False
            return

        full_response = st.write_stream(stream)

    # Update local state with the complete assistant response.
    now = datetime.now(UTC).isoformat()
    for i, conv in enumerate(st.session_state.conversations):
        if conv["id"] == conv_id:
            conv["messages"].append(
                {"role": "assistant", "content": full_response, "timestamp": now}
            )
            conv["updated_at"] = now
            st.session_state.conversations[i] = conv
            break

    # Refresh from backend — but NEVER wipe local state if the
    # backend is unreachable (returns [] on error).
    backend_convs = get_conversations()
    if backend_convs:
        st.session_state.conversations = backend_convs

    # Auto-cleanup: remove any empty conversations that may have
    # accumulated from failed chip clicks.
    _prune_empty_conversations()

    st.session_state.generating = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _relative_time(dt: datetime) -> str:
    """Return a human-friendly relative time string (e.g. '2 min ago')."""
    delta = datetime.now(UTC) - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        mins = int(secs / 60)
        return f"{mins} min ago" if mins == 1 else f"{mins} mins ago"
    if secs < 86400:
        hrs = int(secs / 3600)
        return f"{hrs} hr ago" if hrs == 1 else f"{hrs} hrs ago"
    days = int(secs / 86400)
    return f"{days} day ago" if days == 1 else f"{days} days ago"


def _prune_empty_conversations() -> None:
    """Delete conversations that have zero messages (orphaned chip clicks)."""
    for conv in list(st.session_state.conversations):
        if not conv.get("messages"):
            delete_conversation_api(conv["id"])
            st.session_state.conversations = [
                c for c in st.session_state.conversations
                if c["id"] != conv["id"]
            ]
            if st.session_state.current_conv_id == conv["id"]:
                st.session_state.current_conv_id = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Show a loading placeholder while checking the backend.
    if st.session_state.get("backend_ok") is None:
        ph = st.empty()
        ph.markdown(
            "<div style='text-align:center; padding:4rem; opacity:0.6;'>"
            "🔄 Connecting to backend\u2026</div>",
            unsafe_allow_html=True,
        )

    ok, msg = ensure_backend_running()
    st.session_state.backend_ok = ok
    st.session_state.backend_msg = msg

    # Clear the loading placeholder if it was shown.
    if "ph" in locals():
        ph.empty()

    init_session()
    apply_theme()

    # Clean up orphaned empty conversations from previous sessions.
    _prune_empty_conversations()

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
