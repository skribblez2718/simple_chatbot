# Simple Chatbot

A local AI chatbot capstone project for the **Building Generative AI-Powered Applications with Python** course, part of the **IBM Generative AI Engineering** professional certificate.

The application uses a **FastAPI** backend serving a HuggingFace model of your choice — configurable via `.env` — and a **Streamlit** frontend providing a ChatGPT-like chat interface.

## Features

- 🤖 **Local AI Inference** — Runs a model on your machine via HuggingFace Transformers
- ⚡ **Token Streaming** — Responses stream token-by-token via SSE, rendered incrementally in the UI
- 💬 **ChatGPT-style UI** — User messages on the right (blue-tinted bubble), bot messages on the left
- 📁 **Conversation Management** — Create, switch between, and delete multiple conversations with two-click delete confirmation
- 🎨 **Light / Dark Theme** — Toggle between light and dark modes from the sidebar; persists across page loads
- 📱 **Responsive Design** — Adapts to desktop, tablet, and mobile viewports with collapsing sidebar
- 💾 **SQLite Persistence** — Conversations stored in a local SQLite database, survive page reloads and restarts
- 🚀 **One-Command Startup** — Streamlit auto-launches the FastAPI backend in a background thread
- 🔒 **No Authentication** — Per scope (capstone project)

## Prerequisites

- **Python 3.11+**
- **uv** (package manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Enough free (V)RAM** — the 7B BF16 model needs ~14 GB of combined VRAM/system RAM
- **Internet access** — to download the model from HuggingFace on first run

## Install

```bash
# 1. Clone / cd into the project
cd ~/projects/simple_chatbot

# 2. Create the virtual environment with uv
uv venv .venv

# 3. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 4. Install dependencies (from pyproject.toml)
uv pip install -e .
```

> To run the test suite, install the dev extras: `uv pip install -e ".[dev]"`

## Configuration (`.env`)

All runtime settings live in a `.env` file at the project root. Copy the template and adjust:

```bash
cp .env.example .env
```

The backend and frontend both `load_dotenv()` on startup.

| Variable                   | Default                  | Purpose                                                                          |
| -------------------------- | ------------------------ | -------------------------------------------------------------------------------- |
| `CHATBOT_MODEL_NAME`       | `DeepHat/DeepHat-V1-7B`  | HuggingFace model id used for inference                                          |
| `CHATBOT_BACKEND_HOST`     | `127.0.0.1`              | Host the embedded FastAPI server binds to                                        |
| `CHATBOT_BACKEND_PORT`     | `8000`                   | Port the embedded FastAPI server binds to                                        |
| `CHATBOT_FRONTEND_HOST`    | `localhost`              | Streamlit host — used to build the CORS allow-list                               |
| `CHATBOT_FRONTEND_PORT`    | `8501`                   | Streamlit port — used to build the CORS allow-list                               |
| `CHATBOT_EMBED_BACKEND`    | `1`                      | Set to `0` to disable the auto-start and run the backend manually                |

## GPU Support

The backend auto-detects the best compute device on startup and logs which one it chose:

| Priority | Device | When it is used                                        |
| -------- | ------ | ------------------------------------------------------ |
| 1        | `cuda` | An NVIDIA GPU is present (CUDA) or an AMD GPU is present (ROCm — PyTorch exposes ROCm through the same `torch.cuda` API) |
| 2        | `mps`  | Running on Apple Silicon (M1 / M2 / M3 / M4)           |
| 3        | `cpu`  | No GPU detected — the model loads into system RAM      |

You'll see a log line like one of these on first run:

```
INFO  Detected NVIDIA GPU via CUDA: NVIDIA GeForce RTX 4090
INFO  Detected AMD GPU via ROCm (5.7.0): AMD Radeon RX 7900
INFO  Detected Apple Silicon via MPS
INFO  No GPU detected — using CPU (model will run in system RAM)
```

### Multi-GPU sharding

When `CUDA_VISIBLE_DEVICES` exposes more than one GPU, Accelerate's `device_map="auto"` shards the model across them automatically with an OOM fallback chain:

```
INFO  Loading model DeepHat/DeepHat-V1-7B (visible CUDA devices: ['cuda:0', 'cuda:1'], primary: cuda) ...
INFO  Model sharded: cuda:0=19 layers, cuda:1=13 layers
```

If sharding fails, the loader falls back to single-GPU and then CPU. Every fallback attempt is logged.

## Run (Single Command)

```bash
streamlit run frontend/app.py
```

Streamlit will:

1. Open the chat UI on **http://localhost:8501**
2. Detect that the FastAPI backend isn't running yet
3. Launch it in a **background thread** on **http://127.0.0.1:8000**
4. Pre-load the model on startup (first run downloads the model from HuggingFace)

When the model finishes loading, the sidebar's "🟢 Backend connected" indicator turns green and you can start chatting. **Press `Ctrl + C`** in the terminal to stop both servers.

## Run (Two-Command Mode)

To run the backend and frontend separately (e.g. for debugging or multi-machine setups):

```bash
# Terminal 1 — Backend
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
source .venv/bin/activate
CHATBOT_EMBED_BACKEND=0 streamlit run frontend/app.py
```

## Project Structure

```
simple_chatbot/
├── .env                     # Local config (not committed) — see .env.example
├── .env.example             # Template for .env
├── .gitignore
├── pyproject.toml           # Project config and dependencies
├── README.md
├── chatbot.db               # SQLite database (auto-created, not committed)
├── .streamlit/
│   └── config.toml          # Streamlit theme defaults (dark mode)
├── backend/
│   └── main.py              # FastAPI server — GPU auto-detect, model loading, REST + SSE API
├── frontend/
│   └── app.py               # Streamlit UI — auto-starts backend, chat, theming, responsive CSS
└── tests/
    ├── conftest.py          # Pytest fixtures (mocked model, temp DB)
    ├── test_backend.py      # Backend API tests
    ├── test_config.py       # Env-var + device-detection tests
    ├── test_frontend.py     # Frontend helper tests
    └── test_integration.py  # Real-server integration tests
```

## API Endpoints

| Method | Endpoint                          | Description                                   |
| ------ | --------------------------------- | --------------------------------------------- |
| GET    | `/`                               | API status and model name                     |
| GET    | `/health`                         | Health check (used by the auto-start probe)   |
| POST   | `/chat`                           | Stateless chat (no conversation context)      |
| GET    | `/conversations`                  | List all conversations                        |
| POST   | `/conversations`                  | Create a new conversation                     |
| GET    | `/conversations/{id}`             | Get a single conversation by ID               |
| DELETE | `/conversations/{id}`             | Delete a conversation                         |
| POST   | `/conversations/{id}/messages`    | Add a message, get AI response (sync)         |
| POST   | `/conversations/{id}/messages/stream` | Add a message, stream AI response via SSE |

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The test suite mocks the HuggingFace model and uses a temporary SQLite database so it runs in a few seconds without downloading the model or touching `chatbot.db`. 44 tests cover the backend API, frontend helpers, and configuration.

```bash
# Lint check
ruff check backend/ frontend/ tests/
```

## Security

- **CORS**: Restricted to `http://127.0.0.1:8501` and `http://localhost:8501` only
- **Input Validation**: All API inputs validated with Pydantic schemas (rejects empty/oversized/missing fields)
- **Error Handling**: Generic error messages returned to the client — no stack traces or internal details leaked
- **No Authentication**: Out of scope for the capstone. **Do not expose this app to the internet without adding auth first.**

## Design Decisions

- **Single-command startup** — The Streamlit frontend auto-launches FastAPI via `uvicorn.Server` in a daemon thread, so the capstone student gets chatting with one command.
- **Token streaming** — Responses stream token-by-token via SSE (`/conversations/{id}/messages/stream`) and are rendered incrementally in the chat UI using `st.write_stream()`.
- **Conversation history in the backend** — The model uses chat templates that benefit from the full prior exchange as context, so messages are kept server-side and passed with every request (last 20 exchanges).
- **SQLite persistence** — Conversations and messages are stored in a local SQLite database (`chatbot.db`) with proper schema (foreign keys, indexing, WAL mode). Survives restarts and browser reloads.
- **Model pre-loading** — The model loads on server startup (not lazily on first request) so the first chat interaction is responsive.
- **UUID conversation IDs** — Avoids collisions and prevents trivial enumeration of other users' chats (moot here since there's no auth, but the right pattern).
- **Responsive CSS** — Media queries adapt the layout from desktop (1280px) down to mobile (375px) with collapsing sidebar, wider message bubbles, and larger touch targets.

## Edge Cases Handled

| Scenario                                          | Behaviour                                     |
| ------------------------------------------------- | --------------------------------------------- |
| Model not yet downloaded                         | Frontend shows "🔄 Connecting to backend…" while loading; backend returns a friendly "still loading" message until ready |
| Backend unreachable                               | Frontend sidebar shows "🔴 Backend Unreachable" with a "Retry" button |
| Empty model response                              | Returns the placeholder `"No response generated."` |
| Empty / missing / non-string message              | Rejected with HTTP 422                        |
| Conversation not found                            | Returns 404                                   |
| Frontend tries to chat with no conversation open  | Frontend auto-creates a new conversation      |
| User reloads the browser                          | Conversations reload from SQLite via `GET /conversations` |
| Streaming fails mid-response                      | User's prompt is already saved; orphaned empty conversations are auto-pruned |

## License

This project is for educational use as part of the IBM Generative AI Engineering certificate. The DeepHat-V1-7B model is governed by the Apache-2.0 license plus the DeepHat Extended License (see the [model card](https://huggingface.co/DeepHat/DeepHat-V1-7B) for usage restrictions).
