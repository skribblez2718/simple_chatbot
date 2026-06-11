# ────────────────────────────────────────────────────────────
# simple_chatbot — Makefile
# Targets: install, run, test, lint, format, clean
# ────────────────────────────────────────────────────────────

.PHONY: install run test lint format clean

# ─── Dependencies ──────────────────────────────────────────

UV := uv

# ─── Targets ───────────────────────────────────────────────

install:  ## Create .venv and install all dependencies (including dev)
	$(UV) venv .venv
	$(UV) pip install -e ".[dev]"
	@echo ""
	@echo "✅  Done. Activate with:  source .venv/bin/activate"
	@echo "    Or use:               uv run <command>"

run:  ## Start the Streamlit frontend (auto-launches the FastAPI backend)
	$(UV) run streamlit run frontend/app.py

test:  ## Run the full test suite (models mocked — no downloads needed)
	$(UV) run pytest tests/ -v

lint:  ## Lint-check all source files with ruff
	$(UV) run ruff check backend/ frontend/ tests/

format:  ## Auto-format all source files with ruff
	$(UV) run ruff format backend/ frontend/ tests/

clean:  ## Remove virtual environment, caches, and runtime artifacts
	rm -rf .venv/
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/
	rm -rf *.egg-info/
	rm -rf .uv/
	rm -f chatbot.db
	@echo "✅  Done."