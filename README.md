# Prompta
Prompta is a FastAPI-based backend for running open-source LLMs (starting with `openai/gpt-oss-20b`) and exposing OpenAI-compatible APIs (beginning with the legacy `v1/completions` endpoint).

⚡ FastAPI core – lightweight, async, production-ready API.

🧩 Inference – integrates with Hugging Face `transformers` and `accelerate`.

🔌 Extensible design – plug in different models, prompts, or pipelines.

📦 Developer-friendly – clean project structure, uv-based workflow.

## Quickstart (uv)

Prereqs: Python 3.12+, and `uv` installed (see https://docs.astral.sh/uv/).

1) Copy environment template and adjust as needed:

```bash
cp .env.example .env
```

2) Sync and lock dependencies with uv:

```bash
uv lock
uv sync
```

3) Run the API (development hot reload):

```bash
uv run serve-reload
```

Or run in production mode:

```bash
uv run serve
```

The server defaults to `http://0.0.0.0:8000`. Adjust with `.env` or pass `--host/--port` to `uvicorn` as needed.

### Optional: Faster HF downloads

Enable `hf_transfer` for faster model downloads:

```bash
uv add .[hf]
export HF_HUB_ENABLE_HF_TRANSFER=1
```

If using a gated/private model, set `PROMPTA_HF_TOKEN` in `.env`.

## Project structure

```
app/
  api/            # Routers (health, completions, batch, ...)
  core/           # Config, logging, middleware
  schemas/        # Pydantic models (requests/responses)
  services/       # Inference and related services
```

## Scripts

- `uv run serve-reload` – start dev server with hot reload.
- `uv run serve` – start server for production.

## License

See LICENSE.
