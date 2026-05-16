# Phase 0 Research: Local Prediction REST API

**Status**: Complete — no NEEDS CLARIFICATION items remained after spec validation.

## Decisions

### Web Framework

- **Decision**: FastAPI + uvicorn, single worker
- **Rationale**: Auto-generates `/openapi.json` from Pydantic models (constitution II); async-ready but no async needed for file I/O at this scale; minimal boilerplate for 6 endpoints.
- **Alternatives considered**: Flask (no native Pydantic/OpenAPI), aiohttp (too low-level), Django REST (too heavy for a single-user local tool).

### Pydantic Version

- **Decision**: Pydantic v2 (bundled with FastAPI 0.110+)
- **Rationale**: V2 validators are stricter by default — catches bad inputs (negative `min_edge`, invalid date strings) before route logic runs, satisfying FR-009.
- **Alternatives considered**: Pydantic v1 — would work but v2 is the current default and has better performance.

### Hot-Reload Strategy

- **Decision**: Background thread polling `run_meta.json` every 60 seconds (configurable via `RELOAD_INTERVAL_SECONDS` env var). On `run_date` change, reload `model.pkl` and in-memory bets cache.
- **Rationale**: Constitution V requires hot-reload without restart. A background thread is the simplest approach for a single-worker process; avoids the complexity of `watchdog` or `inotify` on Windows.
- **Alternatives considered**: `watchdog` file-system events (cross-platform complexity), FastAPI lifespan with asyncio periodic task (works but overkill for 60s polling).

### Shared Pipeline Module

- **Decision**: Extract `resolve_team()`, `era_adjustment()`, and `kelly_fraction()` into `pipeline/utils.py` before writing `api.py`.
- **Rationale**: Constitution Development Workflow item 4: "api.py MUST NOT import from the notebook or any patch_*.py file." Both `daily_runner.py` and `api.py` need these functions.
- **Alternatives considered**: Copy-paste into api.py (violates DRY and constitution), import from notebook (explicitly forbidden by constitution).

### Team Name Resolution

- **Decision**: Reuse the existing `resolve_team()` function from the notebook (moved to `pipeline/utils.py`). No additional library needed.
- **Rationale**: Already handles the direct override dict, case-insensitive exact match, St→State normalization, and difflib fuzzy match at cutoff=0.72. Tested against the known-teams set from parquet.
- **Alternatives considered**: `rapidfuzz` library (better performance but unnecessary at this scale), `thefuzz` (same).

### On-Demand Prediction

- **Decision**: `POST /predict` loads the XGBClassifier from `model.pkl` (already in memory), runs inference with the 11 training features, applies ERA adjustment, returns win probabilities.
- **Rationale**: Model is already loaded at startup; no re-training. Satisfies FR-012.
- **Alternatives considered**: Re-running daily_runner.py for each prediction (too slow, violates constitution I).

### Testing

- **Decision**: pytest + `httpx.AsyncClient` (FastAPI's recommended test client).
- **Rationale**: Allows end-to-end route testing without starting a real server; fixtures can swap in test artifact files.
- **Alternatives considered**: `requests` with real server (slower, stateful), unittest (more boilerplate).

### Error Responses

- **Decision**: FastAPI's default `HTTPException` mechanism with Pydantic `ErrorResponse` model for all non-validation errors. Validation errors use FastAPI's built-in 422 response (already Pydantic-structured).
- **Rationale**: Never raw tracebacks (FR-009). FastAPI's exception handlers make this straightforward.
