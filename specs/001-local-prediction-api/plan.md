# Implementation Plan: Local Prediction REST API

**Branch**: `feature/local-prediction-api` | **Date**: 2026-05-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-local-prediction-api/spec.md`

## Summary

Build a FastAPI REST API (`api.py`) that serves the daily baseball prediction pipeline outputs to local Claude Code sessions and scripts. The API reads pre-computed artifacts from `data/` (model.pkl, run_meta.json, best_bets_*.csv), exposes six read-only endpoints, and hot-reloads state when `daily_runner.py` produces a fresh run — all without re-training or recomputing Elo at request time.

## Technical Context

**Language/Version**: Python 3.11+ (same interpreter as `daily_runner.py`)

**Primary Dependencies**: FastAPI 0.110+, uvicorn[standard], Pydantic v2, httpx (tests only)

**Storage**: Filesystem only — `data/model.pkl`, `data/run_meta.json`, `data/best_bets_YYYY-MM-DD.csv`, `data/team_season_stats_2021_2026.parquet`

**Testing**: pytest + httpx.AsyncClient (no live server required)

**Target Platform**: Windows 11, localhost:8000, single worker

**Project Type**: Local web-service (REST API)

**Performance Goals**: < 500 ms p95 for pre-computed reads; < 2 s p95 for on-demand `POST /predict`

**Constraints**: Single worker, no auth, no external network calls, localhost only; zero raw tracebacks to callers

**Scale/Scope**: Single local user, 6 endpoints, ~300 teams, ~30k game rows in memory

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Artifact-First Pipeline | `api.py` reads artifacts; never re-trains or recomputes Elo | PASS |
| II. Schema-First API Contract | All Pydantic models defined before routes; `/openapi.json` accurate | PASS |
| III. Read-Only API in v1 | Only GET + `POST /predict` (no state mutation); localhost only; no auth | PASS |
| IV. Graceful Degradation | All errors return `ErrorResponse`; `resolve_team()` on all inputs; 404 + suggestions | PASS |
| V. Freshness Transparency | `model_version` + `generated_at` on all artifact responses; 60s poll + hot-reload | PASS |
| Dev Workflow IV | `api.py` imports from `pipeline/utils.py`, not notebook or patch scripts | PASS — pipeline/ module created first |

No violations. Complexity Tracking table omitted.

## Project Structure

### Documentation (this feature)

```text
specs/001-local-prediction-api/
├── plan.md              ← this file
├── research.md          ← Phase 0 output (complete)
├── data-model.md        ← Phase 1 output (complete)
├── quickstart.md        ← Phase 1 output (complete)
├── contracts/
│   └── endpoints.md     ← Phase 1 output (complete)
└── tasks.md             ← Phase 2 output (/speckit-tasks — not yet created)
```

### Source Code (repository root)

```text
pipeline/
├── __init__.py
└── utils.py             ← resolve_team(), era_adjustment(), kelly_fraction()

api.py                   ← FastAPI app (main deliverable)
tests/
└── test_api.py          ← pytest + httpx fixtures

data/                    ← gitignored artifacts (runtime reads)
    model.pkl
    run_meta.json
    best_bets_YYYY-MM-DD.csv
    team_season_stats_2021_2026.parquet
```

**Structure Decision**: Single-project layout. `pipeline/` is a proper Python package shared between `daily_runner.py` and `api.py`. No frontend or mobile components in v1.

## Implementation Order

The steps must be followed in sequence — each unblocks the next.

### Step 1 — Extract `pipeline/utils.py`

Copy `resolve_team()`, `era_adjustment()`, and `kelly_fraction()` from the notebook cells and `daily_runner.py` into `pipeline/utils.py`. Update `daily_runner.py` to `from pipeline.utils import resolve_team, era_adjustment, kelly_fraction`.

**Why first**: `api.py` cannot import from the notebook (constitution Dev Workflow IV). This module is the shared foundation for both the runner and the API.

**Files**: `pipeline/__init__.py` (empty), `pipeline/utils.py`, `daily_runner.py` (import update only — no logic changes)

### Step 2 — Define Pydantic schemas (constitution II)

Write all request/response models at the top of `api.py` before any route handlers. See `data-model.md` for full field specifications.

Models to define:
- Request: `GameRequest`, `PredictionsQuery`
- Response: `HealthStatus`, `ModelMeta`, `BetRecommendation`, `PredictionsResponse`, `GamePrediction`, `ErrorResponse`

### Step 3 — AppState + startup loading

Create `AppState` dataclass holding `clf`, `feats`, `run_meta`, `known_teams`, `bets_cache`, `last_reload`. Implement `load_artifacts(data_dir)` to populate it. Wire into FastAPI `lifespan`. If `model.pkl` is missing at startup, set `model_loaded = False` and continue (graceful degradation — constitution IV).

### Step 4 — Hot-reload background thread

Daemon thread polls `run_meta.json` every `RELOAD_INTERVAL_SECONDS` (env var, default 60). On `run_date` change, atomically replace the module-level `AppState` reference by calling `load_artifacts()` again (constitution V).

### Step 5 — GET endpoints (simplest → most complex)

1. `GET /health` — reads AppState fields directly
2. `GET /meta` — reads `AppState.run_meta`
3. `GET /predictions/{date}` — loads/caches CSV for that date; 404 if missing
4. `GET /predictions` — applies `date`, `team`, `market`, `min_edge` filters over date cache
5. `GET /teams/{team}/predictions` — scans available CSV files, filters rows by resolved team name

Apply `resolve_team()` on all team name inputs before any lookup. Return `ErrorResponse` with `suggestions` on team 404.

### Step 6 — POST /predict

Resolve team names → look up stats from in-memory parquet data → build feature vector → `clf.predict_proba()` → apply ERA adjustment → compute ATS probability via `math.erf` Normal CDF → return `GamePrediction`. Return 503 if `model_loaded = False`.

### Step 7 — Tests

pytest fixtures: temp `data/` directory with minimal valid artifacts (small parquet, serialized tiny sklearn model, one CSV). Test happy path and key error conditions for all 6 endpoints.
