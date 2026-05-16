<!-- Sync Impact Report
Version change: (template) → 1.0.0
Added sections: Core Principles (I–V), API Constraints, Development Workflow, Governance
Removed sections: n/a (initial population from template)
Templates updated:
  ✅ .specify/templates/constitution-template.md (source)
  ✅ .specify/memory/constitution.md (this file)
  ⚠ .specify/templates/spec-template.md (review before /speckit-specify)
  ⚠ .specify/templates/plan-template.md (review before /speckit-plan)
Follow-up TODOs: none
-->

# College Baseball Analytics Constitution

## Core Principles

### I. Artifact-First Pipeline

The daily runner (`daily_runner.py`) MUST produce all prediction artifacts to disk before
the API or any consumer reads them. Artifacts are the source of truth:
- `data/model.pkl` — trained XGBClassifier + feature list
- `data/run_meta.json` — freshness signal (run_date, generated_at, feature list, counts)
- `data/best_bets_YYYY-MM-DD.csv` — pre-computed recommendations
- `data/game_results_2021_2026.parquet` — canonical game log
- `data/team_season_stats_2021_2026.parquet` — canonical team stats

No component may re-train the model or recompute Elo at request time.
Rationale: reproducibility, latency, and separation of the training pipeline from the
serving layer.

### II. Schema-First API Contract

All API request and response shapes MUST be defined as Pydantic models before any route
handler is written. The Pydantic schema is the contract; the implementation must conform
to it, not the other way around.

FastAPI's auto-generated `/openapi.json` MUST remain accurate and complete at all times —
it is the primary interface description for Claude Code sessions and future mobile clients.

Rationale: stable contracts survive model changes; OpenAPI enables tool-driven querying
without separate documentation.

### III. Read-Only API in v1

The API MUST expose no write, delete, or admin endpoints in v1. All endpoints are GET
except `POST /predict` (on-demand single-game prediction — read-only in effect, no
state mutation).

No authentication, no rate limiting, no deployment to public hosts in v1.
Rationale: smallest useful surface; localhost-only removes auth complexity entirely.

### IV. Graceful Degradation Over Errors

When a requested team cannot be resolved, a date has no pre-computed bets, or
`model.pkl` is stale, the API MUST return a structured error response (never a 500
traceback) with a human-readable `detail` field and appropriate HTTP status code.

`resolve_team()` fuzzy matching MUST be applied on all team name inputs before any
lookup. Unknown teams return 404 with suggestions where possible.

Rationale: the primary consumers are Claude Code sessions and scripts; machine-readable
errors are more useful than stack traces.

### V. Freshness Transparency

Every response that derives from pre-computed artifacts MUST include `model_version` and
`generated_at` from `run_meta.json`. The `/health` endpoint MUST surface whether the
last run was today's date.

The API MUST poll `run_meta.json` for a changed `run_date` at startup and on a
configurable interval (default 60 s) and hot-reload in-memory state when freshness
changes. No manual restart required after the daily runner completes.

Rationale: a consumer must always know how old the predictions are.

## API Constraints

**In scope for v1:**
- `GET /health` — freshness, last run timestamp, model loaded flag
- `GET /meta` — train years, feature list, TEST_YEAR, calibration info
- `GET /predictions` — query params: `date`, `team`, `market` (ML|ATS), `min_edge`
- `GET /predictions/{date}` — all pre-computed recommendations for a date (YYYY-MM-DD)
- `GET /teams/{team}/predictions` — upcoming predictions filtered by team name
- `POST /predict` — on-demand single-game prediction body: `{home, away, neutral, date}`

**Explicitly out of scope for v1:**
- Historical backtests or ATS records
- Multi-model comparison endpoints
- Live odds fetching at request time
- Kelly scenario sweep endpoints
- Team profile / radar chart endpoints
- Auth, rate limiting, deployment
- Any endpoint that mutates state or triggers re-training

**Runtime:** FastAPI + uvicorn, localhost:8000 only, single worker for v1.

## Development Workflow

1. All new pipeline steps (data pull, feature engineering, model training) belong in
   `daily_runner.py` or standalone `*.py` scripts — never inside the notebook.
2. Notebook cells are JSON-edited programmatically via `patch_*.py` scripts; never edit
   `CollegeBaseballAnalytics_Master.ipynb` directly in a text editor.
3. After any notebook change: update `push_notebook.py` SHA and run it. Confirm the
   GitHub push succeeded before closing the session.
4. The API (`api.py`) MUST NOT import from the notebook or any `patch_*.py` file.
   Shared logic (e.g., `resolve_team`, `era_adjustment`, `kelly_fraction`) MUST be
   extracted to a `pipeline/` module importable by both `daily_runner.py` and `api.py`.
5. `data/` is gitignored (binary artifacts). `run_meta.json` MAY be committed as a
   lightweight freshness record if helpful for CI.

## Governance

This constitution supersedes informal conventions. Amendments require:
1. A clear rationale (what changed and why).
2. An update to this file with version bump per semantic versioning rules.
3. A review of affected templates in `.specify/templates/`.

All spec, plan, and task documents generated by Spec Kit MUST be consistent with
these principles. Conflicts are resolved in favor of the constitution.

**Version**: 1.0.0 | **Ratified**: 2026-05-16 | **Last Amended**: 2026-05-16
