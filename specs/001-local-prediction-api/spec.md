# Feature Specification: Local Prediction REST API

**Feature Branch**: `feature/local-prediction-api`

**Created**: 2026-05-16

**Status**: Draft

**Input**: User description: "A local REST API that exposes the daily prediction pipeline outputs so I can query them easily from other Claude Code sessions (and eventually mobile / other tools). Endpoints include health/meta, predictions by date/team/market, on-demand single-game prediction, and freshness transparency throughout."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Query Today's Best Bets (Priority: P1)

A Claude Code session (or curl command) sends a GET request for today's pre-computed best bets and receives a structured list with market, edge, Kelly size, and freshness metadata — without needing to open the notebook or run any Python manually.

**Why this priority**: This is the primary daily use case. The pipeline already produces `best_bets_YYYY-MM-DD.csv`; the API just needs to serve it. All other features are secondary to this one.

**Independent Test**: Start the API server after `daily_runner.py` has run once. A `GET /predictions/{today}` call must return a non-empty list of bets with `market`, `edge`, `kelly_size`, `model_version`, and `generated_at` fields. Delivers value immediately even if no other endpoints exist.

**Acceptance Scenarios**:

1. **Given** `daily_runner.py` has run today and produced `best_bets_2026-05-16.csv`, **When** `GET /predictions/2026-05-16` is called, **Then** the response contains a list of bet objects each with `home`, `away`, `market`, `edge`, `kelly_size`, `model_version`, `generated_at` and HTTP 200.
2. **Given** no bets file exists for a requested date, **When** `GET /predictions/2026-05-17` is called, **Then** the response returns HTTP 404 with `{"detail": "No pre-computed bets found for 2026-05-17"}`.
3. **Given** the bets file for today exists, **When** `GET /predictions` is called with `?market=ATS&min_edge=0.05`, **Then** only rows with `market == "ATS"` and `edge >= 0.05` are returned.

---

### User Story 2 — Check Pipeline Freshness (Priority: P1)

Before querying predictions, a user or script checks whether the pipeline ran today so they know if the data is current or stale.

**Why this priority**: Stale predictions are worse than no predictions. Freshness must be surfaced at every query. Tied for P1 with bet querying because it gates trust in all other responses.

**Independent Test**: Call `GET /health` immediately after starting the server. Response must include `last_run_date`, `is_fresh` (bool), `model_loaded`, and `generated_at`. Independently verifiable without any other endpoint.

**Acceptance Scenarios**:

1. **Given** `daily_runner.py` ran today, **When** `GET /health` is called, **Then** the response includes `"is_fresh": true`, `"last_run_date": "2026-05-16"`, and `"model_loaded": true`.
2. **Given** `daily_runner.py` last ran yesterday, **When** `GET /health` is called, **Then** `"is_fresh": false` and `"last_run_date": "2026-05-15"`.
3. **Given** `daily_runner.py` finishes while the API is running, **When** the API detects the changed `run_meta.json` (within 60 seconds), **Then** subsequent `/health` calls reflect the new run without restarting the server.

---

### User Story 3 — On-Demand Single-Game Prediction (Priority: P2)

A user wants to predict a specific game not yet in the pre-computed bets — for example, a neutral-site tournament game or a future regular-season matchup — by providing home team, away team, optional neutral site flag, and date.

**Why this priority**: The pre-computed bets cover scheduled games for the current day. On-demand prediction fills the gap for arbitrary matchups. P2 because it requires the model to be loaded in memory and is more compute-intensive than a file read.

**Independent Test**: With `model.pkl` present, `POST /predict` with `{"home": "Texas", "away": "LSU", "neutral": true, "date": "2026-05-20"}` must return a win probability, predicted run differential, and both ML and ATS recommendations. No bet file needed.

**Acceptance Scenarios**:

1. **Given** `model.pkl` is loaded, **When** `POST /predict` is called with valid home/away/neutral/date body, **Then** response includes `home_wp`, `away_wp`, `pred_run_diff`, `ml_recommendation`, `ats_recommendation`, `model_version`, `generated_at`.
2. **Given** the request uses a fuzzy team name (e.g., "arizona st"), **When** `POST /predict` is called, **Then** the name resolves to "Arizona State" and prediction proceeds normally (not a 404).
3. **Given** a team name that cannot be resolved, **When** `POST /predict` is called with `{"home": "Zzzteam", "away": "LSU"}`, **Then** HTTP 422 is returned with `{"detail": "Cannot resolve team 'Zzzteam'", "suggestions": [...]}`.

---

### User Story 4 — Filter Predictions by Team (Priority: P2)

A user wants to see all upcoming predictions for a specific team across multiple dates — for example, all games involving "Texas" in the next week.

**Why this priority**: Useful for tracking a team's schedule and betting slate. Requires iterating across multiple date files, so more complex than single-date queries.

**Independent Test**: With at least two days of bet files containing "Texas" games, `GET /teams/Texas/predictions` must return all matching rows across both files, each with a `date` field.

**Acceptance Scenarios**:

1. **Given** bet files exist for May 16 and May 17 each containing at least one Texas game, **When** `GET /teams/Texas/predictions` is called, **Then** all Texas rows from both files are returned with a `date` field on each.
2. **Given** the team name is provided in a different case or abbreviation (e.g., "texas" or "Tex"), **When** `GET /teams/texas/predictions` is called, **Then** the name resolves correctly and returns the same results as "Texas".
3. **Given** no bet files contain the requested team, **When** `GET /teams/XYZTeam/predictions` is called, **Then** HTTP 404 with `{"detail": "No predictions found for team 'XYZTeam'"}`.

---

### User Story 5 — Inspect Model Metadata (Priority: P3)

A developer or Claude Code session inspects the model's training configuration — features used, test year, calibration info — to understand what the predictions are based on.

**Why this priority**: Useful for debugging and transparency but not needed for daily betting use.

**Independent Test**: `GET /meta` returns `train_cutoff`, `test_year`, `feature_list`, `model_version`, and `generated_at` from `run_meta.json`. No model inference required.

**Acceptance Scenarios**:

1. **Given** `run_meta.json` exists, **When** `GET /meta` is called, **Then** response includes `train_cutoff`, `test_year`, `feature_list` (array), `model_version`, `generated_at`.
2. **Given** `run_meta.json` is missing, **When** `GET /meta` is called, **Then** HTTP 503 with `{"detail": "Pipeline artifacts not found — run daily_runner.py first"}`.

---

### User Story 6 — Automatic Model Improvement (Priority: P2)

When the daily pipeline retrains the model and the new version outperforms the stored baseline on a held-out validation set, the improvement is automatically committed to git, pushed to GitHub, and deployed to the VPS — with no manual steps required.

**Why this priority**: Manual deploy is a friction point that delays improvements reaching production. With a fixed held-out evaluation set, automatic promotion is safe: the model only ships if it demonstrably improves on data it never trained on.

**Independent Test**: Run `daily_runner.py` twice in sequence. After the first run, `data/model_baseline.json` must exist. After the second run, `daily_log.txt` must contain either `"auto-improved"` or `"baseline retained"`. If improved, a new git commit must appear in the log matching the auto-commit message pattern.

**Acceptance Scenarios**:

1. **Given** no baseline exists (first run), **When** `daily_runner.py` completes training, **Then** `data/model_baseline.json` is written with current AUC, Brier, feature list, timestamp, and git SHA; a commit is created and pushed.
2. **Given** a baseline exists and the new model improves AUC by ≥ 0.002, **When** `daily_runner.py` completes training, **Then** a git commit is created with message matching `chore(model): auto-improve AUC=... Brier=... [date]`, pushed to origin, `model_baseline.json` is updated, and `sync_data_to_vps()` deploys the new model to the VPS.
3. **Given** a baseline exists and the new model does NOT improve (Δ AUC < 0.002 AND Δ Brier < 0.002), **When** `daily_runner.py` completes training, **Then** no git commit is created, baseline is unchanged, and `daily_log.txt` logs `"Model retrained but did not improve — baseline retained"`.
4. **Given** a git push fails (no remote, auth error), **When** `daily_runner.py` auto-commit fires, **Then** the failure is logged to `daily_log.txt` and the pipeline run completes normally (push failure must not abort the run).
5. **Given** the VPS deploy step fails (SSH unreachable, wrong password), **When** `daily_runner.py` auto-commit fires, **Then** the failure is logged and the rest of the pipeline completes — the local commit and push still succeed.

---

### Edge Cases

- What happens when `model.pkl` is missing at startup? API starts but `POST /predict` returns 503 until artifacts are present.
- What happens when `best_bets_YYYY-MM-DD.csv` is malformed (missing columns)? API returns 500 with a human-readable detail, never a raw traceback.
- What happens when a team name contains special characters or extra whitespace? `resolve_team()` normalizes whitespace and strips non-alphanumeric before matching.
- What happens when the API is queried while `daily_runner.py` is actively writing artifacts? File-write atomicity is assumed (write to temp → rename); API reads completed files only.
- What happens when `min_edge` is negative or greater than 1? API returns HTTP 422 with a validation error from Pydantic before touching any data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The API MUST expose `GET /health` returning pipeline freshness status, last run date, and model-loaded flag.
- **FR-002**: The API MUST expose `GET /meta` returning model training configuration from `run_meta.json`.
- **FR-003**: The API MUST expose `GET /predictions` with optional query parameters `date` (YYYY-MM-DD), `team` (fuzzy-matched), `market` (`ML` | `ATS`), and `min_edge` (float 0–1).
- **FR-004**: The API MUST expose `GET /predictions/{date}` returning all pre-computed bet recommendations for that date.
- **FR-005**: The API MUST expose `GET /teams/{team}/predictions` returning all pre-computed recommendations for a given team across available dates, with fuzzy name resolution applied to `{team}`.
- **FR-006**: The API MUST expose `POST /predict` accepting `{home, away, neutral, date}` and returning an on-demand single-game prediction using the loaded `model.pkl`.
- **FR-007**: The API MUST apply `resolve_team()` fuzzy matching to all team name inputs before any lookup or prediction.
- **FR-008**: The API MUST include `model_version` and `generated_at` fields on every response that derives from pre-computed artifacts.
- **FR-009**: The API MUST return structured error responses (never raw tracebacks) with a `detail` field and appropriate HTTP status codes for all error conditions.
- **FR-010**: The API MUST poll `run_meta.json` on a configurable interval (default 60 seconds) and hot-reload in-memory state when the `run_date` changes, without requiring a server restart.
- **FR-011**: The API MUST NOT expose any write, delete, admin, or state-mutating endpoints.
- **FR-012**: The API MUST NOT re-train the model or recompute Elo at request time.
- **FR-013**: Unknown team names MUST return HTTP 404 with a `suggestions` field listing close matches where available.
- **FR-014**: All request and response shapes MUST be defined as Pydantic models; the auto-generated `/openapi.json` MUST remain accurate at all times.
- **FR-015**: The API MUST run on `localhost:8000` only in v1 with a single worker process.
- **FR-016**: After each successful model retrain, `daily_runner.py` MUST write `data/model_baseline.json` containing AUC, Brier score, feature list, trained-at timestamp, and current git SHA.
- **FR-017**: A model is considered improved if new AUC exceeds baseline AUC by ≥ 0.002 OR new Brier score is lower than baseline by ≥ 0.002, evaluated on a fixed held-out validation slice (the most recent training year, never the live season).
- **FR-018**: When improvement is detected, `daily_runner.py` MUST create a git commit with the updated `model.pkl` and `model_baseline.json`, push to origin, and call `sync_data_to_vps()`.
- **FR-019**: When no improvement is detected, `daily_runner.py` MUST log the verdict to `daily_log.txt` and leave baseline and git history unchanged.
- **FR-020**: Auto-commit/deploy failures (git errors, SSH errors) MUST be caught and logged; they MUST NOT abort the daily pipeline run.

### Key Entities

- **Prediction**: A pre-computed bet recommendation for a specific game. Key attributes: home team, away team, date, market (ML or ATS), edge (float), kelly_size (float), model_version (str), generated_at (ISO datetime).
- **GameRequest**: Input for on-demand prediction. Key attributes: home (str), away (str), neutral (bool, default false), date (YYYY-MM-DD).
- **GamePrediction**: Output of on-demand prediction. Key attributes: home_wp (float), away_wp (float), pred_run_diff (float), ml_recommendation (str), ats_recommendation (str), model_version (str), generated_at (ISO datetime).
- **HealthStatus**: Pipeline health snapshot. Key attributes: is_fresh (bool), last_run_date (date), model_loaded (bool), generated_at (ISO datetime).
- **ModelMeta**: Training configuration. Key attributes: train_cutoff (int), test_year (int), feature_list (list[str]), model_version (str), generated_at (ISO datetime).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A Claude Code session can retrieve today's best bets in under 2 seconds from the time the request is sent.
- **SC-002**: All six v1 endpoints return correct responses 100% of the time when artifacts are present and valid.
- **SC-003**: Fuzzy team name resolution correctly matches common name variants (abbreviations, case differences, St vs State) with no false positives on a test set of 20 known team aliases.
- **SC-004**: The API detects a new `daily_runner.py` run and refreshes in-memory state within 60 seconds, without a server restart.
- **SC-005**: All error responses include a human-readable `detail` field; zero raw Python tracebacks reach the caller under any input condition.
- **SC-006**: The auto-generated `/openapi.json` accurately describes all endpoints and schemas with no missing fields or incorrect types.
- **SC-007**: Every `daily_runner.py` run produces a `daily_log.txt` entry that explicitly states either `"auto-improved"` (with AUC/Brier delta) or `"baseline retained"` (with delta and reason); zero silent failures.
- **SC-008**: A model improvement (AUC Δ ≥ 0.002) detected during a pipeline run reaches the production VPS API within the same run, requiring zero manual steps.

## Assumptions

- The daily runner (`daily_runner.py`) has run at least once before the API is queried for predictions; the API does not bootstrap its own data.
- The API serves a single local user (developer / Claude Code session) — no concurrent-user or rate-limiting requirements in v1.
- `resolve_team()` and `era_adjustment()` shared logic will be extracted from the notebook into a `pipeline/` module before `api.py` imports them, per the constitution's Development Workflow.
- Mobile client support is a future consideration; v1 is localhost-only with no CORS or auth requirements.
- Artifact files are written atomically by `daily_runner.py` (write to temp path, then rename), so the API will never read a partially-written file.
- Windows Task Scheduler handles the daily `daily_runner.py` execution; the API does not need to trigger or schedule the runner.
