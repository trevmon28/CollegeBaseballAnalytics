# Tasks: Local Prediction REST API

**Input**: Design documents from `specs/001-local-prediction-api/`

**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/endpoints.md ✅

**Organization**: Tasks grouped by user story. Each phase is independently testable and deliverable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no in-flight dependencies)
- **[Story]**: User story this task serves (US1–US5 from spec.md)

---

## Phase 1: Setup

**Purpose**: Create the package structure and install runtime dependencies.

- [ ] T001 Create `pipeline/__init__.py` (empty) at repository root
- [ ] T002 Install API dependencies: `pip install fastapi "uvicorn[standard]" httpx pytest`
- [ ] T003 [P] Create `tests/` directory with empty `tests/__init__.py` and `tests/conftest.py` stub

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared code that both `daily_runner.py` and `api.py` depend on. Must be complete before any endpoint can be implemented.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [ ] T004 Extract `resolve_team()` from notebook Cell 12 into `pipeline/utils.py` (include `_ODDS_TO_ESPN` dict, `_norm()` helper, `difflib` import)
- [ ] T005 [P] Extract `era_adjustment()` from notebook Cell 12 into `pipeline/utils.py`
- [ ] T006 [P] Extract `kelly_fraction(edge, fraction=0.25, cap=0.10)` from `daily_runner.py` into `pipeline/utils.py`
- [ ] T007 Update `daily_runner.py` to `from pipeline.utils import resolve_team, era_adjustment, kelly_fraction` (remove inline definitions)
- [ ] T008 Define all Pydantic **request** models at top of `api.py`: `GameRequest`, `PredictionsQuery` (with `min_edge` validator 0–1)
- [ ] T009 [P] Define all Pydantic **response** models in `api.py`: `HealthStatus`, `ModelMeta`, `BetRecommendation`, `PredictionsResponse`, `GamePrediction`, `ErrorResponse`
- [ ] T010 Implement `AppState` dataclass in `api.py` with fields: `clf`, `feats`, `run_meta`, `known_teams`, `bets_cache`, `last_reload`, `model_loaded`
- [ ] T011 Implement `load_artifacts(data_dir)` in `api.py`: load `model.pkl`, `run_meta.json`, build `known_teams` set from `team_season_stats` parquet; set `model_loaded=False` gracefully if pkl missing
- [ ] T012 Wire FastAPI `lifespan` context manager to call `load_artifacts()` at startup and start hot-reload background daemon thread (poll `run_meta.json` every `RELOAD_INTERVAL_SECONDS`, default 60; replace `AppState` reference on `run_date` change)

**Checkpoint**: `pipeline/utils.py` importable, `api.py` starts without error, `AppState` populated from real artifacts.

---

## Phase 3: User Story 2 — Pipeline Freshness (Priority: P1)

**Goal**: Any script or Claude Code session can verify the pipeline ran today before trusting any predictions.

**Independent Test**: Start the server (`uvicorn api:app --port 8000`). `curl http://localhost:8000/health` returns JSON with `is_fresh`, `last_run_date`, `model_loaded`, and `generated_at`. Server never crashes regardless of artifact state.

- [ ] T013 [US2] Implement `GET /health` route in `api.py`: read from module-level `AppState`; return `HealthStatus`; never raise (return `model_loaded=False` if pkl absent)
- [ ] T014 [US2] Manual smoke test: start server, run `curl http://localhost:8000/health`, confirm all four fields present and `is_fresh` matches whether `daily_runner.py` ran today

**Checkpoint**: `/health` returns correct freshness status. US2 fully functional.

---

## Phase 4: User Story 1 — Query Today's Best Bets (Priority: P1) 🎯 MVP

**Goal**: Retrieve pre-computed bet recommendations by date with optional market and edge filters.

**Independent Test**: `curl http://localhost:8000/predictions/2026-05-16` returns a list of `BetRecommendation` objects each with `market`, `edge`, `kelly_size`, `recommended_bet`, `model_version`, `generated_at`. `curl "http://localhost:8000/predictions?market=ATS&min_edge=0.05"` returns only ATS rows above the threshold.

- [ ] T015 [US1] Implement `load_bets_for_date(date, data_dir)` helper in `api.py`: read `best_bets_{date}.csv` into list of `BetRecommendation`; cache result in `AppState.bets_cache`; return `None` if file missing
- [ ] T016 [US1] Implement `GET /predictions/{date}` route in `api.py`: call `load_bets_for_date`, return `PredictionsResponse`; 404 with `ErrorResponse` if file not found
- [ ] T017 [US1] Implement `GET /predictions` route in `api.py`: load bets for `date` param (or today if omitted), apply `team` (fuzzy), `market`, and `min_edge` filters; return `PredictionsResponse`; 404 if no results after filtering
- [ ] T018 [US1] Manual smoke test: `curl http://localhost:8000/predictions/2026-05-16` confirms bet list; `curl "http://localhost:8000/predictions?market=ML&min_edge=0.03"` returns subset; `curl http://localhost:8000/predictions/1999-01-01` returns 404 with `detail` field

**Checkpoint**: `/predictions` and `/predictions/{date}` fully functional. MVP is complete — US1 + US2 both working.

---

## Phase 5: User Story 3 — On-Demand Single-Game Prediction (Priority: P2)

**Goal**: Predict any arbitrary matchup on demand using the loaded model, without re-training.

**Independent Test**: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"home":"Texas","away":"LSU","neutral":false}'` returns `home_wp + away_wp ≈ 1.0` and both recommendation fields. Fuzzy name `"arizona st"` resolves correctly. Unknown team returns 422 with `suggestions`.

- [ ] T019 [US3] Implement `build_feature_vector(home, away, neutral, team_stats_df)` helper in `api.py`: look up stats for both teams from the in-memory parquet data, compute differential features matching the 11 model features in `AppState.feats`; return as dict or raise if team not in stats
- [ ] T020 [US3] Implement `POST /predict` route in `api.py`: resolve team names → `build_feature_vector` → `clf.predict_proba` → apply `era_adjustment` → compute ATS probability via `math.erf` Normal CDF (RD_SIGMA=3.5) → return `GamePrediction`; return 503 if `model_loaded=False`
- [ ] T021 [US3] Manual smoke test: POST Texas vs LSU; verify `home_wp + away_wp` rounds to 1.0; POST `"arizona st"` vs LSU; POST unknown team and confirm 422 with `suggestions` field

**Checkpoint**: `POST /predict` functional for arbitrary matchups with fuzzy name resolution.

---

## Phase 6: User Story 4 — Filter Predictions by Team (Priority: P2)

**Goal**: Retrieve all upcoming bet recommendations for a specific team across multiple dates.

**Independent Test**: `curl http://localhost:8000/teams/Texas/predictions` returns all Texas rows from all available bet CSVs, each with a `date` field. `curl http://localhost:8000/teams/texas/predictions` (lowercase) returns identical results.

- [ ] T022 [US4] Implement `available_bet_dates(data_dir)` helper in `api.py`: glob `data/best_bets_*.csv`, extract dates from filenames, return sorted list
- [ ] T023 [US4] Implement `GET /teams/{team}/predictions` route in `api.py`: resolve `{team}` via `resolve_team()`; if unresolvable return 404 with `suggestions`; iterate `available_bet_dates()`, load each CSV, filter rows where `home` or `away` matches resolved name; return `PredictionsResponse`; 404 if no rows found
- [ ] T024 [US4] Manual smoke test: `curl http://localhost:8000/teams/Texas/predictions` and `curl http://localhost:8000/teams/texas/predictions` return same results; `curl http://localhost:8000/teams/ZZZTeam/predictions` returns 404 with `suggestions`

**Checkpoint**: `/teams/{team}/predictions` functional with fuzzy resolution.

---

## Phase 7: User Story 5 — Model Metadata (Priority: P3)

**Goal**: Expose model training configuration so consumers know what drives the predictions.

**Independent Test**: `curl http://localhost:8000/meta` returns all fields from `run_meta.json` including `feature_list` as an array, `train_cutoff`, `test_year`, `model_version`, `generated_at`. Returns 503 if `run_meta.json` missing.

- [ ] T025 [US5] Implement `GET /meta` route in `api.py`: read from `AppState.run_meta`; return `ModelMeta`; return 503 `ErrorResponse` if `run_meta` is `None`
- [ ] T026 [US5] Manual smoke test: `curl http://localhost:8000/meta` confirms all `ModelMeta` fields; temporarily rename `run_meta.json` and confirm 503 with `detail` field

**Checkpoint**: All 6 endpoints functional. Full v1 API complete.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [x] T027 Write `tests/test_api.py` using `httpx.AsyncClient` with `pytest` fixtures — 31 tests, all passing
- [ ] T028 [P] Verify `/openapi.json` accuracy: `curl http://localhost:8000/openapi.json` and confirm all 6 endpoints, all request/response schemas, and correct status codes are present
- [ ] T029 [P] Verify hot-reload: touch `data/run_meta.json` with a new `run_date`, wait ≤60s, confirm `GET /health` reflects the new date without restarting the server
- [ ] T030 Update `quickstart.md` with any corrections discovered during smoke testing

---

## Phase 9: Model Enhancement

**Goal**: Improve probability calibration and discrimination by switching to logistic regression
and enriching features with pitcher-level data. Diagnostic (2026-05-17) showed XGBoost offers
no AUC advantage over logistic regression (0.7585 vs 0.7600) and has worse Brier score —
the bottleneck is features, not model complexity.

- [x] T031 Replace XGBoost with `LogisticRegression(C=1.0)` in `daily_runner.py`: swap `train_model()`, update `model.pkl` bundle key `clf` (interface unchanged), re-run `daily_runner.py` and verify AUC ≥ 0.758 and Brier ≤ 0.192 via `diagnostics.py`
- [x] T032 Pull starting-pitcher stats from ESPN box scores into `pull_pitching_stats.py`: aggregate to team-season `k_per_game`, `bb_per_game`, `k_bb_ratio`; save to `data/team_pitching_stats_YYYY.parquet`
- [x] T033 Add pitching K/BB differential features to feature matrix in `daily_runner.py`: `d_k_per_game`, `d_bb_per_game`, `d_k_bb_ratio`; fill 0 when pitcher data unavailable (backward compat)
- [x] T034 Pull team-level batting (BA, OBP) and pitching ERA/WHIP from ESPN box scores: `pull_batting_stats.py` → `data/team_adv_stats_YYYY.parquet`; `daily_runner.py` merges `ba`, `obp`, `era`, `whip` into `FEAT_COLS` with 0-fallback when missing
- [ ] T035 Pull historical stats for all training years then re-run diagnostics; target AUC ≥ 0.780, Brier ≤ 0.185; **baseline (2026-05-18): LR AUC=0.757, Brier=0.194** — gap requires historical pull:
  ```
  python pull_pitching_stats.py --seasons 2021 2022 2023 2024 2025   # ~50 min
  python pull_batting_stats.py  --seasons 2021 2022 2023 2024 2025   # ~50 min
  python daily_runner.py                                               # retrain
  python diagnostics.py                                                # verify
  ```
  Results documented in `specs/diagnostics_log.md`

**Checkpoint**: `diagnostics.py` shows AUC ≥ 0.780, Brier ≤ 0.185, train-test gap < 0.015.

---

## Phase 11: Team Profile & Comparison API

**Goal**: Expose the full model knowledge — every stat, rating, and ranking the pipeline computes —
through queryable endpoints so teams can be browsed, profiled, and directly compared. Enables
natural-language queries like "Who is the best pitching team?" and "Compare Texas vs LSU" from
Claude Desktop.

- [x] T041 Add `rankings_df` and `elo_df` fields to `AppState` in `api.py`; update `load_artifacts()` to load `rankings.parquet` and `elo_ratings.parquet` from `DATA_DIR`
- [x] T042 Add `TeamProfile` Pydantic response model to `api.py`: all stat columns from `rankings.parquet` (rank, conference, wins, losses, win_pct, pythagorean_win_pct, avg_runs_scored, avg_runs_allowed, avg_run_diff, recent_win_pct, elo, avg_opp_elo, runs_scored_std, runs_allowed_std, shutout_pct, close_win_pct, k_per_game, bb_per_game, k_bb_ratio, power_score); add `TeamListResponse` and `TeamComparison` models
- [x] T043 Implement `GET /teams` in `api.py`: return all teams for `season` (default `TEST_YEAR`); support `conference`, `min_games`, `sort_by` query params; source from `rankings_df` for current year else `team_stats_df`
- [x] T044 Implement `GET /teams/{team}` in `api.py`: resolve team name (fuzzy), return full `TeamProfile`; prefer `rankings_df` (has rank + elo), fall back to `team_stats_df`
- [x] T045 Implement `GET /teams/{team}/compare/{other_team}` in `api.py`: resolve both teams, return `TeamComparison` with both `TeamProfile` objects, key stat differentials dict, and an embedded `GamePrediction`; refactor inner predict logic into `_predict_matchup()` helper shared with `POST /predict`
- [x] T046 Add `get_team_profile(team)` and `compare_teams(team_a, team_b, neutral?)` tools to `mcp_server.py`

**Checkpoint**: `GET /teams` returns ranked list; `GET /teams/Texas` returns full profile; `GET /teams/Texas/compare/LSU` returns side-by-side stats + prediction. Claude Desktop can answer "Who has the best pitching?" and "Compare Texas and LSU."

---

## Phase 10: MCP Server

**Goal**: Expose the prediction API as Claude tools so the user can query predictions,
run game predictions, and check pipeline health via natural language in Claude Desktop or Claude Code.

- [ ] T036 Create `mcp_server.py` at repository root: implement MCP server using `anthropic` SDK with three tools — `get_best_bets(date?, min_edge?)`, `predict_game(home, away, neutral?)`, `check_health()`; each tool calls the local REST API at `http://localhost:8000`
- [ ] T037 Add `get_team_predictions(team)` tool to MCP server wrapping `GET /teams/{team}/predictions`
- [ ] T038 Add `get_model_meta()` tool to MCP server wrapping `GET /meta`
- [ ] T039 Register MCP server in Claude Desktop config (`claude_desktop_config.json`): add entry under `mcpServers` pointing to `mcp_server.py` with `python` command
- [ ] T040 Smoke test: open Claude Desktop, ask "What are today's best bets?" and "Predict Texas vs LSU" — confirm tool calls fire and responses are readable plain English

**Checkpoint**: Natural language questions in Claude Desktop trigger correct API calls and return formatted answers.

---

## Phase 12: Post-Update Regression & Validation Testing

**Goal**: Automatically verify pipeline integrity after every daily runner execution and model update, so regressions are caught before bad data reaches the API or MCP server.

**Trigger points**: run after `daily_runner.py` completes, and on-demand via `pytest tests/ -m post_update`.

### T047 — Pipeline Artifact Validation

- [ ] T047 [P] Add `tests/test_post_update.py`: fixture loads `run_meta.json` and validates freshness (`run_date == today`), `teams_ranked >= 300`, `games_loaded > 30000`, and all expected feature keys present in `run_meta["features"]`; mark tests with `@pytest.mark.post_update`

### T048 — Model Sanity Checks

- [ ] T048 [P] Add model sanity tests to `tests/test_post_update.py`: load `model.pkl`, assert `clf` key present, assert `feats` list length matches `run_meta["features"]` length, assert `clf.predict_proba` returns shape `(1, 2)` for a dummy feature vector of correct length, assert probabilities sum to 1.0 ± 0.001

### T049 — API Regression Suite (post-reload)

- [ ] T049 Add `tests/test_api_regression.py` using `httpx.AsyncClient` (live server required, skip if unreachable): after each runner run, hit all 6 endpoints and assert: `/health` → `is_fresh=True` and `model_loaded=True`; `/predictions` → 200 or 404 (no 5xx); `POST /predict` with known teams → `home_wp + away_wp` rounds to 1.0; `/teams` → list non-empty; `/meta` → `model_version == today`; mark with `@pytest.mark.regression`

### T050 — MCP Server Tool Smoke Tests

- [ ] T050 [P] Add `tests/test_mcp_smoke.py`: import `mcp_server` tools directly (bypass HTTP), call `check_health()` and assert returns string containing `"fresh"` or `"stale"`; call `get_best_bets()` and assert result is a non-empty string or "no bets"; call `predict_game("Texas", "LSU")` and assert result contains a probability; mark `@pytest.mark.mcp`

### T051 — Scheduler Miss Detection

- [ ] T051 Add `tests/test_scheduler_health.py`: read `run_meta.json` and assert `run_date` is within the last 2 calendar days; if stale, print actionable message: `"Pipeline is stale — run: python daily_runner.py"` and fail with exit code 1; wire this test into a lightweight cron check (`schtasks /create ... /sc daily /st 09:00`) that emails or logs on failure

### T052 — Post-Update Pytest Hook in daily_runner.py

- [ ] T052 Add a final step in `daily_runner.py` after VPS sync: run `subprocess.run(["pytest", "tests/", "-m", "post_update", "-q", "--tb=short"])`, log pass/fail count to `daily_log.txt`; do not fail the runner on test failure — log a warning instead so the pipeline always completes

**Checkpoint**: After `daily_runner.py` completes, `tests/test_post_update.py` passes automatically and any artifact/model regression is logged within the same run. `/health` and model state are verified fresh before MCP tools serve data.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user story phases
- **Phases 3–7 (User Stories)**: All depend on Phase 2; can proceed sequentially in priority order
- **Phase 8 (Polish)**: Depends on all desired user stories being complete

### User Story Dependencies

| Story | Depends On | Blocks |
|-------|-----------|--------|
| US2 — /health (Phase 3) | Phase 2 | nothing |
| US1 — /predictions (Phase 4) | Phase 2 | nothing |
| US3 — POST /predict (Phase 5) | Phase 2 | nothing |
| US4 — /teams/{team} (Phase 6) | Phase 2, US1 helpers (T015) | nothing |
| US5 — /meta (Phase 7) | Phase 2 | nothing |

### Within Each Phase

- T004–T006 can run in parallel (different functions in same file — write to different sections)
- T008–T009 can run in parallel (request vs response model sections)
- All smoke tests are independent checkpoints, not blocking tasks

### Parallel Opportunities

```
# Phase 2 — run these together:
T004 resolve_team() extraction
T005 era_adjustment() extraction   [P]
T006 kelly_fraction() extraction   [P]

# Phase 2 — then run together:
T008 request models in api.py
T009 response models in api.py     [P]
```

---

## Implementation Strategy

### MVP (US2 + US1 only — Phases 1–4)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (pipeline/utils.py + AppState + schemas)
3. Complete Phase 3: GET /health
4. Complete Phase 4: GET /predictions + GET /predictions/{date}
5. **STOP and VALIDATE**: Daily use case works end-to-end
6. Continue to Phases 5–7 to complete full v1

### Incremental Delivery

- After Phase 3: Health checking works
- After Phase 4: Daily bet querying works (MVP!)
- After Phase 5: On-demand predictions work
- After Phase 6: Team-filtered queries work
- After Phase 7: Full v1 complete
- After Phase 8: Tests + hot-reload verified

---

## Notes

- All code goes in `api.py` and `pipeline/utils.py` — no additional source directories needed
- `data/` is gitignored; tests use fixture artifacts, not production data files
- Smoke tests (T014, T018, T021, T024, T026) are manual curl commands — run them in the terminal as checkpoints
- The `math.erf` Normal CDF implementation is already in `daily_runner.py` — copy it into `pipeline/utils.py` or inline in `api.py`
- Total tasks: 30 | Setup: 3 | Foundational: 9 | US tasks: 15 | Polish: 3
