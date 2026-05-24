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

- [ ] T027 Write `tests/test_api.py` using `httpx.AsyncClient` with `pytest` fixtures: temp `data/` dir with minimal valid artifacts (tiny serialized sklearn model, one-row CSV, valid `run_meta.json`); cover happy path + 404 + 422 + 503 for all 6 endpoints
- [ ] T028 [P] Verify `/openapi.json` accuracy: `curl http://localhost:8000/openapi.json` and confirm all 6 endpoints, all request/response schemas, and correct status codes are present
- [ ] T029 [P] Verify hot-reload: touch `data/run_meta.json` with a new `run_date`, wait ≤60s, confirm `GET /health` reflects the new date without restarting the server
- [ ] T030 Update `quickstart.md` with any corrections discovered during smoke testing

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
