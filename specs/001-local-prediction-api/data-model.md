# Data Model: Local Prediction REST API

All shapes are Pydantic v2 models. Every response that derives from pre-computed artifacts
includes `model_version` and `generated_at` (constitution V).

---

## Request Models

### GameRequest (`POST /predict` body)

```
home        str     Home team name (fuzzy-resolved via resolve_team())
away        str     Away team name (fuzzy-resolved via resolve_team())
neutral     bool    Neutral-site game? (default: false)
date        date    Game date in YYYY-MM-DD format (default: today)
```

Validation rules:
- `home` and `away` must be non-empty strings after stripping whitespace
- `date` must be a valid calendar date; future dates allowed
- `home` != `away` (same-team error → HTTP 422)

### PredictionsQuery (`GET /predictions` query params)

```
date        date | None    Filter to this date (YYYY-MM-DD)
team        str  | None    Filter by team (fuzzy-resolved)
market      "ML" | "ATS" | None    Filter by market type
min_edge    float | None   Minimum edge threshold [0.0, 1.0] (default: 0.0)
```

Validation rules:
- `min_edge` must be in [0.0, 1.0]; values outside → HTTP 422

---

## Response Models

### HealthStatus (`GET /health`)

```
is_fresh        bool     True if last_run_date == today
last_run_date   date     Date of the most recent daily_runner.py run
model_loaded    bool     True if model.pkl is loaded in memory
generated_at    datetime ISO 8601 timestamp from run_meta.json
```

### ModelMeta (`GET /meta`)

```
train_cutoff    int          Last training year (e.g., 2025)
test_year       int          Current test/prediction year (e.g., 2026)
feature_list    list[str]    Names of the 11 model features
games_loaded    int          Total game rows in training data
teams_ranked    int          Number of teams with Elo ratings
model_version   str          Date string of model build (YYYY-MM-DD)
generated_at    datetime     ISO 8601 timestamp from run_meta.json
```

### BetRecommendation (`GET /predictions`, `GET /predictions/{date}`, `GET /teams/{team}/predictions`)

```
date            date     Game date
home            str      Home team (resolved name)
away            str      Away team (resolved name)
market          str      "ML" or "ATS"
edge            float    Edge over implied probability [0.0, 1.0]
kelly_size      float    Recommended bet fraction of bankroll [0.0, KELLY_CAP]
model_wp        float    Model win probability for favored side [0.0, 1.0]
implied_prob    float    Market-implied probability [0.0, 1.0]
recommended_bet str      Human-readable bet description (e.g., "Texas ML")
model_version   str      Date string of model build (YYYY-MM-DD)
generated_at    datetime ISO 8601 timestamp from run_meta.json
```

### PredictionsResponse (wrapper for list endpoints)

```
bets            list[BetRecommendation]
count           int
model_version   str
generated_at    datetime
```

### GamePrediction (`POST /predict` response)

```
home            str      Resolved home team name
away            str      Resolved away team name
date            date     Game date
home_wp         float    Home team win probability [0.0, 1.0]
away_wp         float    Away team win probability [0.0, 1.0]
pred_run_diff   float    Predicted run differential (positive = home favored)
ml_edge         float    Moneyline edge if odds provided (0.0 if no live odds)
ml_recommendation   str  "Bet home ML", "Bet away ML", or "No edge"
ats_recommendation  str  "Bet home ATS", "Bet away ATS", or "No edge"
model_version   str
generated_at    datetime
```

### ErrorResponse (all HTTP 4xx/5xx errors)

```
detail          str              Human-readable error message
suggestions     list[str] | None Fuzzy-match suggestions for team name errors
```

---

## In-Memory State (loaded at startup, refreshed on run_meta.json change)

```
AppState:
  clf             XGBClassifier        Loaded from data/model.pkl
  feats           list[str]            Feature names from model.pkl
  run_meta        dict                 Parsed run_meta.json
  known_teams     set[str]             Canonical team names from team_season_stats parquet
  bets_cache      dict[date, list]     Loaded best_bets CSV files keyed by date
  last_reload     datetime             Timestamp of last hot-reload
```

State transitions:
- **Startup**: Load all artifacts → populate AppState; if model.pkl missing, `model_loaded = False`
- **Hot-reload trigger**: `run_meta.json` `run_date` changes → reload all AppState fields
- **Cache miss**: `GET /predictions/{date}` for a date not in `bets_cache` → try loading CSV from disk; 404 if not found
