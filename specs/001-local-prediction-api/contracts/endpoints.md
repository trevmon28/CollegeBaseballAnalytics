# API Endpoint Contracts

**Base URL**: `http://localhost:8000`
**Auto-generated schema**: `GET /openapi.json` (authoritative — this doc is a human summary only)

---

## GET /health

Returns pipeline freshness and model load status.

**Response 200**:
```json
{
  "is_fresh": true,
  "last_run_date": "2026-05-16",
  "model_loaded": true,
  "generated_at": "2026-05-16T08:03:42Z"
}
```

**Never returns an error** — if `run_meta.json` is missing, returns `is_fresh: false, model_loaded: false`.

---

## GET /meta

Returns model training configuration.

**Response 200**:
```json
{
  "train_cutoff": 2025,
  "test_year": 2026,
  "feature_list": ["d_avg_runs_scored", "d_elo", "..."],
  "games_loaded": 30320,
  "teams_ranked": 321,
  "model_version": "2026-05-16",
  "generated_at": "2026-05-16T08:03:42Z"
}
```

**Response 503**: `run_meta.json` is missing → `{"detail": "Pipeline artifacts not found — run daily_runner.py first"}`

---

## GET /predictions

Returns pre-computed bet recommendations, optionally filtered.

**Query parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `date` | YYYY-MM-DD | none | Filter to this date |
| `team` | string | none | Filter by team (fuzzy-resolved) |
| `market` | ML \| ATS | none | Filter by market type |
| `min_edge` | float [0,1] | 0.0 | Minimum edge threshold |

**Response 200**:
```json
{
  "bets": [
    {
      "date": "2026-05-16",
      "home": "Texas",
      "away": "LSU",
      "market": "ML",
      "edge": 0.087,
      "kelly_size": 0.022,
      "model_wp": 0.623,
      "implied_prob": 0.536,
      "recommended_bet": "Texas ML",
      "model_version": "2026-05-16",
      "generated_at": "2026-05-16T08:03:42Z"
    }
  ],
  "count": 1,
  "model_version": "2026-05-16",
  "generated_at": "2026-05-16T08:03:42Z"
}
```

**Response 404**: No bets found for the given filters.
**Response 422**: Invalid `min_edge` (< 0 or > 1), invalid `date` format, invalid `market` value.

---

## GET /predictions/{date}

Returns all pre-computed bets for a specific date.

**Path parameter**: `date` — YYYY-MM-DD

**Response 200**: Same shape as `GET /predictions` response.
**Response 404**: `{"detail": "No pre-computed bets found for 2026-05-17"}`
**Response 422**: Invalid date format.

---

## GET /teams/{team}/predictions

Returns all pre-computed bets involving the specified team across all available dates.

**Path parameter**: `team` — team name string (fuzzy-resolved)

**Response 200**: Same shape as `GET /predictions` response, each bet includes `date` field.
**Response 404**:
- Team cannot be resolved: `{"detail": "Cannot resolve team 'XYZ'", "suggestions": ["Xavier", "..."]}`
- Team resolved but no bets found: `{"detail": "No predictions found for team 'XYZTeam'"}`

---

## POST /predict

On-demand single-game prediction.

**Request body**:
```json
{
  "home": "Texas",
  "away": "LSU",
  "neutral": false,
  "date": "2026-05-20"
}
```

**Response 200**:
```json
{
  "home": "Texas",
  "away": "LSU",
  "date": "2026-05-20",
  "home_wp": 0.581,
  "away_wp": 0.419,
  "pred_run_diff": 0.83,
  "ml_edge": 0.0,
  "ml_recommendation": "Bet home ML",
  "ats_recommendation": "No edge",
  "model_version": "2026-05-16",
  "generated_at": "2026-05-16T14:22:11Z"
}
```

**Response 422**: Unresolvable team name, same home/away team, invalid date.
**Response 503**: `model.pkl` not loaded → `{"detail": "Model not loaded — run daily_runner.py first"}`
