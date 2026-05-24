# Quickstart: Local Prediction REST API

## Prerequisites

1. `daily_runner.py` has run at least once (produces `data/model.pkl`, `data/run_meta.json`, `data/best_bets_YYYY-MM-DD.csv`)
2. Python dependencies installed:
   ```
   pip install fastapi uvicorn[standard] httpx
   ```

## Start the API

```powershell
# From C:\Users\trevm\Projects\CFBBaseballAnalytics
uvicorn api:app --host 127.0.0.1 --port 8000
```

The server starts and immediately loads all artifacts from `data/`. It then polls `data/run_meta.json` every 60 seconds for freshness changes.

## Common Queries

```powershell
# Check pipeline health
curl http://localhost:8000/health

# Today's best bets
curl "http://localhost:8000/predictions/2026-05-16"

# Only ATS bets with edge > 5%
curl "http://localhost:8000/predictions?market=ATS&min_edge=0.05"

# All upcoming Texas bets
curl "http://localhost:8000/teams/Texas/predictions"

# On-demand prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"home": "Texas", "away": "LSU", "neutral": false}'

# Browse interactive docs
Start-Process "http://localhost:8000/docs"
```

## Pipeline Module

`api.py` imports shared logic from `pipeline/utils.py`:
- `resolve_team(name, known_teams)` — fuzzy team name resolution
- `era_adjustment(team, team_stats)` — ERA-based Elo adjustment
- `kelly_fraction(edge, fraction, cap)` — Kelly criterion sizing

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RELOAD_INTERVAL_SECONDS` | `60` | How often to poll run_meta.json |
| `DATA_DIR` | `data/` | Path to pipeline artifact directory |
| `PORT` | `8000` | Uvicorn port (pass via uvicorn CLI instead) |

## Running Tests

```powershell
pytest tests/test_api.py -v
```

Tests use `httpx.AsyncClient` with fixture artifact files — no live server required.
