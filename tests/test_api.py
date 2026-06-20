"""
tests/test_api.py — Integration tests for the College Baseball Prediction REST API.

Uses httpx.AsyncClient + ASGITransport against a fixture-backed temp data directory.
No live server or production data files required.

Install test deps (once):
    pip install pytest-asyncio httpx
"""

import csv
import json
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── constants shared across fixtures and tests ────────────────────────────────

FEATS = [
    "d_avg_runs_scored", "d_avg_runs_allowed", "d_avg_run_diff",
    "d_pythagorean_win_pct", "d_win_pct", "d_recent_win_pct",
    "d_avg_runs_scored_z", "d_avg_runs_allowed_z", "d_pythagorean_win_pct_z",
    "is_home", "neutral",
]

TODAY = date.today().isoformat()
GEN_AT = datetime.now(timezone.utc).isoformat()
HOME_TEAM = "Texas Longhorns"
AWAY_TEAM = "LSU Tigers"
MISSING_DATE = "1999-01-01"


# ── minimal classifier (no sklearn needed) ────────────────────────────────────

class _TinyClf:
    """Predict-proba stub: returns a deterministic probability from feature[0]."""

    def predict_proba(self, X):
        p = float(np.clip(0.5 + 0.15 * X[0, 0], 0.1, 0.9))
        return np.array([[1.0 - p, p]])


# ── fixture helpers ───────────────────────────────────────────────────────────

def _base_stats(team: str, **overrides) -> dict:
    row = {
        "team": team, "season": 2026,
        "games": 40, "wins": 25, "losses": 15,
        "rs_total": 240, "ra_total": 170, "rd_total": 70,
        "win_pct": 0.625, "avg_runs_scored": 6.0, "avg_runs_allowed": 4.25,
        "avg_run_diff": 1.75, "pythagorean_win_pct": 0.66, "recent_win_pct": 0.65,
        "avg_runs_scored_z": 0.5, "avg_runs_allowed_z": -0.5,
        "avg_run_diff_z": 0.5, "pythagorean_win_pct_z": 0.5,
    }
    row.update(overrides)
    return row


def _write_bets_csv(d: Path, bet_date: str, edge: float = 0.115) -> Path:
    path = d / f"best_bets_{bet_date}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "matchup", "bet_on", "side", "ml",
            "model_wp", "implied_prob", "edge", "ev_per_unit", "kelly_pct", "kelly_$",
        ])
        w.writeheader()
        w.writerow({
            "date": bet_date,
            "matchup": f"{HOME_TEAM} vs {AWAY_TEAM}",
            "bet_on": HOME_TEAM,
            "side": "Home",
            "ml": -130,
            "model_wp": 0.680,
            "implied_prob": round(0.680 - edge, 4),
            "edge": edge,
            "ev_per_unit": 0.205,
            "kelly_pct": 0.050,
            "kelly_$": 50.00,
        })
    return path


# ── session fixture: create all artifact files once per test run ──────────────

@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("api_data")

    # model.pkl — lightweight stub classifier
    with open(d / "model.pkl", "wb") as f:
        pickle.dump({"clf": _TinyClf(), "feats": FEATS}, f)

    # run_meta.json — today's successful run
    meta = {
        "run_date": TODAY,
        "generated_at": GEN_AT,
        "games_loaded": 500,
        "teams_ranked": 2,
        "bets_found": 1,
        "model_version": TODAY,
        "train_cutoff": 2025,
        "test_year": 2026,
        "features": FEATS,
    }
    (d / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # team_season_stats parquet — two teams for 2026
    pd.DataFrame([
        _base_stats(HOME_TEAM, avg_runs_scored=6.8, avg_runs_allowed=3.6,
                    win_pct=0.75, pythagorean_win_pct=0.74,
                    avg_runs_scored_z=1.2, avg_runs_allowed_z=-1.3),
        _base_stats(AWAY_TEAM, avg_runs_scored=5.1, avg_runs_allowed=5.0,
                    win_pct=0.55, pythagorean_win_pct=0.51,
                    avg_runs_scored_z=-0.1, avg_runs_allowed_z=0.5),
    ]).to_parquet(d / "team_season_stats_2021_2026.parquet", index=False)

    # today's best-bets CSV
    _write_bets_csv(d, TODAY)

    return d


# ── async client fixtures ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(data_dir):
    """Client wired to fixture data_dir; model and bets pre-loaded."""
    import api
    preloaded = api.load_artifacts(data_dir)
    with patch.object(api, "DATA_DIR", data_dir):
        async with AsyncClient(
            transport=ASGITransport(app=api.app),
            base_url="http://test",
        ) as ac:
            # Lifespan loaded from real data/ — overwrite with fixture state
            api._state = preloaded
            yield ac


@pytest_asyncio.fixture
async def client_empty(tmp_path):
    """Client with no artifacts (simulates pre-first-run state)."""
    import api
    from api import AppState

    empty = AppState()

    # Patch DATA_DIR so route handlers find nothing on disk.
    # Explicitly set api._state because ASGITransport may not trigger the lifespan.
    with patch.object(api, "DATA_DIR", tmp_path):
        async with AsyncClient(
            transport=ASGITransport(app=api.app),
            base_url="http://test",
        ) as ac:
            api._state = empty
            yield ac


# ── GET /health ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client):
    r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_fields(client):
    body = (await client.get("/health")).json()
    assert "is_fresh" in body
    assert "last_run_date" in body
    assert "model_loaded" in body
    assert "generated_at" in body


@pytest.mark.asyncio
async def test_health_fresh_and_loaded(client):
    body = (await client.get("/health")).json()
    assert body["is_fresh"] is True
    assert body["model_loaded"] is True
    assert body["last_run_date"] == TODAY


@pytest.mark.asyncio
async def test_health_never_errors_without_artifacts(client_empty):
    r = await client_empty.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["is_fresh"] is False
    assert body["model_loaded"] is False


# ── GET /meta ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_returns_200(client):
    r = await client.get("/meta")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_meta_fields(client):
    body = (await client.get("/meta")).json()
    assert body["train_cutoff"] == 2025
    assert body["test_year"] == 2026
    assert body["games_loaded"] == 500
    assert body["teams_ranked"] == 2
    assert isinstance(body["feature_list"], list)
    assert len(body["feature_list"]) == len(FEATS)
    assert body["model_version"] == TODAY


@pytest.mark.asyncio
async def test_meta_503_without_artifacts(client_empty):
    r = await client_empty.get("/meta")
    assert r.status_code == 503
    assert "detail" in r.json()


# ── GET /predictions/{date} ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predictions_by_date_ok(client):
    r = await client.get(f"/predictions/{TODAY}")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["model_version"] == TODAY


@pytest.mark.asyncio
async def test_predictions_by_date_bet_shape(client):
    bet = (await client.get(f"/predictions/{TODAY}")).json()["bets"][0]
    assert bet["market"] == "ML"
    assert bet["recommended_bet"] == HOME_TEAM
    assert bet["side"] == "Home"
    assert abs(bet["edge"] - 0.115) < 0.001
    assert 0 < bet["model_wp"] < 1
    assert 0 < bet["implied_prob"] < 1
    assert bet["kelly_size"] > 0


@pytest.mark.asyncio
async def test_predictions_by_date_404(client):
    r = await client.get(f"/predictions/{MISSING_DATE}")
    assert r.status_code == 404
    assert "detail" in r.json()


# ── GET /predictions ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predictions_today_default(client):
    r = await client.get("/predictions")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


@pytest.mark.asyncio
async def test_predictions_date_query_param(client):
    r = await client.get(f"/predictions?date={TODAY}")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


@pytest.mark.asyncio
async def test_predictions_min_edge_passes(client):
    r = await client.get("/predictions?min_edge=0.10")
    assert r.status_code == 200
    for bet in r.json()["bets"]:
        assert bet["edge"] >= 0.10


@pytest.mark.asyncio
async def test_predictions_min_edge_excludes_all(client):
    r = await client.get("/predictions?min_edge=0.99")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_predictions_unknown_market_returns_no_results(client):
    # market is Optional[str] with no route-level validation; unknown values
    # pass through but match nothing, so the API returns 404 not 422.
    r = await client.get("/predictions?market=FUTURES")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_predictions_min_edge_out_of_range_422(client):
    r = await client.get("/predictions?min_edge=1.5")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predictions_missing_date_404(client):
    r = await client.get(f"/predictions?date={MISSING_DATE}")
    assert r.status_code == 404


# ── GET /teams/{team}/predictions ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_team_predictions_exact_name(client):
    r = await client.get(f"/teams/{HOME_TEAM}/predictions")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    for bet in body["bets"]:
        involved = {bet["home"].lower(), bet["away"].lower()}
        assert HOME_TEAM.lower() in involved


@pytest.mark.asyncio
async def test_team_predictions_fuzzy_name(client):
    r = await client.get("/teams/texas longhorns/predictions")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


@pytest.mark.asyncio
async def test_team_predictions_case_insensitive(client):
    r_upper = await client.get(f"/teams/{HOME_TEAM}/predictions")
    r_lower = await client.get(f"/teams/{HOME_TEAM.lower()}/predictions")
    assert r_upper.status_code == 200
    assert r_lower.status_code == 200
    assert r_upper.json()["count"] == r_lower.json()["count"]


@pytest.mark.asyncio
async def test_team_predictions_unknown_team_404(client):
    r = await client.get("/teams/ZZZUnknownTeamXXX/predictions")
    assert r.status_code == 404
    assert "detail" in r.json()


# ── POST /predict ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_ok(client):
    r = await client.post("/predict", json={"home": HOME_TEAM, "away": AWAY_TEAM})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_predict_probabilities_sum_to_one(client):
    body = (await client.post(
        "/predict", json={"home": HOME_TEAM, "away": AWAY_TEAM}
    )).json()
    assert abs(body["home_wp"] + body["away_wp"] - 1.0) < 1e-4
    assert 0 < body["home_wp"] < 1


@pytest.mark.asyncio
async def test_predict_recommendation_fields(client):
    body = (await client.post(
        "/predict", json={"home": HOME_TEAM, "away": AWAY_TEAM}
    )).json()
    assert "ml_recommendation" in body
    assert "ats_recommendation" in body
    assert "pred_run_diff" in body
    assert body["home"] == HOME_TEAM
    assert body["away"] == AWAY_TEAM


@pytest.mark.asyncio
async def test_predict_neutral_site(client):
    body = (await client.post(
        "/predict", json={"home": HOME_TEAM, "away": AWAY_TEAM, "neutral": True}
    )).json()
    assert abs(body["home_wp"] + body["away_wp"] - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_predict_fuzzy_team_name(client):
    # difflib cutoff=0.72 requires enough overlap; full team names resolve reliably
    r = await client.post("/predict", json={"home": "texas longhorns", "away": "lsu tigers"})
    assert r.status_code == 200
    assert abs(r.json()["home_wp"] + r.json()["away_wp"] - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_predict_same_team_422(client):
    r = await client.post("/predict", json={"home": HOME_TEAM, "away": HOME_TEAM})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_unknown_home_422(client):
    r = await client.post("/predict", json={"home": "ZZZTeamXXX", "away": AWAY_TEAM})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_unknown_away_422(client):
    r = await client.post("/predict", json={"home": HOME_TEAM, "away": "ZZZTeamXXX"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_team_name_422(client):
    r = await client.post("/predict", json={"home": "", "away": AWAY_TEAM})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_503_without_model(client_empty):
    r = await client_empty.post(
        "/predict", json={"home": HOME_TEAM, "away": AWAY_TEAM}
    )
    assert r.status_code == 503
    assert "detail" in r.json()
