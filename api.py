"""
api.py — Local REST API for the college baseball prediction pipeline.
Serves pre-computed artifacts from data/ with hot-reload on daily_runner.py completion.
Run: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import csv
import glob
import json
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from pipeline.utils import era_adjustment, kelly_fraction, norm_cdf, resolve_team

# ── configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
RELOAD_INTERVAL = int(os.environ.get("RELOAD_INTERVAL_SECONDS", "60"))
TEST_YEAR = 2026
RD_SIGMA = 3.5
ERA_ELO_SCALE = 25

# ── Pydantic request models ────────────────────────────────────────────────────

class GameRequest(BaseModel):
    home: str
    away: str
    neutral: bool = False
    date: date = None

    @model_validator(mode="after")
    def check_teams_differ(self):
        if self.home.strip().lower() == self.away.strip().lower():
            raise ValueError("home and away teams must be different")
        if self.date is None:
            self.date = date.today()
        return self

    @field_validator("home", "away")
    @classmethod
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("team name must not be empty")
        return v.strip()


class PredictionsQuery(BaseModel):
    date: Optional[date] = None
    team: Optional[str] = None
    market: Optional[str] = None
    min_edge: float = 0.0

    @field_validator("market")
    @classmethod
    def valid_market(cls, v):
        if v is not None and v.upper() not in ("ML", "ATS"):
            raise ValueError("market must be 'ML' or 'ATS'")
        return v.upper() if v else v

    @field_validator("min_edge")
    @classmethod
    def edge_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_edge must be between 0.0 and 1.0")
        return v

# ── Pydantic response models ───────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    suggestions: Optional[list[str]] = None


class HealthStatus(BaseModel):
    is_fresh: bool
    last_run_date: Optional[date]
    model_loaded: bool
    generated_at: Optional[datetime]


class ModelMeta(BaseModel):
    train_cutoff: int
    test_year: int
    feature_list: list[str]
    games_loaded: int
    teams_ranked: int
    model_version: str
    generated_at: datetime


class BetRecommendation(BaseModel):
    date: date
    home: str
    away: str
    market: str
    recommended_bet: str
    side: str
    odds: Optional[float]
    model_wp: float
    implied_prob: float
    edge: float
    kelly_size: float
    model_version: str
    generated_at: datetime


class PredictionsResponse(BaseModel):
    bets: list[BetRecommendation]
    count: int
    model_version: str
    generated_at: datetime


class GamePrediction(BaseModel):
    home: str
    away: str
    date: date
    home_wp: float
    away_wp: float
    pred_run_diff: float
    ml_recommendation: str
    ats_recommendation: str
    model_version: str
    generated_at: datetime

# ── AppState ───────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    clf: object = None
    feats: list[str] = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)
    known_teams: list[str] = field(default_factory=list)
    team_stats_df: object = None
    bets_cache: dict = field(default_factory=dict)
    last_reload: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_loaded: bool = False


_state = AppState()
_state_lock = threading.Lock()


def load_artifacts(data_dir: Path = DATA_DIR) -> AppState:
    s = AppState()

    # run_meta.json
    meta_path = data_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            s.run_meta = json.load(f)
        s.feats = s.run_meta.get("features", [])

    # model.pkl
    pkl_path = data_dir / "model.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        s.clf = bundle.get("clf")
        s.feats = bundle.get("feats", s.feats)
        s.model_loaded = True

    # team_season_stats parquet
    stat_files = sorted(data_dir.glob("team_season_stats_*.parquet"))
    if stat_files:
        s.team_stats_df = pd.read_parquet(stat_files[-1])
        s.known_teams = sorted(s.team_stats_df["team"].unique().tolist())

    s.last_reload = datetime.now(timezone.utc)
    return s


def _hot_reload_loop():
    meta_path = DATA_DIR / "run_meta.json"
    last_run_date = None
    while True:
        time.sleep(RELOAD_INTERVAL)
        try:
            if not meta_path.exists():
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            run_date = meta.get("run_date")
            if run_date != last_run_date:
                new_state = load_artifacts()
                with _state_lock:
                    global _state
                    _state = new_state
                last_run_date = run_date
        except Exception:
            pass


# ── FastAPI app + lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = load_artifacts()
    t = threading.Thread(target=_hot_reload_loop, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="College Baseball Prediction API",
    version="1.0.0",
    description="Serves daily prediction pipeline outputs. Read-only v1.",
    lifespan=lifespan,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _meta_fields():
    with _state_lock:
        s = _state
    mv = s.run_meta.get("model_version", "unknown")
    ga_raw = s.run_meta.get("generated_at")
    ga = datetime.fromisoformat(ga_raw) if ga_raw else datetime.now(timezone.utc)
    return s, mv, ga


def _resolve_or_404(name: str, candidates: list[str]) -> str:
    resolved, suggestions = resolve_team(name, candidates)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot resolve team '{name}'",
            headers={"X-Suggestions": json.dumps(suggestions)},
        )
    return resolved


def _load_bets_file(file_path: Path, mv: str, ga: datetime) -> list[BetRecommendation]:
    bets = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matchup = row.get("matchup", "")
            parts = matchup.split(" vs ")
            home = parts[0].strip() if len(parts) == 2 else matchup
            away = parts[1].strip() if len(parts) == 2 else ""
            try:
                edge = float(row.get("edge", 0))
                kelly = float(row.get("kelly_pct", 0))
                model_wp = float(row.get("model_wp", 0))
                implied = float(row.get("implied_prob", 0))
                ml_raw = row.get("ml", "")
                ml = float(ml_raw) if ml_raw else None
                raw_date = row.get("date", str(date.today()))
                bet_date = date.fromisoformat(raw_date)
            except (ValueError, TypeError):
                continue
            bets.append(BetRecommendation(
                date=bet_date,
                home=home,
                away=away,
                market="ML",
                recommended_bet=row.get("bet_on", ""),
                side=row.get("side", ""),
                odds=ml,
                model_wp=model_wp,
                implied_prob=implied,
                edge=edge,
                kelly_size=kelly,
                model_version=mv,
                generated_at=ga,
            ))
    return bets


def _bets_for_date(target: date, s: AppState, mv: str, ga: datetime) -> list[BetRecommendation]:
    if target in s.bets_cache:
        return s.bets_cache[target]
    csv_path = DATA_DIR / f"best_bets_{target.isoformat()}.csv"
    if not csv_path.exists():
        return []
    bets = _load_bets_file(csv_path, mv, ga)
    with _state_lock:
        s.bets_cache[target] = bets
    return bets


def _available_dates() -> list[date]:
    paths = sorted(DATA_DIR.glob("best_bets_*.csv"))
    dates = []
    for p in paths:
        stem = p.stem  # best_bets_YYYY-MM-DD
        date_part = stem.replace("best_bets_", "")
        try:
            dates.append(date.fromisoformat(date_part))
        except ValueError:
            pass
    return dates

# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthStatus)
def health():
    with _state_lock:
        s = _state
    run_date_raw = s.run_meta.get("run_date")
    run_date = date.fromisoformat(run_date_raw) if run_date_raw else None
    ga_raw = s.run_meta.get("generated_at")
    ga = datetime.fromisoformat(ga_raw) if ga_raw else None
    return HealthStatus(
        is_fresh=run_date == date.today() if run_date else False,
        last_run_date=run_date,
        model_loaded=s.model_loaded,
        generated_at=ga,
    )

# ── GET /meta ─────────────────────────────────────────────────────────────────

@app.get("/meta", response_model=ModelMeta)
def meta():
    s, mv, ga = _meta_fields()
    if not s.run_meta:
        raise HTTPException(503, "Pipeline artifacts not found — run daily_runner.py first")
    return ModelMeta(
        train_cutoff=s.run_meta.get("train_cutoff", 0),
        test_year=s.run_meta.get("test_year", TEST_YEAR),
        feature_list=s.run_meta.get("features", s.feats),
        games_loaded=s.run_meta.get("games_loaded", 0),
        teams_ranked=s.run_meta.get("teams_ranked", 0),
        model_version=mv,
        generated_at=ga,
    )

# ── GET /predictions/{date} ───────────────────────────────────────────────────

@app.get("/predictions/{bet_date}", response_model=PredictionsResponse)
def predictions_by_date(bet_date: date):
    s, mv, ga = _meta_fields()
    bets = _bets_for_date(bet_date, s, mv, ga)
    if not bets:
        raise HTTPException(404, f"No pre-computed bets found for {bet_date}")
    return PredictionsResponse(bets=bets, count=len(bets), model_version=mv, generated_at=ga)

# ── GET /predictions ──────────────────────────────────────────────────────────

@app.get("/predictions", response_model=PredictionsResponse)
def predictions(
    bet_date: Optional[date] = Query(None, alias="date"),
    team: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    min_edge: float = Query(0.0, ge=0.0, le=1.0),
):
    s, mv, ga = _meta_fields()
    target = bet_date or date.today()
    bets = _bets_for_date(target, s, mv, ga)

    if team:
        resolved, suggestions = resolve_team(team, s.known_teams)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot resolve team '{team}'. Suggestions: {suggestions}",
            )
        bets = [b for b in bets if resolved.lower() in b.home.lower() or resolved.lower() in b.away.lower()]

    if market:
        bets = [b for b in bets if b.market.upper() == market.upper()]

    if min_edge > 0:
        bets = [b for b in bets if b.edge >= min_edge]

    if not bets:
        raise HTTPException(404, "No bets found matching the given filters")

    return PredictionsResponse(bets=bets, count=len(bets), model_version=mv, generated_at=ga)

# ── GET /teams/{team}/predictions ─────────────────────────────────────────────

@app.get("/teams/{team}/predictions", response_model=PredictionsResponse)
def team_predictions(team: str):
    s, mv, ga = _meta_fields()
    resolved, suggestions = resolve_team(team, s.known_teams)
    if resolved is None:
        raise HTTPException(
            404,
            f"Cannot resolve team '{team}'. Suggestions: {suggestions}",
        )

    all_bets = []
    for d in _available_dates():
        day_bets = _bets_for_date(d, s, mv, ga)
        all_bets.extend(
            b for b in day_bets
            if resolved.lower() in b.home.lower() or resolved.lower() in b.away.lower()
        )

    if not all_bets:
        raise HTTPException(404, f"No predictions found for team '{resolved}'")

    return PredictionsResponse(bets=all_bets, count=len(all_bets), model_version=mv, generated_at=ga)

# ── POST /predict ─────────────────────────────────────────────────────────────

@app.post("/predict", response_model=GamePrediction)
def predict(req: GameRequest):
    s, mv, ga = _meta_fields()

    if not s.model_loaded or s.clf is None:
        raise HTTPException(503, "Model not loaded — run daily_runner.py first")

    home_resolved, h_suggestions = resolve_team(req.home, s.known_teams)
    if home_resolved is None:
        raise HTTPException(422, f"Cannot resolve team '{req.home}'. Suggestions: {h_suggestions}")

    away_resolved, a_suggestions = resolve_team(req.away, s.known_teams)
    if away_resolved is None:
        raise HTTPException(422, f"Cannot resolve team '{req.away}'. Suggestions: {a_suggestions}")

    year = req.date.year if req.date else date.today().year
    df = s.team_stats_df
    home_row = df[(df["team"] == home_resolved) & (df["season"] == year)]
    away_row = df[(df["team"] == away_resolved) & (df["season"] == year)]

    # Fall back to most recent season if current year not yet available
    if home_row.empty:
        home_row = df[df["team"] == home_resolved].sort_values("season").tail(1)
    if away_row.empty:
        away_row = df[df["team"] == away_resolved].sort_values("season").tail(1)

    if home_row.empty or away_row.empty:
        raise HTTPException(422, "Insufficient stats for one or both teams")

    h = home_row.iloc[0]
    a = away_row.iloc[0]

    feature_vec = []
    for feat in s.feats:
        if feat == "is_home":
            feature_vec.append(0.0 if req.neutral else 1.0)
        elif feat == "neutral":
            feature_vec.append(1.0 if req.neutral else 0.0)
        elif feat.startswith("d_"):
            col = feat[2:]
            hv = float(h.get(col, 0.0)) if col in h.index else 0.0
            av = float(a.get(col, 0.0)) if col in a.index else 0.0
            feature_vec.append(hv - av)
        else:
            feature_vec.append(float(h.get(feat, 0.0)) if feat in h.index else 0.0)

    X = np.array([feature_vec])
    home_wp = float(s.clf.predict_proba(X)[0, 1])

    # ERA adjustment (Elo shift, not baked into model — prediction-time only)
    era_h = era_adjustment(home_resolved, df, year, ERA_ELO_SCALE)
    era_a = era_adjustment(away_resolved, df, year, ERA_ELO_SCALE)
    era_delta = (era_h - era_a) / 400.0
    home_wp = min(max(home_wp + era_delta * home_wp * (1 - home_wp), 0.01), 0.99)
    away_wp = 1.0 - home_wp

    # Run differential estimate (rough): logit-scale
    import math
    pred_rd = math.log(home_wp / max(away_wp, 1e-6)) * 2.5

    # ATS cover probability (no live spread — informational)
    home_ats_prob = norm_cdf(0.0, mu=pred_rd, sigma=RD_SIGMA)
    ml_rec = "Bet home ML" if home_wp >= 0.55 else ("Bet away ML" if away_wp >= 0.55 else "No clear edge")
    ats_rec = "Bet home ATS" if home_ats_prob >= 0.55 else ("Bet away ATS" if home_ats_prob <= 0.45 else "No clear edge")

    return GamePrediction(
        home=home_resolved,
        away=away_resolved,
        date=req.date or date.today(),
        home_wp=round(home_wp, 4),
        away_wp=round(away_wp, 4),
        pred_run_diff=round(pred_rd, 3),
        ml_recommendation=ml_rec,
        ats_recommendation=ats_rec,
        model_version=mv,
        generated_at=ga,
    )
