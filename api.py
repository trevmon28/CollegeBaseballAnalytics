"""
api.py — Local REST API for the college baseball prediction pipeline.
Serves pre-computed artifacts from data/ with hot-reload on daily_runner.py completion.
Run: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import csv
import glob
import json
import math
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


class TeamProfile(BaseModel):
    team: str
    season: int
    conference: Optional[str] = None
    rank: Optional[int] = None
    power_score: Optional[float] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    games: Optional[int] = None
    win_pct: Optional[float] = None
    pythagorean_win_pct: Optional[float] = None
    avg_runs_scored: Optional[float] = None
    avg_runs_allowed: Optional[float] = None
    avg_run_diff: Optional[float] = None
    recent_win_pct: Optional[float] = None
    elo: Optional[float] = None
    avg_opp_elo: Optional[float] = None
    runs_scored_std: Optional[float] = None
    runs_allowed_std: Optional[float] = None
    shutout_pct: Optional[float] = None
    close_win_pct: Optional[float] = None
    k_per_game: Optional[float] = None
    bb_per_game: Optional[float] = None
    k_bb_ratio: Optional[float] = None


class TeamListResponse(BaseModel):
    teams: list[TeamProfile]
    count: int
    season: int


class TeamComparison(BaseModel):
    home: TeamProfile
    away: TeamProfile
    differentials: dict[str, float]
    prediction: GamePrediction
    generated_at: datetime

# ── AppState ───────────────────────────────────────────────────────────────────

@dataclass
class AppState:
    clf: object = None
    feats: list[str] = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)
    known_teams: list[str] = field(default_factory=list)
    team_stats_df: object = None
    rankings_df: object = None
    elo_df: object = None
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

    # rankings.parquet (current season — rank, conference, elo, sos, all stats)
    rank_path = data_dir / "rankings.parquet"
    if rank_path.exists():
        s.rankings_df = pd.read_parquet(rank_path)

    # elo_ratings.parquet (latest end-of-season elo per team)
    elo_path = data_dir / "elo_ratings.parquet"
    if elo_path.exists():
        s.elo_df = pd.read_parquet(elo_path)

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
            detail={"message": f"Cannot resolve team '{name}'", "team": name, "suggestions": suggestions},
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

# ── team profile helpers ───────────────────────────────────────────────────────

def _flt(v) -> Optional[float]:
    try:
        f = float(v)
        return round(f, 4) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _int(v) -> Optional[int]:
    try:
        f = float(v)
        return int(f) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def _row_to_profile(row, season: int) -> TeamProfile:
    get = lambda k: row[k] if k in row.index else None
    return TeamProfile(
        team=str(row["team"]),
        season=season,
        conference=_str(get("conference")),
        rank=_int(get("rank")),
        power_score=_flt(get("power_score")),
        wins=_int(get("wins")),
        losses=_int(get("losses")),
        games=_int(get("games")),
        win_pct=_flt(get("win_pct")),
        pythagorean_win_pct=_flt(get("pythagorean_win_pct")),
        avg_runs_scored=_flt(get("avg_runs_scored")),
        avg_runs_allowed=_flt(get("avg_runs_allowed")),
        avg_run_diff=_flt(get("avg_run_diff")),
        recent_win_pct=_flt(get("recent_win_pct")),
        elo=_flt(get("elo")),
        avg_opp_elo=_flt(get("avg_opp_elo")),
        runs_scored_std=_flt(get("runs_scored_std")),
        runs_allowed_std=_flt(get("runs_allowed_std")),
        shutout_pct=_flt(get("shutout_pct")),
        close_win_pct=_flt(get("close_win_pct")),
        k_per_game=_flt(get("k_per_game")),
        bb_per_game=_flt(get("bb_per_game")),
        k_bb_ratio=_flt(get("k_bb_ratio")),
    )


def _profile_for_team(resolved: str, season: int, s: AppState) -> Optional[TeamProfile]:
    """Return TeamProfile for a resolved team name, preferring rankings_df."""
    if s.rankings_df is not None and season == TEST_YEAR:
        df = s.rankings_df[s.rankings_df["team"] == resolved]
        if df.empty:
            alt, _ = resolve_team(resolved, s.rankings_df["team"].tolist())
            if alt:
                df = s.rankings_df[s.rankings_df["team"] == alt]
        if not df.empty:
            return _row_to_profile(df.iloc[0], season)
    if s.team_stats_df is not None:
        df = s.team_stats_df[(s.team_stats_df["team"] == resolved) &
                             (s.team_stats_df["season"] == season)]
        if df.empty:
            df = s.team_stats_df[s.team_stats_df["team"] == resolved].sort_values("season").tail(1)
        if not df.empty:
            row = df.iloc[0].copy()
            if s.elo_df is not None and "elo" not in row.index:
                elo_lk = s.elo_df.set_index("team")["elo"]
                row["elo"] = elo_lk.get(resolved)
            return _row_to_profile(row, int(row.get("season", season)))
    return None


# ── predict helper (shared by POST /predict and GET compare) ───────────────────

def _predict_matchup(
    home_resolved: str,
    away_resolved: str,
    neutral: bool,
    req_date: date,
    s: AppState,
    mv: str,
    ga: datetime,
) -> GamePrediction:
    if not s.model_loaded or s.clf is None:
        raise HTTPException(503, "Model not loaded — run daily_runner.py first")

    year = req_date.year
    df = s.team_stats_df
    home_row = df[(df["team"] == home_resolved) & (df["season"] == year)]
    away_row = df[(df["team"] == away_resolved) & (df["season"] == year)]

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
            feature_vec.append(0.0 if neutral else 1.0)
        elif feat == "neutral":
            feature_vec.append(1.0 if neutral else 0.0)
        elif feat.startswith("d_"):
            col = feat[2:]
            hv = float(h.get(col, 0.0)) if col in h.index else 0.0
            av = float(a.get(col, 0.0)) if col in a.index else 0.0
            feature_vec.append(hv - av)
        else:
            feature_vec.append(float(h.get(feat, 0.0)) if feat in h.index else 0.0)

    X = np.array([feature_vec])
    home_wp_log = float(s.clf.predict_proba(X)[0, 1])

    # Elo blend — matches compute_best_bets() in daily_runner.py (60/40)
    elo_lk = s.elo_df.set_index("team")["elo"] if s.elo_df is not None else {}
    elo_init = 1500.0
    home_adv_elo = 0.0 if neutral else 35.0
    era_h = era_adjustment(home_resolved, df, year, ERA_ELO_SCALE)
    era_a = era_adjustment(away_resolved, df, year, ERA_ELO_SCALE)
    elo_h = float(elo_lk.get(home_resolved, elo_init)) + home_adv_elo + era_h
    elo_a = float(elo_lk.get(away_resolved, elo_init)) + era_a
    wp_elo = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400.0))
    home_wp = min(max(0.6 * home_wp_log + 0.4 * wp_elo, 0.01), 0.99)
    away_wp = 1.0 - home_wp

    pred_rd = math.log(home_wp / max(away_wp, 1e-6)) * 2.5
    home_ats_prob = norm_cdf(0.0, mu=pred_rd, sigma=RD_SIGMA)
    ml_rec  = "Bet home ML"  if home_wp >= 0.55 else ("Bet away ML"  if away_wp >= 0.55 else "No clear edge")
    ats_rec = "Bet home ATS" if home_ats_prob >= 0.55 else ("Bet away ATS" if home_ats_prob <= 0.45 else "No clear edge")

    return GamePrediction(
        home=home_resolved,
        away=away_resolved,
        date=req_date,
        home_wp=round(home_wp, 4),
        away_wp=round(away_wp, 4),
        pred_run_diff=round(pred_rd, 3),
        ml_recommendation=ml_rec,
        ats_recommendation=ats_rec,
        model_version=mv,
        generated_at=ga,
    )


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
                detail={"message": f"Cannot resolve team '{team}'", "team": team, "suggestions": suggestions},
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
            detail={"message": f"Cannot resolve team '{team}'", "team": team, "suggestions": suggestions},
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

# ── GET /teams ────────────────────────────────────────────────────────────────

_SORT_FIELDS = {"rank", "elo", "win_pct", "pythagorean_win_pct", "avg_run_diff",
                "avg_runs_scored", "avg_runs_allowed", "power_score", "k_per_game"}

@app.get("/teams", response_model=TeamListResponse)
def list_teams(
    season: int = Query(TEST_YEAR),
    conference: Optional[str] = Query(None),
    min_games: int = Query(10, ge=1),
    sort_by: str = Query("rank"),
):
    with _state_lock:
        s = _state

    if s.rankings_df is not None and season == TEST_YEAR:
        df = s.rankings_df.copy()
    elif s.team_stats_df is not None:
        df = s.team_stats_df[s.team_stats_df["season"] == season].copy()
        if s.elo_df is not None:
            df = df.merge(s.elo_df, on="team", how="left")
    else:
        raise HTTPException(503, "Team stats not available — run daily_runner.py first")

    if conference:
        if "conference" in df.columns:
            df = df[df["conference"].str.lower() == conference.lower()]
        else:
            raise HTTPException(400, "Conference data not available for this season")

    if "games" in df.columns:
        df = df[df["games"] >= min_games]

    sort_col = sort_by if sort_by in _SORT_FIELDS and sort_by in df.columns else "rank"
    ascending = sort_col == "rank"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    profiles = [_row_to_profile(row, int(row.get("season", season)))
                for _, row in df.iterrows()]
    return TeamListResponse(teams=profiles, count=len(profiles), season=season)


# ── GET /teams/{team} ─────────────────────────────────────────────────────────

@app.get("/teams/{team}", response_model=TeamProfile)
def team_profile(team: str, season: int = Query(TEST_YEAR)):
    s, mv, ga = _meta_fields()
    resolved, suggestions = resolve_team(team, s.known_teams)
    if resolved is None:
        raise HTTPException(404, detail={"message": f"Cannot resolve team '{team}'", "team": team, "suggestions": suggestions})
    profile = _profile_for_team(resolved, season, s)
    if profile is None:
        raise HTTPException(404, f"No profile data found for '{resolved}' season {season}")
    return profile


# ── GET /teams/{team}/compare/{other_team} ────────────────────────────────────

_DIFF_KEYS = [
    "win_pct", "pythagorean_win_pct", "avg_run_diff",
    "avg_runs_scored", "avg_runs_allowed", "elo", "avg_opp_elo",
    "recent_win_pct", "shutout_pct", "close_win_pct",
    "k_per_game", "bb_per_game", "k_bb_ratio",
]

@app.get("/teams/{team}/compare/{other_team}", response_model=TeamComparison)
def compare_teams(team: str, other_team: str, neutral: bool = Query(False),
                  season: int = Query(TEST_YEAR)):
    s, mv, ga = _meta_fields()

    home_resolved, h_sugg = resolve_team(team, s.known_teams)
    if home_resolved is None:
        raise HTTPException(404, detail={"message": f"Cannot resolve team '{team}'", "team": team, "suggestions": h_sugg})

    away_resolved, a_sugg = resolve_team(other_team, s.known_teams)
    if away_resolved is None:
        raise HTTPException(404, detail={"message": f"Cannot resolve team '{other_team}'", "team": other_team, "suggestions": a_sugg})

    home_profile = _profile_for_team(home_resolved, season, s)
    away_profile = _profile_for_team(away_resolved, season, s)
    if home_profile is None or away_profile is None:
        raise HTTPException(404, "Profile data unavailable for one or both teams")

    diffs: dict[str, float] = {}
    for key in _DIFF_KEYS:
        hv = getattr(home_profile, key)
        av = getattr(away_profile, key)
        if hv is not None and av is not None:
            diffs[key] = round(hv - av, 4)

    prediction = _predict_matchup(home_resolved, away_resolved, neutral,
                                  date.today(), s, mv, ga)

    return TeamComparison(
        home=home_profile,
        away=away_profile,
        differentials=diffs,
        prediction=prediction,
        generated_at=ga,
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@app.post("/predict", response_model=GamePrediction)
def predict(req: GameRequest):
    s, mv, ga = _meta_fields()

    home_resolved, h_suggestions = resolve_team(req.home, s.known_teams)
    if home_resolved is None:
        raise HTTPException(422, detail={"message": f"Cannot resolve team '{req.home}'", "team": req.home, "suggestions": h_suggestions})

    away_resolved, a_suggestions = resolve_team(req.away, s.known_teams)
    if away_resolved is None:
        raise HTTPException(422, detail={"message": f"Cannot resolve team '{req.away}'", "team": req.away, "suggestions": a_suggestions})

    return _predict_matchup(home_resolved, away_resolved, req.neutral,
                            req.date or date.today(), s, mv, ga)
