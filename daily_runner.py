"""
daily_runner.py — runs automatically every day via Windows Task Scheduler.

What it does:
  1. Pulls fresh ESPN game data (today through yesterday)
  2. Recomputes team stats + Elo
  3. Fetches today's live odds from The Odds API
  4. Runs best_bets() and saves results to data/best_bets_YYYY-MM-DD.csv
  5. Appends a summary line to data/daily_log.txt

Schedule setup (run once in PowerShell as admin):
  $action  = New-ScheduledTaskAction -Execute "python" `
               -Argument "C:\\Users\\trevm\\Projects\\CFBBaseballAnalytics\\daily_runner.py" `
               -WorkingDirectory "C:\\Users\\trevm\\Projects\\CFBBaseballAnalytics"
  $trigger = New-ScheduledTaskTrigger -Daily -At "08:00AM"
  Register-ScheduledTask -TaskName "BaseballAnalyticsDaily" -Action $action -Trigger $trigger -RunLevel Highest
"""

import os, time, warnings, pathlib, pickle, json
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd
import requests
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR    = pathlib.Path(__file__).parent / "data"
LOG_FILE     = CACHE_DIR / "daily_log.txt"
GAME_PATH    = CACHE_DIR / "game_results_2021_2026.parquet"
STAT_PATH    = CACHE_DIR / "team_season_stats_2021_2026.parquet"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_BASE    = "https://api.the-odds-api.com/v4"
SPORT_KEY    = "baseball_ncaa"

TRAIN_YEARS       = [2021, 2022, 2023, 2024, 2025]
TEST_YEAR         = 2026
STARTING_BANKROLL = 1000.0
KELLY_FRACTION    = 0.25
KELLY_CAP         = 0.10
ERA_ELO_SCALE     = 25
ELO_K             = 20
ELO_HOME          = 35
ELO_INIT          = 1500
ELO_MEAN_REVERT   = 0.75
ELO_REVERT_TO     = 1505

CONF_TIER = {
    'SEC':1.00,'ACC':1.00,'Big 12':1.00,
    'Sun Belt':0.90,'Mountain West':0.90,'Big Ten':0.90,'Pac-12':0.90,'American Athletic':0.90,
    'Conference USA':0.78,'WAC':0.78,'MAC':0.78,'Missouri Valley':0.78,'Atlantic 10':0.78,
    'Ohio Valley':0.65,'ASUN':0.65,'Big South':0.65,'Southern':0.65,'Northeast':0.65,
    'Patriot':0.65,'MAAC':0.65,'America East':0.65,'Southland':0.65,
}
CONF_DEFAULT = 0.72

# College World Series is always at Charles Schwab Field, Omaha — no home advantage.
# Super Regionals (real home sites) end by ~June 9; CWS runs June 13–25.
CWS_NEUTRAL_START = (6, 13)  # (month, day)
CWS_NEUTRAL_END   = (6, 26)


def _is_cws_neutral(game_date: date) -> bool:
    """True when game_date falls inside the CWS neutral-site window."""
    m, d = game_date.month, game_date.day
    sm, sd = CWS_NEUTRAL_START
    em, ed = CWS_NEUTRAL_END
    start_ok = (m, d) >= (sm, sd)
    end_ok   = (m, d) <= (em, ed)
    return start_ok and end_ok

TEAM_CONF = {
    'Alabama Crimson Tide':'SEC','Arkansas Razorbacks':'SEC','Auburn Tigers':'SEC',
    'Florida Gators':'SEC','Georgia Bulldogs':'SEC','Kentucky Wildcats':'SEC',
    'LSU Tigers':'SEC','Ole Miss Rebels':'SEC','Mississippi State Bulldogs':'SEC',
    'Missouri Tigers':'SEC','South Carolina Gamecocks':'SEC',
    'Tennessee Volunteers':'SEC','Texas A&M Aggies':'SEC',
    'Vanderbilt Commodores':'SEC','Oklahoma Sooners':'SEC','Texas Longhorns':'SEC',
    'Clemson Tigers':'ACC','Duke Blue Devils':'ACC','Florida State Seminoles':'ACC',
    'Georgia Tech Yellow Jackets':'ACC','Louisville Cardinals':'ACC',
    'Miami Hurricanes':'ACC','NC State Wolfpack':'ACC','North Carolina Tar Heels':'ACC',
    'Notre Dame Fighting Irish':'ACC','Virginia Cavaliers':'ACC',
    'Virginia Tech Hokies':'ACC','Wake Forest Demon Deacons':'ACC',
    'Boston College Eagles':'ACC','Pittsburgh Panthers':'ACC',
    'Baylor Bears':'Big 12','BYU Cougars':'Big 12','Cincinnati Bearcats':'Big 12',
    'Houston Cougars':'Big 12','Iowa State Cyclones':'Big 12','Kansas Jayhawks':'Big 12',
    'Kansas State Wildcats':'Big 12','Oklahoma State Cowboys':'Big 12',
    'TCU Horned Frogs':'Big 12','Texas Tech Red Raiders':'Big 12',
    'UCF Knights':'Big 12','West Virginia Mountaineers':'Big 12',
    'Coastal Carolina Chanticleers':'Sun Belt','Georgia Southern Eagles':'Sun Belt',
    'Old Dominion Monarchs':'Sun Belt','James Madison Dukes':'Sun Belt',
    'South Alabama Jaguars':'Sun Belt','Southern Miss Golden Eagles':'Sun Belt',
    'Troy Trojans':'Sun Belt',"Louisiana Ragin' Cajuns":'Sun Belt',
    'Arkansas State Red Wolves':'Sun Belt','Texas State Bobcats':'Sun Belt',
    'Louisiana Monroe Warhawks':'Sun Belt','Marshall Thundering Herd':'Sun Belt','East Carolina Pirates':'American Athletic',
    'Florida Atlantic Owls':'American Athletic','South Florida Bulls':'American Athletic',
    'Wichita State Shockers':'American Athletic','Rice Owls':'American Athletic',
    'Tulane Green Wave':'American Athletic','UAB Blazers':'American Athletic',
    'Indiana Hoosiers':'Big Ten','Illinois Fighting Illini':'Big Ten',
    'Maryland Terrapins':'Big Ten','Michigan Wolverines':'Big Ten',
    'Minnesota Golden Gophers':'Big Ten','Nebraska Cornhuskers':'Big Ten',
    'Northwestern Wildcats':'Big Ten','Ohio State Buckeyes':'Big Ten',
    'Penn State Nittany Lions':'Big Ten','Rutgers Scarlet Knights':'Big Ten',
    'Oregon State Beavers':'Pac-12','Dallas Baptist Patriots':'Missouri Valley',
    'Illinois State Redbirds':'Missouri Valley','Indiana State Sycamores':'Missouri Valley',
    'Jacksonville State Gamecocks':'Conference USA','Liberty Flames':'Conference USA',
}

FEAT_COLS = [
    'avg_runs_scored', 'avg_runs_allowed', 'avg_run_diff',
    'pythagorean_win_pct', 'win_pct', 'recent_win_pct',
    'avg_runs_scored_z', 'avg_runs_allowed_z', 'pythagorean_win_pct_z',
    # consistency & situational (derived from game scores)
    'runs_scored_std', 'runs_allowed_std',
    'shutout_pct', 'close_win_pct',
    # pitching strikeout/walk rates (from pull_pitching_stats.py)
    'k_per_game', 'bb_per_game', 'k_bb_ratio',
    # batting & ERA features (from pull_batting_stats.py — 0-filled when missing)
    'ba', 'obp', 'era', 'whip',
]

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

from pipeline.utils import american_to_prob, kelly_fraction, era_adjustment as _era_adj_util

def expected_win(ea, eb):
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400))

def margin_mult_538(rd, elo_diff_pre):
    movm = np.log(abs(rd) + 1) * 2.2
    return movm * (2.2 / (elo_diff_pre * 0.001 + 2.2))

# ── step 1: pull fresh ESPN data ──────────────────────────────────────────────
def pull_espn_recent(days_back=2):
    BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"
    HDR  = {"User-Agent": "Mozilla/5.0"}
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(days_back + 1)]
    rows  = []
    for d in dates:
        try:
            r = requests.get(BASE, params={"dates": d.strftime("%Y%m%d"), "limit": 200},
                             headers=HDR, timeout=15)
            if r.status_code != 200: continue
            for ev in r.json().get("events", []):
                comp = ev["competitions"][0]
                if comp["status"]["type"]["name"] != "STATUS_FINAL": continue
                ts   = comp["competitors"]
                home = next((t for t in ts if t.get("homeAway") == "home"), ts[0])
                away = next((t for t in ts if t.get("homeAway") == "away"), ts[1])
                rows.append({
                    "season":     d.year,
                    "date":       pd.to_datetime(ev["date"][:10]),
                    "home_team":  home["team"]["displayName"],
                    "away_team":  away["team"]["displayName"],
                    "home_score": float(home.get("score", "nan")),
                    "away_score": float(away.get("score", "nan")),
                    "neutral":    comp.get("neutralSite", False),
                })
            time.sleep(0.05)
        except Exception as e:
            log(f"  ESPN pull error for {d}: {e}")
    if not rows:
        return pd.DataFrame(columns=["season","date","home_team","away_team","home_score","away_score","neutral"])
    return pd.DataFrame(rows).dropna(subset=["home_score", "away_score"])

def update_game_data():
    if GAME_PATH.exists():
        existing = pd.read_parquet(GAME_PATH)
    else:
        existing = pd.DataFrame()

    new_rows = pull_espn_recent(days_back=2)
    if new_rows.empty:
        log("No new games found today.")
        return existing

    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"])
        new_rows["date"] = pd.to_datetime(new_rows["date"])
        dedup_key = new_rows.apply(lambda r: f"{r['date']}|{r['home_team']}|{r['away_team']}", axis=1)
        ex_key    = existing.apply(lambda r: f"{r['date']}|{r['home_team']}|{r['away_team']}", axis=1)
        new_rows  = new_rows[~dedup_key.isin(ex_key)]

    if new_rows.empty:
        log("No new games to add (already up to date).")
        return existing

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["season"] = combined["season"].astype(int)
    combined["neutral"] = combined["neutral"].astype(bool)
    combined.to_parquet(GAME_PATH, index=False)
    log(f"Added {len(new_rows)} new games. Total: {len(combined):,}")
    return combined

# ── step 2: recompute stats + Elo ─────────────────────────────────────────────
def compute_stats(games):
    home = games[["season","date","home_team","home_score","away_score","neutral"]].copy()
    home.columns = ["season","date","team","rs","ra","neutral"]
    away = games[["season","date","away_team","away_score","home_score","neutral"]].copy()
    away.columns = ["season","date","team","rs","ra","neutral"]
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["rs"] > long["ra"]).astype(int)
    long["rd"]  = long["rs"] - long["ra"]

    agg = (long.groupby(["team","season"])
               .agg(games=("win","count"), wins=("win","sum"),
                    rs_total=("rs","sum"), ra_total=("ra","sum"), rd_total=("rd","sum"))
               .reset_index())
    agg["losses"]           = agg["games"] - agg["wins"]
    agg["win_pct"]          = agg["wins"] / agg["games"]
    agg["avg_runs_scored"]  = agg["rs_total"] / agg["games"]
    agg["avg_runs_allowed"] = agg["ra_total"] / agg["games"]
    agg["avg_run_diff"]     = agg["rd_total"] / agg["games"]
    rs, ra = agg["rs_total"], agg["ra_total"]
    agg["pythagorean_win_pct"] = rs**1.83 / (rs**1.83 + ra**1.83).replace(0, float("nan"))

    # Scoring consistency (std dev of per-game runs) — independent of the mean
    std_rs = (long.groupby(["team","season"])["rs"]
                  .std(ddof=1).reset_index()
                  .rename(columns={"rs": "runs_scored_std"}))
    std_ra = (long.groupby(["team","season"])["ra"]
                  .std(ddof=1).reset_index()
                  .rename(columns={"ra": "runs_allowed_std"}))
    agg = agg.merge(std_rs, on=["team","season"], how="left")
    agg = agg.merge(std_ra, on=["team","season"], how="left")

    # Shutout rate — fraction of games allowing ≤2 runs (elite pitching signal)
    long["shutout"] = (long["ra"] <= 2).astype(int)
    shutout = (long.groupby(["team","season"])["shutout"]
                   .mean().reset_index()
                   .rename(columns={"shutout": "shutout_pct"}))
    agg = agg.merge(shutout, on=["team","season"], how="left")

    # Close-game win rate — win% in games decided by ≤2 runs (clutch performance)
    long["close"] = (long["rd"].abs() <= 2).astype(int)
    close_wins = (long[long["close"] == 1]
                  .groupby(["team","season"])["win"]
                  .mean().reset_index()
                  .rename(columns={"win": "close_win_pct"}))
    agg = agg.merge(close_wins, on=["team","season"], how="left")

    # Normalised features
    norm_cols = ["avg_runs_scored", "avg_runs_allowed", "avg_run_diff",
                 "pythagorean_win_pct", "runs_scored_std", "runs_allowed_std",
                 "shutout_pct", "close_win_pct"]
    for yr in agg["season"].unique():
        mask = agg["season"] == yr
        for col in norm_cols:
            if col not in agg.columns:
                continue
            mu = agg.loc[mask, col].mean()
            sd = max(float(agg.loc[mask, col].std()), 1e-6)
            agg.loc[mask, col + "_z"] = (agg.loc[mask, col] - mu) / sd

    # Recent form (last 15 games)
    long = long.sort_values("date")
    long["recent_win_pct"] = (long.groupby(["team","season"])["win"]
                                   .transform(lambda x: x.rolling(15, min_periods=3).mean()))
    form = long.groupby(["team","season"])["recent_win_pct"].last().reset_index()
    agg  = agg.merge(form, on=["team","season"], how="left")

    # Pitching K/BB stats — merge from pre-built parquet files if available
    pit_files = sorted(CACHE_DIR.glob("team_pitching_stats_*.parquet"))
    if pit_files:
        pit = pd.concat([pd.read_parquet(p) for p in pit_files], ignore_index=True)
        pit = pit[["team", "season", "k_per_game", "bb_per_game", "k_bb_ratio"]]
        for col in ["k_per_game", "bb_per_game", "k_bb_ratio"]:
            if col in agg.columns:
                agg = agg.drop(columns=[col])
        agg = agg.merge(pit, on=["team", "season"], how="left")
        for col in ["k_per_game", "bb_per_game", "k_bb_ratio"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0)

    # Batting + ERA features — merge from pull_batting_stats.py output if available
    adv_files = sorted(CACHE_DIR.glob("team_adv_stats_*.parquet"))
    if adv_files:
        adv = pd.concat([pd.read_parquet(p) for p in adv_files], ignore_index=True)
        adv = adv[["team", "season"] + [c for c in ["ba", "obp", "hr_rate", "era", "whip"]
                                         if c in adv.columns]]
        for col in [c for c in adv.columns if c not in ("team", "season")]:
            if col in agg.columns:
                agg = agg.drop(columns=[col])
        agg = agg.merge(adv, on=["team", "season"], how="left")
        # Fill 0 so rows without batting data are still usable for training
        for col in ["ba", "obp", "hr_rate", "era", "whip"]:
            if col in agg.columns:
                agg[col] = agg[col].fillna(0)

    return agg

def compute_elo(games):
    elo = {}
    cur_season = None
    gs = games.sort_values(["season","date"]).dropna(subset=["date"])
    for _, g in gs.iterrows():
        yr = int(g["season"])
        if yr != cur_season:
            if cur_season is not None:
                elo = {t: ELO_MEAN_REVERT * v + (1 - ELO_MEAN_REVERT) * ELO_REVERT_TO
                       for t, v in elo.items()}
            cur_season = yr
        ht, at = g["home_team"], g["away_team"]
        base_h, base_a = elo.get(ht, ELO_INIT), elo.get(at, ELO_INIT)
        eh = base_h + (ELO_HOME if not g.get("neutral", False) else 0)
        ea = base_a
        exp    = expected_win(eh, ea)
        actual = 1 if g["home_score"] > g["away_score"] else 0
        rd     = g["home_score"] - g["away_score"]
        winner_elo = eh if rd > 0 else ea
        loser_elo  = ea if rd > 0 else eh
        elo_diff_pre = max(winner_elo - loser_elo, 0)
        mult  = margin_mult_538(rd, elo_diff_pre)
        tier  = (CONF_TIER.get(TEAM_CONF.get(ht,""), CONF_DEFAULT) +
                 CONF_TIER.get(TEAM_CONF.get(at,""), CONF_DEFAULT)) / 2
        delta = ELO_K * tier * mult * (actual - exp)
        elo[ht] = base_h + delta
        elo[at] = base_a - delta
    return pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})


def compute_sos(games_df, elo_df):
    """Average opponent Elo per team-season (strength of schedule proxy)."""
    elo_lk = elo_df.set_index("team")["elo"]
    home = games_df[["season", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp"})
    away = games_df[["season", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"})
    long = pd.concat([home, away], ignore_index=True)
    long["opp_elo"] = long["opp"].map(elo_lk).fillna(ELO_INIT)
    return (long.groupby(["team", "season"])["opp_elo"].mean()
               .reset_index().rename(columns={"opp_elo": "avg_opp_elo"}))


def build_rankings(team_stats, elo_df, sos_df, year=None):
    """Composite power ranking for a single season, saved to data/rankings.parquet."""
    if year is None:
        year = TEST_YEAR
    df = team_stats[team_stats["season"] == year].copy()
    df = df.merge(elo_df, on="team", how="left")
    df = df.merge(sos_df[sos_df["season"] == year][["team", "avg_opp_elo"]],
                  on="team", how="left")
    df["elo"]         = df["elo"].fillna(ELO_INIT)
    df["avg_opp_elo"] = df["avg_opp_elo"].fillna(df["avg_opp_elo"].mean())
    df["conference"]  = df["team"].map(TEAM_CONF).fillna("Unknown")

    def z(s):
        sd = s.std()
        return (s - s.mean()) / (sd if sd > 1e-6 else 1.0)

    score = pd.Series(0.0, index=df.index)
    for col, wt, higher in [
        ("pythagorean_win_pct", 0.25, True),
        ("avg_run_diff",        0.20, True),
        ("elo",                 0.20, True),
        ("avg_runs_scored",     0.10, True),
        ("avg_runs_allowed",    0.10, False),
        ("avg_opp_elo",         0.15, True),
    ]:
        if col in df.columns:
            score += wt * (z(df[col]) if higher else -z(df[col]))

    df["power_score"] = score
    df["rank"]        = df["power_score"].rank(ascending=False).astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def generate_static_dashboard(rankings, bets, year):
    """Regenerate docs/index.html by injecting fresh rankings JSON into the template."""
    import json as _json, re

    html_path = pathlib.Path(__file__).parent / "docs" / "index.html"
    if not html_path.exists():
        log("docs/index.html not found — skipping static dashboard generation.")
        return

    # Build TEAMS array — every column the JS needs
    cols = ["rank", "team", "conference", "wins", "losses", "win_pct",
            "pythagorean_win_pct", "avg_runs_scored", "avg_runs_allowed",
            "avg_run_diff", "elo", "avg_opp_elo", "recent_win_pct", "power_score"]
    teams_data = []
    for _, row in rankings.iterrows():
        rec = {}
        for col in cols:
            if col not in row.index:
                rec[col] = None
                continue
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (int, np.integer)):
                rec[col] = int(val)
            elif isinstance(val, float):
                rec[col] = round(float(val), 4)
            else:
                rec[col] = val
        teams_data.append(rec)

    teams_json = _json.dumps(teams_data, separators=(",", ":"))

    # Build BETS array from today's best_bets CSV (empty list if no bets)
    bets_data = bets.to_dict("records") if not bets.empty else []
    bets_json = _json.dumps(bets_data, separators=(",", ":"))

    html = html_path.read_text(encoding="utf-8")

    # Replace data blobs — each is a single line in the file
    # Use lambda replacers so JSON strings with backslashes aren't mis-parsed
    _teams = f"const TEAMS = {teams_json};"
    _season = f"const SEASON   = {year};"
    _bets   = f"const BETS = {bets_json};"
    html = re.sub(r"const TEAMS\s*=\s*\[.*?\];",
                  lambda _: _teams, html, flags=re.DOTALL)
    html = re.sub(r"const SEASON\s*=\s*\d+;",
                  lambda _: _season, html)
    html = re.sub(r"const BETS\s*=\s*\[.*?\];",
                  lambda _: _bets, html, flags=re.DOTALL)

    # Stamp generated date in the header badge
    today = date.today().isoformat()
    html = re.sub(r"Season \d{4}",  f"Season {year}", html)
    html = re.sub(r"Updated \d{4}-\d{2}-\d{2}", f"Updated {today}", html)

    html_path.write_text(html, encoding="utf-8")
    log(f"docs/index.html regenerated — {len(teams_data)} teams, {len(bets_data)} bets.")


# ── step 3: train logistic regression classifier ───────────────────────────────
def train_model(games, team_stats):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = team_stats.set_index(["team", "season"])
    feat_cols = [c for c in FEAT_COLS if c in team_stats.columns]
    diff_feats = [f"d_{c}" for c in feat_cols] + ["is_home", "neutral"]
    rows = []
    for _, g in games.iterrows():
        yr = int(g["season"])
        ht, at = g["home_team"], g["away_team"]
        hk, ak = (ht, yr), (at, yr)
        if hk not in idx.index or ak not in idx.index:
            continue
        hf, af = idx.loc[hk, feat_cols], idx.loc[ak, feat_cols]
        row = {
            "is_home": 1,
            "neutral": int(g.get("neutral", False)),
            "win":     int(g["home_score"] > g["away_score"]),
        }
        for c in feat_cols:
            row[f"d_{c}"] = hf.get(c, np.nan) - af.get(c, np.nan)
        rows.append(row)
    gm = pd.DataFrame(rows).dropna()
    feats = [c for c in diff_feats if c in gm.columns]
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    # Isotonic calibration via 5-fold CV — fixes non-monotonic over-confidence
    # in the 55-60% bucket without touching the training data directly.
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(gm[feats].values, gm["win"].values)
    return clf, feats

# ── step 4: fetch live odds + compute best bets ────────────────────────────────
ODDS_LOOKAHEAD_DAYS = 3  # college baseball books typically post 2-3 days out

def fetch_odds(days_ahead: int = ODDS_LOOKAHEAD_DAYS):
    """Fetch moneyline odds for games from now through +days_ahead days.

    The Odds API commenceTimeTo filter captures lines posted in advance so the
    runner surfaces bets on upcoming games, not just today's slate.
    """
    if not ODDS_API_KEY:
        log("ODDS_API_KEY not set — skipping odds fetch.")
        return pd.DataFrame()
    from datetime import timezone as _tz
    now    = datetime.now(_tz.utc)
    cutoff = now + timedelta(days=days_ahead)
    r = requests.get(f"{ODDS_BASE}/sports/{SPORT_KEY}/odds",
                     params=dict(apiKey=ODDS_API_KEY, regions="us",
                                 markets="h2h", oddsFormat="american",
                                 commenceTimeFrom=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 commenceTimeTo=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")),
                     timeout=15)
    if r.status_code != 200:
        log(f"Odds API error {r.status_code}: {r.text[:200]}")
        return pd.DataFrame()
    rows = []
    for game in r.json():
        home, away = game["home_team"], game["away_team"]
        # Odds API doesn't expose a neutral-site flag. Detect CWS window by
        # game date: June 13-26 games are at Charles Schwab Field (neutral).
        commence_str = game.get("commence_time", "")
        try:
            game_date = datetime.fromisoformat(commence_str.replace("Z", "+00:00")).date()
            neutral = _is_cws_neutral(game_date)
        except (ValueError, AttributeError):
            neutral = False
        for bkm in game.get("bookmakers", [])[:1]:
            for mkt in bkm.get("markets", []):
                if mkt["key"] == "h2h":
                    out = {o["name"]: o["price"] for o in mkt["outcomes"]}
                    rows.append({
                        "home":    home,
                        "away":    away,
                        "ml_home": out.get(home),
                        "ml_away": out.get(away),
                        "commence":game.get("commence_time", ""),
                        "neutral": neutral,
                    })
    log(f"Fetched odds for {len(rows)} games "
        f"(now → +{days_ahead}d). "
        f"Requests remaining: {r.headers.get('x-requests-remaining', '?')}")
    return pd.DataFrame(rows)


def era_adjustment_runner(team, team_stats, year=TEST_YEAR):
    col = 'avg_runs_allowed_z'
    row = team_stats[(team_stats['team'] == team) & (team_stats['season'] == year)]
    if row.empty or col not in team_stats.columns:
        return 0.0
    return -ERA_ELO_SCALE * float(row[col].iloc[0])

def compute_best_bets(odds_df, team_stats, elo_df, clf, feats, edge_min=0.03):
    idx = team_stats[team_stats["season"] == TEST_YEAR].set_index("team")
    feat_base = [c.replace("d_","") for c in feats if c.startswith("d_")]
    elo_lk = elo_df.set_index("team")["elo"]
    rows = []
    for _, g in odds_df.iterrows():
        home, away = g["home"], g["away"]
        if home not in idx.index or away not in idx.index:
            missing = [t for t in (home, away) if t not in idx.index]
            log(f"  Skipping {home} vs {away} — unknown team(s): {missing}")
            continue
        hf, af = idx.loc[home], idx.loc[away]
        is_neutral = int(bool(g.get("neutral", False)))
        # Neutral-site games: strip home advantage from Elo and model features
        home_adv = 0 if is_neutral else ELO_HOME
        feat_row = {"is_home": 1 - is_neutral, "neutral": is_neutral}
        for c in feat_base:
            feat_row[f"d_{c}"] = hf.get(c, np.nan) - af.get(c, np.nan)
        X = np.array([[feat_row.get(f, 0) for f in feats]])
        wp_home = float(clf.predict_proba(X)[0, 1])
        era_h = era_adjustment_runner(home, team_stats)
        era_a = era_adjustment_runner(away, team_stats)
        elo_h = elo_lk.get(home, ELO_INIT) + home_adv + era_h
        elo_a = elo_lk.get(away, ELO_INIT) + era_a
        wp_elo = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400))
        wp_home = 0.6 * wp_home + 0.4 * wp_elo
        wp_away = 1.0 - wp_home

        for side, team, ml, model_wp in [
            ("Home", home, g.get("ml_home"), wp_home),
            ("Away", away, g.get("ml_away"), wp_away),
        ]:
            if pd.isna(ml) or ml is None: continue
            ml = float(ml)
            implied = american_to_prob(ml)
            edge = model_wp - implied
            if edge < edge_min: continue
            b  = (100/abs(ml)) if ml < 0 else (ml/100)
            ev = model_wp * b - (1.0 - model_wp)
            kf = kelly_fraction(model_wp, odds=ml)
            game_date = str(g.get("commence", ""))[:10] or date.today().isoformat()
            rows.append({
                "run_date":     date.today().isoformat(),
                "game_date":    game_date,
                "matchup":      f"{home} vs {away}",
                "bet_on":       team,
                "side":         side,
                "neutral":      bool(is_neutral),
                "ml":           int(ml),
                "model_wp":     round(model_wp, 4),
                "implied_prob": round(implied, 4),
                "edge":         round(edge, 4),
                "ev_per_unit":  round(ev, 4),
                "kelly_pct":    round(kf, 4),
                "kelly_$":      round(STARTING_BANKROLL * kf, 2),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev_per_unit", ascending=False).reset_index(drop=True)

# ── step 7: sync artifacts to VPS ────────────────────────────────────────────
def sync_data_to_vps():
    """SFTP the pipeline artifacts that the remote API reads to the VPS."""
    vps_host = os.environ.get("VPS_HOST", "").strip()
    vps_user = os.environ.get("VPS_USER", "root").strip()
    vps_pass = os.environ.get("VPS_PASSWORD", "").strip()
    vps_path = os.environ.get("VPS_DATA_PATH", "/opt/baseball/data").strip()

    if not vps_host or not vps_pass:
        log("VPS_HOST or VPS_PASSWORD not set — skipping remote sync.")
        return

    files = [
        CACHE_DIR / "run_meta.json",
        CACHE_DIR / "model.pkl",
        CACHE_DIR / "rankings.parquet",
        CACHE_DIR / "elo_ratings.parquet",
        STAT_PATH,
        CACHE_DIR / f"best_bets_{date.today().isoformat()}.csv",
    ]
    files = [p for p in files if p.exists()]

    try:
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(vps_host, username=vps_user, password=vps_pass, timeout=30)
        sftp = client.open_sftp()
        for local in files:
            remote = f"{vps_path}/{local.name}"
            sftp.put(str(local), remote)
        sftp.close()
        client.close()
        log(f"VPS sync complete — {len(files)} files pushed to {vps_host}.")
    except Exception as e:
        log(f"VPS sync failed: {e}")


# ── step 8: push docs/index.html to GitHub ────────────────────────────────────
def push_dashboard_to_github():
    """Commit docs/index.html and push to origin/master."""
    import subprocess

    repo = pathlib.Path(__file__).parent
    today = date.today().isoformat()

    def run(cmd):
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  git error: {result.stderr.strip()}")
        return result.returncode == 0

    # Guard against ever repeating the 002-ecliptic-works-fx incident: if this
    # checkout isn't on master (e.g. someone switched branches for other work
    # in the same working directory), skip the commit entirely rather than
    # silently pushing dashboard updates onto whatever branch is checked out.
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    if branch != "master":
        log(f"Current branch is '{branch}', not master — skipping dashboard auto-commit/push.")
        return

    # Sync with origin/master before committing. Local and the VPS both run
    # this daily and both push to master — without pulling first, whichever
    # runs second gets a non-fast-forward rejection. `--ff-only` is safe here:
    # if the just-regenerated docs/index.html would collide with an incoming
    # change, git refuses the pull rather than overwriting anything, and we
    # just skip this run's push — tomorrow's regeneration catches up either way.
    if not run(["git", "pull", "--ff-only", "origin", "master"]):
        log("Could not sync with origin/master before commit — skipping push this run.")
        return

    # Only stage the static dashboard — never auto-commit data files or secrets
    run(["git", "add", "docs/index.html"])

    # Check if there's anything to commit
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo, capture_output=True
    )
    if status.returncode == 0:
        log("Dashboard unchanged — nothing to push.")
        return

    ok = run(["git", "commit", "-m", f"chore: auto-update dashboard {today}"])
    if not ok:
        log("Git commit failed — skipping push.")
        return

    # Explicit target (not HEAD) — HEAD is whatever's checked out, which is
    # exactly how this ended up on the wrong branch for two months.
    ok = run(["git", "push", "origin", "HEAD:master"])
    if ok:
        log(f"Dashboard pushed to GitHub ({today}).")
    else:
        log("Git push failed — check credentials/network.")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        log("=" * 60)
        log("Daily runner started")

        # 1. Update data
        games = update_game_data()
        if games.empty:
            log("No game data — aborting."); return

        # 2. Recompute stats + Elo + rankings
        team_stats = compute_stats(games)
        elo_df     = compute_elo(games)
        sos_df     = compute_sos(games, elo_df)
        rankings   = build_rankings(team_stats, elo_df, sos_df)
        team_stats.to_parquet(STAT_PATH, index=False)
        elo_df.to_parquet(CACHE_DIR / "elo_ratings.parquet", index=False)
        rankings.to_parquet(CACHE_DIR / "rankings.parquet", index=False)
        log(f"Stats recomputed: {len(team_stats)} team-seasons | {len(elo_df)} Elo ratings | {len(rankings)} ranked")

        # 3. Train model + persist artifact
        try:
            clf, feats = train_model(games, team_stats)
            model_artifact = {"clf": clf, "feats": feats}
            with open(CACHE_DIR / "model.pkl", "wb") as pf:
                pickle.dump(model_artifact, pf)
            log("Model trained (LogisticRegression) and saved to model.pkl.")
        except Exception as e:
            log(f"Model training failed: {e}"); return

        # Write run_meta.json immediately after model training so the freshness
        # signal is always updated even when odds are unavailable or bets fail.
        def write_run_meta(bets_found=0):
            meta = {
                "run_date":      date.today().isoformat(),
                "generated_at":  datetime.now(timezone.utc).isoformat(),
                "games_loaded":  len(games),
                "teams_ranked":  int(team_stats[team_stats["season"] == TEST_YEAR]["team"].nunique()),
                "bets_found":    bets_found,
                "model_version": date.today().isoformat(),
                "train_cutoff":  max(TRAIN_YEARS),
                "test_year":     TEST_YEAR,
                "features":      feats,
            }
            with open(CACHE_DIR / "run_meta.json", "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)
            log("run_meta.json written.")

        write_run_meta(bets_found=0)

        # 4. Fetch odds + best bets
        bets = pd.DataFrame()
        odds_df = fetch_odds()
        if odds_df.empty:
            log("No odds data — skipping best bets.")
        else:
            bets = compute_best_bets(odds_df, team_stats, elo_df, clf, feats, edge_min=0.03)
            if bets.empty:
                log("No bets found with edge > 3% today.")
            else:
                out_path = CACHE_DIR / f"best_bets_{date.today().isoformat()}.csv"
                bets.to_csv(out_path, index=False)
                log(f"Saved {len(bets)} best bets -> {out_path}")
                log("Top bets:")
                for _, r in bets.head(5).iterrows():
                    log(f"  {r['bet_on']:30s}  ML {r['ml']:+5.0f}  "
                        f"model {r['model_wp']:.1%}  edge {r['edge']:+.1%}  "
                        f"EV {r['ev_per_unit']:+.3f}  Kelly ${r['kelly_$']:.0f}")
            write_run_meta(bets_found=len(bets))

        # 5. Regenerate static dashboard
        generate_static_dashboard(rankings, bets, TEST_YEAR)

        # 6. Push docs/index.html to GitHub so trevormonroe.com reflects today's data
        push_dashboard_to_github()

        # 7. Sync pipeline artifacts to VPS so the remote API hot-reloads fresh data
        sync_data_to_vps()

        log("Daily runner complete.")

    except Exception as e:
        log(f"UNHANDLED ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
