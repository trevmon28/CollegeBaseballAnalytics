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
    'Troy Trojans':'Sun Belt','East Carolina Pirates':'American Athletic',
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

FEAT_COLS = ['avg_runs_scored','avg_runs_allowed','avg_run_diff',
             'pythagorean_win_pct','win_pct','recent_win_pct',
             'avg_runs_scored_z','avg_runs_allowed_z','pythagorean_win_pct_z']

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

    # Normalised features
    for yr in agg["season"].unique():
        mask = agg["season"] == yr
        for col in ["avg_runs_scored","avg_runs_allowed","avg_run_diff","pythagorean_win_pct"]:
            mu = agg.loc[mask, col].mean()
            sd = max(float(agg.loc[mask, col].std()), 1e-6)
            agg.loc[mask, col + "_z"] = (agg.loc[mask, col] - mu) / sd

    # Recent form (last 15 games)
    long = long.sort_values("date")
    long["recent_win_pct"] = (long.groupby(["team","season"])["win"]
                                   .transform(lambda x: x.rolling(15, min_periods=3).mean()))
    form = long.groupby(["team","season"])["recent_win_pct"].last().reset_index()
    agg  = agg.merge(form, on=["team","season"], how="left")
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

# ── step 3: train a quick XGBoost classifier ──────────────────────────────────
def train_model(games, team_stats):
    import xgboost as xgb
    idx = team_stats.set_index(["team","season"])
    feat_cols = [c for c in FEAT_COLS if c in team_stats.columns]
    diff_feats = [f"d_{c}" for c in feat_cols] + ["is_home","neutral"]
    rows = []
    for _, g in games.iterrows():
        yr = int(g["season"])
        ht, at = g["home_team"], g["away_team"]
        hk, ak = (ht,yr), (at,yr)
        if hk not in idx.index or ak not in idx.index: continue
        hf, af = idx.loc[hk, feat_cols], idx.loc[ak, feat_cols]
        row = {"is_home":1,"neutral":int(g.get("neutral",False)),
               "win":int(g["home_score"]>g["away_score"])}
        for c in feat_cols:
            row[f"d_{c}"] = hf.get(c,np.nan) - af.get(c,np.nan)
        rows.append(row)
    gm = pd.DataFrame(rows).dropna()
    tr = gm[gm.index < len(gm)*0.85]
    feats = [c for c in diff_feats if c in gm.columns]
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             eval_metric="logloss", random_state=42, n_jobs=-1)
    clf.fit(tr[feats].values, tr["win"].values, verbose=False)
    return clf, feats

# ── step 4: fetch live odds + compute best bets ────────────────────────────────
def fetch_odds():
    if not ODDS_API_KEY:
        log("ODDS_API_KEY not set — skipping odds fetch.")
        return pd.DataFrame()
    r = requests.get(f"{ODDS_BASE}/sports/{SPORT_KEY}/odds",
                     params=dict(apiKey=ODDS_API_KEY, regions="us",
                                 markets="h2h", oddsFormat="american"),
                     timeout=15)
    if r.status_code != 200:
        log(f"Odds API error {r.status_code}: {r.text[:200]}")
        return pd.DataFrame()
    rows = []
    for game in r.json():
        home, away = game["home_team"], game["away_team"]
        for bkm in game.get("bookmakers",[])[:1]:
            for mkt in bkm.get("markets",[]):
                if mkt["key"] == "h2h":
                    out = {o["name"]: o["price"] for o in mkt["outcomes"]}
                    rows.append({"home":home,"away":away,
                                 "ml_home":out.get(home),"ml_away":out.get(away),
                                 "commence":game.get("commence_time","")})
    log(f"Fetched odds for {len(rows)} games. "
        f"Requests remaining: {r.headers.get('x-requests-remaining','?')}")
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
    rows = []
    for _, g in odds_df.iterrows():
        home, away = g["home"], g["away"]
        hk, ak = home, away
        if hk not in idx.index or ak not in idx.index: continue
        hf, af = idx.loc[hk], idx.loc[ak]
        row = {"is_home":1,"neutral":0}
        for c in feat_base:
            row[f"d_{c}"] = hf.get(c,np.nan) - af.get(c,np.nan)
        X = np.array([[row.get(f,0) for f in feats]])
        wp_home = float(clf.predict_proba(X)[0,1])
        era_h = era_adjustment_runner(home, team_stats)
        era_a_val = era_adjustment_runner(away, team_stats)
        elo_h = elo_df.set_index("team")["elo"].get(home, ELO_INIT) + ELO_HOME + era_h
        elo_a = elo_df.set_index("team")["elo"].get(away, ELO_INIT) + era_a_val
        wp_elo = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400))
        wp_home = 0.6 * wp_home + 0.4 * wp_elo   # blend model + ERA-adjusted Elo
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
            rows.append({
                "date":         date.today().isoformat(),
                "matchup":      f"{home} vs {away}",
                "bet_on":       team,
                "side":         side,
                "ml":           int(ml),
                "model_wp":     round(model_wp, 4),
                "implied_prob": round(implied, 4),
                "edge":         round(edge, 4),
                "ev_per_unit":  round(ev, 4),
                "kelly_pct":    round(kf, 4),
                "kelly_$":      round(STARTING_BANKROLL * kf, 2),
            })
    return pd.DataFrame(rows).sort_values("ev_per_unit", ascending=False).reset_index(drop=True)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    log("=" * 60)
    log("Daily runner started")

    # 1. Update data
    games = update_game_data()
    if games.empty:
        log("No game data — aborting."); return

    # 2. Recompute stats + Elo
    team_stats = compute_stats(games)
    elo_df     = compute_elo(games)
    team_stats.to_parquet(STAT_PATH, index=False)
    log(f"Stats recomputed: {len(team_stats)} team-seasons | {len(elo_df)} Elo ratings")

    # 3. Train model + persist artifact
    try:
        clf, feats = train_model(games, team_stats)
        model_artifact = {"clf": clf, "feats": feats}
        with open(CACHE_DIR / "model.pkl", "wb") as pf:
            pickle.dump(model_artifact, pf)
        log("XGBoost model trained and saved to model.pkl.")
    except Exception as e:
        log(f"Model training failed: {e}"); return

    # 4. Fetch odds + best bets
    odds_df = fetch_odds()
    if odds_df.empty:
        log("No odds data — skipping best bets."); return

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

    # 5. Write run_meta.json — freshness signal for the API
    run_meta = {
        "run_date":       date.today().isoformat(),
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "games_loaded":   len(games),
        "teams_ranked":   int(team_stats[team_stats["season"] == TEST_YEAR]["team"].nunique()),
        "bets_found":     len(bets) if not bets.empty else 0,
        "model_version":  date.today().isoformat(),
        "train_cutoff":   max(TRAIN_YEARS),
        "test_year":      TEST_YEAR,
        "features":       feats,
    }
    with open(CACHE_DIR / "run_meta.json", "w", encoding="utf-8") as mf:
        json.dump(run_meta, mf, indent=2)
    log("run_meta.json written.")

    log("Daily runner complete.")

if __name__ == "__main__":
    main()
