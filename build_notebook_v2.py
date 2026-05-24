"""
Rebuild CollegeBaseballAnalytics_Master.ipynb using the new runs-based data schema.
Data comes from pull_ncaa_data.py output:
  - game_results_2021_2025.parquet   (one row per game)
  - team_season_stats_2021_2025.parquet (one row per team × season)
"""
import json

def code_cell(src):
    source = [line + "\n" for line in src.split("\n")]
    if source: source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def md_cell(src):
    source = [line + "\n" for line in src.split("\n")]
    if source: source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("""\
# College Baseball Analytics — Power Rankings & Game Predictor
**Sections:** Setup → Data → Elo Rankings → Run-Diff Model → Win Probability → Ensemble → Game Predictor → Betting → Kelly

*Google Colab + VS Code compatible. Training data: 2021–2025 NCAA Division I.*
*Data sourced from sportsdataverse (2021-2023) + ESPN scoreboard API (2024-2025) via `pull_ncaa_data.py`.*"""))

# ── SECTION 1: SETUP ──────────────────────────────────────────────────────────
cells.append(md_cell("## Section 1 — Setup & Data Loading"))

cells.append(code_cell("""\
!pip install -q xgboost scikit-learn fastai requests pyarrow python-dotenv"""))

cells.append(code_cell("""\
import warnings, os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import xgboost as xgb
warnings.filterwarnings("ignore")

TRAIN_YEARS       = [2021, 2022, 2023, 2024, 2025]
TEST_YEAR         = 2026
ALL_YEARS         = TRAIN_YEARS + [TEST_YEAR]
STARTING_BANKROLL = 1000.0
KELLY_FRACTION    = 0.25
KELLY_CAP         = 0.10

print("Imports complete.")"""))

cells.append(code_cell("""\
# ── Environment detection ─────────────────────────────────────────────────────
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/CollegeBaseballAnalytics/'
else:
    import pathlib
    CACHE_DIR = str(pathlib.Path('data')) + '/'
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Environment : {'Google Colab' if IN_COLAB else 'Local / VS Code'}")
print(f"Cache dir   : {CACHE_DIR}")"""))

cells.append(md_cell("""\
### Loading pre-built data files
Run `pull_ncaa_data.py` locally once to generate the parquet files, then upload to Drive.
The script pulls from sportsdataverse (2021-2023) and ESPN scoreboard API (2024-2025) —
no stats.ncaa.org required."""))

cells.append(code_cell("""\
GAME_PATH = CACHE_DIR + 'game_results_2021_2026.parquet'
STAT_PATH = CACHE_DIR + 'team_season_stats_2021_2026.parquet'

# ── Inline data pull (runs when parquet files are missing) ────────────────────
def _pull_sdv(years=(2021,2022,2023)):
    SDV = "https://raw.githubusercontent.com/sportsdataverse/baseballr-data/main/ncaa/schedules/csv"
    frames = []
    for yr in years:
        url = f"{SDV}/ncaa_baseball_schedule_{yr}.csv"
        print(f"  SDV {yr} ...", end="", flush=True)
        try:
            df = pd.read_csv(url)
            df = df[(df["home_team_division"]==1) & (df["away_team_division"]==1)].copy()
            df["season"]   = yr
            df["neutral"]  = df.get("neutral_site", False).fillna(False)
            df["date"]     = pd.to_datetime(df["date"], errors="coerce")
            df = df.rename(columns={"home_team_score":"home_score","away_team_score":"away_score"})
            df = df[["season","date","home_team","away_team","home_score","away_score","neutral"]]
            df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
            df = df.dropna(subset=["home_score","away_score"])
            frames.append(df); print(f" {len(df)} games")
        except Exception as e: print(f" FAILED: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _pull_espn(years=(2024,2025,2026)):
    from datetime import date, timedelta
    BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"
    HDR  = {"User-Agent": "Mozilla/5.0"}
    frames = []
    for yr in years:
        rows, start, end = [], date(yr,2,14), date(yr,6,28)
        d, dates = start, []
        while d <= end: dates.append(d); d += timedelta(days=1)
        print(f"  ESPN {yr}: scanning {len(dates)} dates", end="", flush=True)
        for i, d in enumerate(dates):
            try:
                r = requests.get(BASE, params={"dates":d.strftime("%Y%m%d"),"limit":200},
                                 headers=HDR, timeout=10)
                if r.status_code != 200: continue
                for ev in r.json().get("events", []):
                    comp = ev["competitions"][0]
                    if comp["status"]["type"]["name"] != "STATUS_FINAL": continue
                    ts   = comp["competitors"]
                    home = next((t for t in ts if t.get("homeAway")=="home"), ts[0])
                    away = next((t for t in ts if t.get("homeAway")=="away"), ts[1])
                    rows.append({"season":yr,"date":pd.to_datetime(ev["date"][:10]),
                                 "home_team":home["team"]["displayName"],
                                 "away_team":away["team"]["displayName"],
                                 "home_score":float(home.get("score","nan")),
                                 "away_score":float(away.get("score","nan")),
                                 "neutral":comp.get("neutralSite",False)})
            except Exception: pass
            if (i+1) % 30 == 0: print(".", end="", flush=True)
            time.sleep(0.05)
        df = pd.DataFrame(rows).dropna(subset=["home_score","away_score"])
        frames.append(df); print(f" >> {len(df)} games")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _compute_stats(games):
    home = games[["season","date","home_team","home_score","away_score","neutral"]].copy()
    home.columns = ["season","date","team","rs","ra","neutral"]; home["is_home"]=1
    away = games[["season","date","away_team","away_score","home_score","neutral"]].copy()
    away.columns = ["season","date","team","rs","ra","neutral"]; away["is_home"]=0
    long = pd.concat([home,away], ignore_index=True)
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
    return agg

if not os.path.exists(GAME_PATH) or not os.path.exists(STAT_PATH):
    print("Parquet files not found — pulling data now (takes ~3 min in Colab) ...")
    sdv   = _pull_sdv([2021,2022,2023])
    espn  = _pull_espn([2024,2025,2026])
    games = pd.concat([sdv, espn], ignore_index=True)
    games["neutral"] = games["neutral"].astype(bool)
    team_stats = _compute_stats(games)
    games.to_parquet(GAME_PATH, index=False)
    team_stats.to_parquet(STAT_PATH, index=False)
    print(f"Saved to {CACHE_DIR}")
else:
    games      = pd.read_parquet(GAME_PATH)
    team_stats = pd.read_parquet(STAT_PATH)

games['season']      = games['season'].astype(int)
team_stats['season'] = team_stats['season'].astype(int)

print(f"Games loaded:      {len(games):,} rows  |  seasons: {sorted(games['season'].unique())}")
print(f"Team stats loaded: {len(team_stats):,} rows  |  {team_stats['season'].nunique()} seasons")
team_stats.head(3)"""))

# ── SECTION 1b: FEATURE ENGINEERING ───────────────────────────────────────────
cells.append(md_cell("## Section 1b — Feature Engineering"))

cells.append(code_cell("""\
# Pythagorean win% already computed; add league-normalised versions
def add_normalised_features(df):
    df = df.copy()
    for yr in df['season'].unique():
        mask = df['season'] == yr
        for col in ['avg_runs_scored','avg_runs_allowed','avg_run_diff','pythagorean_win_pct']:
            if col in df.columns:
                mu = df.loc[mask, col].mean()
                sd = max(float(df.loc[mask, col].std()), 1e-6)
                df.loc[mask, col + '_z'] = (df.loc[mask, col] - mu) / sd
    return df

team_stats = add_normalised_features(team_stats)

# Recent-form rolling (last 15 games win%) — computed from game log
def compute_recent_form(games, n=15):
    home = games[['season','date','home_team','home_score','away_score']].copy()
    home.columns = ['season','date','team','rs','ra']
    away = games[['season','date','away_team','away_score','home_score']].copy()
    away.columns = ['season','date','team','rs','ra']
    long = pd.concat([home, away]).sort_values('date')
    long['win'] = (long['rs'] > long['ra']).astype(int)
    long['recent_win_pct'] = (long.groupby(['team','season'])['win']
                                   .transform(lambda x: x.rolling(n, min_periods=3).mean()))
    # Season-end recent form per team
    form = (long.groupby(['team','season'])['recent_win_pct'].last().reset_index())
    return form

recent_form = compute_recent_form(games)
team_stats  = team_stats.merge(recent_form, on=['team','season'], how='left')
print("Features computed.")
print(team_stats.columns.tolist())
team_stats[['team','season','avg_runs_scored','avg_runs_allowed','pythagorean_win_pct','recent_win_pct']].head(5)"""))

# ── BUILD GAME MATRIX ──────────────────────────────────────────────────────────
cells.append(md_cell("### Build Game-Level Training Matrix"))

cells.append(code_cell("""\
FEAT_COLS = ['avg_runs_scored','avg_runs_allowed','avg_run_diff',
             'pythagorean_win_pct','win_pct','recent_win_pct',
             'avg_runs_scored_z','avg_runs_allowed_z','pythagorean_win_pct_z']
FEAT_COLS = [c for c in FEAT_COLS if c in team_stats.columns]

def build_game_matrix(games, stats):
    idx = stats.set_index(['team','season'])
    rows = []
    for _, g in games.iterrows():
        yr = int(g['season'])
        ht, at = g['home_team'], g['away_team']
        def feat(t):
            key = (t, yr)
            return idx.loc[key, FEAT_COLS] if key in idx.index else pd.Series(np.nan, index=FEAT_COLS)
        hf, af = feat(ht), feat(at)
        row = {
            'season':   yr,
            'home':     ht,
            'away':     at,
            'is_home':  1,
            'win':      int(g['home_score'] > g['away_score']),
            'run_diff': float(g['home_score'] - g['away_score']),
            'neutral':  int(g.get('neutral', False)),
        }
        for c in FEAT_COLS:
            row[f'h_{c}'] = hf.get(c, np.nan)
            row[f'a_{c}'] = af.get(c, np.nan)
            row[f'd_{c}'] = hf.get(c, np.nan) - af.get(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

print("Building game matrix (may take ~30s) …")
game_matrix = build_game_matrix(games, team_stats)
game_matrix.dropna(subset=['win','run_diff'], inplace=True)
print(f"Game matrix: {game_matrix.shape}")
game_matrix.head(3)"""))

# ── SECTION 2: ELO RANKINGS ───────────────────────────────────────────────────
cells.append(md_cell("## Section 2 — Power Rankings (Elo + Composite)"))

cells.append(code_cell("""\
ELO_K    = 30
ELO_HOME = 35
ELO_INIT = 1500

def expected_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

def margin_mult(rd):
    return np.log(abs(rd) + 1) * 2.0

def compute_elo(games):
    elo = {}
    games_s = games.sort_values('date').dropna(subset=['date'])
    for _, g in games_s.iterrows():
        ht, at = g['home_team'], g['away_team']
        eh = elo.get(ht, ELO_INIT) + (ELO_HOME if not g.get('neutral', False) else 0)
        ea = elo.get(at, ELO_INIT)
        exp = expected_win(eh, ea)
        actual = 1 if g['home_score'] > g['away_score'] else 0
        mult = margin_mult(g['home_score'] - g['away_score'])
        delta = ELO_K * mult * (actual - exp)
        elo[ht] = elo.get(ht, ELO_INIT) + delta
        elo[at] = elo.get(at, ELO_INIT) - delta
    return pd.DataFrame({'team': list(elo.keys()), 'elo': list(elo.values())})

elo_df = compute_elo(games)
print(f"Elo computed for {len(elo_df)} teams")"""))

cells.append(code_cell("""\
def build_rankings(stats, elo, year=TEST_YEAR):
    df = stats[stats['season'] == year].copy()
    df = df.merge(elo, on='team', how='left')
    df['elo'] = df['elo'].fillna(ELO_INIT)

    def z(s): return (s - s.mean()) / max(s.std(), 1e-6)

    score = pd.Series(0.0, index=df.index)
    for col, wt, higher in [
        ('pythagorean_win_pct', 0.30, True),
        ('avg_run_diff',        0.25, True),
        ('elo',                 0.25, True),
        ('avg_runs_scored',     0.10, True),
        ('avg_runs_allowed',    0.10, False),
    ]:
        if col in df.columns:
            score += wt * (z(df[col]) if higher else -z(df[col]))

    df['power_score'] = score
    df['rank'] = df['power_score'].rank(ascending=False).astype(int)
    return df.sort_values('rank')

rankings = build_rankings(team_stats, elo_df, year=TEST_YEAR)
show = ['rank','team','elo','pythagorean_win_pct','avg_run_diff','avg_runs_scored','avg_runs_allowed','power_score']
show = [c for c in show if c in rankings.columns]
print(f"=== {TEST_YEAR} College Baseball Power Rankings (Top 25) ===")
rankings[show].head(25).style.format({
    'elo':'{:.0f}','pythagorean_win_pct':'{:.3f}','avg_run_diff':'{:+.2f}',
    'avg_runs_scored':'{:.2f}','avg_runs_allowed':'{:.2f}','power_score':'{:.3f}'
})"""))

cells.append(code_cell("""\
# Print exact team name strings to copy into predict_game() below
print("Exact team names for Top 25 (copy/paste into predict_game):")
for _, r in rankings.head(25).iterrows():
    print(f"  #{r['rank']:2d}  {r['team']}")"""))

cells.append(code_cell("""\
top25 = rankings.head(25)
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top25)))[::-1]
ax.barh(top25['team'][::-1], top25['power_score'][::-1], color=colors[::-1])
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Composite Power Score')
ax.set_title(f'{TEST_YEAR} College Baseball Power Rankings — Top 25', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()"""))

# ── SECTION 3: RUN DIFF MODEL ──────────────────────────────────────────────────
cells.append(md_cell("## Section 3 — Run Differential Prediction (XGBoost)"))

cells.append(code_cell("""\
DIFF_FEATS = [c for c in game_matrix.columns if c.startswith('d_')] + ['is_home','neutral']
DIFF_FEATS = [c for c in DIFF_FEATS if c in game_matrix.columns]

def split_train_test(matrix, target, feats):
    df = matrix.dropna(subset=[target]+feats).copy()
    tr = df[df['season'].isin(TRAIN_YEARS)]
    te = df[df['season'] == TEST_YEAR]
    return tr[feats].values, tr[target].values, te[feats].values, te[target].values, tr, te

X_tr, y_tr, X_te, y_te, tr_df, te_df = split_train_test(game_matrix, 'run_diff', DIFF_FEATS)
print(f"Train: {len(X_tr):,} games  |  Test: {len(X_te):,} games  |  Features: {len(DIFF_FEATS)}")

reg = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_alpha=0.5, random_state=42, n_jobs=-1)
reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

from sklearn.metrics import mean_absolute_error, r2_score
preds_reg = reg.predict(X_te)
print(f"MAE: {mean_absolute_error(y_te, preds_reg):.2f} runs  |  R2: {r2_score(y_te, preds_reg):.3f}")

fi = pd.Series(reg.feature_importances_, index=DIFF_FEATS).sort_values(ascending=False)
fi.head(12).plot(kind='bar', figsize=(10,4), color='steelblue', title='Feature Importances (Run Diff)')
plt.tight_layout(); plt.show()"""))

# ── SECTION 4a: WIN PROB XGBOOST ──────────────────────────────────────────────
cells.append(md_cell("## Section 4a — Win Probability (XGBoost Classifier)"))

cells.append(code_cell("""\
X_tr_c, y_tr_c, X_te_c, y_te_c, _, _ = split_train_test(game_matrix, 'win', DIFF_FEATS)

clf_xgb = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              eval_metric='logloss', random_state=42, n_jobs=-1)
clf_xgb.fit(X_tr_c, y_tr_c, eval_set=[(X_te_c, y_te_c)], verbose=False)

proba_xgb = clf_xgb.predict_proba(X_te_c)[:, 1]
acc_xgb   = accuracy_score(y_te_c, (proba_xgb > 0.5).astype(int))
auc_xgb   = roc_auc_score(y_te_c, proba_xgb)
brier_xgb = brier_score_loss(y_te_c, proba_xgb)
print(f"XGBoost  |  Acc: {acc_xgb:.3f}  AUC: {auc_xgb:.3f}  Brier: {brier_xgb:.4f}")"""))

# ── SECTION 4b: FASTAI ────────────────────────────────────────────────────────
cells.append(md_cell("## Section 4b — Win Probability (FastAI Tabular)"))

cells.append(code_cell("""\
try:
    from fastai.tabular.all import *

    feat_df = game_matrix[DIFF_FEATS + ['win','season']].dropna().copy().reset_index(drop=True)
    feat_df['win'] = feat_df['win'].astype(str)
    valid_idx = feat_df[feat_df['season'] == TEST_YEAR].index.tolist()
    if len(valid_idx) == 0:
        valid_idx = feat_df.index[-max(1, len(feat_df)//10):].tolist()

    to = TabularDataLoaders.from_df(
        feat_df, path='.', y_names='win', y_block=CategoryBlock(),
        cat_names=[], cont_names=DIFF_FEATS,
        procs=[FillMissing, Normalize], valid_idx=valid_idx, bs=256)

    learn = tabular_learner(to, layers=[200, 100], metrics=accuracy)
    learn.fit_one_cycle(10, 1e-3)

    preds_fa, targs_fa = learn.get_preds(dl=to.valid)
    proba_fastai = preds_fa[:, 1].numpy()
    y_fa_valid   = (np.array(targs_fa) == 1).astype(int)
    acc_fa   = accuracy_score(y_fa_valid, (proba_fastai > 0.5).astype(int))
    auc_fa   = roc_auc_score(y_fa_valid, proba_fastai)
    brier_fa = brier_score_loss(y_fa_valid, proba_fastai)
    _fastai_ok = True
    print(f"FastAI  |  Acc: {acc_fa:.3f}  AUC: {auc_fa:.3f}  Brier: {brier_fa:.4f}")
except Exception as e:
    print(f"FastAI skipped: {e}")
    proba_fastai = proba_xgb
    y_fa_valid   = y_te_c
    _fastai_ok   = False"""))

# ── SECTION 4c: MODEL COMPARISON ─────────────────────────────────────────────
cells.append(md_cell("## Section 4c — Model Comparison"))

cells.append(code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for proba, y_true, label, color in [
    (proba_xgb,   y_te_c,     'XGBoost', 'royalblue'),
    (proba_fastai, y_fa_valid, 'FastAI',  'darkorange'),
]:
    fp, mp = calibration_curve(y_true, proba, n_bins=10, strategy='uniform')
    axes[0].plot(mp, fp, marker='o', label=label, color=color)
axes[0].plot([0,1],[0,1],'k--', label='Perfect')
axes[0].set_title('Calibration Curves'); axes[0].legend()

metrics = pd.DataFrame({
    'Model': ['XGBoost','FastAI'],
    'Acc':   [acc_xgb,  acc_fa  if _fastai_ok else acc_xgb],
    'AUC':   [auc_xgb,  auc_fa  if _fastai_ok else auc_xgb],
}).set_index('Model')
metrics.plot(kind='bar', ax=axes[1], colormap='Set2')
axes[1].set_title('Model Metrics'); axes[1].set_ylim(0.4, 1.0)
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()"""))

# ── SECTION 5: ENSEMBLE ───────────────────────────────────────────────────────
cells.append(md_cell("## Section 5 — Stacked Ensemble with Isotonic Calibration"))

cells.append(code_cell("""\
from sklearn.isotonic import IsotonicRegression

# Align FastAI predictions to XGBoost test length before stacking
_fa_for_stack = proba_fastai if len(proba_fastai) == len(proba_xgb) else proba_xgb
meta_X = np.column_stack([proba_xgb, _fa_for_stack, preds_reg])
meta_y = y_te_c

meta_lr = LogisticRegression(C=1.0, max_iter=500)
meta_lr.fit(meta_X, meta_y)
proba_ensemble = meta_lr.predict_proba(meta_X)[:, 1]

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(proba_ensemble, meta_y)
proba_calibrated = iso.predict(proba_ensemble)

acc_ens   = accuracy_score(meta_y, (proba_calibrated > 0.5).astype(int))
auc_ens   = roc_auc_score(meta_y, proba_calibrated)
brier_ens = brier_score_loss(meta_y, proba_calibrated)
print(f"Ensemble  |  Acc: {acc_ens:.3f}  AUC: {auc_ens:.3f}  Brier: {brier_ens:.4f}")
print(f"Weights   |  XGB: {meta_lr.coef_[0][0]:.3f}  FA: {meta_lr.coef_[0][1]:.3f}  RD: {meta_lr.coef_[0][2]:.3f}")"""))

# ── SECTION 6: GAME PREDICTOR ─────────────────────────────────────────────────
cells.append(md_cell("""\
## Section 6 — Game Predictor Tool

### How to use

**Step 1 — Find the exact team name**
Run the rankings cell above. It prints a numbered list like:
```
 #1  Tennessee Volunteers
 #2  LSU Tigers
```
Copy the name exactly as shown — spacing and capitalization matter.

**Step 2 — Call `predict_game()`**
```python
predict_game("Tennessee Volunteers", "LSU Tigers", is_home_a=True)
```
| Argument | Description |
|----------|-------------|
| `team_a` | First team (home by default) |
| `team_b` | Second team (away by default) |
| `is_home_a` | `True` if team_a is at home, `False` for neutral site or away |
| `year` | Season to pull stats from (default: 2026) |

**Step 3 — Read the output**
- **Ensemble win prob** — blended XGBoost + FastAI + run-diff prediction (most reliable)
- **XGBoost win prob** — standalone classifier
- **Elo win prob** — based purely on head-to-head history
- **Predicted run diff** — positive = team_a wins by that margin

**Step 4 — Predict multiple games at once**
```python
predict_games_table([
    ("Tennessee Volunteers", "LSU Tigers",       True),
    ("Florida Gators",       "Arkansas Razorbacks", False),
])
```
Returns a formatted table with win probabilities and predicted margins for all matchups."""))

cells.append(code_cell("""\
def predict_game(team_a, team_b, is_home_a=True, year=TEST_YEAR, verbose=True):
    s_idx  = team_stats[team_stats['season'] == year].set_index('team')
    elo_lk = elo_df.set_index('team')['elo']

    def get(t): return s_idx.loc[t] if t in s_idx.index else pd.Series(dtype=float)
    sa, sb = get(team_a), get(team_b)

    row = {'is_home': int(is_home_a), 'neutral': 0}
    for c in FEAT_COLS:
        row[f'd_{c}'] = sa.get(c, np.nan) - sb.get(c, np.nan)

    X = np.array([[row.get(f, 0) for f in DIFF_FEATS]])
    rd_pred = float(reg.predict(X)[0])
    wp_xgb  = float(clf_xgb.predict_proba(X)[0, 1])
    try:
        meta_in = np.array([[wp_xgb, wp_xgb, rd_pred]])
        wp_ens  = float(iso.predict(meta_lr.predict_proba(meta_in)[:, 1])[0])
    except Exception:
        wp_ens = wp_xgb

    ea = elo_lk.get(team_a, ELO_INIT) + (ELO_HOME if is_home_a else 0)
    eb = elo_lk.get(team_b, ELO_INIT)
    wp_elo = expected_win(ea, eb)

    if verbose:
        side = 'Home' if is_home_a else 'Away/Neutral'
        fav  = team_a if wp_ens > 0.5 else team_b
        print(f"\\n{'='*52}")
        print(f"  {team_a} ({side}) vs {team_b}")
        print(f"{'='*52}")
        print(f"  Ensemble win prob : {wp_ens:.1%}  [favors {fav}]")
        print(f"  XGBoost win prob  : {wp_xgb:.1%}")
        print(f"  Elo win prob      : {wp_elo:.1%}")
        print(f"  Predicted run diff: {rd_pred:+.1f}")
        print(f"  Elo ratings       : {elo_lk.get(team_a,ELO_INIT):.0f} vs {elo_lk.get(team_b,ELO_INIT):.0f}")
    return {'team_a':team_a,'team_b':team_b,'win_prob':wp_ens,'run_diff':rd_pred,'elo_wp':wp_elo}

# Edit team names to match exactly what appears in the Top 25 rankings above
sample = [('Tennessee Volunteers','Arkansas Razorbacks',True),
          ('LSU Tigers','Florida Gators',False),
          ('Texas Longhorns','Oklahoma State Cowboys',True)]
for a, b, home in sample:
    try: predict_game(a, b, is_home_a=home)
    except Exception as e: print(f"{a} vs {b}: {e}")"""))

cells.append(code_cell("""\
def predict_games_table(matchups, year=TEST_YEAR):
    rows = []
    for a, b, home in matchups:
        try:
            r = predict_game(a, b, is_home_a=home, year=year, verbose=False)
            rows.append({'Home/Neutral': a, 'Away': b,
                         'Win Prob (A)': f"{r['win_prob']:.1%}",
                         'Pred Margin': f"{r['run_diff']:+.1f}",
                         'Elo WP': f"{r['elo_wp']:.1%}"})
        except Exception as e:
            rows.append({'Home/Neutral': a, 'Away': b, 'Error': str(e)})
    return pd.DataFrame(rows)

# NCAA Tournament bracket — edit with actual matchups from the rankings above
upcoming = [('Tennessee Volunteers','Arkansas Razorbacks',True),
            ('LSU Tigers','Florida Gators',False)]
predict_games_table(upcoming)"""))

# ── SECTION 7: BETTING ────────────────────────────────────────────────────────
cells.append(md_cell("## Section 7 — Betting Lines (Odds API)"))

cells.append(code_cell("""\
if IN_COLAB:
    try:
        from google.colab import userdata
        ODDS_API_KEY = (userdata.get('ODDS_API_KEY') or '').strip()
    except Exception:
        ODDS_API_KEY = ''
else:
    ODDS_API_KEY = (os.environ.get('ODDS_API_KEY') or '').strip()

ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
SPORT_KEY     = None   # discovered below

def american_to_prob(odds):
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 100/(100+o) if o > 0 else abs(o)/(abs(o)+100)

if not ODDS_API_KEY:
    hint = 'Colab Secrets sidebar (key icon) -> ODDS_API_KEY' if IN_COLAB else '.env file'
    print(f"ODDS_API_KEY not set -- add it to {hint}")
else:
    print(f"Key loaded ({len(ODDS_API_KEY)} chars). Checking available sports ...")
    _r = requests.get(ODDS_API_BASE + '/sports',
                      params=dict(apiKey=ODDS_API_KEY), timeout=10)
    if _r.status_code == 401:
        print("401 Unauthorized -- key is wrong or expired.")
        print("Go to https://the-odds-api.com/account/ and copy your key fresh.")
    elif _r.status_code != 200:
        print(f"Sports list error {_r.status_code}: {_r.text[:300]}")
    else:
        _sports  = _r.json()
        _baseball = [s for s in _sports
                     if 'baseball' in s.get('key','').lower()
                     or 'baseball' in s.get('title','').lower()]
        print("Baseball sports on your plan:")
        for s in _baseball:
            print(f"  {s['key']:40s} {s['title']}")
        for _cand in ['baseball_ncaa','baseball_college','baseball_ncaab']:
            if any(s['key'] == _cand for s in _baseball):
                SPORT_KEY = _cand; break
        if not SPORT_KEY and _baseball:
            SPORT_KEY = _baseball[0]['key']
        print(f"\\nUsing sport key : {SPORT_KEY}")
        print(f"Requests remaining: {_r.headers.get('x-requests-remaining','unknown')}")"""))

cells.append(code_cell("""\
def fetch_live_odds():
    if not ODDS_API_KEY or not SPORT_KEY:
        print("Key or sport key missing -- run the cell above first."); return pd.DataFrame()
    _r = requests.get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
                      params=dict(apiKey=ODDS_API_KEY, regions='us',
                                  markets='h2h,spreads', oddsFormat='american'),
                      timeout=10)
    if _r.status_code != 200:
        print(f"Odds fetch error {_r.status_code}: {_r.text[:300]}"); return pd.DataFrame()
    rows = []
    for game in _r.json():
        home, away = game['home_team'], game['away_team']
        for bkm in game.get('bookmakers', [])[:1]:
            for mkt in bkm.get('markets', []):
                if mkt['key'] == 'h2h':
                    out = {o['name']: o['price'] for o in mkt['outcomes']}
                    rows.append({'home': home, 'away': away,
                                 'ml_home': out.get(home), 'ml_away': out.get(away)})
    print(f"Requests remaining: {_r.headers.get('x-requests-remaining','unknown')}")
    return pd.DataFrame(rows)

live_odds = fetch_live_odds()
live_odds"""))

# ── SECTION 8: BACKTESTING ────────────────────────────────────────────────────
cells.append(md_cell("## Section 8 — Backtesting"))

cells.append(code_cell("""\
def backtest(matrix, model_proba, year=None, edge_threshold=0.03):
    df = matrix.copy()
    if year: df = df[df['season'] == year]
    df = df.dropna(subset=['win'] + DIFF_FEATS).copy()
    if model_proba is not None and len(model_proba) == len(df):
        df['model_wp'] = model_proba
    if 'model_wp' not in df.columns:
        print("model_wp not set — run ensemble section first"); return
    df['implied_prob'] = 0.524  # -110 juice
    df['edge'] = df['model_wp'] - df['implied_prob']
    bets = df[df['edge'] > edge_threshold].copy()
    if len(bets) == 0:
        print(f"No bets at edge > {edge_threshold:.0%}"); return
    bets['payout'] = bets['win'].apply(lambda w: 100/110 if w == 1 else -1)
    pnl    = bets['payout'].sum()
    roi    = pnl / len(bets)
    record = f"{int(bets['win'].sum())}-{int((1-bets['win']).sum())}"
    print(f"Bets: {len(bets)}  Record: {record}  P&L: {pnl:+.2f}u  ROI: {roi:+.1%}")
    return bets

test_rows = game_matrix[game_matrix['season'] == TEST_YEAR].dropna(subset=['win']+DIFF_FEATS).copy()
if len(test_rows) == len(proba_calibrated):
    test_rows['model_wp'] = proba_calibrated
    backtest(test_rows, None, edge_threshold=0.03)
else:
    print(f"Length mismatch: {len(test_rows)} rows vs {len(proba_calibrated)} proba")"""))

# ── SECTION 9: KELLY ──────────────────────────────────────────────────────────
cells.append(md_cell("## Section 9 — Kelly Criterion Bankroll Simulation"))

cells.append(code_cell("""\
def kelly_fraction(wp, odds=-110, frac=KELLY_FRACTION, cap=KELLY_CAP):
    imp = american_to_prob(odds)
    if wp <= imp: return 0.0
    b = (100/abs(odds)) if odds < 0 else (odds/100)
    return min(max(frac * (wp*(b+1)-1)/b, 0), cap)

def simulate_bankroll(bets_df, br=STARTING_BANKROLL, frac=KELLY_FRACTION, cap=KELLY_CAP, odds=-110):
    history = [br]
    for _, row in bets_df.iterrows():
        f = kelly_fraction(row.get('model_wp', 0.5), odds=odds, frac=frac, cap=cap)
        b = (100/abs(odds)) if odds < 0 else (odds/100)
        br += br*f*b if row.get('win',0) == 1 else -br*f
        history.append(br)
    return br, (br - STARTING_BANKROLL)/STARTING_BANKROLL, history

if 'model_wp' in test_rows.columns:
    bets = test_rows[test_rows['model_wp'] > 0.55].copy()
    final, roi, hist = simulate_bankroll(bets)
    print(f"Bets: {len(bets)}  Final: ${final:,.2f}  ROI: {roi:+.1%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist, color='royalblue', linewidth=1.5)
    ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--', linewidth=0.8)
    ax.fill_between(range(len(hist)), STARTING_BANKROLL, hist, alpha=0.15, color='royalblue')
    ax.set_title(f'{TEST_YEAR} Kelly Bankroll Simulation (Quarter Kelly)')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout(); plt.show()"""))

cells.append(code_cell("""\
# Scenario explorer: sweep edge threshold x Kelly fraction
def scenario_explorer(df, thresholds=None, fractions=None):
    if thresholds is None: thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
    if fractions  is None: fractions  = [0.10, 0.25, 0.50]
    rows = []
    df['edge'] = df.get('model_wp', 0.5) - 0.524
    for t in thresholds:
        sub = df[df['edge'] > t]
        for f in fractions:
            if len(sub) == 0:
                rows.append({'Edge':f'{t:.0%}','Kelly':f'{f:.0%}','Bets':0,'ROI':'N/A','Final':'N/A'})
                continue
            _, roi, hist = simulate_bankroll(sub, frac=f)
            rows.append({'Edge':f'{t:.0%}','Kelly':f'{f:.0%}',
                         'Bets':len(sub),'ROI':f'{roi:+.1%}','Final':f'${hist[-1]:,.0f}'})
    return pd.DataFrame(rows)

if 'model_wp' in test_rows.columns:
    scenario_explorer(test_rows)"""))

# ── ASSEMBLE ──────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
