"""
backtest_2025_oot.py — Out-of-time backtest: train 2021-2024, holdout 2025.

What this measures:
  - Model calibration: do 60% predictions actually win 60% of the time?
  - Brier score / AUC vs a naive Elo-only baseline
  - P&L simulation at assumed lines (-110 flat, and -110/-130/-150/-175 bands)
  - Favorite/underdog bias: does the model systematically pick Elo underdogs?
  - Kelly bankroll simulation across full season

Key design choices to avoid leakage:
  - Model trained strictly on 2021-2024 games (no 2025 rows)
  - 2025 game features built from 2024 season-aggregate stats (prior year)
    This mirrors real-world usage: you only know last year's stats at season start.

Run:
    python backtest_2025_oot.py
"""

import sys
import warnings
import pathlib
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from daily_runner import (
    compute_stats, compute_elo, compute_sos, train_model,
    ELO_INIT, ELO_HOME, ERA_ELO_SCALE, KELLY_FRACTION, KELLY_CAP,
    STARTING_BANKROLL, era_adjustment_runner,
)
from pipeline.utils import american_to_prob, kelly_fraction

DATA = pathlib.Path(__file__).parent / "data"

# ----------------------------------------------------------------------------─
# 1. Load raw data
# ----------------------------------------------------------------------------─
print("Loading data…")
games_all = pd.read_parquet(DATA / "game_results_2021_2026.parquet")
games_all["season"] = games_all["season"].astype(int)
games_all["date"]   = pd.to_datetime(games_all["date"])

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR   = 2025

games_train = games_all[games_all["season"].isin(TRAIN_YEARS)].copy()
games_test  = games_all[games_all["season"] == TEST_YEAR].copy()

print(f"  Train: {len(games_train):,} games ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})")
print(f"  Test:  {len(games_test):,} games ({TEST_YEAR})")

# ----------------------------------------------------------------------------─
# 2. Build features + train model on 2021-2024 only
# ----------------------------------------------------------------------------─
print("\nBuilding training stats + Elo…")
train_stats = compute_stats(games_train)
train_elo   = compute_elo(games_train)

print("Training model on 2021-2024…")
clf, feats = train_model(games_train, train_stats)
print(f"  Features: {feats}")

# Elo lookup (end-of-2024 ratings) used for 2025 predictions
elo_lk = train_elo.set_index("team")["elo"]

# 2024 season stats used as the feature source for 2025 game predictions
# (no 2025 data touches the model input)
stats_2024 = train_stats[train_stats["season"] == 2024].set_index("team")

# ----------------------------------------------------------------------------─
# 3. Generate predictions for every 2025 game
# ----------------------------------------------------------------------------─
print("\nGenerating 2025 predictions…")
feat_base = [c.replace("d_", "") for c in feats if c.startswith("d_")]

rows = []
for _, g in games_test.iterrows():
    home, away = g["home_team"], g["away_team"]
    if home not in stats_2024.index or away not in stats_2024.index:
        continue
    hf, af = stats_2024.loc[home], stats_2024.loc[away]
    is_neutral = int(bool(g.get("neutral", False)))
    home_adv   = 0 if is_neutral else ELO_HOME

    # Model win probability
    feat_row = {"is_home": 1 - is_neutral, "neutral": is_neutral}
    for c in feat_base:
        hval = hf[c] if c in hf.index else np.nan
        aval = af[c] if c in af.index else np.nan
        # Fill NaN with 0 (league-average diff) — same as production
        feat_row[f"d_{c}"] = float(0 if (pd.isna(hval) or pd.isna(aval))
                                   else hval - aval)
    X = np.array([[feat_row.get(f, 0) for f in feats]], dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    wp_model = float(clf.predict_proba(X)[0, 1])

    # ERA-adjusted Elo win probability (baseline)
    era_h  = float(-ERA_ELO_SCALE * hf.get("avg_runs_allowed_z", 0))
    era_a  = float(-ERA_ELO_SCALE * af.get("avg_runs_allowed_z", 0))
    elo_h  = elo_lk.get(home, ELO_INIT) + home_adv + era_h
    elo_a  = elo_lk.get(away, ELO_INIT) + era_a
    wp_elo = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400))

    # Blended (same blend used in production)
    wp_blend = 0.6 * wp_model + 0.4 * wp_elo

    home_won = bool(g["home_score"] > g["away_score"])
    rows.append({
        "date":       g["date"],
        "home":       home,
        "away":       away,
        "neutral":    bool(is_neutral),
        "home_won":   home_won,
        "wp_model":   wp_model,
        "wp_elo":     wp_elo,
        "wp_blend":   wp_blend,
        # Elo favorite flag: home team is Elo favourite?
        "elo_favors_home": elo_lk.get(home, ELO_INIT) + home_adv > elo_lk.get(away, ELO_INIT),
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
print(f"  {len(df):,} games with full feature coverage "
      f"({len(games_test) - len(df)} skipped — missing 2024 stats)")

# ----------------------------------------------------------------------------─
# 4. Calibration metrics
# ----------------------------------------------------------------------------─
try:
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

y_true = df["home_won"].astype(int).values

brier_blend = np.mean((df["wp_blend"] - y_true) ** 2)
brier_elo   = np.mean((df["wp_elo"]   - y_true) ** 2)
brier_naive = np.mean((np.full(len(y_true), 0.5) - y_true) ** 2)

if HAS_SKLEARN:
    auc_blend = roc_auc_score(y_true, df["wp_blend"])
    auc_elo   = roc_auc_score(y_true, df["wp_elo"])
    ll_blend  = log_loss(y_true, df[["wp_blend"]].assign(p0=lambda x: 1 - x["wp_blend"])[["p0","wp_blend"]].values)
    ll_elo    = log_loss(y_true, df[["wp_elo"]].assign(p0=lambda x: 1 - x["wp_elo"])[["p0","wp_elo"]].values)

acc_blend = (((df["wp_blend"] > 0.5) == df["home_won"])).mean()
acc_elo   = (((df["wp_elo"]   > 0.5) == df["home_won"])).mean()

# ----------------------------------------------------------------------------─
# 5. Calibration curve: predicted bucket vs actual win rate
# ----------------------------------------------------------------------------─
bins = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
labels = ["45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"]

df["bucket"] = pd.cut(df["wp_blend"], bins=bins, labels=labels, right=False)

calib = (df.groupby("bucket", observed=True)
           .agg(n=("home_won", "count"),
                actual_win=("home_won", "mean"),
                avg_pred=("wp_blend", "mean"))
           .reset_index())

# ----------------------------------------------------------------------------─
# 6. P&L simulation across confidence bands + line assumptions
# ----------------------------------------------------------------------------─
# Build a "bets" frame: for each game, consider both home and away sides
# where our blended wp > implied odds probability at that line.
LINE_SCENARIOS = {
    "-110 (sim)":  -110,   # most generous baseline (no vig on favourites)
    "-120":        -120,
    "-130":        -130,
    "-150":        -150,   # typical shaded favourite closing price
}
FLAT_BET  = 20.0          # $20/game flat — interpretable, no compounding distortion
EDGE_MIN  = 0.03
CONF_MIN  = 0.53

# NOTE ON LINES: No 2025 closing lines are stored, so all four scenarios
# are simulated. -110 is the most optimistic; real closing lines on
# 60-65% confidence games are typically -130 to -175. The spread across
# columns shows how sensitive the edge is to line quality.

def build_bets(df, line, edge_min=EDGE_MIN, conf_min=CONF_MIN):
    implied = american_to_prob(line)
    b_dec   = abs(line) / 100 if line < 0 else line / 100
    records = []
    for _, r in df.iterrows():
        for side, wp, won in [
            ("home", r["wp_blend"],     r["home_won"]),
            ("away", 1-r["wp_blend"],   not r["home_won"]),
        ]:
            if wp < conf_min:
                continue
            edge = wp - implied
            if edge < edge_min:
                continue
            ev = wp * b_dec - (1 - wp)
            records.append({
                "date":    r["date"],
                "wp":      wp,
                "won":     won,
                "edge":    edge,
                "ev":      ev,
                "b_dec":   b_dec,
                "elo_fav": (r["elo_favors_home"] if side == "home"
                            else not r["elo_favors_home"]),
            })
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def simulate_flat(bets_df, flat=FLAT_BET):
    pnl = bets_df.apply(
        lambda b: flat * b["b_dec"] if b["won"] else -flat, axis=1)
    bets_df = bets_df.copy()
    bets_df["flat_pnl"]   = pnl
    bets_df["flat_cumul"] = STARTING_BANKROLL + pnl.cumsum()
    return bets_df


pnl_results = {}
for label, line in LINE_SCENARIOS.items():
    bets = build_bets(df, line)
    if bets.empty:
        pnl_results[label] = None
        continue
    bets = simulate_flat(bets)
    w    = int(bets["won"].sum())
    n    = len(bets)
    net  = bets["flat_pnl"].sum()
    cumul = bets["flat_cumul"]
    pnl_results[label] = {
        "bets": n, "wins": w, "win_pct": w / n,
        "net":      net,
        "roi_flat": net / (n * FLAT_BET),   # ROI per dollar wagered
        "avg_edge": bets["edge"].mean(),
        "avg_ev":   bets["ev"].mean(),
        "max_dd":   (cumul - cumul.cummax()).min(),
        "end_br":   cumul.iloc[-1],
        "bets_df":  bets,
    }

# placeholders — bias + monthly computed inline during printing

# ----------------------------------------------------------------------------─
# 9. Print report
# ----------------------------------------------------------------------------─
SEP = "=" * 65

print(f"\n{SEP}")
print("  OUT-OF-TIME BACKTEST: 2025 HOLDOUT")
print(f"  Train: {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}   |   Test: {TEST_YEAR}")
print(SEP)
print(f"\n  Games evaluated:  {len(df):,}")

print(f"\n  -- CALIBRATION ------------------------------------------")
print(f"  {'Metric':<28} {'Blend':>10} {'Elo-only':>10} {'Naive':>10}")
print(f"  {'-'*58}")
print(f"  {'Brier Score (lower=better)':<28} {brier_blend:>10.4f} {brier_elo:>10.4f} {brier_naive:>10.4f}")
print(f"  {'Accuracy (>50% threshold)':<28} {acc_blend:>10.1%} {acc_elo:>10.1%} {'50.0%':>10}")
if HAS_SKLEARN:
    print(f"  {'ROC-AUC':<28} {auc_blend:>10.4f} {auc_elo:>10.4f} {'0.5000':>10}")

print(f"\n  -- RELIABILITY (blended model) --------------------------")
print(f"  {'Bucket':<10} {'Count':>7} {'Pred%':>7} {'Actual%':>8} {'Gap':>7}")
print(f"  {'-'*44}")
for _, row in calib.iterrows():
    if row["n"] == 0:
        continue
    gap = row["actual_win"] - row["avg_pred"]
    marker = "  ◄ underestimates" if gap >  0.04 else \
             "  ◄ overestimates"  if gap < -0.04 else ""
    print(f"  {str(row['bucket']):<10} {int(row['n']):>7,} "
          f"{row['avg_pred']:>7.1%} {row['actual_win']:>8.1%} "
          f"{gap:>+7.1%}{marker}")

print(f"\n  -- P&L SIMULATION (flat ${FLAT_BET:.0f}/game, edge>{EDGE_MIN:.0%}, wp>{CONF_MIN:.0%}) -------")
print(f"  * Lines are simulated — no actual 2025 closing lines stored *")
print(f"  {'Line':<13} {'Bets':>6} {'W-L':>9} {'Win%':>6} {'Net P&L':>10} {'ROI/bet':>8} {'AvgEdge':>8} {'MaxDD':>9}")
print(f"  {'-'*73}")
for label, res in pnl_results.items():
    if res is None:
        print(f"  {label:<13}  -- no qualifying bets --")
        continue
    wl = f"{res['wins']}-{res['bets']-res['wins']}"
    print(f"  {label:<13} {res['bets']:>6} {wl:>9} {res['win_pct']:>6.1%} "
          f"${res['net']:>+9.2f} {res['roi_flat']:>+7.1%} "
          f"{res['avg_edge']:>+7.1%} ${res['max_dd']:>+8.2f}")

bets_110_key = "-110 (sim)"
bets_110 = pnl_results.get(bets_110_key, {})
if bets_110 and bets_110.get("bets_df") is not None:
    bdf = bets_110["bets_df"]
    fav_bets = bdf[bdf["elo_fav"] == True]
    dog_bets = bdf[bdf["elo_fav"] == False]
    fav_win  = fav_bets["won"].mean() if len(fav_bets) else float("nan")
    dog_win  = dog_bets["won"].mean() if len(dog_bets) else float("nan")
    fav_pnl  = fav_bets["flat_pnl"].sum() if len(fav_bets) else 0
    dog_pnl  = dog_bets["flat_pnl"].sum() if len(dog_bets) else 0
else:
    fav_win = dog_win = float("nan")
    fav_pnl = dog_pnl = 0
    fav_bets = dog_bets = pd.DataFrame()

print(f"\n  -- FAVORITE/UNDERDOG BIAS (-110 bets) ------------------")
print(f"  {'Side':<20} {'Bets':>6} {'Win%':>7} {'Net P&L':>10}")
print(f"  {'-'*46}")
print(f"  {'Elo favourite':<20} {len(fav_bets):>6} {fav_win:>7.1%} ${fav_pnl:>+9.2f}")
print(f"  {'Elo underdog':<20} {len(dog_bets):>6} {dog_win:>7.1%} ${dog_pnl:>+9.2f}")
if len(fav_bets) > 0 and len(dog_bets) > 0:
    dog_share = len(dog_bets) / (len(fav_bets) + len(dog_bets))
    print(f"\n  Model backed the Elo underdog {dog_share:.0%} of the time.")
    if dog_share > 0.40:
        verdict = ("dogs outperformed — possible genuine edge vs market."
                   if dog_win > fav_win else
                   "dogs underperformed — possible model bias toward underdogs.")
        print(f"  Dog win% {dog_win:.1%} vs fav win% {fav_win:.1%}: {verdict}")

if bets_110 and bets_110.get("bets_df") is not None:
    bdf = bets_110["bets_df"].copy()
    bdf["month"] = bdf["date"].dt.to_period("M")
    monthly = (bdf.groupby("month")
                  .agg(n=("won","count"), wins=("won","sum"), pnl=("flat_pnl","sum"))
                  .reset_index())
    monthly["win_pct"] = monthly["wins"] / monthly["n"]
    print(f"\n  -- MONTHLY BREAKDOWN (-110 sim) -------------------------")
    print(f"  {'Month':<9} {'Bets':>5} {'W-L':>7} {'Win%':>6} {'Flat Net':>10}")
    print(f"  {'-'*42}")
    for _, r in monthly.iterrows():
        wl = f"{int(r['wins'])}-{int(r['n']-r['wins'])}"
        print(f"  {str(r['month']):<9} {int(r['n']):>5} {wl:>7} {r['win_pct']:>6.1%} ${r['pnl']:>+9.2f}")

print(f"\n{SEP}")
print("  INTERPRETATION NOTES")
print(SEP)
print("""
  Brier Score: 0.25 = coin flip, lower is better. A well-calibrated
  model on balanced outcomes typically lands 0.21-0.23.

  Reliability table: 'Gap' is actual − predicted. Positive means the
  model is conservative (actual wins more than predicted). Negative
  means overconfident. Gaps > ±4pp in high-volume buckets are meaningful.

  P&L: No actual 2025 odds are stored, so all lines are simulated.
  The −110 scenario is the most generous; −130/−150 show what happens
  when books shade favourites, which is the realistic market.

  Favorite bias: If the model bets the Elo underdog >60% of the time
  AND those bets lose at a higher rate, it is a calibration problem.
  If they win at a similar or higher rate, it may be a genuine edge.
""")

# ----------------------------------------------------------------------------─
# 10. Save artefacts
# ----------------------------------------------------------------------------─
out_path = DATA / "backtest_2025_results.csv"
if bets_110 and bets_110.get("bets_df") is not None:
    bets_110["bets_df"].to_csv(out_path, index=False)
    print(f"  Detailed bet log saved → {out_path}")

model_path = DATA / "model_2021_2024.pkl"
with open(model_path, "wb") as f:
    pickle.dump({"clf": clf, "feats": feats,
                 "train_years": TRAIN_YEARS, "test_year": TEST_YEAR}, f)
print(f"  OOT model saved        → {model_path}")
print()
