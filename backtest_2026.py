"""
backtest_2026.py — realistic 2026 ML backtest
- Uses 2025 stats to predict 2026 games (no data leakage)
- Only bets games where model confidence is 55-65% (range where -110 is realistic market price)
- Flat $20/game and rolling quarter-Kelly on a $1,000 starting bankroll
"""
import pickle
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

bundle = pickle.load(open(DATA / "model.pkl", "rb"))
clf, FEATS = bundle["clf"], bundle["feats"]
games = pd.read_parquet(DATA / "game_results_2021_2026.parquet")
stats = pd.read_parquet(DATA / "team_season_stats_2021_2026.parquet")

stats25 = stats[stats.season == 2025].set_index("team")
games26 = games[games.season == 2026].copy()
games26["date"] = pd.to_datetime(games26["date"])

rows = []
for _, g in games26.iterrows():
    h, a = g.home_team, g.away_team
    if h not in stats25.index or a not in stats25.index:
        continue
    hs, as_ = stats25.loc[h], stats25.loc[a]
    fvec = []
    for feat in FEATS:
        if feat == "is_home":
            fvec.append(0.0 if g.neutral else 1.0)
        elif feat == "neutral":
            fvec.append(1.0 if g.neutral else 0.0)
        elif feat.startswith("d_"):
            col = feat[2:]
            fvec.append(float(hs.get(col, 0)) - float(as_.get(col, 0)))
        else:
            fvec.append(float(hs.get(feat, 0)))
    home_wp = float(clf.predict_proba([fvec])[0, 1])
    rows.append({
        "date": g.date, "home": h, "away": a,
        "home_wp": home_wp, "home_won": g.home_score > g.away_score,
        "home_score": int(g.home_score), "away_score": int(g.away_score),
    })

df = pd.DataFrame(rows)

# Only bet 55-65% confidence range: where -110 is a realistic market price
bets = []
for _, r in df.iterrows():
    for side in ("home", "away"):
        wp = r.home_wp if side == "home" else 1 - r.home_wp
        won = bool(r.home_won) if side == "home" else not bool(r.home_won)
        if 0.55 <= wp <= 0.65:
            bets.append({
                "date": r.date, "side": side, "wp": wp, "won": won,
                "home": r.home, "away": r.away,
                "score": f"{r.home_score}-{r.away_score}",
            })

bets_df = pd.DataFrame(bets).sort_values("date").reset_index(drop=True)

BANKROLL = 1000.0
FLAT = 20.0
PAYOUT = 100 / 110   # -110 payout: win $0.909 per $1 bet
IMPLIED = 100 / 210  # 52.38% break-even

# --- flat betting ---
flat_pnls = [FLAT * PAYOUT if w else -FLAT for w in bets_df.won]
bets_df["flat_pnl"] = flat_pnls
bets_df["flat_cumul"] = BANKROLL + bets_df.flat_pnl.cumsum()

# --- rolling Kelly ---
bankroll = BANKROLL
kelly_pnls, kelly_bets = [], []
for _, b in bets_df.iterrows():
    k = min(max(0.25 * (b.wp * (PAYOUT + 1) - 1) / PAYOUT, 0.0), 0.10)
    bet = round(k * bankroll, 2)
    pnl = bet * PAYOUT if b.won else -bet
    kelly_pnls.append(pnl)
    kelly_bets.append(bet)
    bankroll += pnl
bets_df["kelly_bet"] = kelly_bets
bets_df["kelly_pnl"] = kelly_pnls
bets_df["kelly_cumul"] = BANKROLL + bets_df.kelly_pnl.cumsum()

wins = int(bets_df.won.sum())
total = len(bets_df)
flat_net = bets_df.flat_pnl.sum()
kelly_net = bets_df.kelly_pnl.sum()
kelly_end = BANKROLL + kelly_net

print()
print("=" * 60)
print("  2026 SEASON BACKTEST — ML @ -110 (55-65% confidence band)")
print("=" * 60)
print(f"  Games analyzed:        {len(df):,}")
print(f"  Bets placed:           {total:,}")
print(f"  Record:                {wins}W / {total-wins}L  ({wins/total*100:.1f}%)")
print(f"  Break-even @ -110:     52.4%")
print()
print(f"  FLAT ($20/game)")
print(f"    Net P&L:             ${flat_net:+,.2f}")
print(f"    Ending bankroll:     ${BANKROLL + flat_net:,.2f}")
print(f"    ROI on bankroll:     {flat_net/BANKROLL*100:+.1f}%")
print()
print(f"  KELLY (rolling, quarter-Kelly, 10% cap)")
print(f"    Avg bet size:        ${sum(kelly_bets)/len(kelly_bets):.2f}")
print(f"    Net P&L:             ${kelly_net:+,.2f}")
print(f"    Ending bankroll:     ${kelly_end:,.2f}")
print(f"    ROI on bankroll:     {kelly_net/BANKROLL*100:+.1f}%")
print()

bets_df["month"] = bets_df.date.dt.to_period("M")
print(f"  Monthly breakdown:")
header = f"  {'Month':<9} {'Bets':>5} {'W-L':>8} {'Win%':>6} {'Flat':>9} {'Kelly':>9}"
print(header)
print("  " + "-" * 50)
for month, grp in bets_df.groupby("month"):
    w = int(grp.won.sum()); n = len(grp)
    fp = grp.flat_pnl.sum(); kp = grp.kelly_pnl.sum()
    wl = f"{w}-{n-w}"
    print(f"  {str(month):<9} {n:>5} {wl:>8} {w/n*100:>5.1f}% {fp:>+9.2f} {kp:>+9.2f}")
print("=" * 60)
print()
print("  NOTE: Assumes flat -110 on all bets. Real market odds on")
print("  55-65% confidence games typically range -120 to -175.")
print("  Actual returns would be lower. This shows directional edge.")
