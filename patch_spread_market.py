"""
Adds spread (ATS) market to fetch_live_odds() and best_bets().
- fetch_live_odds: returns one row per game with both ML and spread columns
- best_bets: evaluates ML and ATS bets in a unified table sorted by EV
  - ATS cover probability uses Normal CDF around predicted run differential
  - RD_SIGMA = 3.5 runs (empirical college baseball prediction uncertainty)
"""

import json, math

NB = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def src(c): return ''.join(c.get('source', []))

fo_idx = next(i for i, c in enumerate(cells) if 'def fetch_live_odds' in src(c))
bb_idx = next(i for i, c in enumerate(cells) if 'def best_bets' in src(c))

# ── 1. Replace fetch_live_odds ────────────────────────────────────────────────
NEW_FO = '''\
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
        raw_time = game.get('commence_time', '')
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
            game_date = dt.strftime('%b %-d')
            game_time = dt.strftime('%I:%M %p UTC')
        except Exception:
            game_date = raw_time[:10]
            game_time = ''
        for bkm in game.get('bookmakers', [])[:1]:
            row = {
                'date': game_date, 'time': game_time,
                'home': home, 'away': away,
                'ml_home': None, 'ml_away': None,
                'spread_home': None, 'spread_home_price': None,
                'spread_away': None, 'spread_away_price': None,
            }
            for mkt in bkm.get('markets', []):
                if mkt['key'] == 'h2h':
                    out = {o['name']: o['price'] for o in mkt['outcomes']}
                    row['ml_home'] = out.get(home)
                    row['ml_away'] = out.get(away)
                elif mkt['key'] == 'spreads':
                    for o in mkt['outcomes']:
                        if o['name'] == home:
                            row['spread_home']       = o.get('point')
                            row['spread_home_price'] = o.get('price')
                        else:
                            row['spread_away']       = o.get('point')
                            row['spread_away_price'] = o.get('price')
            rows.append(row)
    print(f"Requests remaining: {_r.headers.get('x-requests-remaining','unknown')}")
    df = pd.DataFrame(rows)
    col_order = ['date','time','home','away',
                 'ml_home','ml_away',
                 'spread_home','spread_home_price','spread_away','spread_away_price']
    return df[[c for c in col_order if c in df.columns]] if not df.empty else df

live_odds = fetch_live_odds()
live_odds
'''

cells[fo_idx]['source'] = NEW_FO
print(f"[OK] Replaced fetch_live_odds() in cell {fo_idx}")

# ── 2. Replace best_bets ──────────────────────────────────────────────────────
NEW_BB = '''\
import math as _math

RD_SIGMA = 3.5   # std dev of run-differential prediction error (college baseball empirical)

def _norm_cdf(x, mu, sigma):
    """P(X <= x) for X ~ N(mu, sigma) — uses stdlib math.erf, no scipy needed."""
    return 0.5 * (1.0 + _math.erf((x - mu) / (sigma * _math.sqrt(2))))

def best_bets(date=None, teams=None, budget=None, edge_min=0.03, year=TEST_YEAR):
    """
    Returns ranked betting recommendations from live odds (ML and ATS combined).

    Args:
        date    : filter by date string (e.g., 'May 15') — partial match, case-insensitive
        teams   : list of team names to include, or None for all games
        budget  : total $ budget to allocate; triggers bet_$ column showing Kelly-sized amounts
        edge_min: minimum model edge required (default 0.03 = 3%)
        year    : season for model stats (default TEST_YEAR)

    Returns:
        DataFrame sorted by expected value (best bets first), ML and ATS combined
    """
    from IPython.display import display as ipy_display

    odds = fetch_live_odds()
    if odds.empty:
        print("No live odds available — check ODDS_API_KEY and run Section 7 first.")
        return pd.DataFrame()

    if date:
        odds = odds[odds['date'].str.contains(str(date), case=False, na=False)]
    if teams:
        mask = odds['home'].isin(teams) | odds['away'].isin(teams)
        odds = odds[mask]
    if odds.empty:
        print("No games match your filters."); return pd.DataFrame()

    rows = []
    known_teams = set(team_stats[team_stats['season'] == year]['team'])

    for _, g in odds.iterrows():
        home = resolve_team(g['home'], year, silent=True)
        away = resolve_team(g['away'], year, silent=True)

        if home not in known_teams or away not in known_teams:
            continue
        try:
            pred = predict_game(home, away, is_home_a=True, year=year, verbose=False)
        except Exception:
            continue
        if pred.get('elo_only'):
            continue

        wp_home   = pred['win_prob']
        wp_away   = 1.0 - wp_home
        pred_rd   = pred['run_diff']          # positive = home wins by this margin
        matchup   = f"{home} vs {away}"
        game_date = g.get('date', '')
        game_time = g.get('time', '')

        # ── Moneyline bets ────────────────────────────────────────────────────
        for side, team, ml, model_prob in [
            ('Home', home, g.get('ml_home'), wp_home),
            ('Away', away, g.get('ml_away'), wp_away),
        ]:
            if pd.isna(ml) or ml is None: continue
            ml = float(ml)
            implied = american_to_prob(ml)
            edge = model_prob - implied
            if edge < edge_min: continue
            b  = (100 / abs(ml)) if ml < 0 else (ml / 100)
            ev = model_prob * b - (1.0 - model_prob)
            kf = kelly_fraction(model_prob, odds=ml)
            rows.append({
                'date':       game_date, 'time': game_time,
                'matchup':    matchup,   'bet_on': team,
                'side':       side,      'market': 'ML',
                'handicap':   None,      'price':  int(ml),
                'model_prob': model_prob,
                'implied':    implied,
                'edge':       edge,
                'ev':         ev,
                'kelly_pct':  kf,
                'bet_$':      round(budget * kf, 2) if budget else None,
            })

        # ── Spread (ATS) bets ─────────────────────────────────────────────────
        sh = g.get('spread_home')
        shp = g.get('spread_home_price')
        sap = g.get('spread_away_price')

        if pd.notna(sh) and sh is not None:
            sh = float(sh)
            # P(home covers sh): home wins by more than |sh| when sh < 0
            # General: P(run_diff > -sh) = 1 - norm_cdf(-sh, pred_rd, sigma)
            p_home_cover = 1.0 - _norm_cdf(-sh, pred_rd, RD_SIGMA)
            p_away_cover = 1.0 - p_home_cover

            for side, team, price, model_prob, handicap in [
                ('Home ATS', home, shp, p_home_cover,  sh),
                ('Away ATS', away, sap, p_away_cover, -sh),
            ]:
                if pd.isna(price) or price is None: continue
                price = float(price)
                implied = american_to_prob(price)
                edge = model_prob - implied
                if edge < edge_min: continue
                b  = (100 / abs(price)) if price < 0 else (price / 100)
                ev = model_prob * b - (1.0 - model_prob)
                kf = kelly_fraction(model_prob, odds=price)
                rows.append({
                    'date':       game_date, 'time': game_time,
                    'matchup':    matchup,   'bet_on': team,
                    'side':       side,      'market': 'ATS',
                    'handicap':   f"{handicap:+.1f}",
                    'price':      int(price),
                    'model_prob': model_prob,
                    'implied':    implied,
                    'edge':       edge,
                    'ev':         ev,
                    'kelly_pct':  kf,
                    'bet_$':      round(budget * kf, 2) if budget else None,
                })

    if not rows:
        print(f"No bets found with edge > {edge_min:.0%}. Try lowering edge_min.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('ev', ascending=False).reset_index(drop=True)

    show = ['date','time','matchup','bet_on','side','market','handicap','price',
            'model_prob','implied','edge','ev','kelly_pct']
    if budget:
        show.append('bet_$')

    fmt = {
        'model_prob': '{:.1%}',
        'implied':    '{:.1%}',
        'edge':       '{:.1%}',
        'ev':         '{:+.3f}',
        'kelly_pct':  '{:.1%}',
    }
    if budget:
        fmt['bet_$'] = '${:.2f}'

    budget_str = f"  |  budget: ${budget:,.0f}" if budget else ""
    print(f"\\n{'='*65}")
    print(f"  BEST BETS  |  {len(df)} edges found (ML + ATS)"
          f"  |  min edge: {edge_min:.0%}{budget_str}")
    ml_n  = (df['market'] == 'ML').sum()
    ats_n = (df['market'] == 'ATS').sum()
    print(f"  Moneyline: {ml_n}  |  Spread (ATS): {ats_n}")
    print(f"{'='*65}")
    ipy_display(df[show].style.format(fmt, na_rep='-')
                .bar(subset=['ev'],   color=['#d65f5f','#5fba7d'], align='zero')
                .bar(subset=['edge'], color='#5fba7d'))

    if budget:
        total = df['bet_$'].sum()
        print(f"\\n  Total allocated: ${total:,.2f} / ${budget:,.0f}"
              f"  |  Remaining: ${budget - total:,.2f}")

    return df

print("best_bets() updated with ML + ATS markets.")
'''

cells[bb_idx]['source'] = NEW_BB
print(f"[OK] Replaced best_bets() in cell {bb_idx}")

# ── Save ──────────────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook saved ({len(cells)} cells).")

# ── Cleanup tmp files ─────────────────────────────────────────────────────────
import os
for f in ['_tmp_fo.txt', '_tmp_bb.txt']:
    try: os.remove(r'C:\Users\trevm\Projects\CFBBaseballAnalytics\\' + f)
    except: pass
