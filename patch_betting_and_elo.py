"""
Patches CollegeBaseballAnalytics_Master.ipynb with:
1. FiveThirtyEight-style Elo (autocorrelation correction + season mean-reversion)
2. Section 7b: best_bets() — actionable betting table with EV and Kelly sizing
3. Section 10b: query_db() — query the game results database
"""

import json

NB = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

# ── 1. Upgrade Elo cell (index 12) to FiveThirtyEight-style ──────────────────
ELO_CELL_SRC = """\
ELO_K    = 20   # FiveThirtyEight-style: lower K; MOVM multiplier handles magnitude
ELO_HOME = 35   # home advantage in Elo points (college, ~35 pts empirically)
ELO_INIT = 1500

# Conference tier multipliers — scales Elo K-factor so beating elite programs
# moves the needle more than beating low-major opponents.
CONF_TIER = {
    # Elite
    'SEC': 1.00, 'ACC': 1.00, 'Big 12': 1.00,
    # High mid-major
    'Sun Belt': 0.90, 'Mountain West': 0.90, 'Big Ten': 0.90,
    'Pac-12': 0.90, 'American Athletic': 0.90,
    # Mid-major
    'Conference USA': 0.78, 'WAC': 0.78, 'MAC': 0.78,
    'Missouri Valley': 0.78, 'Atlantic 10': 0.78,
    # Low mid-major / small conference
    'Ohio Valley': 0.65, 'ASUN': 0.65, 'Big South': 0.65,
    'Southern': 0.65, 'Northeast': 0.65, 'Patriot': 0.65,
    'MAAC': 0.65, 'America East': 0.65, 'Southland': 0.65,
}
CONF_DEFAULT = 0.72

# ESPN display name → conference (covers ~300 D-I baseball programs)
TEAM_CONF = {
    # SEC
    'Alabama Crimson Tide':'SEC','Arkansas Razorbacks':'SEC','Auburn Tigers':'SEC',
    'Florida Gators':'SEC','Georgia Bulldogs':'SEC','Kentucky Wildcats':'SEC',
    'LSU Tigers':'SEC','Ole Miss Rebels':'SEC','Mississippi State Bulldogs':'SEC',
    'Missouri Tigers':'SEC','South Carolina Gamecocks':'SEC',
    'Tennessee Volunteers':'SEC','Texas A&M Aggies':'SEC',
    'Vanderbilt Commodores':'SEC','Oklahoma Sooners':'SEC','Texas Longhorns':'SEC',
    # ACC
    'Clemson Tigers':'ACC','Duke Blue Devils':'ACC','Florida State Seminoles':'ACC',
    'Georgia Tech Yellow Jackets':'ACC','Louisville Cardinals':'ACC',
    'Miami Hurricanes':'ACC','NC State Wolfpack':'ACC','North Carolina Tar Heels':'ACC',
    'Notre Dame Fighting Irish':'ACC','Virginia Cavaliers':'ACC',
    'Virginia Tech Hokies':'ACC','Wake Forest Demon Deacons':'ACC',
    'Boston College Eagles':'ACC','Pittsburgh Panthers':'ACC',
    'Syracuse Orange':'ACC','Stanford Cardinal':'ACC',
    'Cal Bears':'ACC','SMU Mustangs':'ACC',
    # Big 12
    'Baylor Bears':'Big 12','BYU Cougars':'Big 12','Cincinnati Bearcats':'Big 12',
    'Houston Cougars':'Big 12','Iowa State Cyclones':'Big 12','Kansas Jayhawks':'Big 12',
    'Kansas State Wildcats':'Big 12','Oklahoma State Cowboys':'Big 12',
    'TCU Horned Frogs':'Big 12','Texas Tech Red Raiders':'Big 12',
    'UCF Knights':'Big 12','West Virginia Mountaineers':'Big 12',
    'Arizona Wildcats':'Big 12','Arizona State Sun Devils':'Big 12',
    'Colorado Buffaloes':'Big 12','Utah Utes':'Big 12',
    # Sun Belt
    'Appalachian State Mountaineers':'Sun Belt','Arkansas State Red Wolves':'Sun Belt',
    'Coastal Carolina Chanticleers':'Sun Belt','Georgia Southern Eagles':'Sun Belt',
    'Georgia State Panthers':'Sun Belt','James Madison Dukes':'Sun Belt',
    "Louisiana Ragin' Cajuns":'Sun Belt','Louisiana Ragin Cajuns':'Sun Belt',
    'Louisiana Monroe Warhawks':'Sun Belt','Marshall Thundering Herd':'Sun Belt',
    'Old Dominion Monarchs':'Sun Belt','South Alabama Jaguars':'Sun Belt',
    'Southern Miss Golden Eagles':'Sun Belt','Texas State Bobcats':'Sun Belt',
    'Troy Trojans':'Sun Belt',
    # AAC
    'Charlotte 49ers':'American Athletic','East Carolina Pirates':'American Athletic',
    'Florida Atlantic Owls':'American Athletic','Memphis Tigers':'American Athletic',
    'Rice Owls':'American Athletic','South Florida Bulls':'American Athletic',
    'Temple Owls':'American Athletic','Tulane Green Wave':'American Athletic',
    'Tulsa Golden Hurricane':'American Athletic','UAB Blazers':'American Athletic',
    'Wichita State Shockers':'American Athletic','Navy Midshipmen':'American Athletic',
    # Big Ten
    'Indiana Hoosiers':'Big Ten','Illinois Fighting Illini':'Big Ten',
    'Maryland Terrapins':'Big Ten','Michigan Wolverines':'Big Ten',
    'Michigan State Spartans':'Big Ten','Minnesota Golden Gophers':'Big Ten',
    'Nebraska Cornhuskers':'Big Ten','Northwestern Wildcats':'Big Ten',
    'Ohio State Buckeyes':'Big Ten','Penn State Nittany Lions':'Big Ten',
    'Purdue Boilermakers':'Big Ten','Rutgers Scarlet Knights':'Big Ten',
    # Pac-12 remnants
    'Oregon State Beavers':'Pac-12','UCLA Bruins':'Pac-12',
    'Washington Huskies':'Pac-12','Oregon Ducks':'Pac-12',
    # Mountain West
    'Air Force Falcons':'Mountain West','Fresno State Bulldogs':'Mountain West',
    'Nevada Wolf Pack':'Mountain West','UNLV Rebels':'Mountain West',
    'Utah State Aggies':'Mountain West','San Diego State Aztecs':'Mountain West',
    'New Mexico Lobos':'Mountain West','Wyoming Cowboys':'Mountain West',
    'Boise State Broncos':'Mountain West',
    # Conference USA
    'Jacksonville State Gamecocks':'Conference USA','Liberty Flames':'Conference USA',
    'Middle Tennessee Blue Raiders':'Conference USA',
    'New Mexico State Aggies':'Conference USA','UTEP Miners':'Conference USA',
    'Western Kentucky Hilltoppers':'Conference USA','Sam Houston Bearkats':'Conference USA',
    'FIU Panthers':'Conference USA','Louisiana Tech Bulldogs':'Conference USA',
    'UTSA Roadrunners':'Conference USA','Kennesaw State Owls':'Conference USA',
    'Florida International Panthers':'Conference USA',
    # WAC
    'California Baptist Lancers':'WAC','Cal Baptist Lancers':'WAC',
    'Grand Canyon Antelopes':'WAC','Sacramento State Hornets':'WAC',
    'Tarleton State Texans':'WAC','Utah Tech Trailblazers':'WAC',
    'Utah Valley Wolverines':'WAC','Seattle Redhawks':'WAC',
    'Southern Utah Thunderbirds':'WAC','Abilene Christian Wildcats':'WAC',
    "Stephen F. Austin Lumberjacks":'WAC',
    # MAC
    'Bowling Green Falcons':'MAC','Ball State Cardinals':'MAC',
    'Central Michigan Chippewas':'MAC','Eastern Michigan Eagles':'MAC',
    'Kent State Golden Flashes':'MAC','Miami RedHawks':'MAC',
    'Northern Illinois Huskies':'MAC','Ohio Bobcats':'MAC',
    'Toledo Rockets':'MAC','Western Michigan Broncos':'MAC',
    'Akron Zips':'MAC','Buffalo Bulls':'MAC',
    # Missouri Valley
    'Dallas Baptist Patriots':'Missouri Valley','Illinois State Redbirds':'Missouri Valley',
    'Indiana State Sycamores':'Missouri Valley','Missouri State Bears':'Missouri Valley',
    'Southern Illinois Salukis':'Missouri Valley','Bradley Braves':'Missouri Valley',
    'Valparaiso Beacons':'Missouri Valley',
    # OVC
    'Morehead State Eagles':'Ohio Valley','Austin Peay Governors':'Ohio Valley',
    'Eastern Illinois Panthers':'Ohio Valley','Eastern Kentucky Colonels':'Ohio Valley',
    'Murray State Racers':'Ohio Valley','SE Missouri State Redhawks':'Ohio Valley',
    'Tennessee Tech Golden Eagles':'Ohio Valley','UT Martin Skyhawks':'Ohio Valley',
    'SIU Edwardsville Cougars':'Ohio Valley','Lindenwood Lions':'Ohio Valley',
    'Bellarmine Knights':'Ohio Valley',
    # ASUN
    'Central Arkansas Bears':'ASUN','Jacksonville Dolphins':'ASUN',
    'Lipscomb Bisons':'ASUN','North Alabama Lions':'ASUN',
    'North Florida Ospreys':'ASUN','Northern Kentucky Norse':'ASUN',
    'Stetson Hatters':'ASUN','Queens Royals':'ASUN',
    # Big South
    'Campbell Fighting Camels':'Big South','Charleston Southern Buccaneers':'Big South',
    "Gardner-Webb Runnin' Bulldogs":'Big South','High Point Panthers':'Big South',
    'Longwood Lancers':'Big South','Presbyterian Blue Hose':'Big South',
    'Radford Highlanders':'Big South','UNC Asheville Bulldogs':'Big South',
    'USC Upstate Spartans':'Big South','Winthrop Eagles':'Big South',
    # Southern
    'ETSU Buccaneers':'Southern','East Tennessee State Buccaneers':'Southern',
    'Furman Paladins':'Southern','Mercer Bears':'Southern',
    'Samford Bulldogs':'Southern','The Citadel Bulldogs':'Southern',
    'VMI Keydets':'Southern','Western Carolina Catamounts':'Southern',
    'Wofford Terriers':'Southern','Chattanooga Mocs':'Southern',
    # NEC / Patriot / MAAC / America East
    'Bryant Bulldogs':'Northeast','Sacred Heart Pioneers':'Northeast',
    'Wagner Seahawks':'Northeast',
    'Army Black Knights':'Patriot','Bucknell Bison':'Patriot',
    'Holy Cross Crusaders':'Patriot','Lafayette Leopards':'Patriot',
    'Lehigh Mountain Hawks':'Patriot',
    'Fairfield Stags':'MAAC','Manhattan Jaspers':'MAAC','Marist Red Foxes':'MAAC',
    'Niagara Purple Eagles':'MAAC','Quinnipiac Bobcats':'MAAC',
    'Rider Broncs':'MAAC','Siena Saints':'MAAC',
    'Albany Great Danes':'America East','Maine Black Bears':'America East',
    'Stony Brook Seawolves':'America East','Vermont Catamounts':'America East',
    'UMass Lowell River Hawks':'America East','Binghamton Bearcats':'America East',
}

def get_conf_tier(team):
    return CONF_TIER.get(TEAM_CONF.get(team, ''), CONF_DEFAULT)

def expected_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

def margin_mult_538(run_diff, elo_diff_pre):
    \"\"\"
    FiveThirtyEight baseball MOVM formula.
    Autocorrelation correction: strong favorites gain less Elo for expected blowouts.
    elo_diff_pre = winning team Elo - losing team Elo (before the game, always positive).
    \"\"\"
    movm = np.log(abs(run_diff) + 1) * 2.2
    # Autocorrelation correction — dampens margin value when elo gap was already large
    correction = 2.2 / (elo_diff_pre * 0.001 + 2.2)
    return movm * correction

ELO_MEAN_REVERT  = 0.75   # carry fraction from prior season (0.25 reverts to mean)
ELO_REVERT_TO    = 1505   # long-run mean (slightly above 1500 to account for new teams)

def compute_elo(games):
    \"\"\"
    FiveThirtyEight-style Elo:
    - K=20 (lower than naive; MOVM handles magnitude)
    - Autocorrelation correction in MOVM (see margin_mult_538)
    - Season mean-reversion: Elo *= 0.75 + 0.25 * 1505 at season boundary
    - Conference K-tier multiplier preserved
    \"\"\"
    elo = {}
    cur_season = None
    games_s = games.sort_values(['season', 'date']).dropna(subset=['date'])
    for _, g in games_s.iterrows():
        yr = int(g['season'])
        # Mean-revert all ratings at the start of each new season
        if yr != cur_season:
            if cur_season is not None:
                elo = {t: ELO_MEAN_REVERT * v + (1 - ELO_MEAN_REVERT) * ELO_REVERT_TO
                       for t, v in elo.items()}
            cur_season = yr

        ht, at = g['home_team'], g['away_team']
        base_h = elo.get(ht, ELO_INIT)
        base_a = elo.get(at, ELO_INIT)

        # Apply home advantage before computing expected win
        eh = base_h + (ELO_HOME if not g.get('neutral', False) else 0)
        ea = base_a
        exp = expected_win(eh, ea)
        actual = 1 if g['home_score'] > g['away_score'] else 0

        # Pre-game Elo diff from winner's perspective (always positive, used for MOVM)
        rd = g['home_score'] - g['away_score']
        if rd > 0:
            winner_elo, loser_elo = eh, ea
        elif rd < 0:
            winner_elo, loser_elo = ea, eh
        else:
            winner_elo, loser_elo = eh, ea
        elo_diff_pre = max(winner_elo - loser_elo, 0)

        mult  = margin_mult_538(rd, elo_diff_pre)
        tier  = (get_conf_tier(ht) + get_conf_tier(at)) / 2
        delta = ELO_K * tier * mult * (actual - exp)

        elo[ht] = base_h + delta
        elo[at] = base_a - delta

    return pd.DataFrame({'team': list(elo.keys()), 'elo': list(elo.values())})

elo_df = compute_elo(games)
print(f"Elo (FiveThirtyEight-style) computed for {len(elo_df)} teams")
elo_df.sort_values('elo', ascending=False).head(10)
"""

# ── 2. Section 7b: best_bets() ────────────────────────────────────────────────
BEST_BETS_MD = """\
## Section 7b — Best Bets Tool

Query by **date**, **teams**, or **budget** to get ranked betting recommendations with expected value and Kelly sizing.

```python
# All today's games with positive edge
best_bets()

# Filter by date string
best_bets(date='May 15')

# Filter to specific teams
best_bets(teams=['Tennessee Volunteers', 'LSU Tigers'])

# With a $500 budget — shows exact $ to bet per game
best_bets(budget=500)

# Combine filters
best_bets(date='May 15', budget=200, edge_min=0.05)
```

**Columns:**
| Column | Meaning |
|--------|---------|
| `bet_on` | Which team to bet |
| `ml` | Current American moneyline |
| `model_wp` | Model's estimated win probability |
| `implied_prob` | Vegas implied probability from the line |
| `edge` | model_wp − implied_prob (your statistical advantage) |
| `ev_per_unit` | Expected value per $1 wagered (positive = profitable long-run) |
| `kelly_pct` | Recommended bankroll % (quarter-Kelly, 10% cap) |
| `bet_$` | Dollar amount to bet (only shown when budget is provided) |
"""

BEST_BETS_CODE = """\
def best_bets(date=None, teams=None, budget=None, edge_min=0.03, year=TEST_YEAR):
    \"\"\"
    Returns ranked betting recommendations from live odds.

    Args:
        date    : filter by date string (e.g., 'May 15') — partial match, case-insensitive
        teams   : list of team names to include, or None for all games
        budget  : total $ budget to allocate; triggers bet_$ column showing Kelly-sized amounts
        edge_min: minimum model edge required (default 0.03 = 3%)
        year    : season for model stats (default TEST_YEAR)

    Returns:
        DataFrame sorted by expected value (best bets first)
    \"\"\"
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
    for _, g in odds.iterrows():
        home, away = g['home'], g['away']
        ml_home = g.get('ml_home')
        ml_away = g.get('ml_away')
        try:
            pred = predict_game(home, away, is_home_a=True, year=year, verbose=False)
        except Exception:
            continue
        wp_home = pred['win_prob']
        wp_away = 1.0 - wp_home

        for side, team, ml, model_wp in [
            ('Home', home, ml_home, wp_home),
            ('Away', away, ml_away, wp_away),
        ]:
            if pd.isna(ml) or ml is None: continue
            ml = float(ml)
            implied = american_to_prob(ml)
            edge = model_wp - implied
            if edge < edge_min: continue

            b = (100 / abs(ml)) if ml < 0 else (ml / 100)
            ev = model_wp * b - (1.0 - model_wp)   # EV per $1 wagered
            kf = kelly_fraction(model_wp, odds=ml)

            rows.append({
                'date':         g.get('date', ''),
                'time':         g.get('time', ''),
                'matchup':      f"{home} vs {away}",
                'bet_on':       team,
                'side':         side,
                'ml':           int(ml),
                'model_wp':     model_wp,
                'implied_prob': implied,
                'edge':         edge,
                'ev_per_unit':  ev,
                'kelly_pct':    kf,
                'bet_$':        round(budget * kf, 2) if budget else None,
            })

    if not rows:
        print(f"No bets found with edge > {edge_min:.0%}. Try lowering edge_min.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('ev_per_unit', ascending=False).reset_index(drop=True)

    show = ['date', 'time', 'matchup', 'bet_on', 'side', 'ml',
            'model_wp', 'implied_prob', 'edge', 'ev_per_unit', 'kelly_pct']
    if budget:
        show.append('bet_$')

    fmt = {
        'model_wp':     '{:.1%}',
        'implied_prob': '{:.1%}',
        'edge':         '{:.1%}',
        'ev_per_unit':  '{:+.3f}',
        'kelly_pct':    '{:.1%}',
    }
    if budget:
        fmt['bet_$'] = '${:.2f}'

    budget_str = f"  |  budget: ${budget:,.0f}" if budget else ""
    print(f"\\n{'='*65}")
    print(f"  BEST BETS  |  {len(df)} edge{'s' if len(df)!=1 else ''} found"
          f"  |  min edge: {edge_min:.0%}{budget_str}")
    print(f"{'='*65}")
    ipy_display(df[show].style.format(fmt)
                .bar(subset=['ev_per_unit'], color=['#d65f5f','#5fba7d'], align='zero')
                .bar(subset=['edge'], color='#5fba7d'))

    if budget:
        total = df['bet_$'].sum()
        print(f"\\n  Total allocated: ${total:,.2f} / ${budget:,.0f}  |  "
              f"Remaining: ${budget - total:,.2f}")

    return df

# ── example calls (edit to match your query) ──────────────────────────────────
# best_bets()                                    # all today's games
# best_bets(date='May 15', budget=500)           # filter by date + budget
# best_bets(teams=['Tennessee Volunteers'], budget=200, edge_min=0.04)
print("best_bets() loaded. Call it after running Section 7 to fetch live odds.")
"""

# ── 3. Section 10b: query_db() ────────────────────────────────────────────────
QUERY_DB_MD = """\
## Section 10b — Database Query Interface

Query the local game-results database by team, date, season, or result.

```python
# All Tennessee Volunteers games in 2025
query_db(team='Tennessee', season=2025)

# Head-to-head: Tennessee vs Arkansas
query_db(team='Tennessee', opponent='Arkansas')

# Tennessee home wins only
query_db(team='Tennessee', home_only=True, result='W')

# Date range
query_db(date_from='2026-04-01', date_to='2026-05-15')

# Season summary for a team
team_stats_query('LSU Tigers', season=2025)
```
"""

QUERY_DB_CODE = """\
def query_db(team=None, opponent=None, season=None, date_from=None, date_to=None,
             home_only=False, result=None, limit=50):
    \"\"\"
    Query game results.

    Args:
        team      : partial team name match (case-insensitive), e.g. 'Tennessee'
        opponent  : filter to specific opponent (partial match)
        season    : int or list of ints, e.g. 2025 or [2024, 2025]
        date_from : 'YYYY-MM-DD' string (inclusive)
        date_to   : 'YYYY-MM-DD' string (inclusive)
        home_only : only games where `team` was the home team
        result    : 'W' or 'L' (from the perspective of `team`)
        limit     : max rows returned (default 50)

    Returns:
        DataFrame of matching games, sorted newest first
    \"\"\"
    df = games.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if season is not None:
        seasons = [season] if isinstance(season, int) else list(season)
        df = df[df['season'].isin(seasons)]

    if date_from:
        df = df[df['date'] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df['date'] <= pd.to_datetime(date_to)]

    if team:
        if home_only:
            df = df[df['home_team'].str.contains(team, case=False, na=False)]
        else:
            mask = (df['home_team'].str.contains(team, case=False, na=False) |
                    df['away_team'].str.contains(team, case=False, na=False))
            df = df[mask]

        if opponent:
            opp_mask = (df['home_team'].str.contains(opponent, case=False, na=False) |
                        df['away_team'].str.contains(opponent, case=False, na=False))
            df = df[opp_mask]

        if result:
            team_won = (
                (df['home_team'].str.contains(team, case=False, na=False) & (df['home_score'] > df['away_score'])) |
                (df['away_team'].str.contains(team, case=False, na=False) & (df['away_score'] > df['home_score']))
            )
            df = df[team_won] if result.upper() == 'W' else df[~team_won]
    elif opponent:
        mask = (df['home_team'].str.contains(opponent, case=False, na=False) |
                df['away_team'].str.contains(opponent, case=False, na=False))
        df = df[mask]

    df = df.sort_values('date', ascending=False).head(limit).reset_index(drop=True)
    df['run_diff'] = (df['home_score'] - df['away_score']).apply(lambda x: f"{x:+.0f}")
    df['result_str'] = df.apply(
        lambda r: f"{r['home_team']} {r['home_score']:.0f}-{r['away_score']:.0f} {r['away_team']}", axis=1
    )
    print(f"Found {len(df)} games (showing up to {limit}).")
    return df[['season','date','home_team','home_score','away_team','away_score','run_diff','neutral']]


def team_stats_query(team, season=None):
    \"\"\"Season-by-season summary stats for a team.\"\"\"
    df = team_stats.copy()
    mask = df['team'].str.contains(team, case=False, na=False)
    df = df[mask]
    if season:
        df = df[df['season'] == season]
    show = ['season','team','games','wins','losses','win_pct',
            'avg_runs_scored','avg_runs_allowed','avg_run_diff','pythagorean_win_pct']
    show = [c for c in show if c in df.columns]
    fmt = {
        'win_pct':'{:.3f}','avg_runs_scored':'{:.2f}',
        'avg_runs_allowed':'{:.2f}','avg_run_diff':'{:+.2f}',
        'pythagorean_win_pct':'{:.3f}',
    }
    print(f"Season stats for '{team}':")
    from IPython.display import display as ipy_display
    ipy_display(df[show].sort_values('season').style.format(fmt))
    return df[show]


# ── sample queries ─────────────────────────────────────────────────────────────
print("query_db() and team_stats_query() loaded.")
print("Examples:")
print("  query_db(team='Tennessee', season=2025)")
print("  query_db(team='LSU', opponent='Arkansas', result='W')")
print("  team_stats_query('Tennessee Volunteers')")
"""

# ── Locate target cells and apply patches ─────────────────────────────────────
cells = nb['cells']

# Find index of cell 12 (Elo) by matching a unique string
elo_idx = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'def compute_elo' in src and 'ELO_K' in src:
        elo_idx = i
        break

# Find index of fetch_live_odds cell (after which we insert best_bets)
odds_idx = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'def fetch_live_odds' in src:
        odds_idx = i
        break

# Find the Section 10 header cell (after which we insert query_db)
sec10_idx = None
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'Section 10' in src:
        sec10_idx = i
        break

print(f"Elo cell index:      {elo_idx}")
print(f"fetch_live_odds idx: {odds_idx}")
print(f"Section 10 idx:      {sec10_idx}")

assert elo_idx is not None, "Could not find Elo cell"
assert odds_idx is not None, "Could not find fetch_live_odds cell"
assert sec10_idx is not None, "Could not find Section 10 cell"

# 1. Replace Elo cell
cells[elo_idx] = code_cell(ELO_CELL_SRC)
print("[OK] Replaced Elo cell with FiveThirtyEight-style formula")

# 2. Insert best_bets after fetch_live_odds (insert in reverse order: code then md)
#    We insert AFTER odds_idx, so at odds_idx+1
insert_at = odds_idx + 1
cells.insert(insert_at, code_cell(BEST_BETS_CODE))
cells.insert(insert_at, md_cell(BEST_BETS_MD))
print(f"[OK] Inserted best_bets() at cell index {insert_at} (after fetch_live_odds)")

# Recalculate sec10_idx since we inserted 2 cells
sec10_idx += 2

# 3. Insert query_db after Section 10 header + embed cell (skip 2 cells)
insert_at2 = sec10_idx + 2
cells.insert(insert_at2, code_cell(QUERY_DB_CODE))
cells.insert(insert_at2, md_cell(QUERY_DB_MD))
print(f"[OK] Inserted query_db() at cell index {insert_at2}")

nb['cells'] = cells

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved with {len(cells)} cells total.")
print("Next: run push_notebook.py to push to GitHub.")
