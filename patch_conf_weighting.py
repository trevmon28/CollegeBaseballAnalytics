"""
Patches CollegeBaseballAnalytics_Master.ipynb with two-layer conference weighting:
  Layer 1 — SOS (avg opponent Elo) added as 0.15 weight in composite power score
  Layer 2 — Conference tier K-factor multiplier in Elo computation
"""
import json

NB_PATH = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def src(cell):
    return ''.join(cell['source'])

# ── locate target cells ────────────────────────────────────────────────────────
elo_cell_idx      = None   # cell 12 — compute_elo
rankings_cell_idx = None   # cell 13 — build_rankings
names_cell_idx    = None   # cell 14 — print team names

for i, cell in enumerate(nb['cells']):
    s = src(cell)
    if 'def compute_elo(games):' in s and 'ELO_K' in s:
        elo_cell_idx = i
    if 'def build_rankings(stats, elo' in s and 'power_score' in s:
        rankings_cell_idx = i
    if 'Exact team names for Top 25' in s:
        names_cell_idx = i

print(f"Elo cell index      : {elo_cell_idx}")
print(f"Rankings cell index : {rankings_cell_idx}")
print(f"Names cell index    : {names_cell_idx}")

assert elo_cell_idx is not None,      "Could not find Elo cell"
assert rankings_cell_idx is not None, "Could not find rankings cell"
assert names_cell_idx is not None,    "Could not find names cell"

# ── new Elo cell (Layer 2 — conference tier K-factor) ─────────────────────────
new_elo_src = """\
ELO_K    = 30
ELO_HOME = 35
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
CONF_DEFAULT = 0.72   # fallback for unmapped teams

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
    # Conference USA (post-2023 realignment)
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
    # OVC — the overrated-team cluster
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

def margin_mult(rd):
    return np.log(abs(rd) + 1) * 2.0

def compute_elo(games):
    elo = {}
    games_s = games.sort_values('date').dropna(subset=['date'])
    for _, g in games_s.iterrows():
        ht, at = g['home_team'], g['away_team']
        eh  = elo.get(ht, ELO_INIT) + (ELO_HOME if not g.get('neutral', False) else 0)
        ea  = elo.get(at, ELO_INIT)
        exp = expected_win(eh, ea)
        actual = 1 if g['home_score'] > g['away_score'] else 0
        mult   = margin_mult(g['home_score'] - g['away_score'])
        # Average both teams' conference tiers to get per-game K multiplier
        tier  = (get_conf_tier(ht) + get_conf_tier(at)) / 2
        delta = ELO_K * tier * mult * (actual - exp)
        elo[ht] = elo.get(ht, ELO_INIT) + delta
        elo[at] = elo.get(at, ELO_INIT) - delta
    return pd.DataFrame({'team': list(elo.keys()), 'elo': list(elo.values())})

elo_df = compute_elo(games)
print(f"Elo computed for {len(elo_df)} teams")
"""

# ── new rankings cell (Layer 1 — SOS + updated weights) ───────────────────────
new_rankings_src = """\
def compute_sos(games, elo):
    \"\"\"Average Elo of every opponent faced — higher means harder schedule.\"\"\"
    elo_lk = elo.set_index('team')['elo']
    home = games[['season','home_team','away_team']].rename(
        columns={'home_team':'team','away_team':'opp'})
    away = games[['season','away_team','home_team']].rename(
        columns={'away_team':'team','home_team':'opp'})
    long = pd.concat([home, away], ignore_index=True)
    long['opp_elo'] = long['opp'].map(elo_lk).fillna(ELO_INIT)
    return (long.groupby(['team','season'])['opp_elo']
               .mean().reset_index()
               .rename(columns={'opp_elo':'avg_opp_elo'}))

sos_df = compute_sos(games, elo_df)

def build_rankings(stats, elo, sos, year=TEST_YEAR):
    df = stats[stats['season'] == year].copy()
    df = df.merge(elo, on='team', how='left')
    df = df.merge(sos[sos['season'] == year][['team','avg_opp_elo']], on='team', how='left')
    df['elo']         = df['elo'].fillna(ELO_INIT)
    df['avg_opp_elo'] = df['avg_opp_elo'].fillna(df['avg_opp_elo'].mean())

    def z(s): return (s - s.mean()) / max(s.std(), 1e-6)

    score = pd.Series(0.0, index=df.index)
    for col, wt, higher in [
        ('pythagorean_win_pct', 0.25, True),   # was 0.30
        ('avg_run_diff',        0.20, True),    # was 0.25
        ('elo',                 0.20, True),    # was 0.25
        ('avg_runs_scored',     0.10, True),
        ('avg_runs_allowed',    0.10, False),
        ('avg_opp_elo',         0.15, True),    # new SOS component
    ]:
        if col in df.columns:
            score += wt * (z(df[col]) if higher else -z(df[col]))

    df['power_score'] = score
    df['conference']  = df['team'].map(TEAM_CONF).fillna('Unknown')
    df['rank'] = df['power_score'].rank(ascending=False).astype(int)
    return df.sort_values('rank')

rankings = build_rankings(team_stats, elo_df, sos_df, year=TEST_YEAR)
show = ['rank','team','conference','elo','avg_opp_elo','pythagorean_win_pct',
        'avg_run_diff','power_score']
show = [c for c in show if c in rankings.columns]
print(f"=== {TEST_YEAR} College Baseball Power Rankings (Top 25) ===")
rankings[show].head(25).style.format({
    'elo':'{:.0f}','avg_opp_elo':'{:.0f}','pythagorean_win_pct':'{:.3f}',
    'avg_run_diff':'{:+.2f}','power_score':'{:.3f}'
})
"""

# ── new names cell — include conference in print ───────────────────────────────
new_names_src = """\
print("Exact team names for Top 25 (copy/paste into predict_game):")
for _, r in rankings.head(25).iterrows():
    conf = r.get('conference','?')
    print(f"  #{r['rank']:2d}  {r['team']}  [{conf}]")
"""

# ── apply patches ──────────────────────────────────────────────────────────────
def set_source(cell, new_src):
    lines = new_src.splitlines(keepends=True)
    # Ensure last line has no trailing newline (notebook convention)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    cell['source'] = lines

set_source(nb['cells'][elo_cell_idx],      new_elo_src)
set_source(nb['cells'][rankings_cell_idx], new_rankings_src)
set_source(nb['cells'][names_cell_idx],    new_names_src)

# Also update build_rankings call in bar chart cell (cell 15) if it passes no sos arg
for i, cell in enumerate(nb['cells']):
    s = src(cell)
    if 'top25 = rankings.head(25)' in s and 'barh' in s:
        # No structural change needed — rankings is already computed above
        print(f"Bar chart cell at index {i} — no change needed (rankings already computed)")
        break

# Patch the predict_game helper to pass sos_df correctly (build_rankings now needs sos)
# No change needed in predict_game itself; only build_rankings signature changed.

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook patched successfully.")
print(f"  Cell {elo_cell_idx}      → Elo cell: conference tier K-factor + TEAM_CONF dict")
print(f"  Cell {rankings_cell_idx} → Rankings: SOS component (0.15) + updated weights")
print(f"  Cell {names_cell_idx}    → Names print: conference shown in brackets")
