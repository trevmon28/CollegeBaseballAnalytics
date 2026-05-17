"""
build_static.py — Generates docs/index.html for GitHub Pages.
Run: python build_static.py  →  push docs/ to GitHub, enable Pages from /docs.
Live URL: https://trevmon28.github.io/CollegeBaseballAnalytics/
"""
import json, pathlib, numpy as np, pandas as pd

ELO_K = 30; ELO_HOME = 35; ELO_INIT = 1500; TEST_YEAR = 2026

CONF_TIER = {
    'SEC':1.00,'ACC':1.00,'Big 12':1.00,
    'Sun Belt':0.90,'Mountain West':0.90,'Big Ten':0.90,'Pac-12':0.90,'American Athletic':0.90,
    'Conference USA':0.78,'WAC':0.78,'MAC':0.78,'Missouri Valley':0.78,'Atlantic 10':0.78,
    'Ohio Valley':0.65,'ASUN':0.65,'Big South':0.65,'Southern':0.65,
    'Northeast':0.65,'Patriot':0.65,'MAAC':0.65,'America East':0.65,'Southland':0.65,
}
CONF_DEFAULT = 0.72

TEAM_CONF = {
    'Alabama Crimson Tide':'SEC','Arkansas Razorbacks':'SEC','Auburn Tigers':'SEC',
    'Florida Gators':'SEC','Georgia Bulldogs':'SEC','Kentucky Wildcats':'SEC',
    'LSU Tigers':'SEC','Ole Miss Rebels':'SEC','Mississippi State Bulldogs':'SEC',
    'Missouri Tigers':'SEC','South Carolina Gamecocks':'SEC','Tennessee Volunteers':'SEC',
    'Texas A&M Aggies':'SEC','Vanderbilt Commodores':'SEC','Oklahoma Sooners':'SEC','Texas Longhorns':'SEC',
    'Clemson Tigers':'ACC','Duke Blue Devils':'ACC','Florida State Seminoles':'ACC',
    'Georgia Tech Yellow Jackets':'ACC','Louisville Cardinals':'ACC','Miami Hurricanes':'ACC',
    'NC State Wolfpack':'ACC','North Carolina Tar Heels':'ACC','Notre Dame Fighting Irish':'ACC',
    'Virginia Cavaliers':'ACC','Virginia Tech Hokies':'ACC','Wake Forest Demon Deacons':'ACC',
    'Boston College Eagles':'ACC','Pittsburgh Panthers':'ACC','Syracuse Orange':'ACC',
    'Stanford Cardinal':'ACC','Cal Bears':'ACC','SMU Mustangs':'ACC',
    'Baylor Bears':'Big 12','BYU Cougars':'Big 12','Cincinnati Bearcats':'Big 12',
    'Houston Cougars':'Big 12','Iowa State Cyclones':'Big 12','Kansas Jayhawks':'Big 12',
    'Kansas State Wildcats':'Big 12','Oklahoma State Cowboys':'Big 12','TCU Horned Frogs':'Big 12',
    'Texas Tech Red Raiders':'Big 12','UCF Knights':'Big 12','West Virginia Mountaineers':'Big 12',
    'Arizona Wildcats':'Big 12','Arizona State Sun Devils':'Big 12',
    'Colorado Buffaloes':'Big 12','Utah Utes':'Big 12',
    'Appalachian State Mountaineers':'Sun Belt','Arkansas State Red Wolves':'Sun Belt',
    'Coastal Carolina Chanticleers':'Sun Belt','Georgia Southern Eagles':'Sun Belt',
    'Georgia State Panthers':'Sun Belt','James Madison Dukes':'Sun Belt',
    "Louisiana Ragin' Cajuns":'Sun Belt','Louisiana Monroe Warhawks':'Sun Belt',
    'Marshall Thundering Herd':'Sun Belt','Old Dominion Monarchs':'Sun Belt',
    'South Alabama Jaguars':'Sun Belt','Southern Miss Golden Eagles':'Sun Belt',
    'Texas State Bobcats':'Sun Belt','Troy Trojans':'Sun Belt',
    'Charlotte 49ers':'American Athletic','East Carolina Pirates':'American Athletic',
    'Florida Atlantic Owls':'American Athletic','Memphis Tigers':'American Athletic',
    'Rice Owls':'American Athletic','South Florida Bulls':'American Athletic',
    'Tulane Green Wave':'American Athletic','Tulsa Golden Hurricane':'American Athletic',
    'UAB Blazers':'American Athletic','Wichita State Shockers':'American Athletic','Navy Midshipmen':'American Athletic',
    'Indiana Hoosiers':'Big Ten','Illinois Fighting Illini':'Big Ten','Maryland Terrapins':'Big Ten',
    'Michigan Wolverines':'Big Ten','Michigan State Spartans':'Big Ten','Minnesota Golden Gophers':'Big Ten',
    'Nebraska Cornhuskers':'Big Ten','Northwestern Wildcats':'Big Ten','Ohio State Buckeyes':'Big Ten',
    'Penn State Nittany Lions':'Big Ten','Purdue Boilermakers':'Big Ten','Rutgers Scarlet Knights':'Big Ten',
    'Oregon State Beavers':'Pac-12','UCLA Bruins':'Pac-12','Washington Huskies':'Pac-12','Oregon Ducks':'Pac-12',
    'Air Force Falcons':'Mountain West','Fresno State Bulldogs':'Mountain West',
    'Nevada Wolf Pack':'Mountain West','UNLV Rebels':'Mountain West','Utah State Aggies':'Mountain West',
    'San Diego State Aztecs':'Mountain West','New Mexico Lobos':'Mountain West',
    'Wyoming Cowboys':'Mountain West','Boise State Broncos':'Mountain West',
    'Jacksonville State Gamecocks':'Conference USA','Liberty Flames':'Conference USA',
    'Middle Tennessee Blue Raiders':'Conference USA','New Mexico State Aggies':'Conference USA',
    'UTEP Miners':'Conference USA','Western Kentucky Hilltoppers':'Conference USA',
    'Sam Houston Bearkats':'Conference USA','FIU Panthers':'Conference USA',
    'Louisiana Tech Bulldogs':'Conference USA','UTSA Roadrunners':'Conference USA',
    'Kennesaw State Owls':'Conference USA','Florida International Panthers':'Conference USA',
    'California Baptist Lancers':'WAC','Cal Baptist Lancers':'WAC','Grand Canyon Antelopes':'WAC',
    'Sacramento State Hornets':'WAC','Tarleton State Texans':'WAC','Utah Tech Trailblazers':'WAC',
    'Utah Valley Wolverines':'WAC','Seattle Redhawks':'WAC','Southern Utah Thunderbirds':'WAC',
    'Abilene Christian Wildcats':'WAC',"Stephen F. Austin Lumberjacks":'WAC',
    'Bowling Green Falcons':'MAC','Ball State Cardinals':'MAC','Central Michigan Chippewas':'MAC',
    'Eastern Michigan Eagles':'MAC','Kent State Golden Flashes':'MAC','Miami RedHawks':'MAC',
    'Northern Illinois Huskies':'MAC','Ohio Bobcats':'MAC','Toledo Rockets':'MAC',
    'Western Michigan Broncos':'MAC','Akron Zips':'MAC','Buffalo Bulls':'MAC',
    'Dallas Baptist Patriots':'Missouri Valley','Illinois State Redbirds':'Missouri Valley',
    'Indiana State Sycamores':'Missouri Valley','Missouri State Bears':'Missouri Valley',
    'Southern Illinois Salukis':'Missouri Valley','Bradley Braves':'Missouri Valley',
    'Valparaiso Beacons':'Missouri Valley',
    'Morehead State Eagles':'Ohio Valley','Austin Peay Governors':'Ohio Valley',
    'Eastern Illinois Panthers':'Ohio Valley','Eastern Kentucky Colonels':'Ohio Valley',
    'Murray State Racers':'Ohio Valley','SE Missouri State Redhawks':'Ohio Valley',
    'Tennessee Tech Golden Eagles':'Ohio Valley','UT Martin Skyhawks':'Ohio Valley',
    'SIU Edwardsville Cougars':'Ohio Valley','Lindenwood Lions':'Ohio Valley','Bellarmine Knights':'Ohio Valley',
    'Central Arkansas Bears':'ASUN','Jacksonville Dolphins':'ASUN','Lipscomb Bisons':'ASUN',
    'North Alabama Lions':'ASUN','North Florida Ospreys':'ASUN','Northern Kentucky Norse':'ASUN',
    'Stetson Hatters':'ASUN','Queens Royals':'ASUN',
    'Campbell Fighting Camels':'Big South','Charleston Southern Buccaneers':'Big South',
    "Gardner-Webb Runnin' Bulldogs":'Big South','High Point Panthers':'Big South',
    'Longwood Lancers':'Big South','Presbyterian Blue Hose':'Big South',
    'Radford Highlanders':'Big South','UNC Asheville Bulldogs':'Big South',
    'USC Upstate Spartans':'Big South','Winthrop Eagles':'Big South',
    'ETSU Buccaneers':'Southern','East Tennessee State Buccaneers':'Southern',
    'Furman Paladins':'Southern','Mercer Bears':'Southern','Samford Bulldogs':'Southern',
    'The Citadel Bulldogs':'Southern','VMI Keydets':'Southern',
    'Western Carolina Catamounts':'Southern','Wofford Terriers':'Southern','Chattanooga Mocs':'Southern',
    'Bryant Bulldogs':'Northeast','Sacred Heart Pioneers':'Northeast','Wagner Seahawks':'Northeast',
    'Army Black Knights':'Patriot','Bucknell Bison':'Patriot','Holy Cross Crusaders':'Patriot',
    'Lafayette Leopards':'Patriot','Lehigh Mountain Hawks':'Patriot',
    'Fairfield Stags':'MAAC','Manhattan Jaspers':'MAAC','Marist Red Foxes':'MAAC',
    'Niagara Purple Eagles':'MAAC','Quinnipiac Bobcats':'MAAC','Rider Broncs':'MAAC','Siena Saints':'MAAC',
    'Albany Great Danes':'America East','Maine Black Bears':'America East',
    'Stony Brook Seawolves':'America East','Vermont Catamounts':'America East',
    'UMass Lowell River Hawks':'America East','Binghamton Bearcats':'America East',
}

CONF_COLOR = {
    'SEC':'#ff6b35','ACC':'#4361ee','Big 12':'#e63946','Sun Belt':'#06d6a0',
    'American Athletic':'#118ab2','Big Ten':'#cc0000','Pac-12':'#1d3557',
    'Mountain West':'#457b9d','Conference USA':'#6a0572','WAC':'#d4a017',
    'MAC':'#2d6a4f','Missouri Valley':'#e9c46a','Ohio Valley':'#6d6875',
    'ASUN':'#023e8a','Big South':'#780000','Southern':'#386641',
    'Northeast':'#7b2d8b','Patriot':'#1b4332','MAAC':'#9d4edd','America East':'#0077b6',
}

def get_conf_tier(t): return CONF_TIER.get(TEAM_CONF.get(t,''), CONF_DEFAULT)

def compute_elo(gdf):
    elo = {}
    for _, g in gdf.sort_values('date').dropna(subset=['date']).iterrows():
        ht, at = g['home_team'], g['away_team']
        eh = elo.get(ht, ELO_INIT) + (ELO_HOME if not g.get('neutral', False) else 0)
        ea = elo.get(at, ELO_INIT)
        exp = 1.0 / (1.0 + 10**((ea - eh)/400))
        actual = 1 if g['home_score'] > g['away_score'] else 0
        mult = np.log(abs(g['home_score']-g['away_score'])+1)*2.0
        tier = (get_conf_tier(ht)+get_conf_tier(at))/2
        delta = ELO_K*tier*mult*(actual-exp)
        elo[ht] = elo.get(ht, ELO_INIT)+delta
        elo[at] = elo.get(at, ELO_INIT)-delta
    return pd.DataFrame({'team':list(elo),'elo':list(elo.values())})

def compute_sos(gdf, edf):
    lk = edf.set_index('team')['elo']
    h = gdf[['season','home_team','away_team']].rename(columns={'home_team':'team','away_team':'opp'})
    a = gdf[['season','away_team','home_team']].rename(columns={'away_team':'team','home_team':'opp'})
    long = pd.concat([h,a],ignore_index=True)
    long['opp_elo'] = long['opp'].map(lk).fillna(ELO_INIT)
    return long.groupby(['team','season'])['opp_elo'].mean().reset_index().rename(columns={'opp_elo':'avg_opp_elo'})

POWER_TIER = 0.90
POWER_GAME_MIN = 12  # non-power-conf teams need this many games vs power opponents

def compute_form(gdf, n=15):
    h = gdf[['season','date','home_team','home_score','away_score']].copy(); h.columns=['season','date','team','rs','ra']
    a = gdf[['season','date','away_team','away_score','home_score']].copy(); a.columns=['season','date','team','rs','ra']
    long = pd.concat([h,a]).sort_values('date')
    long['win'] = (long['rs']>long['ra']).astype(int)
    long['rwp'] = long.groupby(['team','season'])['win'].transform(lambda x: x.rolling(n,min_periods=3).mean())
    return long.groupby(['team','season'])['rwp'].last().reset_index().rename(columns={'rwp':'recent_win_pct'})

def compute_power_games(gdf, year):
    g = gdf[gdf['season']==year][['home_team','away_team']].copy()
    h = g.rename(columns={'home_team':'team','away_team':'opp'})
    a = g.rename(columns={'away_team':'team','home_team':'opp'})
    long = pd.concat([h,a],ignore_index=True)
    long['opp_tier'] = long['opp'].apply(get_conf_tier)
    return long.groupby('team')['opp_tier'].apply(lambda x:(x>=POWER_TIER).sum()).reset_index(name='power_games')

def build_rankings(stats, edf, sdf, pgdf, year):
    df = stats[stats['season']==year].copy()
    df = df.merge(edf,on='team',how='left').merge(sdf[sdf['season']==year][['team','avg_opp_elo']],on='team',how='left')
    df = df.merge(pgdf,on='team',how='left')
    df['elo'] = df['elo'].fillna(ELO_INIT)
    df['avg_opp_elo'] = df['avg_opp_elo'].fillna(df['avg_opp_elo'].mean())
    df['power_games'] = df['power_games'].fillna(0).astype(int)
    def z(s): return (s-s.mean())/max(s.std(),1e-6)
    sc = pd.Series(0.0, index=df.index)
    for col,wt,hi in [('pythagorean_win_pct',.25,True),('avg_run_diff',.20,True),('elo',.20,True),
                       ('avg_runs_scored',.10,True),('avg_runs_allowed',.10,False),('avg_opp_elo',.15,True)]:
        if col in df.columns: sc += wt*(z(df[col]) if hi else -z(df[col]))
    df['power_score'] = sc
    df['conference'] = df['team'].map(TEAM_CONF).fillna('Unknown')
    df['own_tier'] = df['team'].apply(get_conf_tier)
    df['power_eligible'] = (df['own_tier'] >= POWER_TIER) | (df['power_games'] >= POWER_GAME_MIN)
    df['rank'] = df['power_score'].rank(ascending=False).astype(int)
    return df.sort_values('rank').reset_index(drop=True)

BASE = pathlib.Path(__file__).parent / 'data'
games = pd.read_parquet(BASE/'game_results_2021_2026.parquet')
team_stats = pd.read_parquet(BASE/'team_season_stats_2021_2026.parquet')
games['season'] = games['season'].astype(int)
team_stats['season'] = team_stats['season'].astype(int)

elo_df = compute_elo(games)
sos_df = compute_sos(games, elo_df)
form_df = compute_form(games)
power_df = compute_power_games(games, TEST_YEAR if TEST_YEAR in games['season'].unique() else int(games['season'].max()))
team_stats = team_stats.merge(form_df, on=['team','season'], how='left')
year = TEST_YEAR if TEST_YEAR in team_stats['season'].unique() else int(team_stats['season'].max())
rankings = build_rankings(team_stats, elo_df, sos_df, power_df, year)

def fv(v, d=None):
    if v is None or (isinstance(v, float) and np.isnan(v)): return d
    return float(round(v, 4))

teams_data = []
for _, r in rankings.iterrows():
    teams_data.append({
        'team': r['team'], 'conference': r.get('conference','Unknown'),
        'rank': int(r['rank']), 'wins': int(r.get('wins',0)), 'losses': int(r.get('losses',0)),
        'win_pct':               fv(r.get('win_pct'), 0),
        'pythagorean_win_pct':   fv(r.get('pythagorean_win_pct')),
        'avg_runs_scored':       fv(r.get('avg_runs_scored')),
        'avg_runs_allowed':      fv(r.get('avg_runs_allowed')),
        'avg_run_diff':          fv(r.get('avg_run_diff')),
        'elo':                   fv(r.get('elo', ELO_INIT)),
        'avg_opp_elo':           fv(r.get('avg_opp_elo', ELO_INIT)),
        'recent_win_pct':        fv(r.get('recent_win_pct')),
        'power_score':           fv(r.get('power_score')),
        'power_games':           int(r.get('power_games', 0)),
        'power_eligible':        bool(r.get('power_eligible', True)),
    })

TEAMS_JSON     = json.dumps(teams_data, ensure_ascii=False)
CONF_CLR_JSON  = json.dumps(CONF_COLOR, ensure_ascii=False)
YEAR_VAL       = year

# ── HTML ───────────────────────────────────────────────────────────────────────
_PGMIN  = int(POWER_GAME_MIN)
_TITLE  = f"College Baseball Analytics {year}"
_BADGE  = f"Season {year} &nbsp;&bull;&nbsp; Conference-adjusted Elo + SOS"
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>College Baseball Analytics {YEAR_VAL}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#07090f;color:#e2e8f0;font-family:"Segoe UI",Inter,system-ui,sans-serif;min-height:100vh}}
#header{{background:#0d1526;border-bottom:1px solid #1a2f52;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
#header h1{{font-size:18px;font-weight:700;color:#fff}}
#header .sub{{font-size:11px;color:#4a6080;margin-top:2px}}
#header .badge{{background:#0f1c36;border:1px solid #1a2f52;border-radius:20px;padding:4px 12px;font-size:11px;color:#06b6d4}}
#tab-nav{{background:#0d1526;border-bottom:1px solid #1a2f52;display:flex;padding:0 24px}}
.tab-btn{{background:none;border:none;border-bottom:2px solid transparent;color:#4a6080;cursor:pointer;font-size:13px;font-weight:500;padding:12px 18px;transition:.15s}}
.tab-btn:hover{{color:#e2e8f0}}
.tab-btn.active{{border-bottom-color:#06b6d4;color:#06b6d4}}
#content{{padding:20px 24px;max-width:1400px;margin:0 auto}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}
.kpi-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
.kpi{{background:#0f1c36;border:1px solid #1a2f52;border-radius:10px;padding:14px 18px;min-width:130px}}
.kpi-label{{color:#4a6080;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px}}
.kpi-value{{font-size:24px;font-weight:700;line-height:1.1}}
.kpi-sub{{color:#4a6080;font-size:11px;margin-top:3px}}
.filter-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;align-items:flex-end}}
.filter-group{{display:flex;flex-direction:column;gap:4px}}
.filter-label{{color:#4a6080;font-size:10px;text-transform:uppercase;letter-spacing:1px}}
select,input[type=text]{{background:#0f1c36;border:1px solid #243d65;border-radius:7px;color:#e2e8f0;font-size:13px;padding:8px 12px;outline:none}}
select:focus,input:focus{{border-color:#06b6d4}}
select option{{background:#0f1c36}}
.tbl-wrap{{overflow-x:auto;overflow-y:auto;max-height:520px;border-radius:8px;border:1px solid #1a2f52}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead{{position:sticky;top:0;z-index:10}}
th{{background:#1a2f52;color:#06b6d4;font-size:10px;font-weight:600;letter-spacing:1px;text-transform:uppercase;padding:10px 12px;white-space:nowrap;cursor:pointer;user-select:none}}
th:hover{{background:#243d65}}
td{{padding:8px 12px;border-bottom:1px solid #1a2f52;white-space:nowrap}}
tr:nth-child(even) td{{background:#0d1526}}
tr:hover td{{background:#1a2f52!important}}
.rank-gold{{border-left:3px solid #eab308}}
.conf-badge{{border-radius:4px;font-size:10px;font-weight:600;padding:2px 7px;color:#fff}}
.two-col{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}}
.two-col .col-main{{flex:1;min-width:300px}}
.two-col .col-side{{width:400px;flex-shrink:0}}
.gloss-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;margin-top:16px}}
.gloss-card{{background:#0f1c36;border:1px solid #1a2f52;border-radius:10px;padding:18px 20px}}
.gloss-card h3{{color:#06b6d4;font-size:13px;font-weight:700;letter-spacing:.5px;margin-bottom:8px;text-transform:uppercase}}
.gloss-card p,.gloss-card li{{color:#94a3b8;font-size:13px;line-height:1.6}}
.gloss-card ul{{padding-left:18px;margin-top:6px}}
.gloss-card li{{margin-bottom:4px}}
.tier-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
.tier-chip{{border-radius:4px;font-size:11px;font-weight:600;padding:3px 9px}}
@media(max-width:768px){{.two-col{{flex-direction:column}}.two-col .col-side{{width:100%}}#content{{padding:12px}}}}
</style>
</head>
<body>

<div id="header">
  <div>
    <h1>&#9918; College Baseball Analytics</h1>
    <div class="sub">Power Rankings &middot; Team Profiles &middot; Model Guide</div>
  </div>
  <div class="badge">Season {YEAR_VAL} &nbsp;&bull;&nbsp; Conference-adjusted Elo + SOS</div>
</div>

<div id="tab-nav">
  <button class="tab-btn active" onclick="showTab(event,'rankings')">Power Rankings</button>
  <button class="tab-btn" onclick="showTab(event,'profile')">Team Profile</button>
  <button class="tab-btn" onclick="showTab(event,'glossary')">Model Guide</button>
</div>

<div id="content">

<!-- RANKINGS -->
<div id="tab-rankings" class="tab-panel active">
  <div class="kpi-row" id="kpi-row"></div>
  <div class="filter-row">
    <div class="filter-group">
      <span class="filter-label">Conference</span>
      <select id="conf-filter" onchange="renderRankings()"></select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Search</span>
      <input type="text" id="team-search" placeholder="Filter teams..." oninput="renderRankings()" style="width:200px">
    </div>
  </div>
  <div class="two-col">
    <div class="col-main">
      <div class="tbl-wrap">
        <table><thead id="rthead"></thead><tbody id="rtbody"></tbody></table>
      </div>
    </div>
    <div class="col-side">
      <div id="top25-chart" style="height:540px"></div>
    </div>
  </div>
</div>

<!-- PROFILE -->
<div id="tab-profile" class="tab-panel">
  <div class="filter-row">
    <div class="filter-group">
      <span class="filter-label">Team</span>
      <select id="profile-sel" onchange="renderProfile(this.value)" style="min-width:280px"></select>
    </div>
  </div>
  <div id="profile-kpis" class="kpi-row"></div>
  <div id="profile-chart" style="height:380px"></div>
</div>

<!-- GLOSSARY -->
<div id="tab-glossary" class="tab-panel">
  <div class="gloss-grid">
    <div class="gloss-card">
      <h3>How the Power Score Works</h3>
      <p>Each team gets a composite score built from six metrics, all Z-score normalized so conferences with different run environments are compared fairly. Weights:</p>
      <ul>
        <li><strong>Pythagorean W%</strong> (25%) &mdash; expected win rate from runs scored vs allowed</li>
        <li><strong>Run Differential / G</strong> (20%) &mdash; average margin per game</li>
        <li><strong>Elo Rating</strong> (20%) &mdash; opponent-adjusted performance over all seasons</li>
        <li><strong>Runs Scored / G</strong> (10%) &mdash; offensive output</li>
        <li><strong>Runs Allowed / G</strong> (10%) &mdash; pitching / defense (lower is better)</li>
        <li><strong>SOS &mdash; Avg Opp Elo</strong> (15%) &mdash; strength of schedule</li>
      </ul>
    </div>
    <div class="gloss-card">
      <h3>Elo Rating System</h3>
      <p>Elo is a running skill rating updated after every game. Key adjustments:</p>
      <ul>
        <li><strong>Home advantage:</strong> +35 Elo points added to the home team before each game</li>
        <li><strong>Margin multiplier:</strong> log(|run diff| + 1) x 2 &mdash; bigger wins move the needle more</li>
        <li><strong>Conference tier K-factor:</strong> games between stronger conferences generate more Elo movement (SEC/ACC/Big 12 = 1.0x, down to Ohio Valley = 0.65x)</li>
        <li><strong>Starting value:</strong> 1500 for all teams before their first game</li>
      </ul>
    </div>
    <div class="gloss-card">
      <h3>Conference Tiers</h3>
      <p>Tiers scale the K-factor so low-major games generate less Elo movement per win:</p>
      <div class="tier-row">
        <span class="tier-chip" style="background:#ff6b3540;border:1px solid #ff6b35;color:#ff6b35">SEC 1.00</span>
        <span class="tier-chip" style="background:#4361ee40;border:1px solid #4361ee;color:#4361ee">ACC 1.00</span>
        <span class="tier-chip" style="background:#e6394640;border:1px solid #e63946;color:#e63946">Big 12 1.00</span>
        <span class="tier-chip" style="background:#06d6a040;border:1px solid #06d6a0;color:#06d6a0">Sun Belt 0.90</span>
        <span class="tier-chip" style="background:#118ab240;border:1px solid #118ab2;color:#118ab2">AAC 0.90</span>
        <span class="tier-chip" style="background:#cc000040;border:1px solid #cc0000;color:#cc0000">Big Ten 0.90</span>
        <span class="tier-chip" style="background:#6a057240;border:1px solid #6a0572;color:#c084fc">C-USA 0.78</span>
        <span class="tier-chip" style="background:#d4a01740;border:1px solid #d4a017;color:#d4a017">WAC 0.78</span>
        <span class="tier-chip" style="background:#6d687540;border:1px solid #6d6875;color:#9ca3af">OVC 0.65</span>
      </div>
    </div>
    <div class="gloss-card">
      <h3>Top-25 Eligibility (P-Conf G)</h3>
      <p>To appear in the Top 25 chart and headline ranking, a team must meet one of:</p>
      <ul>
        <li>Play in a power conference (tier &ge; 0.90) &mdash; auto-eligible</li>
        <li>OR have played at least <strong>{_PGMIN} games</strong> against power-conference opponents in the current season</li>
      </ul>
      <p style="margin-top:8px">This prevents mid-major teams that only beat cupcakes from inflating into the top 25. The <strong>P-Conf G</strong> column in the table shows each team's count. Ineligible rows are dimmed.</p>
    </div>
    <div class="gloss-card">
      <h3>Strength of Schedule (SOS)</h3>
      <p>SOS is the average Elo rating of all opponents faced in the current season. A higher number means a harder schedule. It accounts for 15% of the composite power score, so teams that only play weak opponents are penalized even if their win-loss record looks strong.</p>
    </div>
    <div class="gloss-card">
      <h3>Recent Form (L15)</h3>
      <p>Rolling win percentage over the last 15 games (minimum 3 games). Shown in Team Profile to highlight teams on a hot or cold streak that the season-aggregate stats may not yet reflect. Not included in the composite power score &mdash; use it as a qualitative check.</p>
    </div>
    <div class="gloss-card">
      <h3>Pythagorean Win %</h3>
      <p>Expected win percentage based on runs scored vs runs allowed, using the formula popularized by Bill James: RS^2 / (RS^2 + RA^2). It strips out luck in close games and is a better predictor of future performance than actual win percentage.</p>
    </div>
    <div class="gloss-card">
      <h3>Metric Quick Reference</h3>
      <ul>
        <li><strong>Elo</strong> &mdash; opponent-adjusted skill rating (higher = better)</li>
        <li><strong>SOS</strong> &mdash; avg Elo of opponents faced (higher = harder schedule)</li>
        <li><strong>Rd/G</strong> &mdash; run differential per game (+ is good)</li>
        <li><strong>Pythag W%</strong> &mdash; luck-adjusted win rate</li>
        <li><strong>RS/G</strong> &mdash; runs scored per game (offense)</li>
        <li><strong>RA/G</strong> &mdash; runs allowed per game (pitching/defense)</li>
        <li><strong>P-Conf G</strong> &mdash; games played vs power-conf opponents</li>
        <li><strong>Score</strong> &mdash; composite power score (Z-score weighted sum)</li>
      </ul>
    </div>
  </div>
</div>

</div><!-- #content -->

<script>
window.onerror = function(msg,src,line,col,err){{
  var b=document.createElement('div');
  b.style.cssText='position:fixed;top:0;left:0;right:0;padding:12px 20px;background:#7f1d1d;color:#fca5a5;font:13px monospace;z-index:9999;white-space:pre-wrap';
  b.textContent='JS ERROR: '+msg+'  (line '+line+')';
  document.body.prepend(b);
}};
const TEAMS    = {TEAMS_JSON};
const CONF_CLR = {CONF_CLR_JSON};
const SEASON   = {YEAR_VAL};
const ELO_INIT = {ELO_INIT};
const PGMIN    = {_PGMIN};

const BG   = '#0d1526';
const FONT = {{color:'#e2e8f0',family:'"Segoe UI",Inter,system-ui,sans-serif',size:12}};
const LB   = {{
  paper_bgcolor:BG,plot_bgcolor:BG,font:FONT,
  xaxis:{{gridcolor:'#1a2f52',linecolor:'#1a2f52',zerolinecolor:'#243d65'}},
  yaxis:{{gridcolor:'#1a2f52',linecolor:'#1a2f52',zerolinecolor:'#243d65'}},
  margin:{{l:14,r:14,t:38,b:14}},
  hoverlabel:{{bgcolor:'#0f1c36',bordercolor:'#1a2f52',font:{{color:'#e2e8f0'}}}},
}};
const CFG = {{displayModeBar:false,responsive:true}};

function getPct(col,val,higher){{
  if(val==null)return 0;
  const vals=TEAMS.map(t=>t[col]).filter(v=>v!=null);
  return Math.round(vals.filter(v=>higher?v<val:v>val).length/vals.length*100);
}}
function fmtSign(v,d){{d=d||2;return v==null?'--':(v>=0?'+':'')+v.toFixed(d);}}
function kpi(label,value,sub,color){{
  return '<div class="kpi"><div class="kpi-label">'+label+'</div>'
       + '<div class="kpi-value" style="color:'+(color||'#06b6d4')+'">'+value+'</div>'
       + '<div class="kpi-sub">'+(sub||'')+'</div></div>';
}}
function badge(conf){{
  var c=CONF_CLR[conf]||'#4a6080';
  return '<span class="conf-badge" style="background:'+c+'40;border:1px solid '+c+'">'+conf+'</span>';
}}

function showTab(e,name){{
  document.querySelectorAll('.tab-panel').forEach(function(p){{p.classList.remove('active');}});
  document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
  document.getElementById('tab-'+name).classList.add('active');
  e.currentTarget.classList.add('active');
  if(name==='rankings')renderRankings();
}}

var sortCol='rank', sortAsc=true;
var COLS=[
  {{k:'rank',      h:'#',         d:0}},
  {{k:'team',      h:'Team',      d:-1}},
  {{k:'conference',h:'Conf',      d:-2}},
  {{k:'wins',      h:'W',         d:0}},
  {{k:'losses',    h:'L',         d:0}},
  {{k:'pythagorean_win_pct',h:'Pythag W%',d:3}},
  {{k:'avg_run_diff',       h:'Rd/G',    d:2,sign:true}},
  {{k:'elo',               h:'Elo',     d:0}},
  {{k:'avg_opp_elo',       h:'SOS',     d:0}},
  {{k:'power_games',       h:'P-Conf G',d:0,pgcol:true}},
  {{k:'power_score',       h:'Score',   d:3}},
];

function sortBy(col){{
  if(sortCol===col)sortAsc=!sortAsc; else{{sortCol=col;sortAsc=(col==='rank');}}
  renderRankings();
}}

function renderRankings(){{
  var conf  = document.getElementById('conf-filter').value||'ALL';
  var srch  = (document.getElementById('team-search').value||'').toLowerCase();
  var data  = TEAMS.filter(function(t){{
    return (conf==='ALL'||t.conference===conf) && (!srch||t.team.toLowerCase().indexOf(srch)>=0);
  }});
  data = data.slice().sort(function(a,b){{
    var va=a[sortCol], vb=b[sortCol];
    if(va==null)return 1; if(vb==null)return -1;
    return sortAsc?(va<vb?-1:va>vb?1:0):(va>vb?-1:va<vb?1:0);
  }});

  document.getElementById('rthead').innerHTML = '<tr>'+COLS.map(function(c){{
    var arr = sortCol===c.k?(sortAsc?' ^':' v'):'';
    return '<th onclick="sortBy(\\''+c.k+'\\')">'+c.h+arr+'</th>';
  }}).join('')+'</tr>';

  document.getElementById('rtbody').innerHTML = data.map(function(t){{
    var gold = t.rank<=5?'rank-gold':'';
    var dim  = t.power_eligible?'':'opacity:0.65';
    var cells = COLS.map(function(c){{
      if(c.k==='team'){{
        var warn = t.power_eligible?'':' <span style="color:#f59e0b;font-size:10px" title="Fewer than '+PGMIN+' power-conf games">[!]</span>';
        return '<td><strong>'+t.team+'</strong>'+warn+'</td>';
      }}
      if(c.k==='conference') return '<td>'+badge(t.conference)+'</td>';
      if(c.pgcol){{
        var clr = t.power_eligible?'#22c55e':'#f59e0b';
        return '<td style="color:'+clr+'">'+t[c.k]+'</td>';
      }}
      var v=t[c.k];
      if(v==null)return '<td>--</td>';
      var txt = c.sign?fmtSign(v,c.d):(c.d===0?Math.round(v):v.toFixed(c.d));
      return '<td>'+txt+'</td>';
    }}).join('');
    return '<tr class="'+gold+'" style="'+dim+'">'+cells+'</tr>';
  }}).join('');

  var top25 = TEAMS.filter(function(t){{return t.power_eligible;}})
               .slice().sort(function(a,b){{return b.power_score-a.power_score;}})
               .slice(0,25).reverse();
  if(top25.length<2)return;
  Plotly.react('top25-chart',[{{
    type:'bar',orientation:'h',
    x:top25.map(function(t){{return t.power_score;}}),
    y:top25.map(function(t){{return t.team.split(' ').slice(-1)[0]+' ('+( t.conference||'?').split(' ')[0]+')'}}),
    marker:{{color:top25.map(function(t){{return CONF_CLR[t.conference]||'#06b6d4';}})}},
    text:top25.map(function(t){{return t.power_score.toFixed(3);}}),
    textposition:'outside',textfont:{{size:10,color:'#e2e8f0'}},
    hovertemplate:'<b>%{{y}}</b><br>Score: %{{x:.3f}}<extra></extra>',
  }}],Object.assign({{}},LB,{{
    title:{{text:'Top 25 Power Rankings (Eligible)',font:{{color:'#e2e8f0',size:13}}}},
    xaxis:Object.assign({{}},LB.xaxis,{{range:[top25[0].power_score-0.3,top25[top25.length-1].power_score+0.4]}}),
    yaxis:Object.assign({{}},LB.yaxis,{{automargin:true}}),
    margin:Object.assign({{}},LB.margin,{{l:160,r:60}}),
    height:540,
  }}),CFG);
}}

var PMETS=[
  {{col:'pythagorean_win_pct',label:'Pythagorean W%', higher:true, fmt:function(v){{return v.toFixed(3);}}}},
  {{col:'avg_runs_scored',    label:'Offense (RS/G)', higher:true, fmt:function(v){{return v.toFixed(2);}}}},
  {{col:'avg_runs_allowed',   label:'Defense (RA/G)', higher:false,fmt:function(v){{return v.toFixed(2);}}}},
  {{col:'avg_run_diff',       label:'Run Diff/G',     higher:true, fmt:function(v){{return fmtSign(v,2);}}}},
  {{col:'elo',                label:'Elo Rating',     higher:true, fmt:function(v){{return Math.round(v);}}}},
  {{col:'avg_opp_elo',        label:'SOS (Opp Elo)',  higher:true, fmt:function(v){{return Math.round(v);}}}},
  {{col:'recent_win_pct',     label:'Recent Form L15',higher:true, fmt:function(v){{return v.toFixed(3);}}}},
  {{col:'power_score',        label:'Power Score',    higher:true, fmt:function(v){{return v.toFixed(3);}}}},
];

function renderProfile(team){{
  if(!team)return;
  var t=TEAMS.find(function(x){{return x.team===team;}}); if(!t)return;
  var conf=t.conference||'Unknown';
  var cc=CONF_CLR[conf]||'#4a6080';
  var elig=t.power_eligible?'Eligible':'Ineligible (P25)';
  var ec=t.power_eligible?'#22c55e':'#f59e0b';
  document.getElementById('profile-kpis').innerHTML =
    kpi('Rank','#'+t.rank,'of '+TEAMS.length+' teams','#eab308')+
    kpi('Elo Rating',Math.round(t.elo),'init '+ELO_INIT,'#06b6d4')+
    kpi('Record',t.wins+'-'+t.losses,t.win_pct.toFixed(3)+' W%','#22c55e')+
    kpi('Power Score',(t.power_score||0).toFixed(3),'composite','#8b5cf6')+
    kpi('P-Conf Games',t.power_games,elig,ec)+
    '<div class="kpi"><div class="kpi-label">Conference</div>'
    +'<div class="kpi-value" style="font-size:15px;color:'+cc+'">'+conf+'</div></div>';

  var labels=[],vals=[],texts=[],bcolors=[];
  PMETS.forEach(function(m){{
    if(t[m.col]==null)return;
    var p=getPct(m.col,t[m.col],m.higher);
    labels.push(m.label); vals.push(p);
    texts.push(m.fmt(t[m.col])+'  (P'+p+')');
    bcolors.push(p>=75?'#22c55e':p>=50?'#06b6d4':p>=25?'#eab308':'#ef4444');
  }});

  Plotly.react('profile-chart',[{{
    type:'bar',orientation:'h',
    x:vals,y:labels,marker:{{color:bcolors}},
    text:texts,textposition:'outside',textfont:{{size:11,color:'#e2e8f0'}},
    width:0.55,
    hovertemplate:'<b>%{{y}}</b><br>Percentile: %{{x}}<extra></extra>',
  }}],Object.assign({{}},LB,{{
    title:{{text:team+' -- Percentile vs All D-I Teams',font:{{color:'#e2e8f0',size:13}}}},
    xaxis:Object.assign({{}},LB.xaxis,{{range:[0,132],tickvals:[0,25,50,75,100],ticktext:['0','25th','50th','75th','100th']}}),
    yaxis:Object.assign({{}},LB.yaxis,{{autorange:'reversed',automargin:true}}),
    margin:Object.assign({{}},LB.margin,{{l:160,r:80}}),
    height:380,
    shapes:[{{type:'line',x0:50,x1:50,y0:-0.5,y1:labels.length-0.5,line:{{color:'#4a6080',width:1,dash:'dash'}}}}],
  }}),CFG);
}}

(function init(){{
  var eligible = TEAMS.filter(function(t){{return t.power_eligible;}});
  var top = eligible.length ? eligible[0] : TEAMS[0];
  var avgSOS = TEAMS.reduce(function(s,t){{return s+(t.avg_opp_elo||0);}},0)/TEAMS.length;
  var avgRD  = TEAMS.reduce(function(s,t){{return s+(t.avg_run_diff||0);}},0)/TEAMS.length;

  document.getElementById('kpi-row').innerHTML =
    kpi('Top Ranked (P25)',top.team.split(' ').slice(-1)[0],top.conference,'#eab308')+
    kpi('Teams Ranked',TEAMS.length,'Season '+SEASON,'#06b6d4')+
    kpi('Avg Opp Elo',Math.round(avgSOS),'strength of schedule','#8b5cf6')+
    kpi('Avg Run Diff',fmtSign(avgRD,2),'per game','#22c55e');

  var confs=['ALL'];
  var seen={{}};
  TEAMS.forEach(function(t){{if(t.conference&&!seen[t.conference]){{seen[t.conference]=1;confs.push(t.conference);}}}});
  confs.sort(function(a,b){{return a==='ALL'?-1:b==='ALL'?1:a.localeCompare(b);}});
  var cf=document.getElementById('conf-filter');
  cf.innerHTML=confs.map(function(c){{
    return '<option value="'+c+'">'+(c==='ALL'?'All Conferences':c)+'</option>';
  }}).join('');

  var ps=document.getElementById('profile-sel');
  ps.innerHTML='<option value="">Select a team...</option>';
  TEAMS.forEach(function(t){{
    var o=document.createElement('option');
    o.value=t.team; o.textContent=t.team; ps.appendChild(o);
  }});

  renderRankings();
}})();
</script>
</body>
</html>"""

docs = pathlib.Path(__file__).parent / 'docs'
docs.mkdir(exist_ok=True)
out = docs / 'index.html'
out.write_text(HTML, encoding='utf-8')
print(f"Generated {out}  ({out.stat().st_size//1024} KB)")
