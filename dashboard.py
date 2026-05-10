"""
College Baseball Analytics Dashboard — Plotly Dash
Install: pip install dash dash-bootstrap-components plotly pandas numpy pyarrow
Run:     python dashboard.py  →  http://127.0.0.1:8050
"""
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ── palette ────────────────────────────────────────────────────────────────────
C = dict(
    bg='#07090f', panel='#0d1526', card='#0f1c36',
    border='#1a2f52', border2='#243d65',
    cyan='#06b6d4', purple='#8b5cf6', green='#22c55e',
    red='#ef4444', gold='#eab308',
    text='#e2e8f0', muted='#4a6080', white='#ffffff',
)

CHART_LAYOUT = dict(
    paper_bgcolor=C['panel'], plot_bgcolor=C['panel'],
    font=dict(color=C['text'], family='"Segoe UI", Inter, system-ui, sans-serif', size=12),
    colorway=[C['cyan'], C['purple'], C['green'], C['gold'], C['red']],
    xaxis=dict(gridcolor=C['border'], linecolor=C['border'], zerolinecolor=C['border2']),
    yaxis=dict(gridcolor=C['border'], linecolor=C['border'], zerolinecolor=C['border2']),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border']),
    margin=dict(l=12, r=12, t=36, b=12),
    hoverlabel=dict(bgcolor=C['card'], bordercolor=C['border'], font_color=C['text']),
)

# ── constants ──────────────────────────────────────────────────────────────────
ELO_K    = 30
ELO_HOME = 35
ELO_INIT = 1500
TEST_YEAR = 2026

CONF_TIER = {
    'SEC': 1.00, 'ACC': 1.00, 'Big 12': 1.00,
    'Sun Belt': 0.90, 'Mountain West': 0.90, 'Big Ten': 0.90,
    'Pac-12': 0.90, 'American Athletic': 0.90,
    'Conference USA': 0.78, 'WAC': 0.78, 'MAC': 0.78,
    'Missouri Valley': 0.78, 'Atlantic 10': 0.78,
    'Ohio Valley': 0.65, 'ASUN': 0.65, 'Big South': 0.65,
    'Southern': 0.65, 'Northeast': 0.65, 'Patriot': 0.65,
    'MAAC': 0.65, 'America East': 0.65, 'Southland': 0.65,
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
    'Syracuse Orange':'ACC','Stanford Cardinal':'ACC','Cal Bears':'ACC','SMU Mustangs':'ACC',
    'Baylor Bears':'Big 12','BYU Cougars':'Big 12','Cincinnati Bearcats':'Big 12',
    'Houston Cougars':'Big 12','Iowa State Cyclones':'Big 12','Kansas Jayhawks':'Big 12',
    'Kansas State Wildcats':'Big 12','Oklahoma State Cowboys':'Big 12',
    'TCU Horned Frogs':'Big 12','Texas Tech Red Raiders':'Big 12',
    'UCF Knights':'Big 12','West Virginia Mountaineers':'Big 12',
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
    'UAB Blazers':'American Athletic','Wichita State Shockers':'American Athletic',
    'Navy Midshipmen':'American Athletic',
    'Indiana Hoosiers':'Big Ten','Illinois Fighting Illini':'Big Ten',
    'Maryland Terrapins':'Big Ten','Michigan Wolverines':'Big Ten',
    'Michigan State Spartans':'Big Ten','Minnesota Golden Gophers':'Big Ten',
    'Nebraska Cornhuskers':'Big Ten','Northwestern Wildcats':'Big Ten',
    'Ohio State Buckeyes':'Big Ten','Penn State Nittany Lions':'Big Ten',
    'Purdue Boilermakers':'Big Ten','Rutgers Scarlet Knights':'Big Ten',
    'Oregon State Beavers':'Pac-12','UCLA Bruins':'Pac-12',
    'Washington Huskies':'Pac-12','Oregon Ducks':'Pac-12',
    'Air Force Falcons':'Mountain West','Fresno State Bulldogs':'Mountain West',
    'Nevada Wolf Pack':'Mountain West','UNLV Rebels':'Mountain West',
    'Utah State Aggies':'Mountain West','San Diego State Aztecs':'Mountain West',
    'New Mexico Lobos':'Mountain West','Wyoming Cowboys':'Mountain West',
    'Boise State Broncos':'Mountain West',
    'Jacksonville State Gamecocks':'Conference USA','Liberty Flames':'Conference USA',
    'Middle Tennessee Blue Raiders':'Conference USA',
    'New Mexico State Aggies':'Conference USA','UTEP Miners':'Conference USA',
    'Western Kentucky Hilltoppers':'Conference USA','Sam Houston Bearkats':'Conference USA',
    'FIU Panthers':'Conference USA','Louisiana Tech Bulldogs':'Conference USA',
    'UTSA Roadrunners':'Conference USA','Kennesaw State Owls':'Conference USA',
    'Florida International Panthers':'Conference USA',
    'California Baptist Lancers':'WAC','Cal Baptist Lancers':'WAC',
    'Grand Canyon Antelopes':'WAC','Sacramento State Hornets':'WAC',
    'Tarleton State Texans':'WAC','Utah Tech Trailblazers':'WAC',
    'Utah Valley Wolverines':'WAC','Seattle Redhawks':'WAC',
    'Southern Utah Thunderbirds':'WAC','Abilene Christian Wildcats':'WAC',
    "Stephen F. Austin Lumberjacks":'WAC',
    'Bowling Green Falcons':'MAC','Ball State Cardinals':'MAC',
    'Central Michigan Chippewas':'MAC','Eastern Michigan Eagles':'MAC',
    'Kent State Golden Flashes':'MAC','Miami RedHawks':'MAC',
    'Northern Illinois Huskies':'MAC','Ohio Bobcats':'MAC',
    'Toledo Rockets':'MAC','Western Michigan Broncos':'MAC',
    'Akron Zips':'MAC','Buffalo Bulls':'MAC',
    'Dallas Baptist Patriots':'Missouri Valley','Illinois State Redbirds':'Missouri Valley',
    'Indiana State Sycamores':'Missouri Valley','Missouri State Bears':'Missouri Valley',
    'Southern Illinois Salukis':'Missouri Valley','Bradley Braves':'Missouri Valley',
    'Valparaiso Beacons':'Missouri Valley',
    'Morehead State Eagles':'Ohio Valley','Austin Peay Governors':'Ohio Valley',
    'Eastern Illinois Panthers':'Ohio Valley','Eastern Kentucky Colonels':'Ohio Valley',
    'Murray State Racers':'Ohio Valley','SE Missouri State Redhawks':'Ohio Valley',
    'Tennessee Tech Golden Eagles':'Ohio Valley','UT Martin Skyhawks':'Ohio Valley',
    'SIU Edwardsville Cougars':'Ohio Valley','Lindenwood Lions':'Ohio Valley',
    'Bellarmine Knights':'Ohio Valley',
    'Central Arkansas Bears':'ASUN','Jacksonville Dolphins':'ASUN',
    'Lipscomb Bisons':'ASUN','North Alabama Lions':'ASUN',
    'North Florida Ospreys':'ASUN','Northern Kentucky Norse':'ASUN',
    'Stetson Hatters':'ASUN','Queens Royals':'ASUN',
    'Campbell Fighting Camels':'Big South','Charleston Southern Buccaneers':'Big South',
    "Gardner-Webb Runnin' Bulldogs":'Big South','High Point Panthers':'Big South',
    'Longwood Lancers':'Big South','Presbyterian Blue Hose':'Big South',
    'Radford Highlanders':'Big South','UNC Asheville Bulldogs':'Big South',
    'USC Upstate Spartans':'Big South','Winthrop Eagles':'Big South',
    'ETSU Buccaneers':'Southern','East Tennessee State Buccaneers':'Southern',
    'Furman Paladins':'Southern','Mercer Bears':'Southern',
    'Samford Bulldogs':'Southern','The Citadel Bulldogs':'Southern',
    'VMI Keydets':'Southern','Western Carolina Catamounts':'Southern',
    'Wofford Terriers':'Southern','Chattanooga Mocs':'Southern',
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

CONF_COLOR = {
    'SEC':'#ff6b35','ACC':'#4361ee','Big 12':'#e63946',
    'Sun Belt':'#06d6a0','American Athletic':'#118ab2','Big Ten':'#cc0000',
    'Pac-12':'#1d3557','Mountain West':'#457b9d','Conference USA':'#6a0572',
    'WAC':'#d4a017','MAC':'#2d6a4f','Missouri Valley':'#e9c46a',
    'Ohio Valley':'#6d6875','ASUN':'#023e8a','Big South':'#780000',
    'Southern':'#386641','Northeast':'#7b2d8b','Patriot':'#1b4332',
    'MAAC':'#9d4edd','America East':'#0077b6','Southland':'#e07a5f',
}

# ── data loading & computation ─────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent / 'data'

games      = pd.read_parquet(BASE / 'game_results_2021_2026.parquet')
team_stats = pd.read_parquet(BASE / 'team_season_stats_2021_2026.parquet')
games['season']      = games['season'].astype(int)
team_stats['season'] = team_stats['season'].astype(int)

def get_conf_tier(team):
    return CONF_TIER.get(TEAM_CONF.get(team, ''), CONF_DEFAULT)

def expected_elo_win(ea, eb):
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400))

def compute_elo(games_df):
    elo = {}
    for _, g in games_df.sort_values('date').dropna(subset=['date']).iterrows():
        ht, at = g['home_team'], g['away_team']
        eh  = elo.get(ht, ELO_INIT) + (ELO_HOME if not g.get('neutral', False) else 0)
        ea  = elo.get(at, ELO_INIT)
        exp = expected_elo_win(eh, ea)
        actual = 1 if g['home_score'] > g['away_score'] else 0
        mult   = np.log(abs(g['home_score'] - g['away_score']) + 1) * 2.0
        tier   = (get_conf_tier(ht) + get_conf_tier(at)) / 2
        delta  = ELO_K * tier * mult * (actual - exp)
        elo[ht] = elo.get(ht, ELO_INIT) + delta
        elo[at] = elo.get(at, ELO_INIT) - delta
    return pd.DataFrame({'team': list(elo.keys()), 'elo': list(elo.values())})

def compute_sos(games_df, elo_df):
    elo_lk = elo_df.set_index('team')['elo']
    home = games_df[['season','home_team','away_team']].rename(columns={'home_team':'team','away_team':'opp'})
    away = games_df[['season','away_team','home_team']].rename(columns={'away_team':'team','home_team':'opp'})
    long = pd.concat([home, away], ignore_index=True)
    long['opp_elo'] = long['opp'].map(elo_lk).fillna(ELO_INIT)
    return (long.groupby(['team','season'])['opp_elo'].mean().reset_index()
               .rename(columns={'opp_elo':'avg_opp_elo'}))

def compute_recent_form(games_df, n=15):
    home = games_df[['season','date','home_team','home_score','away_score']].copy()
    home.columns = ['season','date','team','rs','ra']
    away = games_df[['season','date','away_team','away_score','home_score']].copy()
    away.columns = ['season','date','team','rs','ra']
    long = pd.concat([home, away]).sort_values('date')
    long['win'] = (long['rs'] > long['ra']).astype(int)
    long['recent_win_pct'] = (long.groupby(['team','season'])['win']
                                  .transform(lambda x: x.rolling(n, min_periods=3).mean()))
    return long.groupby(['team','season'])['recent_win_pct'].last().reset_index()

def build_rankings(stats, elo_df, sos_df, year=TEST_YEAR):
    df = stats[stats['season'] == year].copy()
    df = df.merge(elo_df, on='team', how='left')
    df = df.merge(sos_df[sos_df['season'] == year][['team','avg_opp_elo']], on='team', how='left')
    df['elo']         = df['elo'].fillna(ELO_INIT)
    df['avg_opp_elo'] = df['avg_opp_elo'].fillna(df['avg_opp_elo'].mean())
    def z(s): return (s - s.mean()) / max(s.std(), 1e-6)
    score = pd.Series(0.0, index=df.index)
    for col, wt, higher in [
        ('pythagorean_win_pct', 0.25, True),
        ('avg_run_diff',        0.20, True),
        ('elo',                 0.20, True),
        ('avg_runs_scored',     0.10, True),
        ('avg_runs_allowed',    0.10, False),
        ('avg_opp_elo',         0.15, True),
    ]:
        if col in df.columns:
            score += wt * (z(df[col]) if higher else -z(df[col]))
    df['power_score'] = score
    df['conference']  = df['team'].map(TEAM_CONF).fillna('Unknown')
    df['rank']        = df['power_score'].rank(ascending=False).astype(int)
    return df.sort_values('rank').reset_index(drop=True)

print("Computing rankings…")
elo_df  = compute_elo(games)
sos_df  = compute_sos(games, elo_df)
form_df = compute_recent_form(games)
team_stats = team_stats.merge(form_df, on=['team','season'], how='left')

year = TEST_YEAR if TEST_YEAR in team_stats['season'].unique() else team_stats['season'].max()
rankings = build_rankings(team_stats, elo_df, sos_df, year=year)
ALL_TEAMS  = sorted(rankings['team'].tolist())
ALL_CONFS  = sorted(rankings['conference'].dropna().unique().tolist())
print(f"Ready — {len(rankings)} teams, season {year}")

# ── helpers ────────────────────────────────────────────────────────────────────
def percentile(col, val, higher=True):
    vals = rankings[col].dropna()
    if vals.empty or pd.isna(val): return 0
    return int(((vals < val).sum() / len(vals)) * 100) if higher else int(((vals > val).sum() / len(vals)) * 100)

def norm_0_1(col, val, higher=True):
    vals = rankings[col].dropna()
    if vals.empty or pd.isna(val): return 0.5
    mn, mx = vals.min(), vals.max()
    if mx == mn: return 0.5
    n = (val - mn) / (mx - mn)
    return n if higher else 1.0 - n

PROFILE_METRICS = [
    ('pythagorean_win_pct','Pythagorean W%', True,  '{:.3f}'),
    ('avg_runs_scored',    'Offense (RS/G)', True,  '{:.2f}'),
    ('avg_runs_allowed',   'Defense (RA/G)', False, '{:.2f}'),
    ('avg_run_diff',       'Run Diff/G',     True,  '{:+.2f}'),
    ('elo',                'Elo Rating',     True,  '{:.0f}'),
    ('avg_opp_elo',        'SOS (Opp Elo)',  True,  '{:.0f}'),
    ('recent_win_pct',     'Recent Form L15',True,  '{:.3f}'),
    ('power_score',        'Power Score',    True,  '{:.3f}'),
]

RADAR_COLS   = ['pythagorean_win_pct','avg_runs_scored','avg_runs_allowed',
                'avg_run_diff','elo','avg_opp_elo','recent_win_pct']
RADAR_LABELS = ['Pythag W%','Offense','Defense\n(inv)','Run Diff','Elo','SOS','Recent\nForm']
RADAR_HIGHER = [True, True, False, True, True, True, True]

# ── style helpers ──────────────────────────────────────────────────────────────
CARD_STYLE = dict(backgroundColor=C['card'], border=f"1px solid {C['border']}",
                  borderRadius='10px', padding='16px')

def kpi(title, value, sub='', color=None):
    color = color or C['cyan']
    return html.Div([
        html.P(title, style=dict(color=C['muted'], fontSize='10px', textTransform='uppercase',
                                 letterSpacing='1.5px', margin='0 0 4px')),
        html.Div(value, style=dict(color=color, fontSize='26px', fontWeight='700',
                                   lineHeight='1.1', letterSpacing='-0.5px')),
        html.P(sub, style=dict(color=C['muted'], fontSize='11px', margin='4px 0 0')),
    ], style={**CARD_STYLE, 'minWidth':'120px'})

def team_dd(id_, placeholder='Select team…'):
    return dcc.Dropdown(id=id_,
        options=[{'label': t, 'value': t} for t in ALL_TEAMS],
        placeholder=placeholder, clearable=False,
        style=dict(backgroundColor=C['card'], color=C['text'],
                   border=f'1px solid {C["border2"]}', borderRadius='8px'),
        className='dash-dropdown-dark')

# ── app ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
                suppress_callback_exceptions=True, title='⚾ CBB Analytics')

app.index_string = '''
<!DOCTYPE html><html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body,#react-entry-point{background:''' + C['bg'] + '''!important}
  .dash-dropdown-dark .Select-control{background:''' + C['card'] + '''!important;border-color:''' + C['border2'] + '''!important;color:''' + C['text'] + '''!important}
  .dash-dropdown-dark .Select-menu-outer,.dash-dropdown-dark .VirtualizedSelectFocusedOption{background:''' + C['card'] + '''!important}
  .dash-dropdown-dark .Select-value-label,.dash-dropdown-dark .Select-placeholder{color:''' + C['text'] + '''!important}
  .dash-dropdown-dark .Select-arrow{border-top-color:''' + C['muted'] + '''!important}
  .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th{background-color:''' + C['card'] + '''!important;color:''' + C['text'] + '''!important;border-color:''' + C['border'] + '''!important;font-size:13px}
  .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td{background-color:''' + C['border'] + '''!important}
  .dash-table-container .dash-spreadsheet-container table{border-collapse:collapse}
  .tab-selected{border-top:2px solid ''' + C['cyan'] + '''!important;background:''' + C['card'] + '''!important;color:''' + C['text'] + '''!important}
  .custom-tab{background:''' + C['panel'] + ''';color:''' + C['muted'] + ''';border:none;border-bottom:1px solid ''' + C['border'] + ''';padding:10px 18px;font-size:13px;font-weight:500;letter-spacing:.5px}
  .custom-tab:hover{color:''' + C['text'] + '''}
  .tabs-container{border-bottom:1px solid ''' + C['border'] + '''}
  input[type=text]{background:''' + C['card'] + '''!important;border-color:''' + C['border2'] + '''!important;color:''' + C['text'] + '''!important}
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
'''

# ── tab layouts ────────────────────────────────────────────────────────────────
def rankings_tab():
    return html.Div([
        # KPI row
        html.Div([
            kpi('Top Ranked', rankings.iloc[0]['team'].split()[0] + ' ' + rankings.iloc[0]['team'].split()[-1],
                rankings.iloc[0].get('conference',''), C['gold']),
            kpi('Teams Ranked', f"{len(rankings)}", f"Season {year}", C['cyan']),
            kpi('Avg Elo SOS', f"{rankings['avg_opp_elo'].mean():.0f}", 'Opp Elo', C['purple']),
            kpi('Avg Run Diff', f"{rankings['avg_run_diff'].mean():+.2f}", 'per game', C['green']),
        ], style=dict(display='flex', gap='12px', flexWrap='wrap', marginBottom='18px')),

        # Filter row
        html.Div([
            html.Div([
                html.Label('Conference', style=dict(color=C['muted'], fontSize='11px', marginBottom='4px')),
                dcc.Dropdown(id='conf-filter',
                    options=[{'label':'All Conferences','value':'ALL'}] +
                            [{'label':c,'value':c} for c in ALL_CONFS],
                    value='ALL', clearable=False,
                    style=dict(backgroundColor=C['card'], border=f'1px solid {C["border2"]}',
                               borderRadius='8px', minWidth='220px'),
                    className='dash-dropdown-dark'),
            ], style=dict(display='flex', flexDirection='column')),
        ], style=dict(display='flex', gap='12px', marginBottom='14px')),

        # Table + chart row
        html.Div([
            # DataTable
            html.Div([
                dash_table.DataTable(id='rankings-table',
                    columns=[
                        {'name':'#',           'id':'rank'},
                        {'name':'Team',        'id':'team'},
                        {'name':'Conf',        'id':'conference'},
                        {'name':'W',           'id':'wins'},
                        {'name':'L',           'id':'losses'},
                        {'name':'Pythag W%',   'id':'pythagorean_win_pct'},
                        {'name':'Rd/G',        'id':'avg_run_diff'},
                        {'name':'Elo',         'id':'elo'},
                        {'name':'SOS',         'id':'avg_opp_elo'},
                        {'name':'Score',       'id':'power_score'},
                    ],
                    style_table={'overflowY':'auto', 'maxHeight':'520px'},
                    style_header={'backgroundColor':C['border'], 'color':C['cyan'],
                                  'fontWeight':'600', 'fontSize':'11px',
                                  'textTransform':'uppercase', 'letterSpacing':'1px',
                                  'border':f'1px solid {C["border"]}'},
                    style_cell={'textAlign':'left', 'padding':'8px 12px', 'fontSize':'13px',
                                'backgroundColor':C['card'], 'color':C['text'],
                                'border':f'1px solid {C["border"]}'},
                    style_cell_conditional=[
                        {'if':{'column_id':'rank'},       'width':'42px',  'textAlign':'center'},
                        {'if':{'column_id':'wins'},        'width':'40px',  'textAlign':'center'},
                        {'if':{'column_id':'losses'},      'width':'40px',  'textAlign':'center'},
                        {'if':{'column_id':'conference'},  'width':'110px'},
                    ],
                    style_data_conditional=[
                        {'if':{'row_index':'odd'}, 'backgroundColor':C['panel']},
                        {'if':{'filter_query':'{rank} <= 5'},
                         'borderLeft':f'3px solid {C["gold"]}'},
                    ],
                    sort_action='native', page_size=50,
                    fixed_rows={'headers': True},
                ),
            ], style=dict(flex='1', minWidth='0')),

            # Top 25 bar chart
            html.Div([
                dcc.Graph(id='top25-chart', config={'displayModeBar':False},
                          style={'height':'560px'})
            ], style=dict(width='420px', flexShrink='0')),
        ], style=dict(display='flex', gap='16px', alignItems='flex-start')),
    ], style=dict(padding='18px'))


def profile_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label('Select Team', style=dict(color=C['muted'], fontSize='11px', marginBottom='4px')),
                team_dd('profile-team-dd'),
            ], style=dict(display='flex', flexDirection='column', maxWidth='400px')),
        ], style=dict(marginBottom='18px')),
        html.Div(id='profile-content',
                 children=html.P('Select a team above.', style=dict(color=C['muted']))),
    ], style=dict(padding='18px'))


def compare_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label('Team A', style=dict(color=C['cyan'], fontSize='11px', marginBottom='4px')),
                team_dd('compare-a-dd', 'Select Team A…'),
            ], style=dict(flex='1', display='flex', flexDirection='column')),
            html.Div([
                html.Label('Team B', style=dict(color=C['purple'], fontSize='11px', marginBottom='4px')),
                team_dd('compare-b-dd', 'Select Team B…'),
            ], style=dict(flex='1', display='flex', flexDirection='column')),
            html.Div([
                html.Label('​', style=dict(color='transparent', fontSize='11px', marginBottom='4px')),
                html.Button('Compare', id='compare-btn', n_clicks=0,
                            style=dict(backgroundColor=C['cyan'], color='#000', border='none',
                                       borderRadius='8px', padding='9px 22px',
                                       fontWeight='700', cursor='pointer', fontSize='13px')),
            ], style=dict(display='flex', flexDirection='column', justifyContent='flex-end')),
        ], style=dict(display='flex', gap='12px', marginBottom='18px', alignItems='flex-end')),
        html.Div(id='compare-content',
                 children=html.P('Select two teams above.', style=dict(color=C['muted']))),
    ], style=dict(padding='18px'))


def predictor_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label('Home Team', style=dict(color=C['green'], fontSize='11px', marginBottom='4px')),
                team_dd('pred-home-dd', 'Home team…'),
            ], style=dict(flex='1', display='flex', flexDirection='column')),
            html.Div([
                html.Label('Away Team', style=dict(color=C['red'], fontSize='11px', marginBottom='4px')),
                team_dd('pred-away-dd', 'Away team…'),
            ], style=dict(flex='1', display='flex', flexDirection='column')),
            html.Div([
                html.Label('Site', style=dict(color=C['muted'], fontSize='11px', marginBottom='4px')),
                dcc.Checklist(id='neutral-check', options=[{'label':'  Neutral site','value':'neutral'}],
                              value=[], style=dict(color=C['text'], fontSize='13px', marginTop='6px')),
            ], style=dict(display='flex', flexDirection='column')),
            html.Div([
                html.Label('​', style=dict(color='transparent', fontSize='11px', marginBottom='4px')),
                html.Button('Predict', id='predict-btn', n_clicks=0,
                            style=dict(backgroundColor=C['green'], color='#000', border='none',
                                       borderRadius='8px', padding='9px 24px',
                                       fontWeight='700', cursor='pointer', fontSize='13px')),
            ], style=dict(display='flex', flexDirection='column', justifyContent='flex-end')),
        ], style=dict(display='flex', gap='12px', marginBottom='20px', alignItems='flex-end')),
        html.Div(id='predictor-content',
                 children=html.P('Select teams and click Predict.', style=dict(color=C['muted']))),
    ], style=dict(padding='18px'))


# ── main layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(style=dict(backgroundColor=C['bg'], minHeight='100vh',
                                  fontFamily='"Segoe UI", Inter, system-ui, sans-serif'), children=[
    # Header
    html.Div([
        html.Div([
            html.Span('⚾', style=dict(fontSize='28px', marginRight='10px')),
            html.Div([
                html.H1('College Baseball Analytics',
                        style=dict(color=C['white'], fontSize='22px', fontWeight='700',
                                   margin='0', letterSpacing='-0.3px')),
                html.P(f'Power Rankings · Team Profiles · Comparison · Predictor  |  Season {year}',
                       style=dict(color=C['muted'], fontSize='12px', margin='1px 0 0')),
            ]),
        ], style=dict(display='flex', alignItems='center')),
        html.Div(f'Conference-adjusted Elo + SOS weighting',
                 style=dict(color=C['muted'], fontSize='11px', letterSpacing='.5px')),
    ], style=dict(backgroundColor=C['panel'], padding='14px 24px',
                  borderBottom=f'1px solid {C["border"]}',
                  display='flex', justifyContent='space-between', alignItems='center')),

    # Tabs
    dcc.Tabs(id='main-tabs', value='rankings',
             colors=dict(border=C['border'], primary=C['cyan'], background=C['panel']),
             children=[
        dcc.Tab(label='Power Rankings', value='rankings',
                className='custom-tab', selected_className='tab-selected'),
        dcc.Tab(label='Team Profile',   value='profile',
                className='custom-tab', selected_className='tab-selected'),
        dcc.Tab(label='Compare Teams',  value='compare',
                className='custom-tab', selected_className='tab-selected'),
        dcc.Tab(label='Game Predictor', value='predictor',
                className='custom-tab', selected_className='tab-selected'),
    ]),
    html.Div(id='tab-content'),
])


# ── callbacks ──────────────────────────────────────────────────────────────────
@app.callback(Output('tab-content','children'), Input('main-tabs','value'))
def render_tab(tab):
    return {'rankings':rankings_tab,'profile':profile_tab,
            'compare':compare_tab,'predictor':predictor_tab}[tab]()


@app.callback(Output('rankings-table','data'), Output('top25-chart','figure'),
              Input('conf-filter','value'))
def update_rankings(conf):
    df = rankings if conf == 'ALL' else rankings[rankings['conference'] == conf]
    rows = df[['rank','team','conference','wins','losses',
               'pythagorean_win_pct','avg_run_diff','elo','avg_opp_elo','power_score']].copy()
    rows['pythagorean_win_pct'] = rows['pythagorean_win_pct'].map('{:.3f}'.format)
    rows['avg_run_diff']        = rows['avg_run_diff'].map('{:+.2f}'.format)
    rows['elo']                 = rows['elo'].map('{:.0f}'.format)
    rows['avg_opp_elo']         = rows['avg_opp_elo'].map('{:.0f}'.format)
    rows['power_score']         = rows['power_score'].map('{:.3f}'.format)

    top = df.head(25)
    bar_colors = [CONF_COLOR.get(c, C['cyan']) for c in top['conference']]
    fig = go.Figure(go.Bar(
        x=top['power_score'][::-1], y=top['team'][::-1],
        orientation='h', marker_color=bar_colors[::-1],
        text=top['power_score'][::-1].map('{:.3f}'.format),
        textposition='outside', textfont=dict(color=C['text'], size=10),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>',
    ))
    fig.update_layout(**CHART_LAYOUT, title=dict(text='Top 25 Power Rankings',
                                                  font=dict(color=C['text'], size=13)),
                      xaxis_title='Power Score', yaxis_title=None,
                      xaxis=dict(range=[top['power_score'].min()-0.3, top['power_score'].max()+0.4]))
    return rows.to_dict('records'), fig


@app.callback(Output('profile-content','children'), Input('profile-team-dd','value'))
def update_profile(team):
    if not team: return html.P('Select a team above.', style=dict(color=C['muted']))
    row = rankings[rankings['team'] == team]
    if row.empty: return html.P('No data.', style=dict(color=C['muted']))
    r = row.iloc[0]
    conf  = r.get('conference','Unknown')
    tier  = CONF_TIER.get(conf, CONF_DEFAULT)
    wins  = int(r.get('wins', 0))
    losses= int(r.get('losses', 0))
    rank  = int(r['rank'])
    conf_color = CONF_COLOR.get(conf, C['muted'])

    # KPI row
    kpi_row = html.Div([
        kpi('Rank', f'#{rank}', f'of {len(rankings)} teams', C['gold']),
        kpi('Elo Rating', f"{r['elo']:.0f}", f'Init: {ELO_INIT}', C['cyan']),
        kpi('Record', f'{wins}–{losses}', f'{r.get("win_pct",0):.3f} W%', C['green']),
        kpi('Power Score', f"{r['power_score']:.3f}", 'composite', C['purple']),
        html.Div([
            html.P('Conference', style=dict(color=C['muted'], fontSize='10px',
                                             textTransform='uppercase', letterSpacing='1.5px',
                                             margin='0 0 4px')),
            html.Div(conf, style=dict(color=conf_color, fontSize='15px', fontWeight='700')),
            html.P(f'Tier mult: {tier:.2f}', style=dict(color=C['muted'], fontSize='11px', margin='4px 0 0')),
        ], style={**CARD_STYLE, 'minWidth':'150px'}),
    ], style=dict(display='flex', gap='12px', flexWrap='wrap', marginBottom='18px'))

    # Percentile bars chart
    labels, values, pcts, fmts = [], [], [], []
    for col, lbl, higher, fmt in PROFILE_METRICS:
        val = r.get(col)
        if pd.isna(val): continue
        p = percentile(col, val, higher)
        labels.append(lbl)
        values.append(p)
        pcts.append(f'{fmt.format(val)}  (P{p})')
        fmts.append(fmt)

    bar_colors_p = [C['green'] if v >= 75 else C['cyan'] if v >= 50 else
                    C['gold'] if v >= 25 else C['red'] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=bar_colors_p, text=pcts,
        textposition='outside', textfont=dict(color=C['text'], size=11),
        hovertemplate='<b>%{y}</b><br>Percentile: %{x}<extra></extra>',
        width=0.55,
    ))
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=f'{team} — Percentile Rankings vs All D-I Teams',
                   font=dict(color=C['text'], size=13)),
        xaxis=dict(range=[0, 130], tickvals=[0,25,50,75,100],
                   ticktext=['0','25th','50th','75th','100th'],
                   gridcolor=C['border']),
        yaxis=dict(autorange='reversed'),
        height=380,
    )
    fig.add_vline(x=50, line_dash='dash', line_color=C['muted'], line_width=1)

    return html.Div([kpi_row, dcc.Graph(figure=fig, config={'displayModeBar':False})])


@app.callback(Output('compare-content','children'),
              Input('compare-btn','n_clicks'),
              State('compare-a-dd','value'), State('compare-b-dd','value'),
              prevent_initial_call=True)
def update_compare(_, team_a, team_b):
    if not team_a or not team_b:
        return html.P('Select both teams.', style=dict(color=C['muted']))
    ra = rankings[rankings['team'] == team_a]
    rb = rankings[rankings['team'] == team_b]
    if ra.empty or rb.empty:
        return html.P('Team not found in rankings.', style=dict(color=C['muted']))
    ra, rb = ra.iloc[0], rb.iloc[0]

    # H2H
    h2h = games[((games['home_team']==team_a)&(games['away_team']==team_b)) |
                ((games['home_team']==team_b)&(games['away_team']==team_a))].copy()
    a_w = (((h2h['home_team']==team_a)&(h2h['home_score']>h2h['away_score'])) |
            ((h2h['away_team']==team_a)&(h2h['away_score']>h2h['home_score']))).sum()
    b_w = len(h2h) - a_w

    # Radar
    vals_a = [norm_0_1(c, ra.get(c, np.nan), h) for c, h in zip(RADAR_COLS, RADAR_HIGHER)]
    vals_b = [norm_0_1(c, rb.get(c, np.nan), h) for c, h in zip(RADAR_COLS, RADAR_HIGHER)]
    angles = RADAR_LABELS + [RADAR_LABELS[0]]
    va     = vals_a + [vals_a[0]]
    vb     = vals_b + [vals_b[0]]

    radar = go.Figure()
    for vals, name, color, fill in [(va, team_a, C['cyan'], 'toself'),
                                     (vb, team_b, C['purple'], 'toself')]:
        radar.add_trace(go.Scatterpolar(
            r=vals, theta=angles, name=name.split()[0]+' '+name.split()[-1],
            fill=fill, line=dict(color=color, width=2),
            fillcolor=color.replace(')',',0.12)').replace('rgb','rgba') if color.startswith('rgb') else color+'1e',
            marker=dict(color=color, size=5),
        ))
    radar.update_layout(**{**CHART_LAYOUT,
        'polar': dict(
            bgcolor=C['card'],
            radialaxis=dict(visible=True, range=[0,1], tickvals=[.25,.5,.75],
                            ticktext=['25th','50th','75th'], tickfont=dict(size=9,color=C['muted']),
                            gridcolor=C['border'], linecolor=C['border']),
            angularaxis=dict(tickfont=dict(size=10,color=C['text']), gridcolor=C['border']),
        ),
        'legend': dict(orientation='h', y=-0.08, x=0.5, xanchor='center',
                       bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        'title': dict(text='Percentile Comparison', font=dict(color=C['text'], size=13)),
        'height': 400,
    })

    # Metric table
    table_rows = []
    for col, lbl, higher, fmt in PROFILE_METRICS:
        va_val = ra.get(col, np.nan)
        vb_val = rb.get(col, np.nan)
        if pd.isna(va_val) and pd.isna(vb_val): continue
        a_wins = not pd.isna(va_val) and not pd.isna(vb_val) and ((va_val > vb_val) if higher else (va_val < vb_val))
        b_wins = not pd.isna(va_val) and not pd.isna(vb_val) and ((vb_val > va_val) if higher else (vb_val < va_val))
        fa = fmt.format(va_val) if not pd.isna(va_val) else '–'
        fb = fmt.format(vb_val) if not pd.isna(vb_val) else '–'
        table_rows.append(html.Tr([
            html.Td(lbl, style=dict(color=C['muted'], fontSize='12px', padding='7px 12px')),
            html.Td(fa,  style=dict(color=C['cyan']   if a_wins else C['text'],
                                    fontWeight='700' if a_wins else '400',
                                    fontSize='13px', padding='7px 12px', textAlign='right')),
            html.Td('◀' if a_wins else ('▶' if b_wins else '–'),
                    style=dict(color=C['gold'], textAlign='center', padding='7px 6px', fontSize='11px')),
            html.Td(fb,  style=dict(color=C['purple'] if b_wins else C['text'],
                                    fontWeight='700' if b_wins else '400',
                                    fontSize='13px', padding='7px 12px', textAlign='right')),
        ]))

    metric_table = html.Table([
        html.Thead(html.Tr([
            html.Th('Metric', style=dict(color=C['muted'], fontSize='10px', textTransform='uppercase',
                                         letterSpacing='1px', padding='8px 12px')),
            html.Th(team_a.split()[-1], style=dict(color=C['cyan'], fontSize='11px',
                                                    textAlign='right', padding='8px 12px')),
            html.Th('', style=dict(width='24px')),
            html.Th(team_b.split()[-1], style=dict(color=C['purple'], fontSize='11px',
                                                     textAlign='right', padding='8px 12px')),
        ])),
        html.Tbody(table_rows),
    ], style=dict(width='100%', borderCollapse='collapse'))

    h2h_card = html.Div([
        html.Span('H2H (all seasons):  ', style=dict(color=C['muted'], fontSize='12px')),
        html.Span(f'{team_a.split()[-1]} ', style=dict(color=C['cyan'], fontWeight='700')),
        html.Span(f'{a_w}–{b_w}', style=dict(color=C['text'], fontWeight='700', fontSize='15px')),
        html.Span(f' {team_b.split()[-1]}', style=dict(color=C['purple'], fontWeight='700')),
        html.Span(f'  ({len(h2h)} games recorded)', style=dict(color=C['muted'], fontSize='11px')),
    ], style={**CARD_STYLE, 'marginBottom':'14px'})

    return html.Div([
        h2h_card,
        html.Div([
            html.Div([dcc.Graph(figure=radar, config={'displayModeBar':False})],
                     style=dict(flex='1')),
            html.Div([html.Div(metric_table, style={**CARD_STYLE, 'padding':'8px'})],
                     style=dict(flex='1')),
        ], style=dict(display='flex', gap='16px')),
    ])


@app.callback(Output('predictor-content','children'),
              Input('predict-btn','n_clicks'),
              State('pred-home-dd','value'), State('pred-away-dd','value'),
              State('neutral-check','value'),
              prevent_initial_call=True)
def update_predictor(_, home, away, neutral):
    if not home or not away:
        return html.P('Select both teams.', style=dict(color=C['muted']))
    elo_lk = elo_df.set_index('team')['elo']
    is_neutral = 'neutral' in (neutral or [])
    ea = elo_lk.get(home, ELO_INIT) + (0 if is_neutral else ELO_HOME)
    eb = elo_lk.get(away, ELO_INIT)
    wp_home = expected_elo_win(ea, eb)
    wp_away = 1 - wp_home

    rh = rankings[rankings['team'] == home].iloc[0] if home in rankings['team'].values else None
    ra = rankings[rankings['team'] == away].iloc[0] if away in rankings['team'].values else None

    # Simple run diff estimate from pythagorean and run diff gap
    rd_est = 0.0
    if rh is not None and ra is not None:
        rd_est = float(rh.get('avg_run_diff', 0)) - float(ra.get('avg_run_diff', 0))
        rd_est *= 0.4  # regress toward 0

    fav    = home if wp_home >= 0.5 else away
    wp_fav = max(wp_home, wp_away)
    color  = C['green'] if wp_fav > 0.65 else C['gold'] if wp_fav > 0.55 else C['muted']

    # Probability bar
    prob_bar = go.Figure(go.Bar(
        x=[wp_home * 100, wp_away * 100],
        y=[home.split()[-1], away.split()[-1]],
        orientation='h',
        marker_color=[C['green'], C['red']],
        text=[f'{wp_home:.1%}', f'{wp_away:.1%}'],
        textposition='inside', textfont=dict(color='#000', size=14, family='Inter'),
        hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra></extra>',
        width=0.5,
    ))
    prob_bar.update_layout(**CHART_LAYOUT,
        xaxis=dict(range=[0,100], tickvals=[0,25,50,75,100],
                   ticktext=['0%','25%','50%','75%','100%'],
                   gridcolor=C['border']),
        yaxis=dict(autorange='reversed'),
        title=dict(text='Win Probability (Elo-based)',
                   font=dict(color=C['text'], size=13)),
        height=200,
        showlegend=False,
    )
    prob_bar.add_vline(x=50, line_dash='dash', line_color=C['muted'], line_width=1)

    # Detail cards
    elo_h = elo_lk.get(home, ELO_INIT)
    elo_a = elo_lk.get(away, ELO_INIT)
    rank_h = int(rh['rank']) if rh is not None else '–'
    rank_a = int(ra['rank']) if ra is not None else '–'

    detail = html.Div([
        kpi('Favored', fav.split()[-1], f'{wp_fav:.1%} win prob', color),
        kpi('Pred Margin', f'{abs(rd_est):+.1f}' if rd_est != 0 else 'Even',
            f'favors {fav.split()[-1]}' if abs(rd_est) > 0.1 else 'pick\'em', C['cyan']),
        kpi(f'{home.split()[-1]} Elo', f'{elo_h:.0f}', f'Rank #{rank_h}', C['green']),
        kpi(f'{away.split()[-1]} Elo', f'{elo_a:.0f}', f'Rank #{rank_a}', C['red']),
        kpi('Elo Gap', f'{abs(elo_h - elo_a):.0f}', 'points', C['muted']),
    ], style=dict(display='flex', gap='12px', flexWrap='wrap', marginBottom='16px'))

    matchup_label = html.Div([
        html.Span(home, style=dict(color=C['green'], fontWeight='700', fontSize='18px')),
        html.Span(' vs ', style=dict(color=C['muted'], fontSize='15px', margin='0 8px')),
        html.Span(away, style=dict(color=C['red'], fontWeight='700', fontSize='18px')),
        html.Span('  (Neutral site)' if is_neutral else '  (Home/Away)',
                  style=dict(color=C['muted'], fontSize='12px')),
    ], style=dict(marginBottom='14px'))

    return html.Div([matchup_label, detail,
                     dcc.Graph(figure=prob_bar, config={'displayModeBar':False})])


if __name__ == '__main__':
    app.run(debug=True, port=8050)
