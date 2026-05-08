import json

def code_cell(src):
    if isinstance(src, list):
        source = src
    else:
        source = [line + "\n" for line in src.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def md_cell(src):
    if isinstance(src, list):
        source = src
    else:
        source = [line + "\n" for line in src.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md_cell("""# College Baseball Analytics — Power Rankings & Game Predictor
**Sections:** Setup → Features → Elo Rankings → Run-Diff Model → Win Probability → Ensemble → Game Predictor → Betting → Kelly

*Designed for Google Colab. Training data: 2021–2025 NCAA Division I.*"""))

# ── SECTION 1: SETUP & DATA LOADING ───────────────────────────────────────────
cells.append(md_cell("## Section 1 — Setup & Data Loading"))

cells.append(code_cell("""\
# Install packages
!pip install -q git+https://github.com/nathanblumenfeld/collegebaseball.git
!pip install -q git+https://github.com/CodeMateo15/CollegeBaseballStatsPackage.git
!pip install -q xgboost scikit-learn fastai requests beautifulsoup4 lxml pyarrow"""))

cells.append(code_cell("""\
import warnings, os, json, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import xgboost as xgb
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS   = [2021, 2022, 2023, 2024]
TEST_YEAR     = 2025
ALL_YEARS     = TRAIN_YEARS + [TEST_YEAR]
DIVISION      = 1
STARTING_BANKROLL = 1000.0
KELLY_FRACTION    = 0.25
KELLY_CAP         = 0.10   # max bet as fraction of bankroll

# wOBA linear weights (college baseball approximation, from The Book)
WOBA_WEIGHTS = dict(uBB=0.690, HBP=0.722, single=0.888,
                    double=1.271, triple=1.616, HR=2.101)
WOBA_SCALE   = 0.320   # league-average wOBA target for normalisation

print("Imports complete.")"""))

cells.append(code_cell("""\
# Mount Google Drive and set cache path
from google.colab import drive
drive.mount('/content/drive')

CACHE_DIR = '/content/drive/MyDrive/CollegeBaseballAnalytics/'
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")"""))

# ── DATA LOADING: TEAM SEASON STATS ───────────────────────────────────────────
cells.append(code_cell("""\
# Pull D1 team batting + pitching season stats via collegebaseball package
# Falls back to direct stats.ncaa.org scraping on error
try:
    from collegebaseball import ncaa_scraper
    _scraper_available = True
except ImportError:
    _scraper_available = False
    print("collegebaseball not available — will use CollegeBaseballStatsPackage")

BATTING_CACHE  = CACHE_DIR + 'team_batting_2021_2025.parquet'
PITCHING_CACHE = CACHE_DIR + 'team_pitching_2021_2025.parquet'

def _pull_via_scraper():
    bat_frames, pit_frames = [], []
    for yr in ALL_YEARS:
        try:
            b = ncaa_scraper.ncaa_team_batting(year=yr, division=DIVISION)
            b['season'] = yr
            bat_frames.append(b)
            p = ncaa_scraper.ncaa_team_pitching(year=yr, division=DIVISION)
            p['season'] = yr
            pit_frames.append(p)
            print(f"  {yr}: {len(b)} teams batting, {len(p)} teams pitching")
        except Exception as e:
            print(f"  {yr} error: {e}")
        time.sleep(0.5)
    return pd.concat(bat_frames, ignore_index=True), pd.concat(pit_frames, ignore_index=True)

def _pull_via_stats_package():
    try:
        from ncaa_bbStats import get_team_stats
        frames = []
        for yr in ALL_YEARS:
            try:
                df = get_team_stats(year=yr, division='D1')
                df['season'] = yr
                frames.append(df)
                print(f"  {yr}: {len(df)} teams")
            except Exception as e:
                print(f"  {yr} error: {e}")
        return pd.concat(frames, ignore_index=True)
    except ImportError:
        return None

if os.path.exists(BATTING_CACHE) and os.path.exists(PITCHING_CACHE):
    batting_raw  = pd.read_parquet(BATTING_CACHE)
    pitching_raw = pd.read_parquet(PITCHING_CACHE)
    print(f"Loaded from cache: {len(batting_raw)} batting rows, {len(pitching_raw)} pitching rows")
elif _scraper_available:
    print("Pulling from stats.ncaa.org …")
    batting_raw, pitching_raw = _pull_via_scraper()
    batting_raw.to_parquet(BATTING_CACHE)
    pitching_raw.to_parquet(PITCHING_CACHE)
else:
    alt = _pull_via_stats_package()
    if alt is not None:
        batting_raw = alt
        pitching_raw = alt.copy()
    batting_raw.to_parquet(BATTING_CACHE)
    pitching_raw.to_parquet(PITCHING_CACHE)

batting_raw.head(3)"""))

# ── SECTION 1b: FEATURE ENGINEERING ───────────────────────────────────────────
cells.append(md_cell("## Section 1b — Advanced Feature Engineering"))

cells.append(code_cell("""\
# Standardise column names across sources
def normalise_batting_cols(df):
    rename = {
        'Team': 'team', 'School': 'team', 'school': 'team',
        'G': 'G', 'AB': 'AB', 'H': 'H', '2B': 'dbl', '3B': 'trpl',
        'HR': 'HR', 'BB': 'BB', 'HBP': 'HBP', 'SF': 'SF',
        'K': 'SO', 'SO': 'SO', 'R': 'R', 'RBI': 'RBI',
        'SB': 'SB', 'CS': 'CS', 'IBB': 'IBB',
    }
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

def normalise_pitching_cols(df):
    rename = {
        'Team': 'team', 'School': 'team', 'school': 'team',
        'ERA': 'ERA', 'IP': 'IP', 'H': 'H_allowed', 'R': 'R_allowed',
        'ER': 'ER', 'BB': 'BB_pit', 'SO': 'K_pit', 'K': 'K_pit',
        'HR': 'HR_allowed', 'HBP': 'HBP_pit', 'BF': 'BF',
        'WHIP': 'WHIP', 'G': 'G_pit',
    }
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

batting_raw  = normalise_batting_cols(batting_raw)
pitching_raw = normalise_pitching_cols(pitching_raw)
print("Columns normalised.")
print("Batting cols:", batting_raw.columns.tolist())
print("Pitching cols:", pitching_raw.columns.tolist())"""))

cells.append(code_cell("""\
# ── Advanced batting metrics ───────────────────────────────────────────────────
def compute_batting_metrics(df):
    df = df.copy()
    for c in ['AB','H','dbl','trpl','HR','BB','HBP','SF','SO','IBB','R','G']:
        if c not in df.columns:
            df[c] = 0
    df['IBB'] = df.get('IBB', pd.Series(0, index=df.index)).fillna(0)
    df['PA']      = df['AB'] + df['BB'] + df['HBP'] + df['SF']
    df['1B']      = df['H'] - df['dbl'] - df['trpl'] - df['HR']
    df['uBB']     = df['BB'] - df['IBB']
    df['wOBA']    = (WOBA_WEIGHTS['uBB']*df['uBB']
                   + WOBA_WEIGHTS['HBP']*df['HBP']
                   + WOBA_WEIGHTS['single']*df['1B']
                   + WOBA_WEIGHTS['double']*df['dbl']
                   + WOBA_WEIGHTS['triple']*df['trpl']
                   + WOBA_WEIGHTS['HR']*df['HR']
                    ) / (df['AB'] + df['uBB'] + df['HBP'] + df['SF']).replace(0, np.nan)
    df['ISO']     = (df['dbl'] + 2*df['trpl'] + 3*df['HR']) / df['AB'].replace(0, np.nan)
    df['K_pct']   = df['SO'] / df['PA'].replace(0, np.nan)
    df['BB_pct']  = df['BB'] / df['PA'].replace(0, np.nan)
    df['OBP']     = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF']).replace(0, np.nan)
    df['SLG']     = (df['1B'] + 2*df['dbl'] + 3*df['trpl'] + 4*df['HR']) / df['AB'].replace(0, np.nan)
    df['OPS']     = df['OBP'] + df['SLG']
    df['RPG']     = df['R'] / df['G'].replace(0, np.nan)
    # wRC+ (normalised wOBA relative to division average — computed per season)
    return df

batting_feat = compute_batting_metrics(batting_raw)

# Normalise wOBA to league average per season
def add_wrc_plus(df):
    df = df.copy()
    for yr in df['season'].unique():
        mask = df['season'] == yr
        lg_woba = df.loc[mask, 'wOBA'].median()
        df.loc[mask, 'wRC_plus'] = (df.loc[mask, 'wOBA'] - lg_woba) / lg_woba * 100 + 100
    return df

batting_feat = add_wrc_plus(batting_feat)
print("Batting features computed.")
batting_feat[['team','season','wOBA','ISO','K_pct','BB_pct','OPS','wRC_plus','RPG']].head()"""))

cells.append(code_cell("""\
# ── Advanced pitching metrics (FIP, K-BB%, DER approximation) ─────────────────
def compute_pitching_metrics(df):
    df = df.copy()
    for c in ['IP','K_pit','BB_pit','HR_allowed','HBP_pit','ER','ERA','G_pit']:
        if c not in df.columns:
            df[c] = 0

    # FIP constant: solve so that league FIP == league ERA
    def fip_constant(sub):
        ip    = sub['IP'].sum()
        if ip == 0: return 3.10
        lg_era = sub['ER'].sum() / ip * 9
        raw    = (13*sub['HR_allowed'].sum() + 3*(sub['BB_pit'].sum()+sub['HBP_pit'].sum())
                  - 2*sub['K_pit'].sum()) / ip
        return lg_era - raw

    df['FIP_raw'] = (13*df['HR_allowed'] + 3*(df['BB_pit']+df['HBP_pit'])
                     - 2*df['K_pit']) / df['IP'].replace(0, np.nan)
    # Per-season FIP constant
    for yr in df['season'].unique():
        mask = df['season'] == yr
        c = fip_constant(df[mask])
        df.loc[mask, 'FIP'] = df.loc[mask, 'FIP_raw'] + c

    df['K9']      = df['K_pit'] / df['IP'].replace(0, np.nan) * 9
    df['BB9']     = df['BB_pit'] / df['IP'].replace(0, np.nan) * 9
    df['HR9']     = df['HR_allowed'] / df['IP'].replace(0, np.nan) * 9
    df['K_BB_pct']= (df['K_pit'] - df['BB_pit']) / (df['IP'].replace(0, np.nan) / 9 * 27)
    df['RA9']     = df['ER'] / df['IP'].replace(0, np.nan) * 9

    # ERA- (ERA relative to league; lower is better)
    for yr in df['season'].unique():
        mask = df['season'] == yr
        lg_era = df.loc[mask, 'ERA'].median()
        df.loc[mask, 'ERA_minus'] = df.loc[mask, 'ERA'] / lg_era * 100

    return df

pitching_feat = compute_pitching_metrics(pitching_raw)
print("Pitching features computed.")
pitching_feat[['team','season','ERA','FIP','K9','BB9','HR9','K_BB_pct','ERA_minus']].head()"""))

# ── GAME LOG LOADING ───────────────────────────────────────────────────────────
cells.append(md_cell("### Game Logs — Training Dataset Construction"))

cells.append(code_cell("""\
# Pull per-team game schedules from stats.ncaa.org to build game-level training rows
# Results are cached to Drive since full pull takes ~20 min

GAME_LOG_CACHE = CACHE_DIR + 'game_logs_2021_2025.parquet'

def _fetch_game_logs_collegebaseball():
    from collegebaseball import ncaa_scraper
    # Get all D1 school IDs (package ships a lookup table)
    try:
        from collegebaseball.lookups import get_available_schools
        schools = get_available_schools(division=1)
        ids = schools['school_id'].tolist()
    except Exception:
        # Fallback: use the known D1 school ID range heuristic
        ids = list(range(1, 800))

    frames = []
    for yr in ALL_YEARS:
        print(f"  Pulling {yr} game logs …", end=' ')
        count = 0
        for sid in ids:
            try:
                logs = ncaa_scraper.ncaa_schedule_info(school_id=sid, year=yr)
                if logs is not None and len(logs) > 0:
                    logs['season']    = yr
                    logs['school_id'] = sid
                    frames.append(logs)
                    count += 1
            except Exception:
                pass
            time.sleep(0.05)
        print(f"{count} teams")
    return pd.concat(frames, ignore_index=True)

if os.path.exists(GAME_LOG_CACHE):
    game_logs_raw = pd.read_parquet(GAME_LOG_CACHE)
    print(f"Loaded game logs from cache: {len(game_logs_raw):,} rows")
else:
    print("Pulling game logs (this takes ~20 min, cached after first run) …")
    if _scraper_available:
        game_logs_raw = _fetch_game_logs_collegebaseball()
    else:
        # Minimal fallback: empty frame — model will train on season-aggregate cross-join
        game_logs_raw = pd.DataFrame()
        print("  WARNING: No game logs pulled. Using season-aggregate features only.")
    if len(game_logs_raw) > 0:
        game_logs_raw.to_parquet(GAME_LOG_CACHE)
    print("Done.")

if len(game_logs_raw) > 0:
    game_logs_raw.head(3)"""))

cells.append(code_cell("""\
# Standardise game log schema and parse scores
def parse_game_logs(df):
    if len(df) == 0:
        return df
    df = df.copy()
    # Common column aliases
    rename = {
        'Opponent': 'opponent', 'opponent': 'opponent',
        'Result': 'result', 'result': 'result',
        'Date': 'date', 'date': 'date',
        'score': 'score_str', 'Score': 'score_str',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse "W 7-3" or "L 2-5" result strings
    if 'result' in df.columns:
        df['outcome'] = df['result'].str.extract(r'^([WL])', expand=False)
        df['score_str'] = df['result'].str.extract(r'(\\d+-\\d+)', expand=False)

    if 'score_str' in df.columns:
        split = df['score_str'].str.split('-', expand=True)
        df['team_score'] = pd.to_numeric(split[0], errors='coerce')
        df['opp_score']  = pd.to_numeric(split[1], errors='coerce')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df['is_home'] = df.get('location', pd.Series('', index=df.index)).str.upper().ne('A')
    df['win']     = (df.get('outcome', pd.Series('', index=df.index)) == 'W').astype(int)
    df['run_diff'] = df.get('team_score', 0) - df.get('opp_score', 0)
    return df

game_logs = parse_game_logs(game_logs_raw)
print(f"Game logs parsed: {len(game_logs):,} rows")
if len(game_logs) > 0:
    print(f"Win rate: {game_logs['win'].mean():.3f}  (should be ~0.5)")
    game_logs[['opponent','date','win','team_score','opp_score','run_diff','season']].head()"""))

cells.append(code_cell("""\
# Merge game logs with season-aggregate team features to build training matrix
# Known limitation: season-aggregate stats contain mild data leakage for early-season games.
# For the NCAA tournament, this is moot since the full season has been played.

def build_game_matrix(logs, bat, pit):
    if len(logs) == 0:
        print("No game logs — returning empty matrix.")
        return pd.DataFrame()

    # Index batting/pitching by (team, season)
    bat_idx = bat.set_index(['team','season'])
    pit_idx = pit.set_index(['team','season'])

    bat_cols = ['wOBA','ISO','K_pct','BB_pct','OPS','wRC_plus','RPG']
    pit_cols = ['ERA','FIP','K9','BB9','HR9','K_BB_pct','ERA_minus','RA9']

    rows = []
    for _, g in logs.iterrows():
        team = g.get('team', g.get('school', ''))
        opp  = g.get('opponent', '')
        yr   = g.get('season', TEST_YEAR)

        # Look up team stats
        def get_stats(t, yr, idx, cols):
            key = (t, yr)
            if key in idx.index:
                row = idx.loc[key]
                return {c: row.get(c, np.nan) for c in cols}
            return {c: np.nan for c in cols}

        t_bat = get_stats(team, yr, bat_idx, bat_cols)
        t_pit = get_stats(team, yr, pit_idx, pit_cols)
        o_bat = get_stats(opp,  yr, bat_idx, bat_cols)
        o_pit = get_stats(opp,  yr, pit_idx, pit_cols)

        row = {
            'season':    yr,
            'team':      team,
            'opponent':  opp,
            'is_home':   int(g.get('is_home', 1)),
            'win':       g.get('win', np.nan),
            'run_diff':  g.get('run_diff', np.nan),
        }
        row.update({f't_{k}': v for k, v in {**t_bat, **t_pit}.items()})
        row.update({f'o_{k}': v for k, v in {**o_bat, **o_pit}.items()})
        # Differential features (team minus opponent) — most predictive
        for c in bat_cols:
            row[f'd_{c}'] = t_bat.get(c, np.nan) - o_bat.get(c, np.nan)  if not np.isnan(t_bat.get(c, np.nan)) else np.nan
        for c in pit_cols:
            row[f'd_{c}'] = t_pit.get(c, np.nan) - o_pit.get(c, np.nan)  if not np.isnan(t_pit.get(c, np.nan)) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)

game_matrix = build_game_matrix(game_logs, batting_feat, pitching_feat)
print(f"Game matrix: {game_matrix.shape}")
game_matrix.dropna(subset=['win','run_diff'], inplace=True)
print(f"After dropping NaN targets: {game_matrix.shape}")
if len(game_matrix) > 0:
    game_matrix.head(3)"""))

# ── SECTION 1c: STARTER INTEGRATION ───────────────────────────────────────────
cells.append(md_cell("""\
## Section 1c — Starter-Aware Features
Pull probable/actual starting pitcher for each game and blend their individual FIP into the team-level pitching signal.
Skip this cell if roster data is unavailable — the model degrades gracefully."""))

cells.append(code_cell("""\
# Starter FIP integration (optional — runs only if collegebaseball is available)
STARTER_CACHE = CACHE_DIR + 'starter_fip_2021_2025.parquet'

def pull_starter_fip():
    from collegebaseball import ncaa_scraper
    frames = []
    for yr in ALL_YEARS:
        try:
            from collegebaseball.lookups import get_available_schools
            schools = get_available_schools(division=1)
        except Exception:
            print(f"Cannot get school list for {yr}"); continue
        for _, row in schools.iterrows():
            sid = row['school_id']
            try:
                # Individual pitcher game logs give per-start IP, ER, K, BB, HR
                plogs = ncaa_scraper.ncaa_game_logs(school_id=sid, year=yr, category='pitching')
                if plogs is not None and len(plogs) > 0:
                    plogs['season']    = yr
                    plogs['school_id'] = sid
                    plogs['team']      = row.get('school', str(sid))
                    frames.append(plogs)
            except Exception:
                pass
            time.sleep(0.05)
    return pd.concat(frames, ignore_index=True)

def compute_starter_fip_by_team(starter_logs, fip_constants):
    \"\"\"Aggregate individual pitcher season FIP; return top-starter FIP per team/season.\"\"\"
    df = starter_logs.copy()
    for c in ['IP','K','BB','HR','HBP','ER']:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    df['FIP_raw'] = (13*df['HR'] + 3*(df['BB']+df['HBP']) - 2*df['K']) / df['IP'].replace(0, np.nan)
    df = df.merge(fip_constants, on='season', how='left')
    df['FIP'] = df['FIP_raw'] + df['FIP_const']
    # Top starter = pitcher with most IP on the team
    top = (df.sort_values('IP', ascending=False)
             .groupby(['team','season']).first()[['FIP']]
             .rename(columns={'FIP': 'ace_FIP'})
             .reset_index())
    return top

if _scraper_available:
    if os.path.exists(STARTER_CACHE):
        starter_fip = pd.read_parquet(STARTER_CACHE)
        print(f"Starter FIP loaded from cache: {len(starter_fip)} teams")
    else:
        print("Pulling individual pitcher game logs (slow) …")
        starter_logs = pull_starter_fip()
        # Compute per-season FIP constants from team pitching data
        fip_const_df = (pitching_feat.groupby('season')
                        .apply(lambda s: pd.Series({'FIP_const':
                               s['ERA'].median() - s['FIP_raw'].median()
                               if 'FIP_raw' in s.columns else 3.1}))
                        .reset_index())
        starter_fip = compute_starter_fip_by_team(starter_logs, fip_const_df)
        starter_fip.to_parquet(STARTER_CACHE)
    # Merge ace FIP into batting_feat for downstream use
    batting_feat = batting_feat.merge(starter_fip[['team','season','ace_FIP']],
                                      on=['team','season'], how='left')
    print("Ace FIP merged into batting_feat.")
else:
    batting_feat['ace_FIP'] = np.nan
    print("Starter FIP skipped (package not available).")"""))

# ── SECTION 2: ELO + POWER RANKINGS ───────────────────────────────────────────
cells.append(md_cell("## Section 2 — Team Power Rankings (Elo + Composite)"))

cells.append(code_cell("""\
# Elo rating system
# K-factor: 30 (higher variance than football due to 50-60 game schedule)
# Margin multiplier: log(|run_diff| + 1) * 2.0, capped
# Home-field advantage: +35 Elo points (college baseball HFA is meaningful)

ELO_K    = 30
ELO_HOME = 35
ELO_INIT = 1500

def expected_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

def margin_multiplier(run_diff):
    \"\"\"Diminishing returns on blowouts (from The Book principle of not over-weighting margin).\"\"\"
    return np.log(abs(run_diff) + 1) * 2.0

def update_elo(elo_home, elo_away, home_score, away_score):
    adj_home = elo_home + ELO_HOME
    e_home   = expected_win(adj_home, elo_away)
    actual   = 1 if home_score > away_score else 0
    mult     = margin_multiplier(home_score - away_score)
    delta    = ELO_K * mult * (actual - e_home)
    return elo_home + delta, elo_away - delta

def compute_elo_ratings(logs):
    \"\"\"Replay game history chronologically to produce final Elo per team per season.\"\"\"
    if len(logs) == 0:
        return {}
    elo = {}  # {team: current_elo}
    history = []

    logs_sorted = logs.sort_values('date').dropna(subset=['date','team_score','opp_score'])
    for _, g in logs_sorted.iterrows():
        team = g.get('team', '')
        opp  = g.get('opponent', '')
        if not team or not opp:
            continue
        e_t = elo.get(team, ELO_INIT)
        e_o = elo.get(opp,  ELO_INIT)
        if g.get('is_home', 1):
            new_t, new_o = update_elo(e_t, e_o, g['team_score'], g['opp_score'])
        else:
            new_o, new_t = update_elo(e_o, e_t, g['opp_score'], g['team_score'])
        elo[team] = new_t
        elo[opp]  = new_o
        history.append({'team': team, 'opponent': opp, 'date': g['date'],
                         'season': g.get('season'), 'elo_after': new_t})
    return elo, pd.DataFrame(history)

if len(game_logs) > 0:
    current_elo, elo_history = compute_elo_ratings(game_logs)
    elo_df = pd.DataFrame({'team': list(current_elo.keys()),
                            'elo':  list(current_elo.values())})
    print(f"Elo computed for {len(elo_df)} teams.")
else:
    # Fallback: assign equal Elo and sort by composite stats
    teams_2025 = batting_feat[batting_feat['season'] == TEST_YEAR]['team'].unique()
    elo_df = pd.DataFrame({'team': teams_2025, 'elo': ELO_INIT})
    print("No game log data — Elo initialised at 1500 for all teams.")"""))

cells.append(code_cell("""\
# Composite Power Score = weighted average of Elo + offensive + pitching metrics
# Weights informed by run-environment research in The Book

def build_rankings(bat, pit, elo, year=TEST_YEAR):
    b = bat[bat['season'] == year].copy()
    p = pit[pit['season'] == year].copy()
    merged = b.merge(p, on=['team','season'], how='outer', suffixes=('_bat','_pit'))
    merged = merged.merge(elo[['team','elo']], on='team', how='left')
    merged['elo'] = merged['elo'].fillna(ELO_INIT)

    # Normalise each metric to z-score
    def zscore(s):
        return (s - s.mean()) / s.std().replace(0, 1)

    metrics = {
        'wOBA':      ( 0.25, True),   # (weight, higher=better)
        'K_BB_pct':  ( 0.20, True),
        'FIP':       ( 0.20, False),  # lower FIP is better
        'ISO':       ( 0.10, True),
        'wRC_plus':  ( 0.10, True),
        'elo':       ( 0.15, True),
    }
    score = pd.Series(0.0, index=merged.index)
    for col, (wt, higher_better) in metrics.items():
        if col in merged.columns:
            z = zscore(pd.to_numeric(merged[col], errors='coerce').fillna(merged[col].median() if col in merged.columns else 0))
            score += wt * (z if higher_better else -z)

    merged['power_score'] = score
    merged['rank'] = merged['power_score'].rank(ascending=False).astype(int)
    return merged.sort_values('rank')

rankings = build_rankings(batting_feat, pitching_feat, elo_df, year=TEST_YEAR)
display_cols = ['rank','team','elo','wOBA','FIP','K_BB_pct','ISO','wRC_plus','power_score']
display_cols = [c for c in display_cols if c in rankings.columns]
print(f"=== {TEST_YEAR} College Baseball Power Rankings (Top 25) ===")
rankings[display_cols].head(25).style.format({
    'elo': '{:.0f}', 'wOBA': '{:.3f}', 'FIP': '{:.2f}',
    'K_BB_pct': '{:.3f}', 'ISO': '{:.3f}', 'wRC_plus': '{:.1f}',
    'power_score': '{:.3f}'
})"""))

cells.append(code_cell("""\
# Bar chart: Top 25 teams by composite power score
top25 = rankings.head(25).copy()
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top25)))[::-1]
bars = ax.barh(top25['team'][::-1], top25['power_score'][::-1], color=colors[::-1])
ax.set_xlabel('Composite Power Score (z-score weighted)')
ax.set_title(f'{TEST_YEAR} College Baseball Power Rankings — Top 25', fontsize=14, fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
for bar, val in zip(bars, top25['power_score'][::-1]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}', va='center', fontsize=8)
plt.tight_layout()
plt.show()"""))

# ── SECTION 3: RUN DIFFERENTIAL MODEL ─────────────────────────────────────────
cells.append(md_cell("## Section 3 — Run Differential Prediction (XGBoost Regressor)"))

cells.append(code_cell("""\
# Feature columns for all models
DIFF_FEATS = [c for c in game_matrix.columns if c.startswith('d_')]
TEAM_FEATS = [c for c in game_matrix.columns if c.startswith('t_') or c.startswith('o_')]
BASE_FEATS = DIFF_FEATS + ['is_home']

print(f"Differential features: {len(DIFF_FEATS)}")
print(f"Team-level features:   {len(TEAM_FEATS)}")
print(f"Base feature set used: {len(BASE_FEATS)}")

def get_train_test(matrix, target_col, feature_cols):
    df = matrix.dropna(subset=[target_col] + feature_cols).copy()
    train = df[df['season'].isin(TRAIN_YEARS)]
    test  = df[df['season'] == TEST_YEAR]
    X_tr  = train[feature_cols].values
    y_tr  = train[target_col].values
    X_te  = test[feature_cols].values
    y_te  = test[target_col].values
    return X_tr, y_tr, X_te, y_te, train, test"""))

cells.append(code_cell("""\
if len(game_matrix) > 0:
    X_tr, y_tr, X_te, y_te, train_df, test_df = get_train_test(
        game_matrix, 'run_diff', BASE_FEATS)

    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    reg.fit(X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False)

    preds_reg = reg.predict(X_te)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_te, preds_reg)
    r2  = r2_score(y_te, preds_reg)
    print(f"Run-diff model  |  MAE: {mae:.2f} runs  |  R²: {r2:.3f}")

    # Feature importance
    fi = pd.Series(reg.feature_importances_, index=BASE_FEATS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    fi.head(15).plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('XGBoost Regressor — Top 15 Feature Importances (Run Diff)')
    ax.set_ylabel('Gain')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(); plt.show()
else:
    print("Insufficient game data — skipping regressor training.")
    reg = None"""))

# ── SECTION 4a: WIN PROBABILITY XGBOOST ───────────────────────────────────────
cells.append(md_cell("## Section 4a — Win Probability (XGBoost Classifier)"))

cells.append(code_cell("""\
if len(game_matrix) > 0:
    X_tr_c, y_tr_c, X_te_c, y_te_c, _, _ = get_train_test(
        game_matrix, 'win', BASE_FEATS)

    clf_xgb = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, use_label_encoder=False,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    clf_xgb.fit(X_tr_c, y_tr_c,
                eval_set=[(X_te_c, y_te_c)],
                verbose=False)

    proba_xgb = clf_xgb.predict_proba(X_te_c)[:, 1]
    acc_xgb   = accuracy_score(y_te_c, (proba_xgb > 0.5).astype(int))
    auc_xgb   = roc_auc_score(y_te_c, proba_xgb)
    brier_xgb = brier_score_loss(y_te_c, proba_xgb)
    print(f"XGBoost Classifier  |  Acc: {acc_xgb:.3f}  |  AUC: {auc_xgb:.3f}  |  Brier: {brier_xgb:.4f}")
else:
    print("Insufficient data — skipping XGBoost classifier.")
    proba_xgb = None"""))

# ── SECTION 4b: FASTAI WIN PROBABILITY ────────────────────────────────────────
cells.append(md_cell("## Section 4b — Win Probability (FastAI Tabular)"))

cells.append(code_cell("""\
from fastai.tabular.all import *

if len(game_matrix) > 0:
    feat_df = game_matrix[BASE_FEATS + ['win','season']].dropna().copy()
    feat_df['win'] = feat_df['win'].astype(str)  # fastai needs category target

    train_idx = feat_df[feat_df['season'].isin(TRAIN_YEARS)].index.tolist()
    valid_idx = feat_df[feat_df['season'] == TEST_YEAR].index.tolist()

    if len(valid_idx) == 0:
        valid_idx = train_idx[-max(1, len(train_idx)//10):]

    cont_names = BASE_FEATS
    cat_names  = []

    to = TabularDataLoaders.from_df(
        feat_df, path='.', y_names='win', y_block=CategoryBlock(),
        cat_names=cat_names, cont_names=cont_names,
        procs=[FillMissing, Normalize],
        valid_idx=valid_idx, bs=256
    )

    learn = tabular_learner(to, layers=[200, 100], metrics=accuracy)
    learn.fit_one_cycle(10, 1e-3)

    preds_fastai_raw, targs = learn.get_preds(dl=to.valid)
    proba_fastai = preds_fastai_raw[:, 1].numpy()
    y_valid      = (np.array(targs) == 1).astype(int)
    acc_fa   = accuracy_score(y_valid, (proba_fastai > 0.5).astype(int))
    auc_fa   = roc_auc_score(y_valid, proba_fastai)
    brier_fa = brier_score_loss(y_valid, proba_fastai)
    print(f"FastAI Tabular  |  Acc: {acc_fa:.3f}  |  AUC: {auc_fa:.3f}  |  Brier: {brier_fa:.4f}")
else:
    print("Insufficient data — skipping FastAI model.")
    proba_fastai = None"""))

# ── SECTION 4c: MODEL COMPARISON ──────────────────────────────────────────────
cells.append(md_cell("## Section 4c — Model Comparison & Calibration"))

cells.append(code_cell("""\
if proba_xgb is not None and proba_fastai is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curves
    for proba, label, color in [(proba_xgb, 'XGBoost', 'royalblue'),
                                  (proba_fastai, 'FastAI', 'darkorange')]:
        frac_pos, mean_pred = calibration_curve(y_te_c, proba, n_bins=10, strategy='uniform')
        axes[0].plot(mean_pred, frac_pos, marker='o', label=label, color=color)
    axes[0].plot([0,1],[0,1],'k--', label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curves')
    axes[0].legend()

    # Metrics bar chart
    metrics_df = pd.DataFrame({
        'Model':  ['XGBoost', 'FastAI'],
        'Acc':    [acc_xgb, acc_fa],
        'AUC':    [auc_xgb, auc_fa],
        'Brier':  [1-brier_xgb, 1-brier_fa],  # invert: lower brier = better
    }).set_index('Model')
    metrics_df.plot(kind='bar', ax=axes[1], colormap='Set2')
    axes[1].set_title('Model Metrics (Accuracy, AUC, 1-Brier)')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(loc='lower right')

    plt.tight_layout(); plt.show()
else:
    print("Need both models to compare.")"""))

# ── SECTION 5: STACKED ENSEMBLE ────────────────────────────────────────────────
cells.append(md_cell("## Section 5 — Stacked Ensemble with Isotonic Calibration"))

cells.append(code_cell("""\
# Meta-learner: Logistic Regression on [xgb_proba, fastai_proba, xgb_run_diff_pred]
if proba_xgb is not None and proba_fastai is not None:
    from sklearn.isotonic import IsotonicRegression

    # Align arrays (both from TEST_YEAR valid set)
    meta_X = np.column_stack([proba_xgb, proba_fastai, preds_reg])
    meta_y = y_te_c

    meta_lr = LogisticRegression(C=1.0, max_iter=500)
    meta_lr.fit(meta_X, meta_y)
    proba_ensemble = meta_lr.predict_proba(meta_X)[:, 1]

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba_ensemble, meta_y)
    proba_calibrated = iso.predict(proba_ensemble)

    acc_ens   = accuracy_score(meta_y, (proba_calibrated > 0.5).astype(int))
    auc_ens   = roc_auc_score(meta_y, proba_calibrated)
    brier_ens = brier_score_loss(meta_y, proba_calibrated)
    print(f"Ensemble (calibrated)  |  Acc: {acc_ens:.3f}  |  AUC: {auc_ens:.3f}  |  Brier: {brier_ens:.4f}")

    print(f"\\nMeta-learner coefficients: XGB={meta_lr.coef_[0][0]:.3f}, "
          f"FastAI={meta_lr.coef_[0][1]:.3f}, RunDiff={meta_lr.coef_[0][2]:.3f}")
else:
    print("Models not available — cannot build ensemble.")
    proba_calibrated = proba_xgb"""))

# ── SECTION 6: GAME PREDICTOR ──────────────────────────────────────────────────
cells.append(md_cell("## Section 6 — Game Predictor Tool"))

cells.append(code_cell("""\
def predict_game(team_a, team_b, is_home_a=True, starter_fip_a=None, starter_fip_b=None,
                 year=TEST_YEAR, verbose=True):
    \"\"\"
    Predict a single game. Returns dict with win_prob (team_a wins), predicted run_diff.
    team_a is home if is_home_a=True. starter_fip overrides season ace_FIP if provided.
    \"\"\"
    bat_yr = batting_feat[batting_feat['season'] == year].set_index('team')
    pit_yr = pitching_feat[pitching_feat['season'] == year].set_index('team')
    elo_lkp = elo_df.set_index('team')['elo']

    def get_bat(t):
        return bat_yr.loc[t] if t in bat_yr.index else pd.Series(dtype=float)
    def get_pit(t):
        return pit_yr.loc[t] if t in pit_yr.index else pd.Series(dtype=float)

    ba, bb = get_bat(team_a), get_bat(team_b)
    pa, pb = get_pit(team_a), get_pit(team_b)

    bat_cols = ['wOBA','ISO','K_pct','BB_pct','OPS','wRC_plus','RPG']
    pit_cols = ['ERA','FIP','K9','BB9','HR9','K_BB_pct','ERA_minus','RA9']

    row = {'is_home': int(is_home_a)}
    for c in bat_cols:
        row[f'd_{c}'] = ba.get(c, np.nan) - bb.get(c, np.nan)
    for c in pit_cols:
        row[f'd_{c}'] = pa.get(c, np.nan) - pb.get(c, np.nan)

    # Override FIP with known starter FIP
    if starter_fip_a is not None and starter_fip_b is not None:
        row['d_FIP'] = starter_fip_a - starter_fip_b

    X = np.array([[row.get(f, 0) for f in BASE_FEATS]])

    rd_pred    = float(reg.predict(X)[0]) if reg else 0.0
    wp_xgb     = float(clf_xgb.predict_proba(X)[0, 1]) if clf_xgb is not None else 0.5
    # Ensemble if available
    try:
        fa_proba = float(learn.get_preds(dl=to.test_dl(pd.DataFrame([row])[cont_names]))[0][0, 1])
    except Exception:
        fa_proba = wp_xgb
    meta_input = np.array([[wp_xgb, fa_proba, rd_pred]])
    try:
        wp_ens = float(iso.predict(meta_lr.predict_proba(meta_input)[:, 1])[0])
    except Exception:
        wp_ens = wp_xgb

    elo_a = elo_lkp.get(team_a, ELO_INIT)
    elo_b = elo_lkp.get(team_b, ELO_INIT)
    elo_wp = expected_win(elo_a + (ELO_HOME if is_home_a else 0), elo_b)

    result = {
        'team_a': team_a, 'team_b': team_b,
        'win_prob_ensemble': wp_ens,
        'win_prob_xgb':      wp_xgb,
        'win_prob_elo':      elo_wp,
        'predicted_run_diff': rd_pred,
        'elo_a': elo_a, 'elo_b': elo_b,
    }
    if verbose:
        outcome = 'FAVORED' if wp_ens > 0.5 else 'UNDERDOG'
        print(f"\\n{'='*55}")
        print(f"  {team_a} vs {team_b}  ({'Home' if is_home_a else 'Away'})")
        print(f"{'='*55}")
        print(f"  Ensemble win prob:   {wp_ens:.1%}  [{outcome}]")
        print(f"  XGBoost win prob:    {wp_xgb:.1%}")
        print(f"  Elo win prob:        {elo_wp:.1%}")
        print(f"  Predicted run diff:  {rd_pred:+.1f}")
        print(f"  Elo ratings:         {elo_a:.0f} vs {elo_b:.0f}")
    return result

# Example predictions — edit team names to match your data
sample_games = [
    ('Tennessee', 'Arkansas',   True),
    ('LSU',       'Florida',    False),
    ('Wake Forest','Virginia',  True),
]
for a, b, home in sample_games:
    try:
        predict_game(a, b, is_home_a=home)
    except Exception as e:
        print(f"{a} vs {b}: {e}")"""))

cells.append(code_cell("""\
# Multi-game prediction table (e.g., NCAA Tournament bracket round)
def predict_games_table(matchups, year=TEST_YEAR):
    \"\"\"matchups: list of (team_a, team_b, is_home_a) tuples\"\"\"
    rows = []
    for a, b, home in matchups:
        try:
            r = predict_game(a, b, is_home_a=home, year=year, verbose=False)
            rows.append({
                'Home' if home else 'Neutral': a,
                'Away': b,
                'Win Prob (A)': f"{r['win_prob_ensemble']:.1%}",
                'Predicted Margin': f"{r['predicted_run_diff']:+.1f}",
                'Elo (A)': f"{r['elo_a']:.0f}",
                'Elo (B)': f"{r['elo_b']:.0f}",
            })
        except Exception as e:
            rows.append({'Home' if home else 'Neutral': a, 'Away': b, 'Error': str(e)})
    return pd.DataFrame(rows)

# Edit this list with actual upcoming games or tournament bracket
upcoming = [
    ('Tennessee', 'Arkansas', True),
    ('LSU',       'Florida',  False),
]
predict_games_table(upcoming)"""))

# ── SECTION 7: BETTING / ODDS ──────────────────────────────────────────────────
cells.append(md_cell("## Section 7 — Betting Spread & Moneyline Integration"))

cells.append(code_cell("""\
# The Odds API — free tier (500 req/month). Set your key in Colab Secrets as ODDS_API_KEY
try:
    from google.colab import userdata
    ODDS_API_KEY = userdata.get('ODDS_API_KEY')
except Exception:
    ODDS_API_KEY = None  # set manually: ODDS_API_KEY = 'your_key_here'

ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
SPORT_KEY     = 'baseball_ncaa'  # check https://api.the-odds-api.com/v4/sports for key

def fetch_live_odds():
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set. Add to Colab Secrets.")
        return pd.DataFrame()
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h,spreads', 'oddsFormat': 'american'}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"Odds API error: {resp.status_code} {resp.text[:200]}")
        return pd.DataFrame()
    data = resp.json()
    rows = []
    for game in data:
        home, away = game['home_team'], game['away_team']
        for bkm in game.get('bookmakers', [])[:1]:  # take first bookmaker
            for mkt in bkm.get('markets', []):
                if mkt['key'] == 'h2h':
                    outcomes = {o['name']: o['price'] for o in mkt['outcomes']}
                    rows.append({'home': home, 'away': away,
                                 'ml_home': outcomes.get(home), 'ml_away': outcomes.get(away),
                                 'commence': game.get('commence_time')})
                elif mkt['key'] == 'spreads':
                    for o in mkt['outcomes']:
                        if o['name'] == home:
                            rows.append({'home': home, 'away': away,
                                         'spread': o.get('point'), 'spread_odds': o.get('price'),
                                         'commence': game.get('commence_time')})
    return pd.DataFrame(rows)

def american_to_prob(odds):
    \"\"\"Convert American odds to implied probability.\"\"\"
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 100/(100+o) if o > 0 else abs(o)/(abs(o)+100)

def compare_to_spread(team_a, team_b, model_wp, spread, spread_odds=-110):
    \"\"\"Return edge vs the spread line.\"\"\"
    implied_prob = american_to_prob(spread_odds)
    edge = model_wp - implied_prob
    return {'team_a': team_a, 'team_b': team_b,
            'model_wp': model_wp, 'implied_prob': implied_prob,
            'edge': edge, 'spread': spread,
            'bet': 'team_a' if edge > 0 else 'team_b'}

print("Odds integration ready. Set ODDS_API_KEY in Secrets to fetch live lines.")
live_odds = fetch_live_odds()
live_odds.head()"""))

# ── SECTION 8: BACKTESTING ─────────────────────────────────────────────────────
cells.append(md_cell("## Section 8 — Backtesting (ATS & Moneyline)"))

cells.append(code_cell("""\
def backtest_moneyline(game_matrix, model_proba, year=None, edge_threshold=0.03):
    \"\"\"
    Bet every game where model edge > edge_threshold over the implied ML probability.
    Assumes -110 juice on both sides if no actual line available (flat bet backtesting).
    \"\"\"
    df = game_matrix.copy()
    if year:
        df = df[df['season'] == year]
    if model_proba is not None and len(model_proba) == len(df):
        df['model_wp'] = model_proba
    elif 'model_wp' not in df.columns:
        print("No model predictions available for backtest.")
        return

    # If actual moneyline not in data, use flat -110 (implied ~52.4%)
    if 'ml_home' not in df.columns:
        df['implied_prob'] = 0.524
    else:
        df['implied_prob'] = df['ml_home'].apply(american_to_prob)

    df['edge'] = df['model_wp'] - df['implied_prob']
    df['bet']  = df['edge'] > edge_threshold
    bets = df[df['bet']].copy()

    if len(bets) == 0:
        print(f"No bets with edge > {edge_threshold:.0%}")
        return

    # Flat $1 bet per game, -110 payout
    bets['payout'] = bets['win'].apply(lambda w: 100/110 if w == 1 else -1)
    total  = bets['payout'].sum()
    roi    = total / len(bets)
    record = f"{bets['win'].sum():.0f}-{(1-bets['win']).sum():.0f}"

    print(f"Backtest  |  Bets: {len(bets)}  |  Record: {record}  |  P&L: {total:+.2f} units  |  ROI: {roi:+.1%}")
    return bets

if proba_calibrated is not None and len(game_matrix) > 0:
    # Align calibrated proba with test set
    test_rows = game_matrix[game_matrix['season'] == TEST_YEAR].dropna(subset=['win'] + BASE_FEATS).copy()
    if len(test_rows) == len(proba_calibrated):
        test_rows['model_wp'] = proba_calibrated
        backtest_moneyline(test_rows, None, edge_threshold=0.03)
    else:
        print(f"Length mismatch: test_rows={len(test_rows)}, proba={len(proba_calibrated)}")
else:
    print("No calibrated probabilities available for backtesting.")"""))

# ── SECTION 9: KELLY CRITERION ─────────────────────────────────────────────────
cells.append(md_cell("## Section 9 — Kelly Criterion Bankroll Simulation"))

cells.append(code_cell("""\
def kelly_fraction(model_wp, odds=-110, kelly_frac=KELLY_FRACTION, cap=KELLY_CAP):
    \"\"\"Quarter-Kelly bet sizing relative to current bankroll.\"\"\"
    implied = american_to_prob(odds)
    if model_wp <= implied:
        return 0.0
    # Full Kelly = edge / odds_decimal
    b = (100/abs(odds)) if odds < 0 else (odds/100)  # net payout per unit
    full_kelly = (model_wp * (b + 1) - 1) / b
    frac = kelly_frac * full_kelly
    return min(max(frac, 0.0), cap)

def simulate_bankroll(bets_df, starting_bankroll=STARTING_BANKROLL,
                       kelly_frac=KELLY_FRACTION, cap=KELLY_CAP, odds=-110):
    bankroll = starting_bankroll
    history  = [bankroll]
    for _, row in bets_df.iterrows():
        wp   = row.get('model_wp', 0.5)
        frac = kelly_fraction(wp, odds=odds, kelly_frac=kelly_frac, cap=cap)
        bet  = bankroll * frac
        b    = (100/abs(odds)) if odds < 0 else (odds/100)
        bankroll += bet * b if row.get('win', 0) == 1 else -bet
        history.append(bankroll)
    roi = (bankroll - starting_bankroll) / starting_bankroll
    return bankroll, roi, history

if proba_calibrated is not None and len(game_matrix) > 0:
    test_rows = game_matrix[game_matrix['season'] == TEST_YEAR].dropna(subset=['win'] + BASE_FEATS).copy()
    if len(test_rows) == len(proba_calibrated):
        test_rows['model_wp'] = proba_calibrated
        bets_kelly = test_rows[test_rows['model_wp'] > 0.55].copy()  # only confident bets

        final_bankroll, roi, history = simulate_bankroll(bets_kelly)
        print(f"Kelly Simulation  |  Bets: {len(bets_kelly)}  |  "
              f"Final: ${final_bankroll:,.2f}  |  ROI: {roi:+.1%}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history, color='royalblue', linewidth=1.5)
        ax.axhline(STARTING_BANKROLL, color='gray', linestyle='--', linewidth=0.8)
        ax.fill_between(range(len(history)), STARTING_BANKROLL, history,
                         alpha=0.15, color='royalblue')
        ax.set_xlabel('Bet Number')
        ax.set_ylabel('Bankroll ($)')
        ax.set_title(f'{TEST_YEAR} Kelly Bankroll Simulation (Quarter Kelly, cap={KELLY_CAP:.0%})')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        plt.tight_layout(); plt.show()
else:
    print("No data available for Kelly simulation.")"""))

cells.append(code_cell("""\
# Scenario Explorer: sweep edge threshold and Kelly fraction
def kelly_scenario_explorer(bets_df, thresholds=None, fractions=None):
    if thresholds is None: thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]
    if fractions  is None: fractions  = [0.10, 0.25, 0.50]
    results = []
    for thresh in thresholds:
        sub = bets_df[bets_df.get('edge', bets_df['model_wp'] - 0.524) > thresh]
        for frac in fractions:
            if len(sub) == 0:
                results.append({'Edge Threshold': f'{thresh:.0%}', 'Kelly Frac': f'{frac:.0%}',
                                 'Bets': 0, 'ROI': 'N/A', 'Final': 'N/A'}); continue
            _, roi, hist = simulate_bankroll(sub, kelly_frac=frac)
            results.append({'Edge Threshold': f'{thresh:.0%}', 'Kelly Frac': f'{frac:.0%}',
                             'Bets': len(sub), 'ROI': f'{roi:+.1%}',
                             'Final $': f'${hist[-1]:,.2f}'})
    return pd.DataFrame(results)

if proba_calibrated is not None and len(game_matrix) > 0 and len(test_rows) == len(proba_calibrated):
    test_rows['edge'] = test_rows['model_wp'] - 0.524
    print("=== Kelly Scenario Explorer ===")
    kelly_scenario_explorer(test_rows)
"""))

# ── NOTEBOOK ASSEMBLY ──────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0", "codemirror_mode": {"name": "ipython", "version": 3}}
    },
    "cells": cells
}

out_path = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {out_path}")
print(f"Total cells: {len(cells)}")