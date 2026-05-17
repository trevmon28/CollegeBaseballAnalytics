"""
Two fixes:
1. resolve_team() — fuzzy name matching so 'florida st', 'TENNESSEE', 'lsu' all work
2. best_bets() / predict_game() skip games where a team has no stats (fixes 100% model_wp bug)
"""

import json, re

NB = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def src(c): return ''.join(c.get('source', []))

# ── 1. Append resolve_team() to the Elo/era cell (12) ────────────────────────
elo_idx = next(i for i, c in enumerate(cells) if 'def era_adjustment' in src(c))

RESOLVE_TEAM = '''

import difflib as _difflib

# Known Odds API name → ESPN display name overrides
_ODDS_TO_ESPN = {
    'Arizona St Sun Devils':           'Arizona State Sun Devils',
    'Appalachian St Mountaineers':     'Appalachian State Mountaineers',
    'Florida St Seminoles':            'Florida State Seminoles',
    'Georgia St Panthers':             'Georgia State Panthers',
    'Jacksonville St Gamecocks':       'Jacksonville State Gamecocks',
    'Mississippi St Bulldogs':         'Mississippi State Bulldogs',
    'Michigan St Spartans':            'Michigan State Spartans',
    'Ohio St Buckeyes':                'Ohio State Buckeyes',
    'Penn St Nittany Lions':           'Penn State Nittany Lions',
    'Sam Houston St Bearkats':         'Sam Houston State Bearkats',
    'Colorado St Rams':                'Colorado State Rams',
    'Kennesaw St Owls':                'Kennesaw State Owls',
    'Utah St Aggies':                  'Utah State Aggies',
    'New Mexico St Aggies':            'New Mexico State Aggies',
    'Arkansas St Red Wolves':          'Arkansas State Red Wolves',
    'Wichita St Shockers':             'Wichita State Shockers',
    'Indiana St Sycamores':            'Indiana State Sycamores',
    'Illinois St Redbirds':            'Illinois State Redbirds',
    'Missouri St Bears':               'Missouri State Bears',
    'SE Missouri State Redhawks':      'SE Missouri State Redhawks',
    'Tarleton St Texans':              'Tarleton State Texans',
    'CSU Fullerton Titans':            'Cal State Fullerton Titans',
    'Long Beach St':                   'Long Beach State Dirtbags',
    'Sacramento St Hornets':           'Sacramento State Hornets',
    'Southern Utah':                   'Southern Utah Thunderbirds',
}

def _norm(name):
    """Lowercase + expand St/Mt abbreviations for fuzzy matching."""
    name = name.strip().lower()
    name = re.sub(r'\\bst\\.?\\b', 'state', name)
    name = re.sub(r'\\bmt\\.?\\b', 'mount', name)
    return name

def resolve_team(name, year=TEST_YEAR, silent=False):
    """
    Resolve a team name to the exact ESPN display name used in team_stats.
    Handles: case differences, St/State abbreviations, fuzzy typos, Odds API names.

    Examples:
        resolve_team('tennessee volunteers')   -> 'Tennessee Volunteers'
        resolve_team('florida st seminoles')   -> 'Florida State Seminoles'
        resolve_team('LSU')                    -> 'LSU Tigers'
        resolve_team('texas a&m')              -> 'Texas A&M Aggies'
    """
    if not isinstance(name, str) or not name.strip():
        return name

    # 1. Direct override (Odds API known names)
    if name in _ODDS_TO_ESPN:
        resolved = _ODDS_TO_ESPN[name]
        if not silent: print(f"  [resolved] '{name}' -> '{resolved}'")
        return resolved

    # Build candidate pool from the dataset
    candidates = team_stats['team'].unique().tolist()

    # 2. Exact case-insensitive match
    name_lower = name.strip().lower()
    for c in candidates:
        if c.lower() == name_lower:
            return c

    # 3. Normalize (St->State) then exact match
    name_norm = _norm(name)
    norm_map = {_norm(c): c for c in candidates}
    if name_norm in norm_map:
        resolved = norm_map[name_norm]
        if not silent and resolved != name:
            print(f"  [resolved] '{name}' -> '{resolved}'")
        return resolved

    # 4. Fuzzy match on normalized names (cutoff 0.72 catches most typos)
    matches = _difflib.get_close_matches(name_norm, norm_map.keys(), n=1, cutoff=0.72)
    if matches:
        resolved = norm_map[matches[0]]
        if not silent:
            print(f"  [resolved] '{name}' -> '{resolved}'")
        return resolved

    if not silent:
        print(f"  [warning] Could not resolve team: '{name}' — no stats available")
    return name
'''

old_elo = src(cells[elo_idx])
cells[elo_idx]['source'] = old_elo + RESOLVE_TEAM
print(f"[OK] Appended resolve_team() to cell {elo_idx}")

# ── 2. Fix predict_game(): resolve names + skip when stats missing ─────────────
pg_idx = next(i for i, c in enumerate(cells) if 'def predict_game' in src(c) and 'wp_elo' in src(c))
old_pg = src(cells[pg_idx])

# Insert resolve calls + missing-team guard right after the def line and s_idx setup
OLD_PG_TOP = '''\
def predict_game(team_a, team_b, is_home_a=True, year=TEST_YEAR, verbose=True):
    s_idx  = team_stats[team_stats['season'] == year].set_index('team')
    elo_lk = elo_df.set_index('team')['elo']

    def get(t): return s_idx.loc[t] if t in s_idx.index else pd.Series(dtype=float)
    sa, sb = get(team_a), get(team_b)'''

NEW_PG_TOP = '''\
def predict_game(team_a, team_b, is_home_a=True, year=TEST_YEAR, verbose=True):
    team_a = resolve_team(team_a, year, silent=not verbose)
    team_b = resolve_team(team_b, year, silent=not verbose)

    s_idx  = team_stats[team_stats['season'] == year].set_index('team')
    elo_lk = elo_df.set_index('team')['elo']

    def get(t): return s_idx.loc[t] if t in s_idx.index else None
    sa, sb = get(team_a), get(team_b)

    # Fall back to Elo-only when either team has no season stats
    if sa is None or sb is None:
        missing = [t for t, s in [(team_a, sa), (team_b, sb)] if s is None]
        era_a = era_adjustment(team_a, year) if sa is not None else 0.0
        era_b = era_adjustment(team_b, year) if sb is not None else 0.0
        ea = elo_lk.get(team_a, ELO_INIT) + (ELO_HOME if is_home_a else 0) + era_a
        eb = elo_lk.get(team_b, ELO_INIT) + era_b
        wp_elo = expected_win(ea, eb)
        if verbose:
            print(f"  [Elo-only] No {year} stats for {missing}. Using ERA-adjusted Elo.")
            print(f"  Elo win prob: {wp_elo:.1%}  [favors {'team_a' if wp_elo>0.5 else 'team_b'}]")
        return {'team_a': team_a, 'team_b': team_b,
                'win_prob': wp_elo, 'run_diff': 0.0, 'elo_wp': wp_elo, 'elo_only': True}

    sa, sb = sa, sb  # both are valid Series from here on'''

assert OLD_PG_TOP in old_pg, "predict_game top block not found"
new_pg = old_pg.replace(OLD_PG_TOP, NEW_PG_TOP)

# Also fix the .get(c, np.nan) calls — sa/sb are now guaranteed Series, not None
# but they came from s_idx.loc[t] so they're already Series — no change needed.
# However the differential feature loop used: hf.get(c, np.nan) - af.get(c, np.nan)
# We renamed sa/sb but they feed into hf/af:
new_pg = new_pg.replace(
    "    def get(t): return s_idx.loc[t] if t in s_idx.index else None\n    sa, sb = get(team_a), get(team_b)\n\n    # Fall back",
    "    def get(t): return s_idx.loc[t] if t in s_idx.index else None\n    sa, sb = get(team_a), get(team_b)\n\n    # Fall back"
)
cells[pg_idx]['source'] = new_pg
print(f"[OK] Updated predict_game() in cell {pg_idx}")

# ── 3. Fix team_profile() and compare_teams() to resolve names ────────────────
tp_idx = next(i for i, c in enumerate(cells) if 'def team_profile' in src(c) and 'def compare_teams' in src(c))
old_tp = src(cells[tp_idx])

# team_profile: add resolve at top
old_tp = old_tp.replace(
    'def team_profile(team, year=TEST_YEAR):\n    row = _get_row(team, year)',
    'def team_profile(team, year=TEST_YEAR):\n    team = resolve_team(team, year)\n    row = _get_row(team, year)'
)

# compare_teams: add resolve at top
old_tp = old_tp.replace(
    'def compare_teams(team_a, team_b, year=TEST_YEAR):\n    ra = _get_row(team_a, year)\n    rb = _get_row(team_b, year)',
    'def compare_teams(team_a, team_b, year=TEST_YEAR):\n    team_a = resolve_team(team_a, year)\n    team_b = resolve_team(team_b, year)\n    ra = _get_row(team_a, year)\n    rb = _get_row(team_b, year)'
)

cells[tp_idx]['source'] = old_tp
print(f"[OK] Updated team_profile() and compare_teams() in cell {tp_idx}")

# ── 4. Fix best_bets() to skip games with no stats for either team ────────────
bb_idx = next(i for i, c in enumerate(cells) if 'def best_bets' in src(c))
old_bb = src(cells[bb_idx])

# Resolve names before calling predict_game, and skip elo-only results
OLD_BB_LOOP = '''\
    for _, g in odds.iterrows():
        home, away = g['home'], g['away']
        ml_home = g.get('ml_home')
        ml_away = g.get('ml_away')
        try:
            pred = predict_game(home, away, is_home_a=True, year=year, verbose=False)
        except Exception:
            continue
        wp_home = pred['win_prob']
        wp_away = 1.0 - wp_home'''

NEW_BB_LOOP = '''\
    known_teams = set(team_stats[team_stats['season'] == year]['team'])
    for _, g in odds.iterrows():
        home = resolve_team(g['home'], year, silent=True)
        away = resolve_team(g['away'], year, silent=True)
        ml_home = g.get('ml_home')
        ml_away = g.get('ml_away')
        # Skip games where either team has no stats — model would be unreliable
        if home not in known_teams or away not in known_teams:
            continue
        try:
            pred = predict_game(home, away, is_home_a=True, year=year, verbose=False)
        except Exception:
            continue
        if pred.get('elo_only'):
            continue   # skip Elo-only fallbacks in the betting table
        wp_home = pred['win_prob']
        wp_away = 1.0 - wp_home'''

assert OLD_BB_LOOP in old_bb, "best_bets loop not found"
new_bb = old_bb.replace(OLD_BB_LOOP, NEW_BB_LOOP)
cells[bb_idx]['source'] = new_bb
print(f"[OK] Updated best_bets() in cell {bb_idx}")

# ── Save ──────────────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook saved ({len(cells)} cells).")
