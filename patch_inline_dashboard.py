"""
1. Fixes SyntaxError in Section 2b cell — f-strings can't have literal newlines in Python < 3.12
2. Adds Section 10: inline dashboard (IFrame embed of GitHub Pages site + Dash fallback)
"""
import json

NB_PATH = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def src(cell): return ''.join(cell['source'])

def set_source(cell, new_src):
    lines = new_src.splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    cell['source'] = lines

def code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}

# ── 1. Fix cell 17 (team_profile / compare_teams) ─────────────────────────────
profile_idx = None
for i, cell in enumerate(nb['cells']):
    if 'def team_profile' in src(cell) and 'def compare_teams' in src(cell):
        profile_idx = i
        break

assert profile_idx is not None, "Could not find profile cell"
print(f"Fixing profile cell at index {profile_idx}")

FIXED_PROFILE = r'''# ── helpers ───────────────────────────────────────────────────────────────────
PROFILE_METRICS = [
    ('pythagorean_win_pct', 'Pythag W%',   True),
    ('avg_runs_scored',     'Off (RS/G)',   True),
    ('avg_runs_allowed',    'Def (RA/G)',   False),
    ('avg_run_diff',        'Run Diff',     True),
    ('elo',                 'Elo',          True),
    ('avg_opp_elo',         'SOS',          True),
    ('recent_win_pct',      'Recent Form',  True),
    ('power_score',         'Power Score',  True),
]

def _get_row(team, year=TEST_YEAR):
    rk = rankings[rankings['team'] == team]
    if not rk.empty:
        return rk.iloc[0]
    s = team_stats[(team_stats['team']==team)&(team_stats['season']==year)]
    if s.empty: return None
    row = s.iloc[0].copy()
    row['elo']         = elo_df.set_index('team')['elo'].get(team, ELO_INIT)
    sos_val            = sos_df[(sos_df['team']==team)&(sos_df['season']==year)]
    row['avg_opp_elo'] = sos_val['avg_opp_elo'].values[0] if not sos_val.empty else ELO_INIT
    row['power_score'] = float('nan')
    row['rank']        = float('nan')
    return row

def _percentile(col, val, higher=True):
    vals = rankings[col].dropna()
    if vals.empty or pd.isna(val): return 0
    return int(((vals < val).sum() / len(vals)) * 100) if higher else int(((vals > val).sum() / len(vals)) * 100)

def _bar(p, width=20):
    filled = int(p / 100 * width)
    return '[' + '#'*filled + '.'*(width-filled) + ']'

# ── team_profile ──────────────────────────────────────────────────────────────
def team_profile(team, year=TEST_YEAR):
    row = _get_row(team, year)
    if row is None:
        print("No data for '{}' in {}.".format(team, year)); return

    conf     = TEAM_CONF.get(team, 'Unknown')
    tier     = CONF_TIER.get(conf, CONF_DEFAULT)
    n_teams  = len(rankings)
    rank_str = '#{}'.format(int(row['rank'])) if not pd.isna(row.get('rank', float('nan'))) else 'N/A'
    wins     = int(row.get('wins',   0))
    losses   = int(row.get('losses', 0))

    sep = '=' * 58
    print('\n' + sep)
    print('  ' + team)
    print('  {}  |  Conf tier: {:.2f}  |  Rank: {} / {}'.format(conf, tier, rank_str, n_teams))
    print(sep)
    print('  Record          : {}-{}  ({:.3f})'.format(wins, losses, row.get('win_pct', 0)))
    print()

    labels = {
        'pythagorean_win_pct': ('Pythagorean W%  ', '{:.3f}'.format(row.get('pythagorean_win_pct', 0))),
        'avg_runs_scored':     ('Runs Scored/G   ', '{:.2f}'.format(row.get('avg_runs_scored', 0))),
        'avg_runs_allowed':    ('Runs Allowed/G  ', '{:.2f}'.format(row.get('avg_runs_allowed', 0))),
        'avg_run_diff':        ('Run Diff/G      ', '{:+.2f}'.format(row.get('avg_run_diff', 0))),
        'elo':                 ('Elo Rating      ', '{:.0f}'.format(row.get('elo', ELO_INIT))),
        'avg_opp_elo':         ('SOS (Opp Elo)   ', '{:.0f}'.format(row.get('avg_opp_elo', ELO_INIT))),
        'recent_win_pct':      ('Recent Form L15 ', '{:.3f}'.format(row.get('recent_win_pct', 0))),
        'power_score':         ('Power Score     ', '{:.3f}'.format(row.get('power_score', 0))),
    }

    for col, higher in [(c, h) for c, _, h in PROFILE_METRICS]:
        val = row.get(col, float('nan'))
        if pd.isna(val): continue
        lbl, fmt_val = labels[col]
        p = _percentile(col, val, higher)
        print('  {}: {:>8}   P{:3d}  {}'.format(lbl, fmt_val, p, _bar(p)))
    print(sep)

# ── compare_teams ─────────────────────────────────────────────────────────────
def compare_teams(team_a, team_b, year=TEST_YEAR):
    ra = _get_row(team_a, year)
    rb = _get_row(team_b, year)
    if ra is None: print("No data for '{}'".format(team_a)); return
    if rb is None: print("No data for '{}'".format(team_b)); return

    conf_a = TEAM_CONF.get(team_a, 'Unknown')
    conf_b = TEAM_CONF.get(team_b, 'Unknown')

    h2h  = games[
        ((games['home_team']==team_a)&(games['away_team']==team_b)) |
        ((games['home_team']==team_b)&(games['away_team']==team_a))
    ].copy()
    a_wins = (((h2h['home_team']==team_a)&(h2h['home_score']>h2h['away_score'])) |
               ((h2h['away_team']==team_a)&(h2h['away_score']>h2h['home_score']))).sum()
    b_wins = len(h2h) - a_wins

    col_w = 22
    sep   = '=' * 66
    print('\n' + sep)
    print('  {:<22}       {:<{w}}  {:<{w}}'.format('METRIC', team_a[:col_w], team_b[:col_w], w=col_w))
    print('  {:<22}       [{:<18}]  [{:<18}]'.format('', conf_a[:18], conf_b[:18]))
    print(sep)

    display_metrics = [
        ('pythagorean_win_pct', 'Pythagorean W%',  True,  '{:.3f}'),
        ('avg_runs_scored',     'Runs Scored/G',    True,  '{:.2f}'),
        ('avg_runs_allowed',    'Runs Allowed/G',   False, '{:.2f}'),
        ('avg_run_diff',        'Run Diff/G',       True,  '{:+.2f}'),
        ('elo',                 'Elo Rating',        True,  '{:.0f}'),
        ('avg_opp_elo',         'SOS (Opp Elo)',     True,  '{:.0f}'),
        ('recent_win_pct',      'Recent Form L15',   True,  '{:.3f}'),
        ('power_score',         'Power Score',       True,  '{:.3f}'),
    ]

    for col, label, higher, fmt_str in display_metrics:
        va = ra.get(col, float('nan'))
        vb = rb.get(col, float('nan'))
        if pd.isna(va) and pd.isna(vb): continue
        a_better = not pd.isna(va) and not pd.isna(vb) and ((va > vb) if higher else (va < vb))
        b_better = not pd.isna(va) and not pd.isna(vb) and ((vb > va) if higher else (vb < va))
        fa = fmt_str.format(va) + (' <<' if a_better else '   ')
        fb = fmt_str.format(vb) + (' <<' if b_better else '   ')
        print('  {:<22}       {:<{w}}  {:<{w}}'.format(label, fa, fb, w=col_w))

    print(sep)
    print('  H2H Record (all seasons):  {} {}-{} {}'.format(team_a[:20], a_wins, b_wins, team_b[:20]))
    print(sep)

    radar_cols   = ['pythagorean_win_pct','avg_runs_scored','avg_runs_allowed',
                    'avg_run_diff','elo','avg_opp_elo','recent_win_pct','power_score']
    radar_labels = ['Pythag W%','Offense','Defense\n(inv)','Run Diff',
                    'Elo','SOS','Recent\nForm','Power\nScore']
    radar_higher = [True, True, False, True, True, True, True, True]

    def norm(col, val, higher):
        vals = rankings[col].dropna()
        if vals.empty or pd.isna(val): return 0.5
        mn, mx = vals.min(), vals.max()
        if mx == mn: return 0.5
        n = (val - mn) / (mx - mn)
        return n if higher else 1.0 - n

    vals_a = [norm(c, ra.get(c, float('nan')), h) for c, h in zip(radar_cols, radar_higher)]
    vals_b = [norm(c, rb.get(c, float('nan')), h) for c, h in zip(radar_cols, radar_higher)]

    N      = len(radar_cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]; vals_a += vals_a[:1]; vals_b += vals_b[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_a, 'o-', linewidth=2, color='royalblue', label=team_a)
    ax.fill(angles, vals_a, alpha=0.15, color='royalblue')
    ax.plot(angles, vals_b, 'o-', linewidth=2, color='darkorange', label=team_b)
    ax.fill(angles, vals_b, alpha=0.15, color='darkorange')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25th', '50th', '75th'], size=7, color='gray')
    ax.set_title(team_a + '\nvs\n' + team_b, size=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    plt.tight_layout()
    plt.show()

# ── sample calls ──────────────────────────────────────────────────────────────
team_profile("Tennessee Volunteers")
compare_teams("Tennessee Volunteers", "Morehead State Eagles")
'''

set_source(nb['cells'][profile_idx], FIXED_PROFILE)
print("  Fixed f-string SyntaxErrors in profile cell")

# ── 2. Add Section 10: inline dashboard ───────────────────────────────────────
DASHBOARD_URL = 'https://trevmon28.github.io/CollegeBaseballAnalytics/'

md_src = '## Section 10 — Interactive Dashboard (Inline)\n\nThe full dashboard — rankings, team profiles, comparison radar, and game predictor — is embedded below.\nAlso available as a standalone link: [trevmon28.github.io/CollegeBaseballAnalytics]({})'.format(DASHBOARD_URL)

code_src = (
    "from IPython.display import IFrame, HTML, display\n"
    "\n"
    "DASHBOARD_URL = '" + DASHBOARD_URL + "'\n"
    "\n"
    "try:\n"
    "    import google.colab\n"
    "    display(HTML(\n"
    "        '<div style=\"border:1px solid #1a2f52;border-radius:10px;overflow:hidden\">'\n"
    "        '<iframe src=\"' + DASHBOARD_URL + '\" width=\"100%\" height=\"820\" '\n"
    "        'style=\"border:none;background:#07090f\" allowfullscreen></iframe></div>'\n"
    "    ))\n"
    "except ImportError:\n"
    "    display(IFrame(DASHBOARD_URL, width='100%', height=820))\n"
    "    print('Dashboard also at http://127.0.0.1:8050 (if dashboard.py is running)')\n"
)

# Find last cell index and append Section 10
last_idx = len(nb['cells'])
nb['cells'].append(md_cell(md_src))
nb['cells'].append(code_cell(code_src))
print("Added Section 10 (inline dashboard) at indices {} and {}".format(last_idx, last_idx+1))

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
