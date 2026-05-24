"""
Inserts Section 2b — Team Profiles & Comparison after cell 15 (bar chart).
Adds:
  - team_profile(team)      — detailed stat card with percentile ranks
  - compare_teams(a, b)     — side-by-side table + radar chart + H2H record
"""
import json

NB_PATH = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def src(cell):
    return ''.join(cell['source'])

def code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}

# Find the bar chart cell (cell 15)
bar_chart_idx = None
for i, cell in enumerate(nb['cells']):
    if 'top25 = rankings.head(25)' in src(cell) and 'barh' in src(cell):
        bar_chart_idx = i
        break

assert bar_chart_idx is not None, "Could not find bar chart cell"
print(f"Inserting Section 2b after cell index {bar_chart_idx}")

# ── markdown header ────────────────────────────────────────────────────────────
md_src = """\
## Section 2b — Team Profiles & Comparison

### Usage

```python
# Single team deep-dive
team_profile("Tennessee Volunteers")

# Side-by-side comparison with radar chart
compare_teams("Tennessee Volunteers", "Morehead State Eagles")
```
"""

# ── code cell ─────────────────────────────────────────────────────────────────
code_src = """\
# ── helpers ───────────────────────────────────────────────────────────────────
PROFILE_METRICS = [
    ('pythagorean_win_pct', 'Pythag W%',   True),
    ('avg_runs_scored',     'Off (RS/G)',   True),
    ('avg_runs_allowed',    'Def (RA/G)',   False),  # lower is better
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
    # fall back to team_stats + elo if not in rankings year
    s = team_stats[(team_stats['team']==team)&(team_stats['season']==year)]
    if s.empty: return None
    row = s.iloc[0].copy()
    elo_val = elo_df.set_index('team')['elo'].get(team, ELO_INIT)
    row['elo'] = elo_val
    sos_val = sos_df[(sos_df['team']==team)&(sos_df['season']==year)]
    row['avg_opp_elo'] = sos_val['avg_opp_elo'].values[0] if not sos_val.empty else ELO_INIT
    row['power_score'] = float('nan')
    row['rank'] = float('nan')
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
        print(f"No data for '{team}' in {year}."); return

    conf      = TEAM_CONF.get(team, 'Unknown')
    tier      = CONF_TIER.get(conf, CONF_DEFAULT)
    n_teams   = len(rankings)
    rank_str  = f"#{int(row['rank'])}" if not pd.isna(row.get('rank', float('nan'))) else 'N/A'

    wins   = int(row.get('wins',   0))
    losses = int(row.get('losses', 0))

    print(f"\n{'='*58}")
    print(f"  {team}")
    print(f"  {conf}  |  Conf tier: {tier:.2f}  |  Rank: {rank_str} / {n_teams}")
    print(f"{'='*58}")
    print(f"  Record          : {wins}-{losses}  ({row.get('win_pct',0):.3f})")
    print()

    labels = {
        'pythagorean_win_pct': ('Pythagorean W%  ', f"{row.get('pythagorean_win_pct',0):.3f}"),
        'avg_runs_scored':     ('Runs Scored/G   ', f"{row.get('avg_runs_scored',0):.2f}"),
        'avg_runs_allowed':    ('Runs Allowed/G  ', f"{row.get('avg_runs_allowed',0):.2f}"),
        'avg_run_diff':        ('Run Diff/G      ', f"{row.get('avg_run_diff',0):+.2f}"),
        'elo':                 ('Elo Rating      ', f"{row.get('elo',ELO_INIT):.0f}"),
        'avg_opp_elo':         ('SOS (Opp Elo)   ', f"{row.get('avg_opp_elo',ELO_INIT):.0f}"),
        'recent_win_pct':      ('Recent Form L15 ', f"{row.get('recent_win_pct',0):.3f}"),
        'power_score':         ('Power Score     ', f"{row.get('power_score',0):.3f}"),
    }

    for col, higher in [(c,h) for c,_,h in PROFILE_METRICS]:
        val = row.get(col, float('nan'))
        if pd.isna(val): continue
        lbl, fmt_val = labels[col]
        p = _percentile(col, val, higher)
        print(f"  {lbl}: {fmt_val:>8}   P{p:3d}  {_bar(p)}")
    print(f"{'='*58}")

# ── compare_teams ─────────────────────────────────────────────────────────────
def compare_teams(team_a, team_b, year=TEST_YEAR):
    ra = _get_row(team_a, year)
    rb = _get_row(team_b, year)
    if ra is None: print(f"No data for '{team_a}'"); return
    if rb is None: print(f"No data for '{team_b}'"); return

    conf_a = TEAM_CONF.get(team_a,'Unknown')
    conf_b = TEAM_CONF.get(team_b,'Unknown')

    # ── head-to-head record ───────────────────────────────────────────────────
    h2h = games[
        ((games['home_team']==team_a)&(games['away_team']==team_b)) |
        ((games['home_team']==team_b)&(games['away_team']==team_a))
    ].copy()
    a_wins = ((h2h['home_team']==team_a)&(h2h['home_score']>h2h['away_score'])).sum() + \
             ((h2h['away_team']==team_a)&(h2h['away_score']>h2h['home_score'])).sum()
    b_wins = len(h2h) - a_wins

    # ── side-by-side table ────────────────────────────────────────────────────
    col_w = 22
    def fmt(v, col, higher):
        if pd.isna(v): return '  N/A  '
        va = ra.get(col, float('nan'))
        vb = rb.get(col, float('nan'))
        if pd.isna(va) or pd.isna(vb): return f"{v:>7.3f}" if isinstance(v,float) else str(v)
        better = (va > vb) if higher else (va < vb)
        mark = ' <' if (v==va and better) else (' <' if (v==vb and not better) else '  ')
        if isinstance(v,float):
            return f"{v:>7.3f}{mark}"
        return f"{v!s:>7}{mark}"

    print(f"\n{'='*66}")
    print(f"  {'METRIC':<22} {'':>2}  {team_a[:col_w]:<{col_w}}  {team_b[:col_w]:<{col_w}}")
    print(f"  {'':22}       [{conf_a[:18]:<18}]  [{conf_b[:18]:<18}]")
    print(f"{'='*66}")

    display = [
        ('pythagorean_win_pct', 'Pythagorean W%',  True,  '{:.3f}'),
        ('avg_runs_scored',     'Runs Scored/G',    True,  '{:.2f}'),
        ('avg_runs_allowed',    'Runs Allowed/G',   False, '{:.2f}'),
        ('avg_run_diff',        'Run Diff/G',        True,  '{:+.2f}'),
        ('elo',                 'Elo Rating',        True,  '{:.0f}'),
        ('avg_opp_elo',         'SOS (Opp Elo)',     True,  '{:.0f}'),
        ('recent_win_pct',      'Recent Form L15',   True,  '{:.3f}'),
        ('power_score',         'Power Score',       True,  '{:.3f}'),
    ]

    for col, label, higher, fmt_str in display:
        va = ra.get(col, float('nan'))
        vb = rb.get(col, float('nan'))
        if pd.isna(va) and pd.isna(vb): continue
        fa = (fmt_str.format(va) + (' <<' if (not pd.isna(va) and not pd.isna(vb) and ((va>vb) if higher else (va<vb))) else '   '))
        fb = (fmt_str.format(vb) + (' <<' if (not pd.isna(va) and not pd.isna(vb) and ((vb>va) if higher else (vb<va))) else '   '))
        print(f"  {label:<22}       {fa:<{col_w}}  {fb:<{col_w}}")

    print(f"{'='*66}")
    print(f"  H2H Record (all seasons):  {team_a[:20]} {a_wins}-{b_wins} {team_b[:20]}")
    print(f"{'='*66}")

    # ── radar chart ───────────────────────────────────────────────────────────
    radar_cols  = ['pythagorean_win_pct','avg_runs_scored','avg_runs_allowed',
                   'avg_run_diff','elo','avg_opp_elo','recent_win_pct','power_score']
    radar_labels = ['Pythag W%','Offense','Defense\n(inv)','Run Diff',
                    'Elo','SOS','Recent\nForm','Power\nScore']
    radar_higher = [True, True, False, True, True, True, True, True]

    # normalize each metric to 0-1 using full rankings range
    def norm(col, val, higher):
        vals = rankings[col].dropna()
        if vals.empty or pd.isna(val): return 0.5
        mn, mx = vals.min(), vals.max()
        if mx == mn: return 0.5
        n = (val - mn) / (mx - mn)
        return n if higher else 1.0 - n

    vals_a = [norm(c, ra.get(c, float('nan')), h) for c,h in zip(radar_cols, radar_higher)]
    vals_b = [norm(c, rb.get(c, float('nan')), h) for c,h in zip(radar_cols, radar_higher)]

    N = len(radar_cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]

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
    ax.set_title(f"{team_a}\nvs\n{team_b}", size=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    plt.tight_layout()
    plt.show()

# ── sample calls ──────────────────────────────────────────────────────────────
team_profile("Tennessee Volunteers")
compare_teams("Tennessee Volunteers", "Morehead State Eagles")
"""

# Insert the two new cells after bar_chart_idx
insert_at = bar_chart_idx + 1
nb['cells'].insert(insert_at,     md_cell(md_src))
nb['cells'].insert(insert_at + 1, code_cell(code_src))

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Inserted Section 2b at cell indices {insert_at} and {insert_at+1}")
print("Done.")
