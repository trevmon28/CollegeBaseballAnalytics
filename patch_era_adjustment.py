"""
Adds team ERA adjustment to game predictions.

Changes:
  1. Add ERA_ELO_SCALE constant to Cell 3
  2. Add era_adjustment() helper at end of Elo cell (Cell 12)
  3. Modify predict_game() to apply ERA-adjusted Elo for wp_elo
  4. Update daily_runner.py to include ERA adjustment in compute_best_bets
"""

import json, re

NB  = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\CollegeBaseballAnalytics_Master.ipynb'
RUN = r'C:\Users\trevm\Projects\CFBBaseballAnalytics\daily_runner.py'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def src(cell):
    return ''.join(cell.get('source', []))

# ── 1. Add ERA_ELO_SCALE to the constants cell ────────────────────────────────
const_idx = next(i for i, c in enumerate(cells) if 'KELLY_FRACTION' in src(c) and 'KELLY_CAP' in src(c))
old_const = ''.join(cells[const_idx]['source'])
new_const = old_const.replace(
    'KELLY_CAP         = 0.10',
    'KELLY_CAP         = 0.10\nERA_ELO_SCALE     = 25   # Elo points per 1-SD in team runs-allowed z-score'
)
assert new_const != old_const, "Constants cell replacement failed"
cells[const_idx]['source'] = new_const
print(f"[OK] Added ERA_ELO_SCALE to cell {const_idx}")

# ── 2. Append era_adjustment() to the Elo cell ────────────────────────────────
elo_idx = next(i for i, c in enumerate(cells) if 'def compute_elo' in src(c) and 'margin_mult_538' in src(c))

ERA_HELPER = '''

def era_adjustment(team, year=TEST_YEAR):
    """
    Temporary Elo adjustment for game prediction based on team pitching quality.
    Uses avg_runs_allowed_z (already computed in feature engineering).
    Lower runs allowed = better pitching = positive Elo boost.
    Applied only at prediction time; does NOT modify permanent Elo ratings.
    """
    col = 'avg_runs_allowed_z'
    row = team_stats[(team_stats['team'] == team) & (team_stats['season'] == year)]
    if row.empty or col not in team_stats.columns:
        return 0.0
    era_z = float(row[col].iloc[0])
    return -ERA_ELO_SCALE * era_z   # negative z = fewer runs allowed = positive adj
'''

old_elo = ''.join(cells[elo_idx]['source'])
cells[elo_idx]['source'] = old_elo + ERA_HELPER
print(f"[OK] Appended era_adjustment() to Elo cell {elo_idx}")

# ── 3. Modify predict_game() to apply ERA-adjusted Elo ────────────────────────
pg_idx = next(i for i, c in enumerate(cells) if 'def predict_game' in src(c) and 'wp_elo' in src(c))
old_pg = ''.join(cells[pg_idx]['source'])

# Replace the Elo section inside predict_game
OLD_ELO_BLOCK = (
    '    ea = elo_lk.get(team_a, ELO_INIT) + (ELO_HOME if is_home_a else 0)\n'
    '    eb = elo_lk.get(team_b, ELO_INIT)\n'
    '    wp_elo = expected_win(ea, eb)'
)
NEW_ELO_BLOCK = (
    '    era_a = era_adjustment(team_a, year)\n'
    '    era_b = era_adjustment(team_b, year)\n'
    '    ea = elo_lk.get(team_a, ELO_INIT) + (ELO_HOME if is_home_a else 0) + era_a\n'
    '    eb = elo_lk.get(team_b, ELO_INIT) + era_b\n'
    '    wp_elo = expected_win(ea, eb)'
)
assert OLD_ELO_BLOCK in old_pg, "Elo block not found in predict_game"
new_pg = old_pg.replace(OLD_ELO_BLOCK, NEW_ELO_BLOCK)

# Add ERA adj line to verbose output (after Elo ratings line)
OLD_PRINT = "        print(f\"  Elo ratings       : {elo_lk.get(team_a,ELO_INIT):.0f} vs {elo_lk.get(team_b,ELO_INIT):.0f}\")"
NEW_PRINT = (
    "        print(f\"  Elo ratings       : {elo_lk.get(team_a,ELO_INIT):.0f} vs {elo_lk.get(team_b,ELO_INIT):.0f}\")\n"
    "        print(f\"  ERA adj (Elo)     : {team_a} {era_a:+.0f}  /  {team_b} {era_b:+.0f}\")"
)
assert OLD_PRINT in new_pg, "Elo ratings print line not found in predict_game"
new_pg = new_pg.replace(OLD_PRINT, NEW_PRINT)

cells[pg_idx]['source'] = new_pg
print(f"[OK] Updated predict_game() in cell {pg_idx}")

# ── Save notebook ──────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook saved ({len(cells)} cells).")

# ── 4. Patch daily_runner.py ──────────────────────────────────────────────────
with open(RUN, encoding='utf-8') as f:
    runner = f.read()

# Add ERA_ELO_SCALE constant
runner = runner.replace(
    'KELLY_CAP         = 0.10',
    'KELLY_CAP         = 0.10\nERA_ELO_SCALE     = 25'
)

# Add era_adjustment helper before compute_best_bets
ERA_HELPER_RUNNER = '''
def era_adjustment_runner(team, team_stats, year=TEST_YEAR):
    col = 'avg_runs_allowed_z'
    row = team_stats[(team_stats['team'] == team) & (team_stats['season'] == year)]
    if row.empty or col not in team_stats.columns:
        return 0.0
    return -ERA_ELO_SCALE * float(row[col].iloc[0])

'''

runner = runner.replace(
    'def compute_best_bets(',
    ERA_HELPER_RUNNER + 'def compute_best_bets('
)

# Apply ERA adjustment inside compute_best_bets when computing ea/eb for Elo-based wp
OLD_RUNNER_BLOCK = (
    '        X = np.array([[row.get(f,0) for f in feats]])\n'
    '        wp_home = float(clf.predict_proba(X)[0,1])'
)
NEW_RUNNER_BLOCK = (
    '        X = np.array([[row.get(f,0) for f in feats]])\n'
    '        wp_home = float(clf.predict_proba(X)[0,1])\n'
    '        era_h = era_adjustment_runner(home, team_stats)\n'
    '        era_a_val = era_adjustment_runner(away, team_stats)\n'
    '        elo_h = elo_df.set_index("team")["elo"].get(home, ELO_INIT) + ELO_HOME + era_h\n'
    '        elo_a = elo_df.set_index("team")["elo"].get(away, ELO_INIT) + era_a_val\n'
    '        wp_elo = 1.0 / (1.0 + 10 ** ((elo_a - elo_h) / 400))\n'
    '        wp_home = 0.6 * wp_home + 0.4 * wp_elo   # blend model + ERA-adjusted Elo'
)
assert OLD_RUNNER_BLOCK in runner, "Runner block not found in daily_runner.py"
runner = runner.replace(OLD_RUNNER_BLOCK, NEW_RUNNER_BLOCK)

# Fix compute_best_bets signature to accept elo_df (needed for the Elo lookup above)
runner = runner.replace(
    'def compute_best_bets(odds_df, team_stats, clf, feats, edge_min=0.03):',
    'def compute_best_bets(odds_df, team_stats, elo_df, clf, feats, edge_min=0.03):'
)
runner = runner.replace(
    'bets = compute_best_bets(odds_df, team_stats, clf, feats, edge_min=0.03)',
    'bets = compute_best_bets(odds_df, team_stats, elo_df, clf, feats, edge_min=0.03)'
)

with open(RUN, 'w', encoding='utf-8') as f:
    f.write(runner)
print("[OK] Patched daily_runner.py with ERA adjustment")
print("\nNext: python push_notebook.py")
