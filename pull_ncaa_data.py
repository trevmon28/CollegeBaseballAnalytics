"""
pull_ncaa_data.py — builds NCAA D1 baseball team stats from game results.

Sources (no auth, no blocking):
  - sportsdataverse/baseballr-data GitHub CSVs  → 2021-2023 game results
  - ESPN public scoreboard API                  → 2024-2025 game results

Derives team season stats from scores:
  avg_runs_scored, avg_runs_allowed, run_diff, pythagorean_win_pct,
  wins, losses, games_played

Usage: python pull_ncaa_data.py
Output: data/game_results_2021_2025.parquet   (one row per game)
        data/team_season_stats_2021_2025.parquet (one row per team × season)

Upload both to Google Drive / CollegeBaseballAnalytics / for Colab use.
"""

import os, time, pathlib
from datetime import date, timedelta
import requests
import pandas as pd

OUT_DIR = pathlib.Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

SDV_BASE  = "https://raw.githubusercontent.com/sportsdataverse/baseballr-data/main/ncaa/schedules/csv"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

# ── Source 1: sportsdataverse CSVs 2021-2023 ──────────────────────────────────
def pull_sdv_games(years=(2021, 2022, 2023)):
    frames = []
    for yr in years:
        url = f"{SDV_BASE}/ncaa_baseball_schedule_{yr}.csv"
        print(f"  SDV {yr} … ", end="", flush=True)
        try:
            df = pd.read_csv(url)
            # Keep only D1 games (both teams D1)
            df = df[(df["home_team_division"] == 1) & (df["away_team_division"] == 1)].copy()
            df = df.rename(columns={
                "home_team": "home_team", "away_team": "away_team",
                "home_team_score": "home_score", "away_team_score": "away_score",
            })
            df["season"]     = yr
            df["neutral"]    = df.get("neutral_site", False).fillna(False)
            df["date"]       = pd.to_datetime(df["date"], errors="coerce")
            df = df[["season","date","home_team","away_team","home_score","away_score","neutral"]]
            df = df.dropna(subset=["home_score","away_score"])
            df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
            df = df.dropna(subset=["home_score","away_score"])
            frames.append(df)
            print(f"{len(df)} games")
        except Exception as e:
            print(f"FAILED: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Source 2: ESPN scoreboard API 2024-2025 ───────────────────────────────────
def espn_season_dates(year):
    """Yield all dates in the college baseball season (Feb 14 – June 28)."""
    start = date(year, 2, 14)
    end   = date(year, 6, 28)
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def pull_espn_games(years=(2024, 2025)):
    frames = []
    for yr in years:
        rows = []
        dates = list(espn_season_dates(yr))
        print(f"  ESPN {yr}: scanning {len(dates)} dates", end="", flush=True)
        for i, d in enumerate(dates):
            ds = d.strftime("%Y%m%d")
            try:
                r = requests.get(ESPN_BASE, params={"dates": ds, "limit": 200},
                                 headers=HEADERS, timeout=10)
                if r.status_code != 200:
                    continue
                for event in r.json().get("events", []):
                    comp  = event["competitions"][0]
                    teams = comp["competitors"]
                    # competitors[0] is home, [1] is away (homeAway field confirms)
                    home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
                    away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
                    status = comp["status"]["type"]["name"]
                    if status != "STATUS_FINAL":
                        continue
                    rows.append({
                        "season":     yr,
                        "date":       pd.to_datetime(event["date"][:10]),
                        "home_team":  home["team"]["displayName"],
                        "away_team":  away["team"]["displayName"],
                        "home_score": float(home.get("score", "nan")),
                        "away_score": float(away.get("score", "nan")),
                        "neutral":    comp.get("neutralSite", False),
                    })
            except Exception:
                pass
            if (i + 1) % 30 == 0:
                print(".", end="", flush=True)
            time.sleep(0.05)
        df = pd.DataFrame(rows).dropna(subset=["home_score","away_score"])
        frames.append(df)
        print(f" >> {len(df)} games")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Compute team season stats from game results ───────────────────────────────
def compute_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    # Build long-form: one row per team per game
    home = games[["season","date","home_team","home_score","away_score","neutral"]].copy()
    home.columns = ["season","date","team","runs_scored","runs_allowed","neutral"]
    home["is_home"] = 1

    away = games[["season","date","away_team","away_score","home_score","neutral"]].copy()
    away.columns = ["season","date","team","runs_scored","runs_allowed","neutral"]
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["runs_scored"] > long["runs_allowed"]).astype(int)
    long["run_diff"] = long["runs_scored"] - long["runs_allowed"]

    agg = (long.groupby(["team","season"])
               .agg(
                   games     = ("win",          "count"),
                   wins      = ("win",          "sum"),
                   rs_total  = ("runs_scored",  "sum"),
                   ra_total  = ("runs_allowed", "sum"),
                   rd_total  = ("run_diff",     "sum"),
               )
               .reset_index())

    agg["losses"]             = agg["games"] - agg["wins"]
    agg["win_pct"]            = agg["wins"] / agg["games"]
    agg["avg_runs_scored"]    = agg["rs_total"] / agg["games"]
    agg["avg_runs_allowed"]   = agg["ra_total"] / agg["games"]
    agg["avg_run_diff"]       = agg["rd_total"] / agg["games"]

    # Pythagorean win% (exponent 1.83 for baseball)
    rs, ra = agg["rs_total"], agg["ra_total"]
    agg["pythagorean_win_pct"] = rs**1.83 / (rs**1.83 + ra**1.83).replace(0, float("nan"))

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== NCAA Baseball Data Pull ===\n")

    print("1. Downloading sportsdataverse game results (2021-2023) ...")
    sdv_games = pull_sdv_games(years=[2021, 2022, 2023])

    print("\n2. Fetching ESPN scoreboard results (2024-2025) ...")
    espn_games = pull_espn_games(years=[2024, 2025])

    print("\n3. Combining ...")
    all_games = pd.concat([sdv_games, espn_games], ignore_index=True)
    all_games["neutral"] = all_games["neutral"].astype(bool)
    print(f"   Total games: {len(all_games):,}  across seasons: {sorted(all_games['season'].unique())}")

    print("\n4. Computing team season stats ...")
    team_stats = compute_team_stats(all_games)
    print(f"   Team-seasons: {len(team_stats)}  columns: {team_stats.columns.tolist()}")

    # Save
    game_path = OUT_DIR / "game_results_2021_2025.parquet"
    stat_path = OUT_DIR / "team_season_stats_2021_2025.parquet"
    all_games.to_parquet(game_path,  index=False)
    team_stats.to_parquet(stat_path, index=False)

    print(f"\nSaved:")
    print(f"  {game_path}")
    print(f"  {stat_path}")
    print(f"\nSample team stats:")
    print(team_stats.sort_values("avg_run_diff", ascending=False)
          [["team","season","games","wins","avg_runs_scored","avg_runs_allowed","pythagorean_win_pct"]]
          .head(10).to_string(index=False))

    print("\nNext: upload both .parquet files to Google Drive / CollegeBaseballAnalytics /")
