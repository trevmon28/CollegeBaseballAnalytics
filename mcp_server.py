"""
mcp_server.py — MCP server for the college baseball prediction pipeline.

Exposes the local prediction REST API (api.py, port 8000) as Claude tools so
you can query predictions, run on-demand game predictions, and check pipeline
health via natural language in Claude Desktop or Claude Code.

Prerequisites:
  pip install mcp httpx
  uvicorn api:app --host 127.0.0.1 --port 8000  (must be running)

Usage (Claude Desktop):
  See T039 — register this file in claude_desktop_config.json under mcpServers.

Usage (standalone):
  python mcp_server.py
"""

import json
import os
from datetime import date
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("BASEBALL_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT  = 10.0

# FastMCP 1.x reads host/port from constructor settings (not run() kwargs).
# When MCP_TRANSPORT=http the service file sets MCP_PORT; we pass it here
# so the settings object is configured before the first import-time read.
_mcp_port = int(os.environ.get("MCP_PORT", "9000"))
_mcp_host = os.environ.get("MCP_HOST", "127.0.0.1")
mcp = FastMCP(
    "baseball-predictions",
    host=_mcp_host,
    port=_mcp_port,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_bet(b: dict, bankroll: Optional[float] = None) -> str:
    team  = b.get("recommended_bet", "?")
    mkt   = b.get("market", "?")
    edge  = b.get("edge", 0)
    kelly = b.get("kelly_size", 0)   # fraction (e.g. 0.05 = 5%)
    odds  = b.get("odds", "?")
    wp    = b.get("model_wp", 0)
    if bankroll is not None:
        bet_amt = bankroll * kelly
        kelly_str = f"Bet ${bet_amt:.2f}  ({kelly:.1%} of ${bankroll:,.0f})"
    else:
        kelly_str = f"Kelly: {kelly:.1%}"
    return (
        f"  • {team} ({mkt}) @ {odds}\n"
        f"    Model win prob: {wp:.1%}  |  Edge: {edge:+.1%}  |  {kelly_str}"
    )


def _http_error(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    return f"API error {resp.status_code}: {detail}"


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def check_health() -> str:
    """
    Check whether the prediction pipeline ran today and the model is loaded.
    Returns freshness status, last run date, and model availability.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{API_BASE}/health")
        except httpx.ConnectError:
            return (
                "Cannot reach the prediction API at localhost:8000.\n"
                "Make sure the server is running:\n"
                "  uvicorn api:app --host 127.0.0.1 --port 8000"
            )
        data = r.json()

    fresh      = data.get("is_fresh", False)
    last_run   = data.get("last_run_date", "unknown")
    model_ok   = data.get("model_loaded", False)
    generated  = data.get("generated_at", "?")

    lines = [
        f"Pipeline status as of {generated}:",
        f"  Ran today:    {'YES' if fresh else 'NO — predictions may be stale'}",
        f"  Last run:     {last_run}",
        f"  Model loaded: {'YES' if model_ok else 'NO — POST /predict unavailable'}",
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_best_bets(
    target_date: Optional[str] = None,
    min_edge: float = 0.03,
    market: Optional[str] = None,
    bankroll: Optional[float] = None,
) -> str:
    """
    Get pre-computed best bet recommendations from the pipeline.

    Args:
        target_date: Date in YYYY-MM-DD format (defaults to today).
        min_edge:    Minimum edge threshold, e.g. 0.05 for 5% edge (default 0.03).
        market:      Filter by market — "ML" (moneyline) or "ATS" (spread).
        bankroll:    Your betting budget in dollars (e.g. 500). When provided, each
                     bet shows the exact dollar amount to wager based on Kelly sizing.
    """
    params: dict = {"min_edge": min_edge}
    if target_date:
        params["date"] = target_date
    if market:
        params["market"] = market

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{API_BASE}/predictions", params=params)
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 404:
        d = target_date or str(date.today())
        return f"No bets found for {d} with min_edge={min_edge:.0%}{' market=' + market if market else ''}."
    if r.status_code != 200:
        return _http_error(r)

    data  = r.json()
    bets  = data.get("bets", [])
    d_str = data.get("date", target_date or str(date.today()))

    if not bets:
        return f"No qualifying bets for {d_str}."

    header = f"Best bets for {d_str} (edge ≥ {min_edge:.0%})"
    if bankroll is not None:
        header += f"  |  Bankroll: ${bankroll:,.0f}"
    lines = [header + ":"]
    for b in bets:
        lines.append(_fmt_bet(b, bankroll=bankroll))

    if bankroll is not None:
        total_kelly = sum(b.get("kelly_size", 0) for b in bets)
        total_dollars = bankroll * total_kelly
        lines.append(f"\nTotal allocated: ${total_dollars:.2f}  ({total_kelly:.1%} of bankroll)")

    lines.append(f"{len(bets)} bet(s) found.")
    return "\n".join(lines)


@mcp.tool()
async def predict_game(
    home: str,
    away: str,
    neutral: bool = False,
) -> str:
    """
    Run an on-demand prediction for any college baseball matchup.

    Args:
        home:    Home team name (fuzzy match supported, e.g. "Texas" or "texas longhorns").
        away:    Away team name.
        neutral: True if the game is at a neutral site.
    """
    payload = {"home": home, "away": away, "neutral": neutral}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.post(f"{API_BASE}/predict", json=payload)
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 422:
        detail = r.json().get("detail", r.text)
        if isinstance(detail, list):
            # Pydantic validation errors
            msgs = [e.get("msg", str(e)) for e in detail]
            return "Validation error: " + "; ".join(msgs)
        if isinstance(detail, dict) and "suggestions" in detail:
            sugg = ", ".join(detail["suggestions"][:5])
            team_name = detail.get("team", "?")
            hint = f"  Did you mean: {sugg}?" if sugg else ""
            return f"Unknown team '{team_name}'.{hint}"
        return f"Validation error: {detail}"
    if r.status_code == 503:
        return "Model not loaded — run daily_runner.py first."
    if r.status_code != 200:
        return _http_error(r)

    d = r.json()
    home_name = d.get("home_team", home)
    away_name = d.get("away_team", away)
    home_wp   = d.get("home_wp", 0)
    away_wp   = d.get("away_wp", 0)
    home_rec  = d.get("home_recommendation", "")
    away_rec  = d.get("away_recommendation", "")
    site      = "neutral site" if neutral else f"{home_name} home"

    lines = [
        f"Game prediction ({site}):",
        f"  {home_name}: {home_wp:.1%} win probability  →  {home_rec}",
        f"  {away_name}: {away_wp:.1%} win probability  →  {away_rec}",
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_team_predictions(team: str) -> str:
    """
    Get all available bet recommendations for a specific team across all dates.

    Args:
        team: Team name (fuzzy match supported, e.g. "Texas" or "lsu").
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{API_BASE}/teams/{team}/predictions")
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 404:
        detail = r.json().get("detail", {})
        if isinstance(detail, dict) and "suggestions" in detail:
            sugg = ", ".join(detail["suggestions"][:5])
            team_name = detail.get("team", team)
            hint = f"  Did you mean: {sugg}?" if sugg else ""
            return f"Team '{team_name}' not found.{hint}"
        return f"No predictions found for '{team}'."
    if r.status_code != 200:
        return _http_error(r)

    data = r.json()
    bets = data.get("bets", [])
    if not bets:
        return f"No upcoming bets found for '{team}'."

    resolved = bets[0].get("home_team", team) if bets else team
    lines = [f"Upcoming bets for {resolved}:"]
    for b in bets:
        d = b.get("date", "?")
        lines.append(f"\n  {d}")
        lines.append(_fmt_bet(b))
    lines.append(f"\n{len(bets)} bet(s) found.")
    return "\n".join(lines)


@mcp.tool()
async def get_team_profile(team: str) -> str:
    """
    Get the full statistical profile for a college baseball team — wins, losses,
    run averages, Elo rating, strength of schedule, pitching K/BB, and power rank.

    Args:
        team: Team name (fuzzy match supported, e.g. "Texas" or "lsu tigers").
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{API_BASE}/teams/{team}")
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 404:
        detail = r.json().get("detail", "")
        if isinstance(detail, dict) and "suggestions" in detail:
            sugg = ", ".join(detail["suggestions"][:5])
            team_name = detail.get("team", team)
            hint = f"  Did you mean: {sugg}?" if sugg else ""
            return f"Team '{team_name}' not found.{hint}"
        return f"Team not found: {detail}"
    if r.status_code != 200:
        return _http_error(r)

    d = r.json()
    conf  = d.get("conference") or "Unknown conf"
    rank  = d.get("rank")
    rank_str = f"#{rank}" if rank else "unranked"
    w, l  = d.get("wins", 0), d.get("losses", 0)
    games = d.get("games", (w or 0) + (l or 0))

    lines = [
        f"{d['team']} ({conf})  —  Season {d['season']}",
        f"  Record:           {w}-{l}  ({d.get('win_pct', 0):.1%} win rate)",
        f"  Power rank:       {rank_str}  (score: {d.get('power_score', 0):.3f})",
        f"  Pythagorean W%:   {d.get('pythagorean_win_pct', 0):.1%}",
        f"  Runs scored/g:    {d.get('avg_runs_scored', 0):.2f}",
        f"  Runs allowed/g:   {d.get('avg_runs_allowed', 0):.2f}",
        f"  Run diff/g:       {d.get('avg_run_diff', 0):+.2f}",
        f"  Recent form:      {d.get('recent_win_pct', 0):.1%}  (last 15 games)",
        f"  Elo rating:       {d.get('elo', 0):.0f}",
        f"  Avg opp Elo:      {d.get('avg_opp_elo', 0):.0f}  (strength of schedule)",
        f"  Shutout rate:     {d.get('shutout_pct', 0):.1%}  (≤2 RA)",
        f"  Close-game W%:    {d.get('close_win_pct', 0):.1%}  (1-2 run decisions)",
    ]
    k  = d.get("k_per_game")
    bb = d.get("bb_per_game")
    kr = d.get("k_bb_ratio")
    if k is not None:
        lines.append(f"  K/game:           {k:.2f}   BB/game: {bb:.2f}   K/BB: {kr:.2f}")
    return "\n".join(lines)


@mcp.tool()
async def compare_teams(
    team_a: str,
    team_b: str,
    neutral: bool = False,
) -> str:
    """
    Compare two college baseball teams side-by-side and get a win probability prediction.

    Args:
        team_a:  First team (treated as home unless neutral=True).
        team_b:  Second team (away).
        neutral: True if the game is at a neutral site.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(
                f"{API_BASE}/teams/{team_a}/compare/{team_b}",
                params={"neutral": str(neutral).lower()},
            )
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 404:
        detail = r.json().get("detail", "")
        if isinstance(detail, dict) and "suggestions" in detail:
            sugg = ", ".join(detail["suggestions"][:5])
            team_name = detail.get("team", "?")
            hint = f"  Did you mean: {sugg}?" if sugg else ""
            return f"Team '{team_name}' not found.{hint}"
        return f"Team not found: {detail}"
    if r.status_code != 200:
        return _http_error(r)

    d    = r.json()
    home = d["home"]
    away = d["away"]
    diff = d.get("differentials", {})
    pred = d.get("prediction", {})

    def fmt_diff(key, label, pct=False, invert=False):
        v = diff.get(key)
        if v is None:
            return None
        adv = home["team"] if (v > 0) != invert else away["team"]
        s = f"{abs(v):.1%}" if pct else f"{abs(v):.2f}"
        return f"  {label:22s}  {home.get(key, 0):.2f} vs {away.get(key, 0):.2f}  (Δ {v:+.2f}  → {adv})"

    lines = [
        f"{'─'*60}",
        f"  {home['team']} (home)  vs  {away['team']} (away)",
        f"  {'Neutral site' if neutral else home['team'] + ' hosts'}",
        f"{'─'*60}",
        f"  {'Stat':<22}  {home['team']:<20}  {away['team']:<20}",
        f"{'─'*60}",
    ]

    stat_rows = [
        ("win_pct",            "Win %",          True,  False),
        ("pythagorean_win_pct","Pythag W%",       True,  False),
        ("avg_run_diff",       "Run diff/g",      False, False),
        ("avg_runs_scored",    "Runs scored/g",   False, False),
        ("avg_runs_allowed",   "Runs allowed/g",  False, True),
        ("elo",                "Elo rating",      False, False),
        ("avg_opp_elo",        "SOS (avg opp Elo)",False,False),
        ("recent_win_pct",     "Recent form",     True,  False),
        ("shutout_pct",        "Shutout rate",    True,  False),
        ("k_per_game",         "K/game",          False, False),
        ("bb_per_game",        "BB/game",         False, True),
        ("k_bb_ratio",         "K/BB ratio",      False, False),
    ]

    for key, label, pct, inv in stat_rows:
        hv = home.get(key)
        av = away.get(key)
        if hv is None and av is None:
            continue
        hv = hv or 0.0
        av = av or 0.0
        delta = hv - av
        adv = home["team"] if (delta > 0) != inv else away["team"]
        if pct:
            row = f"  {label:<22}  {hv:.1%}   vs  {av:.1%}   (Δ {delta:+.3f}  → {adv})"
        else:
            row = f"  {label:<22}  {hv:.2f}   vs  {av:.2f}   (Δ {delta:+.2f}  → {adv})"
        lines.append(row)

    lines += [
        f"{'─'*60}",
        f"  PREDICTION ({('neutral' if neutral else 'home field')}):",
        f"  {home['team']:20s}  win prob: {pred.get('home_wp', 0):.1%}  →  {pred.get('ml_recommendation','')}",
        f"  {away['team']:20s}  win prob: {pred.get('away_wp', 0):.1%}",
        f"  Est. run diff: {pred.get('pred_run_diff', 0):+.1f}  |  ATS: {pred.get('ats_recommendation','')}",
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_model_meta() -> str:
    """
    Get metadata about the current prediction model — version, features, training cutoff.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            r = await client.get(f"{API_BASE}/meta")
        except httpx.ConnectError:
            return f"Cannot reach prediction API at {API_BASE}."

    if r.status_code == 503:
        return "Model metadata unavailable — run daily_runner.py to generate run_meta.json."
    if r.status_code != 200:
        return _http_error(r)

    d = r.json()
    feats = d.get("feature_list", [])
    lines = [
        f"Model metadata:",
        f"  Version:        {d.get('model_version', '?')}",
        f"  Generated:      {d.get('generated_at', '?')}",
        f"  Test year:      {d.get('test_year', '?')}",
        f"  Train cutoff:   {d.get('train_cutoff', '?')}",
        f"  Total games:    {d.get('total_games', '?'):,}" if d.get('total_games') else f"  Total games:    {d.get('total_games', '?')}",
        f"  Teams:          {d.get('total_teams', '?')}",
        f"  Features ({len(feats)}): {', '.join(feats)}",
    ]
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────
# MCP_TRANSPORT=http  → streamable-http on port 9000 (VPS / remote Claude.ai)
# MCP_TRANSPORT unset → stdio (local Claude Desktop, default)

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
