"""
tests/test_team_resolution.py

Guards against team name lookup mistakes — the primary failure mode being
short state names (e.g. "Texas") resolving to the wrong team
(e.g. "Texas State Bobcats" instead of "Texas Longhorns").

Run:
    pytest tests/test_team_resolution.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.utils import resolve_team

# ── canonical team list used in production rankings ───────────────────────────

CANDIDATES = [
    "Texas Longhorns",
    "Texas State Bobcats",
    "Texas A&M Aggies",
    "Texas Tech Red Raiders",
    "Kansas Jayhawks",
    "Kansas State Wildcats",
    "Arkansas Razorbacks",
    "Arkansas State Red Wolves",
    "Alabama Crimson Tide",
    "Alabama A&M Bulldogs",
    "Florida Gators",
    "Florida State Seminoles",
    "Florida Atlantic Owls",
    "Georgia Bulldogs",
    "Georgia Tech Yellow Jackets",
    "Georgia State Panthers",
    "Tennessee Volunteers",
    "Tennessee Tech Golden Eagles",
    "Missouri Tigers",
    "Missouri State Bears",
    "Oregon Ducks",
    "Oregon State Beavers",
    "Arizona Wildcats",
    "Arizona State Sun Devils",
    "Colorado Buffaloes",
    "Colorado State Rams",
    "LSU Tigers",
    "Ole Miss Rebels",
    "Mississippi State Bulldogs",
    "South Carolina Gamecocks",
    "North Carolina Tar Heels",
    "NC State Wolfpack",
    "Tarleton State Texans",
    "East Carolina Pirates",
    "USC Upstate Spartans",
    "UCLA Bruins",
    "Louisiana Ragin' Cajuns",
    "TCU Horned Frogs",
    "BYU Cougars",
    "California Golden Bears",
    "Miami Hurricanes",
    "Oklahoma Sooners",
    "Kentucky Wildcats",
    "Virginia Cavaliers",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve(name):
    result, suggestions = resolve_team(name, CANDIDATES)
    return result, suggestions


def assert_resolves_to(query, expected):
    result, suggestions = resolve(query)
    assert result == expected, (
        f"resolve_team({query!r}) → {result!r}, expected {expected!r}. "
        f"Suggestions: {suggestions}"
    )


# ── exact / case-insensitive matches ─────────────────────────────────────────

def test_exact_match():
    assert_resolves_to("Texas Longhorns", "Texas Longhorns")


def test_case_insensitive():
    assert_resolves_to("texas longhorns", "Texas Longhorns")
    assert_resolves_to("LSU TIGERS", "LSU Tigers")


# ── the Texas/Texas State trap ────────────────────────────────────────────────

def test_bare_texas_resolves_to_longhorns():
    """'Texas' must map to Texas Longhorns, not Texas State / Texas A&M / Texas Tech."""
    assert_resolves_to("Texas", "Texas Longhorns")


def test_texas_state_resolves_correctly():
    assert_resolves_to("Texas State", "Texas State Bobcats")


def test_texas_state_bobcats_resolves_correctly():
    assert_resolves_to("Texas State Bobcats", "Texas State Bobcats")


def test_texas_am_resolves_correctly():
    assert_resolves_to("Texas A&M", "Texas A&M Aggies")
    assert_resolves_to("Texas A&M Aggies", "Texas A&M Aggies")


def test_texas_tech_resolves_correctly():
    assert_resolves_to("Texas Tech", "Texas Tech Red Raiders")


# ── other common short-name traps ─────────────────────────────────────────────

def test_bare_kansas_resolves_to_jayhawks():
    assert_resolves_to("Kansas", "Kansas Jayhawks")


def test_kansas_state_resolves_correctly():
    assert_resolves_to("Kansas State", "Kansas State Wildcats")
    assert_resolves_to("Kansas St", "Kansas State Wildcats")


def test_bare_arkansas_resolves_to_razorbacks():
    assert_resolves_to("Arkansas", "Arkansas Razorbacks")


def test_arkansas_state_resolves_correctly():
    assert_resolves_to("Arkansas State", "Arkansas State Red Wolves")
    assert_resolves_to("Arkansas St", "Arkansas State Red Wolves")


def test_bare_alabama_resolves_to_crimson_tide():
    assert_resolves_to("Alabama", "Alabama Crimson Tide")


def test_bare_florida_resolves_to_gators():
    assert_resolves_to("Florida", "Florida Gators")


def test_florida_state_resolves_correctly():
    assert_resolves_to("Florida State", "Florida State Seminoles")
    assert_resolves_to("Florida St", "Florida State Seminoles")


def test_bare_georgia_resolves_to_bulldogs():
    assert_resolves_to("Georgia", "Georgia Bulldogs")


def test_georgia_tech_resolves_correctly():
    assert_resolves_to("Georgia Tech", "Georgia Tech Yellow Jackets")


def test_bare_tennessee_resolves_to_volunteers():
    assert_resolves_to("Tennessee", "Tennessee Volunteers")


def test_bare_missouri_resolves_to_tigers():
    assert_resolves_to("Missouri", "Missouri Tigers")


def test_missouri_state_resolves_correctly():
    assert_resolves_to("Missouri State", "Missouri State Bears")
    assert_resolves_to("Missouri St", "Missouri State Bears")


def test_bare_oregon_resolves_to_ducks():
    assert_resolves_to("Oregon", "Oregon Ducks")


def test_oregon_state_resolves_correctly():
    assert_resolves_to("Oregon State", "Oregon State Beavers")
    assert_resolves_to("Oregon St", "Oregon State Beavers")


def test_bare_arizona_resolves_to_wildcats():
    assert_resolves_to("Arizona", "Arizona Wildcats")


def test_arizona_state_resolves_correctly():
    assert_resolves_to("Arizona State", "Arizona State Sun Devils")
    assert_resolves_to("Arizona St", "Arizona State Sun Devils")


# ── Odds API name overrides ───────────────────────────────────────────────────

def test_odds_api_st_abbreviations():
    assert_resolves_to("Mississippi St Bulldogs", "Mississippi State Bulldogs")
    assert_resolves_to("Florida St Seminoles", "Florida State Seminoles")
    assert_resolves_to("Tarleton St Texans", "Tarleton State Texans")


# ── ambiguous queries must not silently pick the wrong team ───────────────────

def test_ambiguous_single_word_returns_none_with_suggestions():
    """A bare word that matches multiple teams must fail, not guess."""
    # "Tennessee" has override → Volunteers, so it should succeed
    # but if we remove it from the override list, bare "Tennessee" would be ambiguous
    # with "Tennessee Tech" — this test uses a truly ambiguous candidate list.
    ambiguous_candidates = ["Tennessee Volunteers", "Tennessee Tech Golden Eagles"]
    result, suggestions = resolve_team("Tennessee", ambiguous_candidates)
    # Tennessee IS in the override map, so it still resolves
    assert result == "Tennessee Volunteers"


def test_truly_ambiguous_without_override():
    """Bare word not in override map with multiple first-word matches → None + suggestions."""
    custom = ["Fake University Bears", "Fake State Bulldogs"]
    result, suggestions = resolve_team("Fake", custom)
    assert result is None
    assert "Fake University Bears" in suggestions
    assert "Fake State Bulldogs" in suggestions


# ── matchup-level correctness ─────────────────────────────────────────────────

@pytest.mark.parametrize("home_query,away_query,expected_home,expected_away", [
    ("Kansas",        "Arkansas",      "Kansas Jayhawks",       "Arkansas Razorbacks"),
    ("Texas",         "Texas State",   "Texas Longhorns",       "Texas State Bobcats"),
    ("Texas A&M",     "Texas",         "Texas A&M Aggies",      "Texas Longhorns"),
    ("Florida",       "Florida State", "Florida Gators",        "Florida State Seminoles"),
    ("Arizona",       "Arizona State", "Arizona Wildcats",      "Arizona State Sun Devils"),
    ("Missouri",      "Missouri St",   "Missouri Tigers",       "Missouri State Bears"),
    ("Tennessee",     "Tennessee Tech","Tennessee Volunteers",  "Tennessee Tech Golden Eagles"),
    ("Oregon",        "Oregon St",     "Oregon Ducks",          "Oregon State Beavers"),
])
def test_matchup_teams_resolve_correctly(home_query, away_query, expected_home, expected_away):
    """Ensure both sides of a matchup resolve to the intended teams."""
    home_result, _ = resolve_team(home_query, CANDIDATES)
    away_result, _ = resolve_team(away_query, CANDIDATES)
    assert home_result == expected_home, (
        f"Home: resolve_team({home_query!r}) → {home_result!r}, expected {expected_home!r}"
    )
    assert away_result == expected_away, (
        f"Away: resolve_team({away_query!r}) → {away_result!r}, expected {expected_away!r}"
    )


# ── LSU / Louisiana-family aliases ────────────────────────────────────────────

def test_lsu_abbreviation():
    assert_resolves_to("LSU", "LSU Tigers")


def test_louisiana_state_resolves_to_lsu():
    assert_resolves_to("Louisiana State", "LSU Tigers")


def test_louisiana_state_university_resolves_to_lsu():
    """The classic MCP failure: long-form university name for LSU."""
    assert_resolves_to("Louisiana State University", "LSU Tigers")


def test_louisiana_ragin_cajuns_no_apostrophe():
    """LLM inputs often omit the apostrophe — must still resolve correctly."""
    assert_resolves_to("Louisiana Ragin Cajuns", "Louisiana Ragin' Cajuns")


def test_ul_lafayette_resolves_correctly():
    assert_resolves_to("UL Lafayette", "Louisiana Ragin' Cajuns")


def test_louisiana_lafayette_resolves_correctly():
    assert_resolves_to("Louisiana Lafayette", "Louisiana Ragin' Cajuns")


# ── "University of X" patterns ────────────────────────────────────────────────

def test_university_of_florida():
    assert_resolves_to("University of Florida", "Florida Gators")


def test_university_of_texas():
    assert_resolves_to("University of Texas", "Texas Longhorns")


def test_university_of_alabama():
    assert_resolves_to("University of Alabama", "Alabama Crimson Tide")


def test_university_of_georgia():
    assert_resolves_to("University of Georgia", "Georgia Bulldogs")


def test_university_of_tennessee():
    assert_resolves_to("University of Tennessee", "Tennessee Volunteers")


def test_university_of_mississippi():
    assert_resolves_to("University of Mississippi", "Ole Miss Rebels")


def test_university_of_arkansas():
    assert_resolves_to("University of Arkansas", "Arkansas Razorbacks")


def test_university_of_south_carolina():
    assert_resolves_to("University of South Carolina", "South Carolina Gamecocks")


def test_university_of_north_carolina():
    assert_resolves_to("University of North Carolina", "North Carolina Tar Heels")


def test_university_of_missouri():
    assert_resolves_to("University of Missouri", "Missouri Tigers")


def test_university_of_kansas():
    assert_resolves_to("University of Kansas", "Kansas Jayhawks")


def test_university_of_oklahoma():
    assert_resolves_to("University of Oklahoma", "Oklahoma Sooners")


def test_university_of_kentucky():
    assert_resolves_to("University of Kentucky", "Kentucky Wildcats")


def test_university_of_virginia():
    assert_resolves_to("University of Virginia", "Virginia Cavaliers")


# ── common abbreviations / nickname shorthand ─────────────────────────────────

def test_ole_miss_shorthand():
    assert_resolves_to("Ole Miss", "Ole Miss Rebels")


def test_miss_state_shorthand():
    assert_resolves_to("Miss State", "Mississippi State Bulldogs")


def test_tcu_abbreviation():
    assert_resolves_to("TCU", "TCU Horned Frogs")


def test_byu_abbreviation():
    assert_resolves_to("BYU", "BYU Cougars")


def test_ucf_abbreviation():
    assert_resolves_to("UCF", "UCF Knights")


def test_cal_resolves_to_golden_bears():
    assert_resolves_to("Cal", "California Golden Bears")


def test_uc_berkeley_resolves_to_golden_bears():
    assert_resolves_to("UC Berkeley", "California Golden Bears")
