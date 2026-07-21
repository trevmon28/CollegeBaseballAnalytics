import re
import math
import difflib as _difflib

# Known Odds API name → ESPN display name overrides
_ODDS_TO_ESPN = {
    'Arizona St Sun Devils':        'Arizona State Sun Devils',
    'Appalachian St Mountaineers':  'Appalachian State Mountaineers',
    'Florida St Seminoles':         'Florida State Seminoles',
    'Georgia St Panthers':          'Georgia State Panthers',
    'Jacksonville St Gamecocks':    'Jacksonville State Gamecocks',
    'Mississippi St Bulldogs':      'Mississippi State Bulldogs',
    'Michigan St Spartans':         'Michigan State Spartans',
    'Ohio St Buckeyes':             'Ohio State Buckeyes',
    'Penn St Nittany Lions':        'Penn State Nittany Lions',
    'Sam Houston St Bearkats':      'Sam Houston State Bearkats',
    'Colorado St Rams':             'Colorado State Rams',
    'Kennesaw St Owls':             'Kennesaw State Owls',
    'Utah St Aggies':               'Utah State Aggies',
    'New Mexico St Aggies':         'New Mexico State Aggies',
    'Arkansas St Red Wolves':       'Arkansas State Red Wolves',
    'Wichita St Shockers':          'Wichita State Shockers',
    'Indiana St Sycamores':         'Indiana State Sycamores',
    'Illinois St Redbirds':         'Illinois State Redbirds',
    'Missouri St Bears':            'Missouri State Bears',
    'SE Missouri State Redhawks':   'SE Missouri State Redhawks',
    'Tarleton St Texans':           'Tarleton State Texans',
    'CSU Fullerton Titans':         'Cal State Fullerton Titans',
    'Long Beach St':                'Long Beach State Dirtbags',
    'Sacramento St Hornets':        'Sacramento State Hornets',
    'Southern Utah':                'Southern Utah Thunderbirds',
    # UAlbany: two source names (sportsdataverse 2021-23 vs ESPN 2024+) unified here
    'UAlbany':                      'UAlbany Great Danes',
    'Albany Great Danes':           'UAlbany Great Danes',
    'University at Albany':         'UAlbany Great Danes',
    'SUNY Albany':                  'UAlbany Great Danes',
    # Short-name traps: bare state names that must not fuzzy-match the wrong team
    'Texas':                        'Texas Longhorns',
    'Kansas':                       'Kansas Jayhawks',
    'Arkansas':                     'Arkansas Razorbacks',
    'Alabama':                      'Alabama Crimson Tide',
    'Florida':                      'Florida Gators',
    'Georgia':                      'Georgia Bulldogs',
    'Tennessee':                    'Tennessee Volunteers',
    'Missouri':                     'Missouri Tigers',
    'Oregon':                       'Oregon Ducks',
    'Arizona':                      'Arizona Wildcats',
    'Colorado':                     'Colorado Buffaloes',
    # LSU / Louisiana-family aliases (common LLM input patterns)
    'LSU':                              'LSU Tigers',
    'Louisiana State':                  'LSU Tigers',
    'Louisiana State University':       'LSU Tigers',
    "Louisiana Ragin Cajuns":           "Louisiana Ragin' Cajuns",
    'UL Lafayette':                     "Louisiana Ragin' Cajuns",
    'Louisiana Lafayette':              "Louisiana Ragin' Cajuns",
    # "University of X" → canonical forms (common LLM input patterns)
    'University of Alabama':            'Alabama Crimson Tide',
    'University of Arizona':            'Arizona Wildcats',
    'University of Arkansas':           'Arkansas Razorbacks',
    'University of Colorado':           'Colorado Buffaloes',
    'University of Florida':            'Florida Gators',
    'University of Georgia':            'Georgia Bulldogs',
    'University of Kansas':             'Kansas Jayhawks',
    'University of Kentucky':           'Kentucky Wildcats',
    'University of Miami':              'Miami Hurricanes',
    'University of Mississippi':        'Ole Miss Rebels',
    'University of Missouri':           'Missouri Tigers',
    'University of North Carolina':     'North Carolina Tar Heels',
    'University of Oklahoma':           'Oklahoma Sooners',
    'University of Oregon':             'Oregon Ducks',
    'University of South Carolina':     'South Carolina Gamecocks',
    'University of Tennessee':          'Tennessee Volunteers',
    'University of Texas':              'Texas Longhorns',
    'University of Virginia':           'Virginia Cavaliers',
    # Nickname / abbreviation shorthand
    'Ole Miss':                         'Ole Miss Rebels',
    'Miss State':                       'Mississippi State Bulldogs',
    'TCU':                              'TCU Horned Frogs',
    'BYU':                              'BYU Cougars',
    'UCF':                              'UCF Knights',
    'Cal':                              'California Golden Bears',
    'UC Berkeley':                      'California Golden Bears',
}

# Nickname → canonical mapping for common shorthand queries
_NICKNAME_MAP = {
    'longhorns':   'Texas Longhorns',
    'razorbacks':  'Arkansas Razorbacks',
    'crimson tide':'Alabama Crimson Tide',
    'gators':      'Florida Gators',
    'bulldogs':    None,  # ambiguous — fall through to fuzzy
    'aggies':      None,  # ambiguous
    'tigers':      None,  # ambiguous
    'wildcats':    None,  # ambiguous
}


def _norm(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r'\bst\.?\b', 'state', name)
    name = re.sub(r'\bmt\.?\b', 'mount', name)
    return name


def _first_word(name: str) -> str:
    return name.strip().split()[0].lower()


def resolve_team(name: str, candidates: list[str]) -> tuple[str | None, list[str]]:
    """
    Resolve a team name to the exact canonical display name.
    Returns (resolved_name, suggestions).
    resolved_name is None if no match found; suggestions lists close alternatives.

    Ambiguity rule: if the query could match multiple teams that share the same
    first word (e.g. "Texas" → Texas Longhorns AND Texas State Bobcats AND Texas
    A&M Aggies), the override table wins; otherwise the shortest/most-common match
    is preferred and the rest are returned as suggestions so the caller can surface
    a disambiguation error.
    """
    if not isinstance(name, str) or not name.strip():
        return None, []

    # 1. Direct Odds API / short-name override (highest priority)
    if name in _ODDS_TO_ESPN:
        return _ODDS_TO_ESPN[name], []
    if name.strip() in _ODDS_TO_ESPN:
        return _ODDS_TO_ESPN[name.strip()], []

    # 2. Exact case-insensitive match
    name_lower = name.strip().lower()
    for c in candidates:
        if c.lower() == name_lower:
            return c, []

    # 3. Normalize (St→State) then exact match
    name_norm = _norm(name)
    norm_map = {_norm(c): c for c in candidates}
    if name_norm in norm_map:
        return norm_map[name_norm], []

    # 4. Ambiguity guard: single bare word that matches multiple teams by first word
    #    (the Texas/Texas State trap) — refuse rather than guess.
    query_first = _first_word(name)
    if len(name.strip().split()) == 1:
        first_word_hits = [c for c in candidates if _first_word(c) == query_first]
        if len(first_word_hits) > 1:
            return None, sorted(first_word_hits)

    # 5. Prefix match: the query is a leading substring of exactly one candidate
    #    (e.g. "Texas Tech" → "Texas Tech Red Raiders").
    #    If multiple candidates share the prefix, pick the shortest (most specific).
    prefix_hits = [c for c in candidates if _norm(c).startswith(name_norm + " ") or _norm(c) == name_norm]
    if len(prefix_hits) == 1:
        return prefix_hits[0], []
    if len(prefix_hits) > 1:
        # Ambiguous prefix — return the shortest match and suggest the rest
        prefix_hits.sort(key=lambda c: len(c))
        return prefix_hits[0], prefix_hits[1:]

    # 6. Fuzzy match on normalized names
    matches = _difflib.get_close_matches(name_norm, norm_map.keys(), n=5, cutoff=0.72)
    if matches:
        return norm_map[matches[0]], [norm_map[m] for m in matches[1:]]

    # No match — return broader suggestions for the error message
    suggestions = _difflib.get_close_matches(name_norm, norm_map.keys(), n=5, cutoff=0.5)
    return None, [norm_map[s] for s in suggestions]


def era_adjustment(team: str, team_stats_df, year: int, scale: float = 25.0) -> float:
    col = 'avg_runs_allowed_z'
    if col not in team_stats_df.columns:
        return 0.0
    row = team_stats_df[(team_stats_df['team'] == team) & (team_stats_df['season'] == year)]
    if row.empty:
        return 0.0
    return -scale * float(row[col].iloc[0])


def american_to_prob(odds: float) -> float:
    o = float(odds)
    return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)


def kelly_fraction(wp: float, odds: float = -110, frac: float = 0.25, cap: float = 0.10) -> float:
    imp = american_to_prob(odds)
    if wp <= imp:
        return 0.0
    b = (100 / abs(odds)) if odds < 0 else (odds / 100)
    return min(max(frac * (wp * (b + 1) - 1) / b, 0.0), cap)


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
