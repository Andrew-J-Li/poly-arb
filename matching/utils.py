"""
Shared utilities for event-level matching between Kalshi and Polymarket.

Provides:
  - Event loading / grouping (one row per event)
  - Person-name extraction (from original capitalised titles)
  - Year extraction (from cleaned text)
  - Blocking logic (person + date constraints → candidate pairs)
"""

import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "matches"

# ── Files ─────────────────────────────────────────────────────────────────────

KALSHI_CLEAN = DATA_DIR / "kalshi_clean.csv"
KALSHI_ORIG = DATA_DIR / "kalshi_processed.csv"

POLY_CLEAN = DATA_DIR / "polymarket_clean.csv"
POLY_ORIG = DATA_DIR / "polymarket_processed.csv"

# ── Non-person capitalised words to exclude ───────────────────────────────────

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new", "hampshire", "jersey", "mexico", "york",  # parts of multi-word states
    "north", "south", "carolina", "dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode", "island",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west", "wisconsin", "wyoming",
}

COUNTRIES = {
    "iran", "iraq", "china", "russia", "ukraine", "israel", "gaza",
    "lebanon", "syria", "afghanistan", "pakistan", "india", "japan",
    "korea", "taiwan", "mexico", "canada", "cuba", "venezuela",
    "greenland", "germany", "france", "britain", "england", "europe",
    "africa", "brazil", "colombia", "panama", "el", "salvador",
    "honduras", "guatemala", "haiti", "dominican",
}

GOVT_TERMS = {
    "senate", "house", "congress", "president", "governor", "senator",
    "representative", "speaker", "majority", "minority", "leader",
    "whip", "cabinet", "secretary", "administration", "department",
    "justice", "defense", "state", "treasury", "interior", "commerce",
    "energy", "education", "homeland", "security", "attorney", "general",
    "court", "supreme", "federal", "district", "circuit", "judge",
    "chief", "associate", "solicitor", "marshal", "fbi", "cia", "nsa",
    "doj", "dhs", "epa", "fda", "sec", "ftc", "fcc", "usda",
    "director", "chair", "chairman", "chairwoman", "commissioner",
    "inspector", "counsel", "special", "advisor", "aide",
    "ambassador", "envoy", "diplomat",
}

POLITICAL_TERMS = {
    "republican", "democratic", "democrat", "democrats", "gop",
    "party", "primary", "election", "vote", "ballot", "caucus",
    "nomination", "nominee", "candidate", "campaign", "poll",
    "impeachment", "impeach", "resign", "resignation",
    "bill", "act", "law", "resolution", "budget", "tariff", "tariffs",
    "executive", "order", "veto", "pardon", "commute",
    "inauguration", "oath", "swearing",
}

MISC_SKIP = {
    "will", "the", "be", "is", "are", "was", "were", "has", "have",
    "had", "do", "does", "did", "can", "could", "would", "should",
    "shall", "may", "might", "must", "not", "no", "yes",
    "and", "or", "but", "if", "then", "than", "that", "this",
    "for", "from", "with", "without", "before", "after", "during",
    "about", "above", "below", "between", "through", "into",
    "on", "in", "at", "to", "of", "by", "as", "an", "a",
    "any", "all", "each", "every", "some", "more", "most", "many",
    "what", "how", "who", "when", "where", "which", "why",
    "his", "her", "its", "their", "our", "my", "your",
    "he", "she", "it", "they", "we", "you", "us", "me",
    "win", "lose", "pass", "fail", "approve", "confirm", "reject",
    "over", "under", "per", "up", "down", "out",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "special", "general", "national", "annual",
    "first", "second", "third", "next", "last",
    "united", "states", "america", "american", "usa", "us",
    "mr", "mrs", "ms", "dr", "jr", "sr", "st",
}

NON_PERSON_WORDS = US_STATES | COUNTRIES | GOVT_TERMS | POLITICAL_TERMS | MISC_SKIP

# Also skip very short tokens and purely numeric
_MIN_NAME_LEN = 3


# ═════════════════════════════════════════════════════════════════════════════
# Person extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_persons(original_title: str) -> set[str]:
    """
    Extract likely person names from an ORIGINAL (capitalised) title.

    Heuristic: find capitalised words that are not at the very start of the
    sentence and are not known non-person words (states, govt terms, etc.).
    Returns a set of lowercase last-name tokens.
    """
    if not isinstance(original_title, str):
        return set()

    words = original_title.split()
    persons = set()

    for i, raw_word in enumerate(words):
        # Strip punctuation for comparison
        clean = re.sub(r"[^a-zA-Z\-']", "", raw_word)
        if len(clean) < _MIN_NAME_LEN:
            continue
        if not clean[0].isupper():
            continue
        low = clean.lower()
        if low in NON_PERSON_WORDS:
            continue
        # Skip the very first word (often capitalised just as sentence start)
        # unless followed by another capitalised word (likely a name like "Trump ...")
        if i == 0:
            if len(words) > 1:
                next_clean = re.sub(r"[^a-zA-Z]", "", words[1])
                if next_clean and next_clean[0].isupper() and next_clean.lower() not in NON_PERSON_WORDS:
                    persons.add(low)
            continue
        persons.add(low)

    return persons


# ═════════════════════════════════════════════════════════════════════════════
# Year / date extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_years(text: str) -> set[int]:
    """Extract 4-digit years (2020-2029) from text."""
    if not isinstance(text, str):
        return set()
    return {int(y) for y in re.findall(r"\b(20[2-3]\d)\b", text)}


# ═════════════════════════════════════════════════════════════════════════════
# Event loading & grouping
# ═════════════════════════════════════════════════════════════════════════════

def load_kalshi_events() -> pd.DataFrame:
    """
    Load cleaned Kalshi data, group by event_ticker, return one row per event.

    Columns: event_ticker, event_title, event_text, persons, years
    """
    clean = pd.read_csv(KALSHI_CLEAN)
    orig = pd.read_csv(KALSHI_ORIG)

    # Group cleaned data → event-level title & full text
    events = clean.groupby("event_ticker").agg(
        event_title=("event_title", "first"),
        event_rules=("event_rules", "first"),
    ).reset_index()
    events["event_text"] = (events["event_title"] + " " + events["event_rules"]).str.strip()

    # Extract persons from ORIGINAL (capitalised) titles
    orig_titles = orig.groupby("event_ticker")["title"].apply(list).reset_index()
    orig_titles.columns = ["event_ticker", "orig_titles"]

    events = events.merge(orig_titles, on="event_ticker", how="left")
    events["persons"] = events["orig_titles"].apply(
        lambda titles: set().union(*(extract_persons(t) for t in (titles or [])))
    )
    events["years"] = events["event_text"].apply(extract_years)
    events.drop(columns=["orig_titles"], inplace=True)

    print(f"Kalshi: {len(events)} events loaded")
    return events


def load_poly_events() -> pd.DataFrame:
    """
    Load cleaned Polymarket data, group by event_ticker, return one row per event.

    Columns: event_ticker, event_title, event_text, persons, years
    """
    clean = pd.read_csv(POLY_CLEAN)
    orig = pd.read_csv(POLY_ORIG)

    # Group cleaned data → event-level title & full text
    events = clean.groupby("event_ticker").agg(
        event_title=("event_title", "first"),
        event_rules=("event_rules", "first"),
        event_description=("event_description", "first"),
    ).reset_index()
    events["event_description"] = events["event_description"].fillna("")
    events["event_text"] = (
        events["event_title"] + " " + events["event_rules"] + " " + events["event_description"]
    ).str.strip()

    # Extract persons from ORIGINAL (capitalised) titles
    orig_titles = orig.groupby("event_ticker")["title"].apply(list).reset_index()
    orig_titles.columns = ["event_ticker", "orig_titles"]
    events = events.merge(orig_titles, on="event_ticker", how="left")
    events["persons"] = events["orig_titles"].apply(
        lambda titles: set().union(*(extract_persons(t) for t in (titles or [])))
    )
    events["years"] = events["event_text"].apply(extract_years)
    events.drop(columns=["orig_titles"], inplace=True)

    print(f"Polymarket: {len(events)} events loaded")
    return events


# ═════════════════════════════════════════════════════════════════════════════
# Blocking / candidate pair generation
# ═════════════════════════════════════════════════════════════════════════════

def _years_overlap(years_a: set[int], years_b: set[int]) -> bool:
    """
    True if years overlap or are within 1 year of each other.
    Handles "before 2026" vs "before december 31, 2025" by allowing ±1 year.
    """
    for ya in years_a:
        for yb in years_b:
            if abs(ya - yb) <= 1:
                return True
    return False


def get_candidate_pairs(
    kalshi_events: pd.DataFrame,
    poly_events: pd.DataFrame,
) -> list[tuple[int, int]]:
    """
    Apply person and date blocking constraints to generate candidate pairs.

    Returns list of (kalshi_idx, poly_idx) tuples into the respective DataFrames.

    Rules:
      - If EITHER event mentions person(s) → they must share ≥1 person.
      - If BOTH events mention year(s) → years must overlap (±1 year).
    """
    # Build inverted index: person → set of poly event indices
    poly_person_idx: dict[str, set[int]] = {}
    poly_no_person: set[int] = set()
    for i, row in poly_events.iterrows():
        if row["persons"]:
            for p in row["persons"]:
                poly_person_idx.setdefault(p, set()).add(i)
        else:
            poly_no_person.add(i)

    # Pre-extract poly years
    poly_years = {i: row["years"] for i, row in poly_events.iterrows()}

    pairs = []
    for ki, krow in kalshi_events.iterrows():
        k_persons = krow["persons"]
        k_years = krow["years"]

        # ── Person blocking ─────────────────────────────────────────────
        if k_persons:
            # Kalshi has persons → poly must share at least one
            candidates = set()
            for p in k_persons:
                candidates |= poly_person_idx.get(p, set())
        else:
            # Kalshi has no persons → only match poly events with no persons
            candidates = set(poly_no_person)

        # ── Date blocking ───────────────────────────────────────────────
        final = []
        for pi in candidates:
            p_years = poly_years[pi]
            # If both have years → must overlap
            if k_years and p_years:
                if not _years_overlap(k_years, p_years):
                    continue
            final.append((ki, pi))

        pairs.extend(final)

    print(f"Blocking: {len(pairs):,} candidate pairs "
          f"(from {len(kalshi_events):,} × {len(poly_events):,} = "
          f"{len(kalshi_events) * len(poly_events):,} total)")
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_matches(
    matches: list[dict],
    filename: str,
    *,
    score_col: str = "score",
    threshold: float = 0.0,
):
    """Save matched event pairs to CSV, filtered by threshold."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matches)
    if threshold > 0 and score_col in df.columns:
        df = df[df[score_col] >= threshold]
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} matches → {path}")
    return df
