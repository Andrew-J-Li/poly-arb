"""
Event matching: fuzzy on event TITLE + RULES/DESCRIPTION (full text).

Uses rapidfuzz token_sort_ratio on the concatenated event text
after blocking by person + date constraints.
"""

from rapidfuzz import fuzz
from matching.utils import (
    load_kalshi_events,
    load_poly_events,
    get_candidate_pairs,
    save_matches,
)

THRESHOLD = 60  # minimum fuzzy score to keep


def main():
    kalshi = load_kalshi_events()
    poly = load_poly_events()
    pairs = get_candidate_pairs(kalshi, poly)

    print(f"Scoring {len(pairs):,} pairs (fuzzy full) …")
    matches = []
    for ki, pi in pairs:
        k_text = kalshi.at[ki, "event_text"]
        p_text = poly.at[pi, "event_text"]

        score = fuzz.token_sort_ratio(k_text, p_text)

        if score >= THRESHOLD:
            matches.append({
                "kalshi_event": kalshi.at[ki, "event_ticker"],
                "poly_event": poly.at[pi, "event_ticker"],
                "kalshi_title": kalshi.at[ki, "event_title"],
                "poly_title": poly.at[pi, "event_title"],
                "score": round(score, 2),
            })

    df = save_matches(matches, "fuzzy_full.csv", score_col="score", threshold=0)
    print(f"\nTop 10 matches:")
    print(df.head(10)[["kalshi_title", "poly_title", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
