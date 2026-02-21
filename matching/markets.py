"""
Market-level matching for paired Kalshi / Polymarket events.

Pipeline
========
1. **Consolidate** event-level match files from all four methods
   (semantic_title, semantic_full, fuzzy_title, fuzzy_full).
   Normalise scores to 0-1, keep the best score per event pair.

2. **Mutual-best-match filter** — an event pair is kept only when the
   Polymarket event is the *top* match for that Kalshi event *and*
   the Kalshi event is the *top* match for that Polymarket event.
   Each event ticker therefore appears at most once.

3. **Market-level matching** within each paired event:
   • If both event titles contain ``[blank]``, extract the blank fills
     from the individual market titles and compare them with
     ``token_sort_ratio`` (threshold 85).
   • Otherwise, fall back to direct title-vs-title fuzzy matching
     (threshold 90).
   Market pairs are also deduplicated so each market ticker appears
   at most once (mutual best within the event).

Outputs
-------
* ``data/matched_events.csv``  — one row per event pair
* ``data/matched_markets.csv`` — one row per market pair
"""

import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATCH_DIR = DATA_DIR / "event_matches"

# ── Tunable knobs ─────────────────────────────────────────────────────────────

MATCH_FILES = ["semantic_title", "semantic_full", "fuzzy_title", "fuzzy_full"]
FUZZY_SCALE = 100               # fuzzy methods score 0-100; semantic 0-1
MIN_EVENT_SCORE = 0.70          # after normalisation to 0-1

BLANK_MATCH_THRESHOLD = 85      # token_sort_ratio for [blank]-fill comparison
TITLE_MATCH_THRESHOLD = 90      # token_sort_ratio for direct title comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — consolidate event-level matches
# ═══════════════════════════════════════════════════════════════════════════════

def load_event_matches() -> pd.DataFrame:
    """Load all four match files, normalise to 0-1, keep best per pair."""
    frames = []
    for name in MATCH_FILES:
        path = MATCH_DIR / f"{name}.csv"
        if not path.exists():
            print(f"  [skip] {name} — file not found")
            continue
        df = pd.read_csv(path)
        if "fuzzy" in name:
            df["score"] = df["score"] / FUZZY_SCALE
        frames.append(df[["kalshi_event", "poly_event", "score"]])
        print(f"  {name}: {len(df):,} pairs")

    if not frames:
        raise FileNotFoundError("No match files found in " + str(MATCH_DIR))

    combined = pd.concat(frames, ignore_index=True)
    # Best score per unique event pair
    best = (
        combined
        .sort_values("score", ascending=False)
        .drop_duplicates(["kalshi_event", "poly_event"], keep="first")
        .reset_index(drop=True)
    )
    best = best[best["score"] >= MIN_EVENT_SCORE].reset_index(drop=True)
    print(f"  Combined & score >= {MIN_EVENT_SCORE}: {len(best):,} unique event pairs")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — mutual best match
# ═══════════════════════════════════════════════════════════════════════════════

def mutual_best_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only pairs where the match is the highest-scoring for BOTH the
    Kalshi event and the Polymarket event.
    """
    # Best poly match for each Kalshi event
    k_best_idx = df.groupby("kalshi_event")["score"].idxmax()
    k_best = df.loc[k_best_idx]
    k_pairs = set(zip(k_best["kalshi_event"], k_best["poly_event"]))

    # Best Kalshi match for each Poly event
    p_best_idx = df.groupby("poly_event")["score"].idxmax()
    p_best = df.loc[p_best_idx]
    p_pairs = set(zip(p_best["kalshi_event"], p_best["poly_event"]))

    # Intersection: both sides agree
    mutual = k_pairs & p_pairs
    mask = pd.Series(
        [(r["kalshi_event"], r["poly_event"]) in mutual for _, r in df.iterrows()],
        index=df.index,
    )
    result = df[mask].sort_values("score", ascending=False).reset_index(drop=True)
    print(
        f"  Mutual best: {len(result)} event pairs "
        f"(from {df['kalshi_event'].nunique()} K events, "
        f"{df['poly_event'].nunique()} P events)"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — market-level matching
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_blank(template: str, title: str) -> str | None:
    """
    Given an event_title template that contains ``[blank]`` and an individual
    market title, return the text that fills the blank.

    Returns *None* when extraction is impossible.
    """
    if "[blank]" not in template:
        return None

    prefix, suffix = template.split("[blank]", 1)

    # Strip prefix
    if prefix and not title.startswith(prefix):
        return None
    rest = title[len(prefix):] if prefix else title

    # Strip suffix
    suffix_s = suffix.strip()
    rest_s = rest.rstrip()
    if suffix_s and rest_s.endswith(suffix_s):
        blank = rest_s[: -len(suffix_s)]
    elif not suffix_s:
        blank = rest
    else:
        # Suffix mismatch — could be trailing punctuation differences; keep all
        blank = rest

    blank = blank.strip()
    return blank or None


def _market_mutual_best(candidates: list[dict]) -> list[dict]:
    """
    From all above-threshold market pairs within one event pair, keep only
    those that are the best match for both their Kalshi and Polymarket ticker.
    """
    if not candidates:
        return []

    cdf = pd.DataFrame(candidates)

    k_best_idx = cdf.groupby("kalshi_ticker")["market_score"].idxmax()
    k_pairs = set(zip(cdf.loc[k_best_idx, "kalshi_ticker"],
                       cdf.loc[k_best_idx, "poly_ticker"]))

    p_best_idx = cdf.groupby("poly_ticker")["market_score"].idxmax()
    p_pairs = set(zip(cdf.loc[p_best_idx, "kalshi_ticker"],
                       cdf.loc[p_best_idx, "poly_ticker"]))

    mutual = k_pairs & p_pairs
    mask = pd.Series(
        [(r["kalshi_ticker"], r["poly_ticker"]) in mutual for _, r in cdf.iterrows()],
        index=cdf.index,
    )
    return cdf[mask].to_dict("records")


def match_markets_in_pair(
    k_event: str,
    p_event: str,
    event_score: float,
    k_markets: pd.DataFrame,
    p_markets: pd.DataFrame,
) -> list[dict]:
    """
    Match individual markets within one paired event.

    Uses [blank]-fill comparison when both templates contain ``[blank]``,
    otherwise falls back to direct title fuzzy matching.
    """
    k_template = str(k_markets.iloc[0].get("event_title", ""))
    p_template = str(p_markets.iloc[0].get("event_title", ""))

    use_blank = "[blank]" in k_template and "[blank]" in p_template
    threshold = BLANK_MATCH_THRESHOLD if use_blank else TITLE_MATCH_THRESHOLD

    # Pre-extract blanks (or None if not using blank mode)
    k_items = []
    for _, r in k_markets.iterrows():
        blank = _extract_blank(k_template, r["title"]) if use_blank else None
        k_items.append({
            "ticker": r["ticker"], "title": r["title"],
            "result": r["result"], "blank": blank,
        })

    p_items = []
    for _, r in p_markets.iterrows():
        blank = _extract_blank(p_template, r["title"]) if use_blank else None
        p_items.append({
            "ticker": r["ticker"], "title": r["title"],
            "result": r["result"], "blank": blank,
        })

    # Score every (K market, P market) pair
    candidates = []
    for ki in k_items:
        for pi in p_items:
            if use_blank:
                if ki["blank"] is None or pi["blank"] is None:
                    continue
                score = fuzz.token_sort_ratio(ki["blank"], pi["blank"])
            else:
                score = fuzz.token_sort_ratio(ki["title"], pi["title"])

            if score >= threshold:
                candidates.append({
                    "kalshi_event": k_event,
                    "poly_event": p_event,
                    "event_score": round(event_score, 4),
                    "kalshi_ticker": ki["ticker"],
                    "poly_ticker": pi["ticker"],
                    "kalshi_title": ki["title"],
                    "poly_title": pi["title"],
                    "kalshi_blank": ki["blank"] or "",
                    "poly_blank": pi["blank"] or "",
                    "kalshi_result": ki["result"],
                    "poly_result": pi["result"],
                    "market_score": round(score, 2),
                })

    # Mutual best within this event pair
    return _market_mutual_best(candidates)


# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Step 1 ────────────────────────────────────────────────────────────
    print("Step 1: Consolidating event-level matches")
    event_df = load_event_matches()

    # ── Step 2 ────────────────────────────────────────────────────────────
    print("\nStep 2: Mutual-best-match filter")
    event_df = mutual_best_match(event_df)

    # Enrich with event titles from the clean CSVs
    kclean = pd.read_csv(DATA_DIR / "kalshi_clean.csv")
    pclean = pd.read_csv(DATA_DIR / "polymarket_clean.csv")

    k_titles = kclean.drop_duplicates("event_ticker").set_index("event_ticker")["event_title"]
    p_titles = pclean.drop_duplicates("event_ticker").set_index("event_ticker")["event_title"]
    event_df["kalshi_title"] = event_df["kalshi_event"].map(k_titles)
    event_df["poly_title"] = event_df["poly_event"].map(p_titles)

    # ── Step 3 ────────────────────────────────────────────────────────────
    print(f"\nStep 3: Market-level matching ({len(event_df)} event pairs)")
    all_market_matches: list[dict] = []
    blank_events = 0
    direct_events = 0

    for _, erow in event_df.iterrows():
        k_mkts = kclean[kclean["event_ticker"] == erow["kalshi_event"]]
        p_mkts = pclean[pclean["event_ticker"] == erow["poly_event"]]

        if k_mkts.empty or p_mkts.empty:
            continue

        matches = match_markets_in_pair(
            erow["kalshi_event"], erow["poly_event"], erow["score"],
            k_mkts, p_mkts,
        )

        # Track matching method
        k_tmpl = str(k_mkts.iloc[0].get("event_title", ""))
        p_tmpl = str(p_mkts.iloc[0].get("event_title", ""))
        if "[blank]" in k_tmpl and "[blank]" in p_tmpl:
            blank_events += 1
        else:
            direct_events += 1

        all_market_matches.extend(matches)

    # ── Save results ──────────────────────────────────────────────────────
    event_df.to_csv(DATA_DIR / "matched_events.csv", index=False)
    print(f"\nSaved {len(event_df)} event pairs -> {DATA_DIR / 'matched_events.csv'}")

    if all_market_matches:
        mdf = (
            pd.DataFrame(all_market_matches)
            .sort_values("market_score", ascending=False)
            .reset_index(drop=True)
        )
        mdf.to_csv(DATA_DIR / "matched_markets.csv", index=False)
        print(f"Saved {len(mdf)} market pairs -> {DATA_DIR / 'matched_markets.csv'}")
    else:
        mdf = pd.DataFrame()
        print("No market-level matches found.")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Event pairs:       {len(event_df)}")
    print(f"    via [blank]:     {blank_events}")
    print(f"    via direct title:{direct_events}")
    print(f"  Market pairs:      {len(mdf)}")

    if len(mdf) > 0:
        exact = (mdf["market_score"] >= 100).sum()
        high  = ((mdf["market_score"] >= 90) & (mdf["market_score"] < 100)).sum()
        other = len(mdf) - exact - high
        print(f"    exact (100):     {exact}")
        print(f"    high  (90-99):   {high}")
        print(f"    other (85-89):   {other}")

        agree = (mdf["kalshi_result"] == mdf["poly_result"]).sum()
        print(f"  Result agreement:  {agree}/{len(mdf)} ({agree/len(mdf)*100:.1f}%)")

        print(f"\n── Top 15 market matches ──")
        show_cols = ["kalshi_title", "poly_title", "market_score",
                     "kalshi_result", "poly_result"]
        print(mdf.head(15)[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
