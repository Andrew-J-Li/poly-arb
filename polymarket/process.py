import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR.parent / "data" / "polymarket_settled.csv"
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "polymarket_processed.csv"

# Minimum meaningful content (excluding [blank]) for a generalized title
MIN_CONTENT_LEN = 15


def clean_newlines(df):
    """Remove all newlines from description and event_description."""
    for col in ("description", "event_description"):
        df[col] = df[col].fillna("").str.replace(r"\r?\n", " ", regex=True).str.strip()
    return df


def _common_prefix_len(word_lists):
    """Return the number of leading words common to every list."""
    if not word_lists:
        return 0
    min_len = min(len(w) for w in word_lists)
    for i in range(min_len):
        if len({wl[i] for wl in word_lists}) != 1:
            return i
    return min_len


def _common_suffix_len(word_lists):
    """Return the number of trailing words common to every list."""
    if not word_lists:
        return 0
    min_len = min(len(w) for w in word_lists)
    for i in range(1, min_len + 1):
        if len({wl[-i] for wl in word_lists}) != 1:
            return i - 1
    return min_len


def generalize_strings(strings):
    """
    Given a list of strings, find the longest common prefix and suffix of
    words, and replace the variable middle with a single [blank].

    Returns the generalized string.  If all strings are identical, returns
    that string unchanged.  If the group has only one string, returns it as-is.
    """
    non_empty = [s for s in strings if s.strip()]
    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return non_empty[0]

    # If all identical, return as-is
    if len(set(non_empty)) == 1:
        return non_empty[0]

    word_lists = [s.split() for s in non_empty]
    prefix_len = _common_prefix_len(word_lists)
    suffix_len = _common_suffix_len(word_lists)

    prefix_words = word_lists[0][:prefix_len]
    suffix_words = word_lists[0][-suffix_len:] if suffix_len > 0 else []

    parts = []
    if prefix_words:
        parts.append(" ".join(prefix_words))
    parts.append("[blank]")
    if suffix_words:
        parts.append(" ".join(suffix_words))

    return " ".join(parts)


def _title_content(title: str) -> str:
    """Return the title with [blank] removed — the 'meaningful' portion."""
    return title.replace("[blank]", "").strip()


def add_event_columns(df):
    """Add event_title and event_rules columns."""
    event_titles = {}
    event_rules = {}

    for event_ticker, group in df.groupby("event_ticker"):
        titles = group["title"].tolist()
        descriptions = group["description"].tolist()

        gen_title = generalize_strings(titles)

        # Fallback: if generalized title is too generic, use first market title
        if len(_title_content(gen_title)) < MIN_CONTENT_LEN:
            gen_title = titles[0] if titles else gen_title

        event_titles[event_ticker] = gen_title
        event_rules[event_ticker] = generalize_strings(descriptions)

    df["event_title"] = df["event_ticker"].map(event_titles)
    df["event_rules"] = df["event_ticker"].map(event_rules)

    return df


def main():
    print(f"Reading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} markets across {df['event_ticker'].nunique()} events")

    df = clean_newlines(df)
    df = add_event_columns(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")

    # Show a few examples
    sample_events = df["event_ticker"].value_counts().head(3).index
    for evt in sample_events:
        row = df.loc[df["event_ticker"] == evt].iloc[0]
        print(f"\n--- {evt} ---")
        print(f"  event_title: {row['event_title']}")
        print(f"  event_rules: {row['event_rules'][:120]}")


if __name__ == "__main__":
    main()
