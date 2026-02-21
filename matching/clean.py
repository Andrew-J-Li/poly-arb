import re
import calendar
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KALSHI_INPUT = DATA_DIR / "kalshi_processed.csv"
KALSHI_OUTPUT = DATA_DIR / "kalshi_clean.csv"

POLY_INPUT = DATA_DIR / "polymarket_settled.csv"
POLY_OUTPUT = DATA_DIR / "polymarket_clean.csv"

# Columns to clean per dataset
KALSHI_TEXT_COLS = ["title", "rules_primary", "rules_secondary", "event_title", "event_rules"]
POLY_TEXT_COLS = ["title", "description", "event_description"]

# ── Abbreviation maps ────────────────────────────────────────────────────────

MONTH_ABBREVS = {
    r"\bjan\b": "january",
    r"\bfeb\b": "february",
    r"\bmar\b": "march",
    r"\bapr\b": "april",
    r"\bjun\b": "june",
    r"\bjul\b": "july",
    r"\baug\b": "august",
    r"\bsep\b": "september",
    r"\bsept\b": "september",
    r"\boct\b": "october",
    r"\bnov\b": "november",
    r"\bdec\b": "december",
}

COUNTRY_ABBREVS = {
    r"\bu\.s\.a\.": "united states",
    r"\bu\.s\.": "united states",
    r"\busa\b": "united states",
    # Standalone "US" but not inside words like "focus", "us" pronoun at start
    # Match uppercase US before lowercasing — handled specially
}

PARTY_ABBREVS = {
    r"\bgop\b": "republican",
    r"\bdems\b": "democrats",
    r"\bdem\b": "democrat",
    r"\brep\b": "representative",
    r"\(d\)": "(democrat)",
    r"\(r\)": "(republican)",
    r"\(i\)": "(independent)",
}

OTHER_ABBREVS = {
    r"\bd\.c\.": "dc",
    r"\bh\.r\.": "hr",
    r"\bj\.d\.": "jd",
    r"\bgov\b": "governor",
    r"\bsen\b": "senator",
    r"\bpres\b": "president",
    r"\bsecy\b": "secretary",
    r"\bsec\b": "secretary",
    r"\badmin\b": "administration",
}


MONTH_NUM_TO_NAME = {i: calendar.month_name[i].lower() for i in range(1, 13)}


def normalize_us(text: str) -> str:
    """
    Replace uppercase 'US' with 'united states' BEFORE lowercasing,
    so we don't accidentally replace the pronoun 'us'.
    """
    return re.sub(r"\bUS\b", "united states", text)


def normalize_dates(text: str) -> str:
    """
    Standardise dates embedded in free text to 'month day, year' format.

    Handles:
      - ISO timestamps  : 2026-02-01T15:00:00.000Z → february 1, 2026
      - ISO date only   : 2026-02-01 → february 1, 2026
      - Ordinal suffixes: september 25th → september 25
    """
    # ISO timestamps / dates: 2026-02-01T15:00:00.000Z or 2026-02-01
    def _iso_to_text(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{MONTH_NUM_TO_NAME[mo]} {d}, {y}"

    text = re.sub(
        r"\b(\d{4})-(\d{2})-(\d{2})(?:t[\d:.]+z?)?\b",
        _iso_to_text,
        text,
    )

    # Strip ordinal suffixes on day numbers: "25th" → "25", "1st" → "1"
    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)

    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Replace uppercase US -> united states before lowering
    text = normalize_us(text)

    # Lowercase
    text = text.lower()

    # Expand abbreviations (order matters: longest patterns first for U.S.A.)
    for pattern, replacement in COUNTRY_ABBREVS.items():
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in MONTH_ABBREVS.items():
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in PARTY_ABBREVS.items():
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in OTHER_ABBREVS.items():
        text = re.sub(pattern, replacement, text)

    # Normalise dates to consistent "month day, year" format
    text = normalize_dates(text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)
    return df


def main():
    # ── Kalshi ────────────────────────────────────────────────────────────
    print(f"Reading {KALSHI_INPUT}")
    kdf = pd.read_csv(KALSHI_INPUT)
    print(f"  Loaded {len(kdf)} rows, cleaning columns: {KALSHI_TEXT_COLS}")
    kdf = clean_dataframe(kdf, KALSHI_TEXT_COLS)
    kdf.to_csv(KALSHI_OUTPUT, index=False)
    print(f"  Saved → {KALSHI_OUTPUT}")

    # ── Polymarket ────────────────────────────────────────────────────────
    print(f"\nReading {POLY_INPUT}")
    pdf = pd.read_csv(POLY_INPUT)
    print(f"  Loaded {len(pdf)} rows, cleaning columns: {POLY_TEXT_COLS}")
    pdf = clean_dataframe(pdf, POLY_TEXT_COLS)
    pdf.to_csv(POLY_OUTPUT, index=False)
    print(f"  Saved → {POLY_OUTPUT}")

    # ── Quick sample ──────────────────────────────────────────────────────
    print("\n── Kalshi sample ──")
    for col in KALSHI_TEXT_COLS:
        sample = kdf[col].iloc[0]
        print(f"  {col}: {sample[:120]}")

    print("\n── Polymarket sample ──")
    for col in POLY_TEXT_COLS:
        sample = pdf[col].iloc[0]
        print(f"  {col}: {sample[:120]}")


if __name__ == "__main__":
    main()
