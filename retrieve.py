import os
import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "tickers" / "kalshi_settled.csv"
MAX_RETRIES = 5
CATEGORY = "Politics"

FIELDS = ["ticker", "event_ticker", "title", "result", "rules_primary", "rules_secondary"]


def request_with_retries(url, params):
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, params=params)

        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise Exception("Max retries exceeded due to rate limiting")


def fetch_politics_series():
    """Fetch all series tickers in the Politics category."""
    data = request_with_retries(SERIES_URL, {"category": CATEGORY})
    series = data.get("series", [])
    # Filter client-side in case API ignores the category param
    politics_series = [s for s in series if s.get("category", "").lower() == CATEGORY.lower()]
    tickers = [s["ticker"] for s in politics_series]
    print(f"Found {len(tickers)} series in '{CATEGORY}' category (out of {len(series)} total)")
    return tickers


def fetch_settled_markets_for_series(series_ticker):
    """Fetch all settled markets for a single series ticker."""
    markets = []
    cursor = None
    page = 0

    while True:
        params = {
            "status": "settled",
            "mve_filter": "exclude",
            "series_ticker": series_ticker,
            "limit": 1000,
        }

        if cursor:
            params["cursor"] = cursor

        data = request_with_retries(BASE_URL, params)

        batch = data.get("markets", [])
        markets.extend(batch)

        page += 1
        print(f"  [{series_ticker}] Page {page}: fetched {len(batch)} (total {len(markets)})")

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

        time.sleep(0.1)

    return markets


def fetch_settled_markets():
    series_tickers = fetch_politics_series()
    all_markets = []

    for i, series_ticker in enumerate(series_tickers, 1):
        print(f"\n[{i}/{len(series_tickers)}] Fetching series: {series_ticker}")
        markets = fetch_settled_markets_for_series(series_ticker)
        all_markets.extend(markets)
        save_markets(all_markets, OUTPUT_FILE)
        time.sleep(0.1)

    return all_markets


def save_markets(markets, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not markets:
        print("No markets to save yet, skipping.")
        return

    df = pd.DataFrame(markets)[FIELDS]
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} markets to {filepath}")


if __name__ == "__main__":
    markets = fetch_settled_markets()
    save_markets(markets, OUTPUT_FILE)
