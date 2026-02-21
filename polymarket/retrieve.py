import requests
import pandas as pd
import json
import time
from pathlib import Path

BASE_URL = "https://gamma-api.polymarket.com/events"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "test.csv"
MAX_RETRIES = 5
TAG_ID = 2  # Politics
PAGE_SIZE = 100

FIELDS = ["ticker", "event_ticker", "title", "result", "description", "event_description"]


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


def parse_result(outcomes_str, prices_str):
    """
    Parse outcomes and outcomePrices JSON strings.
    Returns the winning outcome ("yes"/"no") or None if not a Yes/No market.
    """
    try:
        outcomes = json.loads(outcomes_str)
        prices = json.loads(prices_str)
    except (json.JSONDecodeError, TypeError):
        return None

    # Only keep Yes/No binary markets
    normalized = [o.strip().lower() for o in outcomes]
    if sorted(normalized) != ["no", "yes"]:
        return None

    # Find the outcome with price == "1" (the winner)
    for outcome, price in zip(outcomes, prices):
        if float(price) == 1.0:
            return outcome.strip().lower()

    return None


def fetch_closed_politics_events():
    """Fetch all closed politics events, paginating with offset/limit."""
    all_rows = []
    offset = 0

    while True:
        params = {
            "tag_id": TAG_ID,
            "closed": True,
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        events = request_with_retries(BASE_URL, params)

        if not events:
            break

        for event in events:
            event_ticker = event.get("slug", "")
            event_description = event.get("description", "") or ""

            for market in event.get("markets", []):
                outcomes_str = market.get("outcomes")
                prices_str = market.get("outcomePrices")

                result = parse_result(outcomes_str, prices_str)
                if result is None:
                    continue

                all_rows.append({
                    "ticker": market.get("slug", "") or market.get("conditionId", ""),
                    "event_ticker": event_ticker,
                    "title": market.get("question", ""),
                    "result": result,
                    "description": (market.get("description", "") or "").replace("\n", " ").strip(),
                    "event_description": event_description.replace("\n", " ").strip(),
                })

        page_num = offset // PAGE_SIZE + 1
        print(f"Page {page_num}: fetched {len(events)} events (total rows: {len(all_rows)})")

        if len(events) < PAGE_SIZE:
            break

        save_markets(all_rows)
        offset += PAGE_SIZE
        time.sleep(0.1)

    return all_rows


def save_markets(rows):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No markets to save yet, skipping.")
        return

    df = pd.DataFrame(rows)[FIELDS]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} markets to {OUTPUT_FILE}")


if __name__ == "__main__":
    rows = fetch_closed_politics_events()
    save_markets(rows)
    print(f"\nDone. Total: {len(rows)} settled Yes/No markets.")
