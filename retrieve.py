import requests
import csv
import time

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
OUTPUT_FILE = "tickers/kalshi_settled.csv"
MAX_RETRIES = 5

FIELDS = [
    "ticker", "event_ticker", "market_type", "title", "subtitle",
    "yes_sub_title", "no_sub_title", "created_time", "updated_time",
    "open_time", "close_time", "expiration_time", "latest_expiration_time",
    "settlement_timer_seconds", "status", "response_price_units",
    "yes_bid", "yes_ask", "no_bid", "no_ask",
    "last_price", "volume", "volume_24h", "result",
    "can_close_early", "fractional_trading_enabled",
    "open_interest", "notional_value",
    "liquidity", "tick_size",
    "expected_expiration_time", "settlement_value",
    "settlement_ts", "expiration_value",
]


def fetch_settled_markets():
    all_markets = []
    cursor = None
    page = 0

    while True:
        params = {
            "status": "settled",
            "mve_filter": "exclude",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        for attempt in range(1, MAX_RETRIES + 1):
            resp = requests.get(BASE_URL, params=params)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"Rate limited, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise Exception(f"Failed after {MAX_RETRIES} retries due to rate limiting")

        data = resp.json()

        markets = data.get("markets", [])
        all_markets.extend(markets)
        page += 1
        print(f"Page {page}: fetched {len(markets)} markets (total: {len(all_markets)})")

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        time.sleep(0.1)

    return all_markets


def save_to_csv(markets, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(markets)
    print(f"Saved {len(markets)} markets to {filepath}")


if __name__ == "__main__":
    markets = fetch_settled_markets()
    save_to_csv(markets, OUTPUT_FILE)
