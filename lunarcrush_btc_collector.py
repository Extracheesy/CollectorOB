import os
import time
import json
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()  # loads .env from current directory (or parent)

# --------------------
# Config
# --------------------
API_URL = "https://lunarcrush.com/api4/public/coins/list/v1"
API_KEY = os.getenv("LUNAR_API_KEY")
OUT_DIR = "results_lunar"
OUT_CSV = os.path.join(OUT_DIR, "lunarcrush_btc.csv")
SIZE_LIMIT_MB = 50
INTERVAL_MINUTES = 5

# BTC symbol we want
TARGET_SYMBOL = "BTC"

# --------------------
# Logging setup
# --------------------
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def get_next_5min_delay() -> float:
    """Return seconds until next 5-min boundary."""
    now = datetime.now(timezone.utc)
    minute = now.minute
    second = now.second
    micro = now.microsecond

    # Next 5-min mark (0, 5, 10, 15, ...)
    next_min_block = (minute // 5 + 1) * 5
    if next_min_block >= 60:
        # next hour, minute = 0
        next_min_block = 0
        # we'll let datetime handle hour increment by just computing delta in seconds

    # Build next boundary same hour for simplicity, then adjust if rolled over
    boundary = now.replace(minute=next_min_block, second=0, microsecond=0)
    if boundary <= now:
        # rolled back, add 1 hour
        boundary = boundary.replace(hour=(boundary.hour + 1) % 24)

    delta = (boundary - now).total_seconds()
    if delta < 0:
        delta = 0
    return delta


def rotate_if_too_big(path: str, size_limit_mb: int = 50) -> None:
    """Rotate CSV if larger than size_limit_mb."""
    if not os.path.exists(path):
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb <= size_limit_mb:
        return

    # Rotate: rename with timestamp
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    new_name = path.replace(".csv", f"_{ts}.csv")
    logger.info(f"Rotating {path} -> {new_name} (size={size_mb:.2f} MB > {size_limit_mb} MB)")
    os.rename(path, new_name)


def fetch_lunarcrush_snapshot() -> dict:
    """Call LunarCrush coins/list/v1 and return JSON dict."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "limit": 200,
        "sort": "galaxy_score",
        "page": 0,
    }

    logger.info("Requesting LunarCrush data ...")
    print("url:", API_URL)
    print("api key", API_KEY)

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    print("HTTP status:", resp.status_code)
    try:
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"HTTP error from LunarCrush: {e}")
        print("Response text:", resp.text[:1000], "...")
        raise

    try:
        data = resp.json()
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from LunarCrush")
        print("Raw text:", resp.text[:1000], "...")
        raise

    # Debug: print a small part of the JSON, not the whole monster
    print("Top-level keys from LunarCrush:", list(data.keys()))
    if "config" in data:
        print("Config snippet:", json.dumps(data["config"], indent=2)[:500], "...")
    if "data" in data:
        print("Number of coins in 'data':", len(data["data"]))

    return data


def extract_btc_row(raw_json: dict, snapshot_ts: datetime) -> dict | None:
    """
    From full LunarCrush JSON, extract a single dict for BTC.
    Returns None if BTC not found.
    """
    coins = raw_json.get("data")
    if coins is None:
        logger.error("JSON has no 'data' field")
        print("Raw JSON keys:", list(raw_json.keys()))
        return None

    # Find BTC by symbol
    btc_coin = None
    for c in coins:
        if c.get("symbol") == TARGET_SYMBOL:
            btc_coin = c
            break

    if btc_coin is None:
        logger.warning(f"No coin with symbol '{TARGET_SYMBOL}' found in LunarCrush response.")
        return None

    # Debug: show BTC coin keys
    print(f"Found BTC coin entry, keys: {list(btc_coin.keys())}")

    def safe_get(key, default=None):
        return btc_coin.get(key, default)

    row = {
        "snapshot_utc": snapshot_ts.isoformat(),
        "symbol": safe_get("symbol"),
        "name": safe_get("name"),
        "price": safe_get("price"),
        "price_btc": safe_get("price_btc"),
        "volume_24h": safe_get("volume_24h"),
        "market_cap": safe_get("market_cap"),
        "market_cap_rank": safe_get("market_cap_rank"),
        "galaxy_score": safe_get("galaxy_score"),
        "galaxy_score_previous": safe_get("galaxy_score_previous"),
        "alt_rank": safe_get("alt_rank"),
        "alt_rank_previous": safe_get("alt_rank_previous"),
        "social_volume_24h": safe_get("social_volume_24h"),
        "interactions_24h": safe_get("interactions_24h"),
        "sentiment": safe_get("sentiment"),
        "percent_change_1h": safe_get("percent_change_1h"),
        "percent_change_24h": safe_get("percent_change_24h"),
        "percent_change_7d": safe_get("percent_change_7d"),
        "categories": safe_get("categories"),
        "topic": safe_get("topic"),
    }

    print("BTC row extracted:", row)
    return row


def append_row_to_csv(row: dict) -> None:
    """Append a single row dict to OUT_CSV with rotation."""
    rotate_if_too_big(OUT_CSV, SIZE_LIMIT_MB)

    df = pd.DataFrame([row])

    if not os.path.exists(OUT_CSV):
        logger.info(f"Creating new CSV file {OUT_CSV}")
        df.to_csv(OUT_CSV, index=False)
    else:
        logger.info(f"Appending row to {OUT_CSV}")
        df.to_csv(OUT_CSV, index=False, mode="a", header=False)


def main_loop():
    logger.info(
        f"Starting LunarCrush BTC collector (coins/list/v1, aligned to {INTERVAL_MINUTES}-min boundaries, "
        f"out_dir={OUT_DIR}, size_limit={SIZE_LIMIT_MB} MB)"
    )

    delay = get_next_5min_delay()
    logger.info(f"Sleeping {delay:.1f} seconds until next {INTERVAL_MINUTES}-min boundary")
    time.sleep(delay)

    while True:
        snapshot_ts = datetime.now(timezone.utc)
        logger.info(f"Collecting snapshot at {snapshot_ts.isoformat()}")

        try:
            raw = fetch_lunarcrush_snapshot()
            row = extract_btc_row(raw, snapshot_ts)
            if row is None:
                logger.warning("No BTC row extracted. Skipping CSV write for this snapshot.")
            else:
                append_row_to_csv(row)
                logger.info("Row successfully written.")
        except Exception as e:
            logger.exception(f"Error during fetch/parse/write: {e}")

        logger.info(f"Sleeping {INTERVAL_MINUTES} minutes until next snapshot ...")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main_loop()
