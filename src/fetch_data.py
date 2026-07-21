# src/fetch_data.py
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from cleaning import clean_pm25_series

API_KEY = os.environ.get("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAQ_API_KEY environment variable is not set. "
        "Set it (e.g. in a .env file loaded via python-dotenv, or `export OPENAQ_API_KEY=...`) before running this script."
    )

BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": API_KEY}

# PM2.5 sensor at OpenAQ location 8118 (New Delhi) — looked up once via
# GET /v3/locations/8118/sensors and hardcoded since it doesn't change.
PM25_SENSOR_ID = 23534

SEQ_LENGTH = 30
FETCH_DAYS = 40  # pull extra days so cleaning still leaves >= 30 valid days
MAX_BAD_DAYS = 5


def fetch_daily_pm25(sensor_id: int = PM25_SENSOR_ID, days: int = FETCH_DAYS) -> pd.Series:
    """Fetch daily-aggregated PM2.5 values for a sensor over the last `days` days."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    params = {
        "date_from": start_date.strftime("%Y-%m-%d"),
        "date_to": end_date.strftime("%Y-%m-%d"),
        "limit": 100,
        "page": 1,
    }

    all_results = []
    while True:
        response = requests.get(
            f"{BASE_URL}/sensors/{sensor_id}/days", headers=HEADERS, params=params
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)

        meta = data.get("meta", {})
        found = meta.get("found", 0)
        if params["page"] * params["limit"] >= found:
            break
        params["page"] += 1

    dates = [
        pd.to_datetime(r["period"]["datetimeFrom"]["local"]).normalize()
        for r in all_results
    ]
    values = [r["value"] for r in all_results]

    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name="pm25")
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series


def get_recent_clean_history(seq_length: int = SEQ_LENGTH) -> pd.Series:
    """Fetch, clean, and return the most recent `seq_length` valid PM2.5 days.

    Raises ValueError if more than MAX_BAD_DAYS of the returned window were
    missing or imputed rather than genuine measurements.
    """
    raw = fetch_daily_pm25()
    cleaned, bad_dates = clean_pm25_series(raw)

    if len(cleaned) < seq_length:
        raise ValueError(
            f"Only {len(cleaned)} valid days available after cleaning; need {seq_length}."
        )

    recent = cleaned.tail(seq_length)
    bad_in_window = [d for d in bad_dates if d in recent.index]

    if len(bad_in_window) > MAX_BAD_DAYS:
        bad_str = ", ".join(d.strftime("%Y-%m-%d") for d in bad_in_window)
        raise ValueError(
            f"{len(bad_in_window)} of the last {seq_length} days were missing or "
            f"imputed (max allowed: {MAX_BAD_DAYS}). Bad dates: {bad_str}"
        )

    return recent


if __name__ == "__main__":
    history = get_recent_clean_history()
    print(f"Fetched {len(history)} clean days:")
    print(history)
