# src/fetch_data.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os


#  Replace with your actual API key
API_KEY = "1c5e8e2ca7f52e01da7c4c2038cd0c95080aa8fa60e350eb16d9374ff7e35db6"

BASE_URL = "https://api.openaq.org/v3/measurements"

HEADERS = {
    "X-API-Key": API_KEY
}

def fetch_data(city="Delhi", parameter="pm25", days=30):
    """
    Fetching air quality data for a city & parameter over certain days.
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    params = {
        "city": city,
        "parameter": parameter,
        "date_from": start_date.isoformat(),
        "date_to": end_date.isoformat(),
        "limit": 100,      # maximum per page
        "page": 1,
        "sort": "desc"     # latest first
    }

    all_results = []

    while True:
        print(f"Fetching page {params['page']}...")
        response = requests.get(BASE_URL, headers=HEADERS, params=params)

        if response.status_code != 200:
            print("Error:", response.json())
            break

        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)

        meta = data.get("meta", {})
        found = meta.get("found", 0)
        page = params["page"]
        limit = params["limit"]

        if page * limit >= found:
            break

        params["page"] += 1  # move to next page

    return all_results


def save_to_csv(data, filename="../data/air_quality.csv"):
    if not data:
        print("No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")


if __name__ == "__main__":
    data = fetch_data(city="Delhi", parameter="pm25", days=365*3)  # 3 years
    save_to_csv(data)
