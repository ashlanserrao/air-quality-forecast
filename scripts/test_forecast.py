# scripts/test_forecast.py — standalone sanity check for forecasting.forecast()
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecasting import forecast

# Last 30 daily PM2.5 values (2024-09-27 -> 2024-10-26) from data/processed/pm25_daily_final.csv
history = [
    7.6, 18.791666666666668, 28.125, 59.0, 60.333333333333336,
    59.125, 48.29166666666666, 44.73913043478261, 33.875, 36.04166666666666,
    49.0, 61.583333333333336, 55.25, 47.04166666666666, 56.54166666666666,
    80.65217391304348, 80.0, 83.58333333333333, 79.75, 96.54166666666669,
    109.83333333333331, 99.08333333333331, 105.125, 121.83333333333331, 132.83333333333334,
    153.91666666666666, 186.13333333333333, 186.13333333333333, 186.13333333333333, 186.13333333333333,
]

if __name__ == "__main__":
    preds = forecast(history, horizon=7)
    print("7-day PM2.5 forecast:")
    for day, val in enumerate(preds, start=1):
        print(f"  Day {day}: {val:.2f}")
