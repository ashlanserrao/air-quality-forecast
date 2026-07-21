# src/cleaning.py — shared PM2.5 cleaning/imputation logic (extracted from pm25_eda.ipynb)
import numpy as np
import pandas as pd


def clean_pm25_series(series: pd.Series) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Clean a daily PM2.5 series and report which dates were missing or imputed.

    Steps (matching pm25_eda.ipynb):
      - treat 0, negative, and >600 values as invalid (nulled)
      - forward/back-fill short gaps (limit 7 days)
      - time-interpolate remaining medium gaps (limit 30 days)
      - drop any dates that still couldn't be filled

    Returns (cleaned_series, bad_dates) where bad_dates lists every date whose
    final value is not the original, valid, as-measured reading (i.e. it was
    missing, out of range, or filled/interpolated).
    """
    s = series.copy()

    valid = s.notna() & (s != 0) & (s >= 0) & (s <= 600)
    bad_dates = s.index[~valid].tolist()

    s = s.where(valid, np.nan)
    s = s.ffill(limit=7).bfill(limit=7)
    s = s.interpolate(method="time", limit=30)
    s = s.dropna()

    return s, sorted(bad_dates)
