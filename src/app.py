# app.py  Streamlit interactive forecaster for daily PM2.5 (Delhi)
import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import timedelta

SEQ_LENGTH = 30
DATA_PATH = "pm25_daily_final.csv"       
MODEL_PATH = "artifacts/best_lstm.keras" 
SCALER_PATH = "artifacts/scaler.joblib"  


st.set_page_config(page_title="Delhi PM2.5 Forecast — LSTM", layout="wide")

@st.cache_data
def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load daily CSV robustly and return DataFrame indexed by datetime with 'pm25' column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load with first column as dates 
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [str(c).strip() for c in df.columns]

    # Normalizing the column names
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "date"})
    if "pm25_clean" in df.columns and "pm25" not in df.columns:
        df = df.rename(columns={"pm25_clean": "pm25"})

    if "date" not in df.columns or "pm25" not in df.columns:
        # If user saved with index only, try to set index as date
        if df.columns.size == 1 and df.index.name:
            df = df.rename(columns={df.columns[0]: "pm25"}).reset_index().rename(columns={df.index.name: "date"})
        else:
            raise ValueError("CSV must contain a datetime column (first col) and a 'pm25' column.")

    # datatype cleaning 
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"]).sort_values("date")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    df = df.set_index("date").sort_index()

   # Making sure there is enough history for prediction
    if len(df) < SEQ_LENGTH + 1:
        raise ValueError(f"Not enough rows for a {SEQ_LENGTH}-day look-back (rows={len(df)}).")

    return df[["pm25"]]

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Did you save it?")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Did you dump it?")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def make_sequences(series_scaled: np.ndarray, seq_len: int = SEQ_LENGTH):
    """Return last seq_len as model input."""
    x = series_scaled[-seq_len:]
    return x.reshape(1, seq_len, 1)

def iterative_forecast(model, scaler, history_values: np.ndarray, horizon: int = 7) -> np.ndarray:
    hist_scaled = scaler.transform(history_values.reshape(-1, 1)).ravel()
    preds = []
    cur = hist_scaled.copy()
    for _ in range(horizon):
        x = make_sequences(cur, SEQ_LENGTH)  # shape (1, seq, 1)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]
        yhat = scaler.inverse_transform(np.array([[yhat_scaled]])).ravel()[0]
        preds.append(float(yhat))
        cur = np.append(cur, yhat_scaled)
    return np.array(preds, dtype=float)

# Interface
st.title("Delhi PM2.5 Forecast — LSTM (Daily)")
st.markdown("""
Forecast **daily PM2.5** with a trained **LSTM**.
- Model: univariate, 30-day look-back → next-day prediction
- Data: Cleaned daily PM2.5 (2016–2025)
""")

with st.sidebar:
    st.header("Settings")
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1)
    allow_manual = st.checkbox("Provide manual last 30 days?", value=False)
    show_hist_days = st.slider("Show recent history (days)", 30, 365, 180, step=30)

# Loading data
try:
    df = load_data()
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {len(df)} daily rows. Last date: {df.index.max().date()}")

# Plot of recent history
st.subheader("Recent observed data")
st.line_chart(df.tail(show_hist_days))

# Manual pm2.5 values entry
if allow_manual:
    st.markdown("Paste **exactly 30** PM2.5 values (one per line). Leave blank to use data history.")
    manual_text = st.text_area("Manual values (µg/m³)", height=200, placeholder="e.g.\n92\n105\n88\n… (30 lines)")
    values = [v.strip() for v in manual_text.splitlines() if v.strip()]
    if len(values) == 0:
        history_raw = df["pm25"].values[-SEQ_LENGTH:]
    elif len(values) != SEQ_LENGTH:
        st.warning(f"Please provide exactly {SEQ_LENGTH} values or clear the box.")
        history_raw = None
    else:
        try:
            history_raw = np.array([float(v) for v in values], dtype=float)
        except ValueError:
            st.error("Some manual entries are not numeric.")
            history_raw = None
else:
    history_raw = df["pm25"].values[-SEQ_LENGTH:]

if history_raw is None or len(history_raw) < SEQ_LENGTH:
    st.info("Provide valid last 30 values (or disable manual mode) to forecast.")
    st.stop()

# Prediction
preds = iterative_forecast(model, scaler, history_raw, horizon=horizon)

# Future index & outputs
last_date = pd.to_datetime(df.index.max())
future_idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
forecast = pd.Series(preds, index=future_idx, name="forecast_pm25")

st.subheader(f"{horizon}-day forecast")
st.line_chart(forecast.to_frame())

# Combining the history with prediction
recent = df.tail(show_hist_days).copy()
combined = pd.concat([recent, forecast.to_frame("pm25")], axis=0)
st.subheader(f"Recent {show_hist_days} days + forecast")
st.line_chart(combined)

# Download function
st.subheader("Forecast table")
st.dataframe(forecast.round(2).to_frame())
os.makedirs("results", exist_ok=True)
forecast.round(2).to_csv("results/pm25_forecast_streamlit.csv", header=True)
st.download_button("Download forecast CSV",
                   forecast.round(2).to_csv().encode("utf-8"),
                   file_name="pm25_forecast.csv",
                   mime="text/csv")
