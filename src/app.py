# app.py — Streamlit UI for the Delhi PM2.5 agent.
#
# Talks to the FastAPI service over HTTP only (POST /forecast, POST /query) — it
# imports no model, TensorFlow, or agent code, so this process stays light and
# the UI is fully decoupled from the ML stack. The API base URL comes from
# API_URL (http://api:8000 under docker-compose; http://localhost:8000 locally).
import os
from datetime import timedelta

import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DATA_PATH = os.environ.get(
    "HISTORY_CSV",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "processed", "pm25_daily_final.csv"),
)
SEQ_LENGTH = 30

st.set_page_config(page_title="Delhi PM2.5 Forecast — LSTM", layout="wide")


@st.cache_data
def load_history(path=DATA_PATH) -> pd.DataFrame:
    """Recent observed PM2.5, for the history chart and as forecast input."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"History CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = [str(c).strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "date"})
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df["date"] = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "pm25"]).drop_duplicates(subset=["date"])
    df = df.set_index("date").sort_index()
    if len(df) < SEQ_LENGTH + 1:
        raise ValueError(f"Not enough rows for a {SEQ_LENGTH}-day look-back (rows={len(df)}).")
    return df[["pm25"]]


def call_forecast(history: list[float], horizon: int) -> pd.Series | None:
    try:
        resp = requests.post(
            f"{API_URL}/forecast",
            json={"history": history, "horizon": horizon},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Forecast API error: {e}")
        return None
    days = resp.json()["forecast"]
    return pd.Series([d["pm25_ugm3"] for d in days], name="forecast_pm25")


def call_query(question: str) -> str | None:
    try:
        resp = requests.post(f"{API_URL}/query", json={"query": question}, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        detail = ""
        if e.response is not None:
            try:
                detail = f" — {e.response.json().get('detail', '')}"
            except Exception:
                pass
        st.error(f"Query API error: {e}{detail}")
        return None
    return resp.json()["answer"]


# --- interface ------------------------------------------------------------
st.title("Delhi PM2.5 Forecast & Air-Quality Agent")
st.caption(f"API: {API_URL}")

with st.sidebar:
    st.header("Settings")
    horizon = st.number_input("Forecast horizon (days)", 1, 30, 7, 1)
    show_hist_days = st.slider("Show recent history (days)", 30, 365, 180, step=30)

try:
    df = load_history()
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {len(df)} daily rows. Last date: {df.index.max().date()}")

st.subheader("Recent observed data")
st.line_chart(df.tail(show_hist_days))

# --- forecast (via POST /forecast) ---------------------------------------
history_raw = df["pm25"].values[-SEQ_LENGTH:].tolist()
preds = call_forecast(history_raw, int(horizon))
if preds is not None:
    last_date = pd.to_datetime(df.index.max())
    future_idx = pd.date_range(last_date + timedelta(days=1), periods=int(horizon), freq="D")
    forecast = pd.Series(preds.values, index=future_idx, name="forecast_pm25")

    st.subheader(f"{int(horizon)}-day forecast")
    combined = pd.concat([df.tail(show_hist_days), forecast.to_frame("pm25")], axis=0)
    st.line_chart(combined)
    st.dataframe(forecast.round(2).to_frame())
    st.download_button(
        "Download forecast CSV",
        forecast.round(2).to_csv().encode("utf-8"),
        file_name="pm25_forecast.csv",
        mime="text/csv",
    )

# --- ask the agent (via POST /query) -------------------------------------
st.subheader("Ask the agent")
st.caption("e.g. \"is Delhi's air worse than what WHO considers safe?\" or "
           "\"will next week trigger GRAP restrictions?\"")
question = st.text_input("Your question", key="agent_q")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking…"):
        answer = call_query(question.strip())
    if answer:
        st.markdown(answer)
