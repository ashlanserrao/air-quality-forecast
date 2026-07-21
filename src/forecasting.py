# src/forecasting.py — standalone PM2.5 forecasting logic (no UI dependencies)
import os
import numpy as np
import joblib
import tensorflow as tf

SEQ_LENGTH = 30
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_REPO_ROOT, "models", "best_lstm.keras")
SCALER_PATH = os.path.join(_REPO_ROOT, "models", "scaler.joblib")

_model = None
_scaler = None


def load_model_and_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Did you save it?")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Did you dump it?")
        _model = tf.keras.models.load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


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


def forecast(history: list[float], horizon: int = 7) -> list[float]:
    """Forecast the next `horizon` days of PM2.5 from the last SEQ_LENGTH days of history."""
    if len(history) < SEQ_LENGTH:
        raise ValueError(f"history must contain at least {SEQ_LENGTH} values (got {len(history)}).")
    model, scaler = load_model_and_scaler()
    history_values = np.array(history[-SEQ_LENGTH:], dtype=float)
    preds = iterative_forecast(model, scaler, history_values, horizon=horizon)
    return preds.tolist()

