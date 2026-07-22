# src/api.py — HTTP surface for the Delhi PM2.5 agent.
#
# Endpoints:
#   GET  /health   — liveness + which API keys are configured (no keys required)
#   POST /query    — natural-language question -> agent.answer() (needs OPENAI key)
#   POST /forecast — LSTM forecast from a supplied history (no keys; pure inference)
#
# `import agent` at module load is deliberate: the heavy stack (TensorFlow, Torch,
# Chroma) warms once at startup rather than on the first request. /health stays
# cheap because everything is already imported by the time it can be hit.
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting import forecast as run_forecast, SEQ_LENGTH
from aqi import pm25_to_aqi
import agent

app = FastAPI(title="Delhi PM2.5 Agent API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str


class ForecastRequest(BaseModel):
    history: list[float] = Field(
        ..., description=f"most recent daily PM2.5 values, at least {SEQ_LENGTH}"
    )
    horizon: int = Field(7, ge=1, le=30)


class ForecastDay(BaseModel):
    day: int
    pm25_ugm3: float
    aqi: int
    aqi_category: str


class ForecastResponse(BaseModel):
    horizon: int
    forecast: list[ForecastDay]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "openaq_key_configured": bool(os.environ.get("OPENAQ_API_KEY")),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        answer = agent.answer(req.query)
    except RuntimeError as e:
        # e.g. OPENAI_API_KEY not set — the request is valid, the server isn't ready.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")
    return {"answer": answer}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if len(req.history) < SEQ_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"history needs at least {SEQ_LENGTH} values (got {len(req.history)})",
        )
    preds = run_forecast(req.history, horizon=req.horizon)
    days = []
    for i, p in enumerate(preds, start=1):
        aqi, category = pm25_to_aqi(p)
        days.append(
            {"day": i, "pm25_ugm3": round(p, 2), "aqi": aqi, "aqi_category": category}
        )
    return {"horizon": req.horizon, "forecast": days}
