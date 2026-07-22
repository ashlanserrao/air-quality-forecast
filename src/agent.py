# src/agent.py — minimal function-calling agent over forecast()
import os
import re
import sys
import json
from openai import OpenAI

from aqi import pm25_to_aqi
from forecasting import forecast, SEQ_LENGTH
from fetch_data import get_recent_clean_history
from rag.retrieval import get_health_guidance

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an assistant for Delhi air quality (PM2.5).

You have two tools:
- get_pm25_forecast: for future, upcoming, or currently-unknown-to-you air
  quality conditions, and for any comparison of Delhi's air against a
  standard or threshold (that comparison needs real numbers, not assumptions).
- get_health_guidance: for anything invoking an official standard, guideline,
  threshold, or action plan — GRAP, WHO, CPCB/EPA/NAAQS, "safe level",
  "acceptable limit", or what actions apply at a given condition.

Some tool results may already be present in this conversation before you
reply; they were retrieved for you because the question clearly needed them.
Use them. Do not re-call a tool whose result you already have. If a question
needs something not already retrieved, call the tool for it yourself.

Never state a regulatory threshold, standard, or agency assignment from your
own training data — these are periodically revised, and the knowledge base
holds the current version. If you don't have grounded guidance for a claim
like that, say so rather than filling it in from memory.

Many queries need BOTH tools — "will next week's air trigger GRAP
restrictions" or "is Delhi's air worse than WHO's safe level" each need a
forecast for the numbers AND health guidance for the standard to judge them
against. Genuinely general/definitional questions with no forward-looking or
standards component (e.g. "what is PM2.5") need neither.

When health guidance is retrieved, cite it by source document and section
(e.g. "per cpcb_about_national_aqi.pdf, Stage III guidance...").

The forecast tool returns, for each day, a pm25_ugm3 concentration AND its
already-computed aqi and aqi_category. Report those AQI values and categories
as given. Never derive a category yourself, and never compare the raw
pm25_ugm3 concentration against AQI ranges — they are different scales and
doing so understates severity. Judge GRAP stages and any other AQI-defined
threshold against the aqi field, not pm25_ugm3.

A WHO guideline is the exception: WHO guideline levels are stated in µg/m³
(e.g. 5 µg/m³ annual, 15 µg/m³ 24-hour), NOT in AQI, so compare them against
the forecast's pm25_ugm3 value — never against the aqi field. When a WHO (or
other µg/m³-based) guideline and a forecast are both in context, state the
forecasted pm25_ugm3 value, the relevant guideline value, and the approximate
multiple over it (e.g. "≈52 µg/m³, about 3× the WHO 24-hour guideline of 15").
Note that CPCB "Satisfactory" and WHO "safe" are different scales that can
disagree — air can be CPCB-Satisfactory while several times over the WHO
guideline; say so plainly rather than collapsing them.

State the category for each forecasted day and give a plain-English
recommendation (e.g. about outdoor activity) based on it. Be concise.

If a tool result contains an "error" field instead of a forecast, do not
fabricate a forecast or guess at air quality. Tell the user honestly and
concisely that you can't generate a reliable forecast right now, based on
the reason given in the error.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pm25_forecast",
            "description": (
                "Forecasts Delhi PM2.5 air quality for the next N days based on the "
                "most recent 30 days of observed data. Use this when the user asks "
                "about future air quality, pollution levels, whether it's safe to be "
                "outside, or anything requiring a forward-looking prediction. Do NOT "
                "use this for general questions about what PM2.5 is, historical facts, "
                "or health information that doesn't require a forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {
                        "type": "integer",
                        "description": (
                            "Number of days ahead to forecast, 1-7. Infer from the "
                            "user's query (e.g. 'next week' -> 7, 'tomorrow' -> 1). "
                            "Default to 7 if unspecified."
                        ),
                        "minimum": 1,
                        "maximum": 7,
                    }
                },
                "required": ["horizon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_guidance",
            "description": (
                "Retrieves official health/regulatory guidance about air quality "
                "- AQI action recommendations, GRAP stage restrictions, or WHO/EPA "
                "guideline levels - from a knowledge base of government and WHO "
                "documents. Use this when the user asks what to do at a given "
                "AQI/pollution level, what a GRAP stage means or requires, or what "
                "official air quality standards/guidelines say. ALWAYS call this "
                "tool for GRAP/AQI-regulatory questions even if you think you "
                "already know the answer - the source documents may be a more "
                "current or specific revision than your training data. Do NOT use "
                "this for forecasting future PM2.5 values - use get_pm25_forecast "
                "for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's health/regulatory question, passed through "
                            "close to verbatim (e.g. 'what to do at AQI 350', "
                            "'GRAP stage 3 restrictions')."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# --- Deterministic pre-routing -------------------------------------------
# Prompt instructions alone did not reliably get the model to call these tools
# for standards/forecast questions — it would answer regulatory questions from
# pretrained knowledge instead. These regexes are a floor under that behavior:
# a match force-calls the tool before the model ever sees the query. The model
# is still handed the full tool list and can call either one on its own for
# anything the regexes miss.
#
# Biased deliberately toward false positives: a spurious retrieval costs a bit
# of latency and context, while a miss produces an ungrounded threshold claim.

STANDARDS_RE = re.compile(
    r"\b(who|world health organization|grap|cpcb|epa|naaqs|"
    r"safe level|safe limit|considered safe|acceptable limit|"
    r"limits?|restrictions?|standards?|guidelines?|thresholds?|"
    r"triggers?|graded response)\b",
    re.IGNORECASE,
)

FORECAST_RE = re.compile(
    r"\b(next week|next \d+ days?|tomorrow|upcoming|coming days|this week|"
    r"forecast|predict|will it be|going to be|compare[ds]?|comparison|"
    r"worse than|better than|higher than|exceeds?)\b",
    re.IGNORECASE,
)

SHORT_HORIZON_RE = re.compile(r"\b(tomorrow|today|tonight)\b", re.IGNORECASE)


def _pre_route(query: str) -> list[tuple[str, dict]]:
    """Tool calls to force before the LLM sees the query, as (name, arguments)."""
    forced = []

    if FORECAST_RE.search(query):
        horizon = 1 if SHORT_HORIZON_RE.search(query) else 7
        forced.append(("get_pm25_forecast", {"horizon": horizon}))

    if STANDARDS_RE.search(query):
        forced.append(("get_health_guidance", {"query": query}))

    return forced


def _prerouted_messages(query: str) -> list[dict]:
    """Run the force-called tools and render them as a synthetic tool-call
    exchange. Using the real tool-call message format (rather than dumping the
    text into a system message) is what the model is trained to ground on, and
    it shows the model the call already happened so it won't repeat it."""
    forced = _pre_route(query)
    if not forced:
        return []

    tool_calls = []
    tool_results = []
    for i, (name, arguments) in enumerate(forced):
        call_id = f"prerouted_{i}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(_run_tool(name, arguments)),
            }
        )

    return [{"role": "assistant", "content": None, "tool_calls": tool_calls}] + tool_results


def _run_tool(name: str, arguments: dict) -> dict:
    if name == "get_pm25_forecast":
        horizon = int(arguments.get("horizon", 7))
        try:
            history = get_recent_clean_history(SEQ_LENGTH).tolist()
        except ValueError as e:
            return {"error": str(e)}
        preds = forecast(history, horizon=horizon)
        return {
            "horizon": horizon,
            "note": (
                "pm25_ugm3 is a concentration; aqi is the CPCB PM2.5 sub-index "
                "computed from it. Use the given aqi/aqi_category values as-is - "
                "do not compare the raw pm25_ugm3 number against AQI ranges."
            ),
            "forecast": [
                {
                    "day": i,
                    "pm25_ugm3": round(p, 2),
                    "aqi": aqi,
                    "aqi_category": category,
                }
                for i, p in enumerate(preds, start=1)
                for aqi, category in [pm25_to_aqi(p)]
            ],
        }

    if name == "get_health_guidance":
        query = arguments.get("query", "")
        try:
            chunks = get_health_guidance(query)
        except Exception as e:
            return {"error": f"Knowledge base retrieval failed: {e}"}
        if not chunks:
            return {"error": "No relevant health guidance found."}
        return {"results": chunks}

    raise ValueError(f"Unknown tool: {name}")


def answer(query: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running this script."
        )
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    messages.extend(_prerouted_messages(query))

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = _run_tool(call.function.name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )

    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Will it be safe to run outside next week?"
    print(answer(query))
