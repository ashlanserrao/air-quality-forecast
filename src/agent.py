# src/agent.py — minimal function-calling agent over forecast()
import os
import sys
import json
from openai import OpenAI

from forecasting import forecast, SEQ_LENGTH
from fetch_data import get_recent_clean_history

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an assistant for Delhi air quality (PM2.5).

Use the get_pm25_forecast tool only for questions that require a prediction about
future conditions (e.g. "next week", "tomorrow", "will it be safe to..."). Answer
general, definitional, or historical questions directly without calling the tool.

When you have a forecast, ground your answer in these US EPA PM2.5 (24-hr average,
ug/m3) breakpoints — do not invent your own thresholds:
  0.0   - 12.0   Good
  12.1  - 35.4   Moderate
  35.5  - 55.4   Unhealthy for Sensitive Groups
  55.5  - 150.4  Unhealthy
  150.5 - 250.4  Very Unhealthy
  250.5 - 500.4  Hazardous

State the category each forecasted day falls into and give a plain-English
recommendation (e.g. about outdoor activity) based on that category. Be concise.

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
    }
]


def _run_tool(name: str, arguments: dict) -> dict:
    if name != "get_pm25_forecast":
        raise ValueError(f"Unknown tool: {name}")
    horizon = int(arguments.get("horizon", 7))

    try:
        history = get_recent_clean_history(SEQ_LENGTH).tolist()
    except ValueError as e:
        return {"error": str(e)}

    preds = forecast(history, horizon=horizon)
    return {"horizon": horizon, "forecast_pm25": [round(p, 2) for p in preds]}


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
