"""CI gate for the eval harness.

Two tiers:
  * The guardrail regression cases run live and PURE here — no API key, no
    network — straight against guardrail.verify_response. These must always pass.
  * The live-agent metrics (routing, retrieval accuracy, faithfulness, context
    relevance, forecast-vs-persistence) are read from eval/report.json, which
    eval/run_eval.py produces. Keeping the expensive, network-bound work in
    run_eval.py keeps this gate fast and deterministic.

Run `python eval/run_eval.py` first to produce the report. Without it these
report-backed tests skip locally; set EVAL_REQUIRE_REPORT=1 (as CI should) to
turn a missing report into a failure instead.
"""
import os
import sys
import json

import yaml
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

CASES_PATH = os.path.join(_REPO_ROOT, "tests", "eval_cases.yaml")
REPORT_PATH = os.path.join(_REPO_ROOT, "eval", "report.json")

import guardrail  # pure module — safe to import without any API key


def _load_cases():
    with open(CASES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["cases"]


# ---- Tier 1: guardrail regression, live + pure (always runs) --------------
_GUARDRAIL_CASES = [c for c in _load_cases() if c["type"] == "guardrail"]


@pytest.mark.parametrize("case", _GUARDRAIL_CASES, ids=[c["id"] for c in _GUARDRAIL_CASES])
def test_guardrail_regression(case):
    flags = guardrail.verify_response(case["response"], case["tool_results"])
    expect = case["expect"]
    assert bool(flags) == bool(expect["flagged"]), f"{case['id']}: flags={flags}"
    for needle in expect.get("flags_contain", []):
        assert any(needle in f for f in flags), f"{case['id']}: missing flag {needle!r} in {flags}"


# ---- Tier 2: live-agent metrics, read from the report ---------------------
def _report():
    if not os.path.exists(REPORT_PATH):
        if os.environ.get("EVAL_REQUIRE_REPORT"):
            pytest.fail("eval/report.json missing — run `python eval/run_eval.py` first")
        pytest.skip("eval/report.json not found; run eval/run_eval.py to generate it")
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


# Hard gates: must be True in the report.
_HARD_GATES = [
    "regression_pass",
    "routing_pass",
    "retrieval_accuracy_pass",
    "faithfulness_pass",
    "context_relevance_floor_pass",
    "context_relevance_drift_pass",
    "forecast_beats_persistence",
]


@pytest.mark.parametrize("gate", _HARD_GATES)
def test_report_gate(gate):
    gates = _report()["gates"]
    assert gate in gates, f"gate {gate!r} not in report"
    assert gates[gate] is True, f"gate {gate!r} failed"


def test_forecast_beats_persistence_margin():
    """Redundant with the gate flag, but asserts on the raw numbers so a
    failure message shows the actual MAEs."""
    fc = _report()["forecast"]
    assert fc["lstm_mae"] <= fc["persistence_mae"], (
        f"LSTM MAE {fc['lstm_mae']} > persistence MAE {fc['persistence_mae']}"
    )
