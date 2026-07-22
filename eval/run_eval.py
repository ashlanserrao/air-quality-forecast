# eval/run_eval.py — run every case in tests/eval_cases.yaml against the live
# agent, score with in-house deterministic metrics (no RAGAS), benchmark the
# forecaster against persistence, and write eval/report.{json,md}.
#
# Metrics:
#   faithfulness      -> guardrail.verify_response(answer, tool_results) == []
#   retrieval_accuracy-> expected source doc is the top retrieved chunk
#   context_relevance -> max cosine (1 - Chroma distance, exact for the
#                        L2-normalized embeddings) among retrieved chunks from
#                        the expected source doc; floor 0.30 + baseline drift 0.10
#   forecast          -> LSTM 1-step MAE/RMSE vs. persistence on the held-out 20%
import os
import sys
import json
import argparse
from datetime import datetime, timezone

import yaml
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

CASES_PATH = os.path.join(_REPO_ROOT, "tests", "eval_cases.yaml")
REPORT_JSON = os.path.join(_REPO_ROOT, "eval", "report.json")
REPORT_MD = os.path.join(_REPO_ROOT, "eval", "report.md")
BASELINES_PATH = os.path.join(_REPO_ROOT, "eval", "baselines.json")

CONTEXT_RELEVANCE_FLOOR = 0.30   # coarse collapse detector, not a quality bar
CONTEXT_RELEVANCE_DRIFT = 0.10   # flag a case that drops this far below baseline


# --------------------------------------------------------------------------
# tool-call capture: monkeypatch agent._run_tool to record every (name, result)
# --------------------------------------------------------------------------
class _ToolTracer:
    def __init__(self, agent_module):
        self._agent = agent_module
        self._orig = agent_module._run_tool
        self.calls = []          # list of (name, result_dict)

    def __enter__(self):
        def traced(name, arguments):
            result = self._orig(name, arguments)
            self.calls.append((name, result))
            return result

        self._agent._run_tool = traced
        return self

    def __exit__(self, *exc):
        self._agent._run_tool = self._orig

    @property
    def tool_names(self):
        return sorted({name for name, _ in self.calls})

    @property
    def results(self):
        return [result for _, result in self.calls]


def _max_similarity_for_source(chunks, source_document):
    """Best cosine among retrieved chunks from the expected source. Injected
    chunks (similarity None) are excluded so they can't mask a regression."""
    sims = [
        c["similarity"]
        for c in chunks
        if c.get("source_document") == source_document and c.get("similarity") is not None
    ]
    return max(sims) if sims else None


def _guidance_chunks(results):
    """Flatten retrieved guidance chunks out of captured tool-result dicts."""
    chunks = []
    for r in results:
        if isinstance(r, dict) and "results" in r:
            chunks.extend(r["results"])
    return chunks


# --------------------------------------------------------------------------
# per-type scoring
# --------------------------------------------------------------------------
def _score_guardrail(case, guardrail):
    flags = guardrail.verify_response(case["response"], case["tool_results"])
    exp = case["expect"]
    reasons = []
    ok = True
    if bool(flags) != bool(exp["flagged"]):
        ok = False
        reasons.append(f"flagged={bool(flags)} expected={bool(exp['flagged'])}")
    for needle in exp.get("flags_contain", []):
        if not any(needle in f for f in flags):
            ok = False
            reasons.append(f"missing flag containing {needle!r}")
    return ok, {"flags": flags, "reasons": reasons}, {}


def _score_routing(case, agent):
    exp = case["expect"]
    with _ToolTracer(agent) as tracer:
        agent.answer(case["query"])
    called = tracer.tool_names
    expected = sorted(exp["tools_called"])
    ok = called == expected
    return ok, {"tools_called": called, "expected": expected}, {}


def _score_retrieval(case, get_health_guidance):
    exp = case["expect"]
    chunks = get_health_guidance(case["query"])
    reasons = []
    ok = True

    top_source = chunks[0]["source_document"] if chunks else None
    expected_source = exp["top_source_document"]
    if top_source != expected_source:
        ok = False
        reasons.append(f"top source {top_source!r} != {expected_source!r}")

    needle = exp.get("any_section_title_contains")
    if needle and not any(needle in c["section_title"] for c in chunks):
        ok = False
        reasons.append(f"no section title contains {needle!r}")

    relevance = _max_similarity_for_source(chunks, expected_source)
    metrics = {"retrieval_accuracy": ok, "context_relevance": relevance}
    return ok, {"top_source": top_source, "reasons": reasons, "context_relevance": relevance}, metrics


def _score_e2e(case, agent, guardrail):
    exp = case["expect"]
    with _ToolTracer(agent) as tracer:
        answer = agent.answer(case["query"])
    reasons = []
    ok = True

    if "tools_called" in exp:
        called = tracer.tool_names
        expected = sorted(exp["tools_called"])
        if called != expected:
            ok = False
            reasons.append(f"tools {called} != {expected}")

    grounded_flags = guardrail.verify_response(answer, tracer.results)
    if exp.get("answer_grounded") and grounded_flags:
        ok = False
        reasons.append(f"ungrounded answer: {grounded_flags}")

    options = exp.get("answer_contains_any")
    if options and not any(o.lower() in (answer or "").lower() for o in options):
        ok = False
        reasons.append(f"answer contains none of {options}")

    src = exp.get("answer_source_document")
    relevance = None
    if src:
        chunks = _guidance_chunks(tracer.results)
        if not any(c.get("source_document") == src for c in chunks):
            ok = False
            reasons.append(f"expected source {src!r} not retrieved")
        relevance = _max_similarity_for_source(chunks, src)

    metrics = {
        "faithfulness": not grounded_flags,
        "context_relevance": relevance,
    }
    details = {"reasons": reasons, "answer": answer, "context_relevance": relevance}
    return ok, details, metrics


# --------------------------------------------------------------------------
# forecast benchmark: LSTM 1-step vs persistence on the held-out 20%
# --------------------------------------------------------------------------
def _run_forecast_benchmark(cfg):
    import pandas as pd
    from forecasting import load_model_and_scaler

    series = pd.read_csv(os.path.join(_REPO_ROOT, cfg["data_file"])).iloc[:, 1].to_numpy(float)
    seq = cfg["seq_length"]

    # Same windowing the model was trained on: window -> next value, no shuffle.
    X = np.array([series[i : i + seq] for i in range(len(series) - seq)])
    y = series[seq:]
    split = int(len(X) * cfg["split"])
    X_test, y_test = X[split:], y[split:]

    model, scaler = load_model_and_scaler()
    scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    lstm_scaled = model.predict(scaled.reshape(X_test.shape[0], seq, 1), verbose=0).ravel()
    lstm_pred = scaler.inverse_transform(lstm_scaled.reshape(-1, 1)).ravel()

    persistence_pred = X_test[:, -1]   # "tomorrow == today"

    def mae(p):
        return float(np.mean(np.abs(p - y_test)))

    def rmse(p):
        return float(np.sqrt(np.mean((p - y_test) ** 2)))

    lstm_mae, persist_mae = mae(lstm_pred), mae(persistence_pred)
    return {
        "test_windows": int(len(X_test)),
        "lstm_mae": round(lstm_mae, 3),
        "lstm_rmse": round(rmse(lstm_pred), 3),
        "persistence_mae": round(persist_mae, 3),
        "persistence_rmse": round(rmse(persistence_pred), 3),
        "beats_persistence": lstm_mae <= persist_mae,
    }


# --------------------------------------------------------------------------
# drift baselines
# --------------------------------------------------------------------------
def _load_baselines():
    if os.path.exists(BASELINES_PATH):
        with open(BASELINES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _check_drift(relevance_by_case, baselines):
    """Return (drift_ok, per-case drift detail). A case fails if it dropped more
    than CONTEXT_RELEVANCE_DRIFT below its recorded baseline."""
    detail = {}
    ok = True
    for case_id, score in relevance_by_case.items():
        base = baselines.get(case_id)
        if score is None or base is None:
            detail[case_id] = {"score": score, "baseline": base, "drift": None}
            continue
        drop = base - score
        failed = drop > CONTEXT_RELEVANCE_DRIFT
        ok = ok and not failed
        detail[case_id] = {"score": round(score, 4), "baseline": round(base, 4),
                           "drop": round(drop, 4), "failed": failed}
    return ok, detail


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(update_baselines=False):
    with open(CASES_PATH, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    import agent
    import guardrail
    from rag.retrieval import get_health_guidance

    results = []
    relevance_by_case = {}

    for case in spec["cases"]:
        ctype = case["type"]
        try:
            if ctype == "guardrail":
                ok, details, metrics = _score_guardrail(case, guardrail)
            elif ctype == "routing":
                ok, details, metrics = _score_routing(case, agent)
            elif ctype == "retrieval":
                ok, details, metrics = _score_retrieval(case, get_health_guidance)
            elif ctype == "e2e":
                ok, details, metrics = _score_e2e(case, agent, guardrail)
            else:
                ok, details, metrics = False, {"reasons": [f"unknown type {ctype}"]}, {}
        except Exception as e:  # a crash is a failure, not an eval abort
            ok, details, metrics = False, {"reasons": [f"exception: {e!r}"]}, {}

        if metrics.get("context_relevance") is not None:
            relevance_by_case[case["id"]] = metrics["context_relevance"]

        results.append(
            {"id": case["id"], "type": ctype, "category": case["category"],
             "passed": ok, "details": details, "metrics": metrics}
        )
        print(f"[{'PASS' if ok else 'FAIL'}] {case['id']}"
              + (f"  {details.get('reasons')}" if not ok and details.get("reasons") else ""))

    forecast = _run_forecast_benchmark(spec["forecast"])
    print(f"[forecast] LSTM MAE {forecast['lstm_mae']} vs persistence "
          f"{forecast['persistence_mae']}  beats={forecast['beats_persistence']}")

    # drift vs. stored baselines
    baselines = _load_baselines()
    drift_ok, drift_detail = _check_drift(relevance_by_case, baselines)
    if update_baselines or not baselines:
        with open(BASELINES_PATH, "w", encoding="utf-8") as f:
            json.dump(relevance_by_case, f, indent=2)
        print(f"[baselines] {'updated' if baselines else 'initialised'} "
              f"{len(relevance_by_case)} case(s)")

    report = _build_report(results, forecast, relevance_by_case, drift_ok, drift_detail)
    _write_reports(report)
    print(f"\nreport -> {REPORT_JSON}\n         {REPORT_MD}")
    return report


def _cat(results, category):
    sub = [r for r in results if r["category"] == category]
    return sum(r["passed"] for r in sub), len(sub)


def _build_report(results, forecast, relevance_by_case, drift_ok, drift_detail):
    relevances = [v for v in relevance_by_case.values() if v is not None]
    e2e = [r for r in results if r["type"] == "e2e"]
    retrieval = [r for r in results if r["type"] == "retrieval"]

    floor_ok = all(v >= CONTEXT_RELEVANCE_FLOOR for v in relevances)

    reg_passed, reg_total = _cat(results, "regression")

    gates = {
        "regression_pass": reg_total > 0 and reg_passed == reg_total,
        "routing_pass": all(r["passed"] for r in results if r["type"] == "routing"),
        "retrieval_accuracy_pass": all(r["passed"] for r in retrieval),
        "faithfulness_pass": all(r["metrics"].get("faithfulness", True) for r in e2e),
        "context_relevance_floor_pass": floor_ok,
        "context_relevance_drift_pass": drift_ok,
        "forecast_beats_persistence": forecast["beats_persistence"],
    }

    categories = {}
    for cat in sorted({r["category"] for r in results}):
        p, t = _cat(results, cat)
        categories[cat] = {"passed": p, "total": t}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"passed": sum(r["passed"] for r in results), "total": len(results)},
        "by_category": categories,
        "metrics": {
            "context_relevance": {k: (round(v, 4) if v is not None else None)
                                  for k, v in relevance_by_case.items()},
            "context_relevance_min": round(min(relevances), 4) if relevances else None,
            "context_relevance_floor": CONTEXT_RELEVANCE_FLOOR,
            "context_relevance_drift": drift_detail,
        },
        "forecast": forecast,
        "gates": gates,
        "cases": results,
    }


def _write_reports(report):
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    g = report["gates"]
    fc = report["forecast"]
    lines = [
        "# Agent evaluation report",
        "",
        f"_Generated {report['generated_at']}_",
        "",
        f"**{report['summary']['passed']}/{report['summary']['total']} cases passed**",
        "",
        "## By category",
        "",
        "| category | passed | total |",
        "| --- | --- | --- |",
    ]
    for cat, v in report["by_category"].items():
        lines.append(f"| {cat} | {v['passed']} | {v['total']} |")

    lines += [
        "",
        "## Forecast benchmark (held-out 20%)",
        "",
        f"- Test windows: {fc['test_windows']}",
        f"- **LSTM**: MAE {fc['lstm_mae']}, RMSE {fc['lstm_rmse']}",
        f"- **Persistence**: MAE {fc['persistence_mae']}, RMSE {fc['persistence_rmse']}",
        f"- Beats persistence: **{fc['beats_persistence']}**",
        "",
        "## Context relevance (cosine of best expected-source chunk)",
        "",
        "| case | cosine |",
        "| --- | --- |",
    ]
    for k, v in report["metrics"]["context_relevance"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"", f"Floor {report['metrics']['context_relevance_floor']}; "
        f"min observed {report['metrics']['context_relevance_min']}.",
        "",
        "## CI gates",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for k, v in g.items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baselines", action="store_true",
                        help="overwrite eval/baselines.json with this run's cosines")
    args = parser.parse_args()
    run(update_baselines=args.update_baselines)
