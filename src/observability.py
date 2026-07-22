# src/observability.py — tracing for the agent, with or without Langfuse.
#
# Every run builds a local span tree (inputs, outputs, latency, token usage,
# estimated cost, guardrail metadata). If Langfuse credentials are present the
# same spans are mirrored to Langfuse as they happen; if not, capture is
# local-only so traces can still be inspected offline. Langfuse calls are all
# defensive — a missing package or API error degrades to local-only and never
# breaks answer().
#
# Env vars to enable Langfuse delivery:
#   LANGFUSE_PUBLIC_KEY   pk-lf-...
#   LANGFUSE_SECRET_KEY   sk-lf-...
#   LANGFUSE_HOST         https://cloud.langfuse.com  (EU)  |
#                         https://us.cloud.langfuse.com (US)
import os
import time
import json
from contextlib import contextmanager
from datetime import datetime, timezone

# Estimated cost only — for the offline preview. When Langfuse is connected it
# computes authoritative cost from model + usage on its side. USD per 1M tokens.
_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_LAST_TRACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "last_trace.json"
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _estimate_cost(model, usage):
    price = _PRICING.get(model)
    if not price or not usage:
        return None
    cost = (usage.get("input", 0) / 1e6) * price["input"] + (
        usage.get("output", 0) / 1e6
    ) * price["output"]
    return round(cost, 8)


class Tracer:
    """A single-run span-tree collector with optional Langfuse mirroring.

    Synchronous, single-threaded: the parent span is just the top of a stack.
    Each stack frame is (local_record, langfuse_object_or_None)."""

    def __init__(self):
        self._stack = []
        self._root = None
        self._lf = None
        self._lf_ready = None  # tri-state: None=unchecked, True/False after first check
        self.last_trace = None

    # -- Langfuse client (lazy, defensive) --------------------------------
    def _langfuse(self):
        if self._lf_ready is None:
            self._lf_ready = False
            if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
                try:
                    from langfuse import Langfuse

                    self._lf = Langfuse()
                    self._lf_ready = True
                except Exception as e:  # package missing or init failed
                    print(f"[observability] Langfuse disabled (local-only): {e}")
        return self._lf if self._lf_ready else None

    @property
    def langfuse_enabled(self):
        return self._langfuse() is not None

    # -- trace (root) -----------------------------------------------------
    @contextmanager
    def trace(self, name, user_input):
        lf = self._langfuse()
        lf_obj = None
        if lf:
            try:
                lf_obj = lf.trace(name=name, input=user_input)
            except Exception:
                lf_obj = None
        rec = {
            "type": "trace", "name": name, "input": user_input,
            "start": _now_iso(), "output": None, "metadata": {}, "children": [],
        }
        self._root = rec
        self._stack.append((rec, lf_obj))
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._stack.pop()
            if lf_obj:
                try:
                    lf_obj.update(output=rec["output"], metadata=rec["metadata"])
                    lf.flush()
                except Exception:
                    pass
            self.last_trace = rec
            try:
                with open(_LAST_TRACE_PATH, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, default=str)
            except Exception:
                pass

    # -- span -------------------------------------------------------------
    @contextmanager
    def span(self, name, input=None):
        if not self._stack:  # called outside a trace (e.g. under eval) -> no-op
            yield {"output": None}
            return
        parent_rec, parent_lf = self._stack[-1]
        lf_obj = None
        if parent_lf:
            try:
                lf_obj = parent_lf.span(name=name, input=input)
            except Exception:
                lf_obj = None
        rec = {"type": "span", "name": name, "input": input,
               "output": None, "children": []}
        parent_rec["children"].append(rec)
        self._stack.append((rec, lf_obj))
        t0 = time.perf_counter()
        try:
            yield rec  # caller sets rec["output"] = ...
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._stack.pop()
            if lf_obj:
                try:
                    lf_obj.end(output=rec["output"])
                except Exception:
                    pass

    # -- generation (LLM call) -------------------------------------------
    def generation(self, name, model, input, output, usage):
        if not self._stack:
            return
        parent_rec, parent_lf = self._stack[-1]
        cost = _estimate_cost(model, usage)
        parent_rec["children"].append({
            "type": "generation", "name": name, "model": model,
            "input_messages": len(input) if isinstance(input, list) else None,
            "output_preview": (output or "")[:160],
            "usage": usage, "estimated_cost_usd": cost,
        })
        if parent_lf:
            try:
                gen = parent_lf.generation(
                    name=name, model=model, input=input, output=output, usage=usage
                )
                gen.end()
            except Exception:
                pass

    # -- trace-level output/metadata -------------------------------------
    def update_trace(self, output=None, metadata=None):
        if self._root is None:
            return
        if output is not None:
            self._root["output"] = output
        if metadata:
            self._root["metadata"].update(metadata)


# module-level singleton the agent imports
tracer = Tracer()


# --------------------------------------------------------------------------
# offline rendering — pretty-print a captured trace so it can be inspected
# without a Langfuse dashboard
# --------------------------------------------------------------------------
def render(rec, indent=0):
    pad = "  " * indent
    lines = []
    if rec["type"] == "trace":
        lines.append(f"{pad}● TRACE {rec['name']}  ({rec.get('latency_ms')} ms)")
        lines.append(f"{pad}  input:  {_short(rec['input'])}")
        lines.append(f"{pad}  output: {_short(rec['output'])}")
        g = rec["metadata"].get("guardrail")
        if g:
            verdict = "PASS" if g.get("passed") else "FAIL"
            extra = []
            if g.get("regenerated"):
                extra.append("regenerated")
            if g.get("fallback"):
                extra.append("fallback")
            flags = g.get("flags") or []
            lines.append(
                f"{pad}  guardrail: {verdict}"
                + (f" [{', '.join(extra)}]" if extra else "")
                + (f"  flags={flags}" if flags else "")
            )
        for c in rec["children"]:
            lines.append(render(c, indent + 1))
    elif rec["type"] == "span":
        lines.append(f"{pad}├─ span {rec['name']}  ({rec.get('latency_ms')} ms)")
        lines.append(f"{pad}│    in:  {_short(rec.get('input'))}")
        lines.append(f"{pad}│    out: {_short(rec.get('output'))}")
        for c in rec["children"]:
            lines.append(render(c, indent + 1))
    elif rec["type"] == "generation":
        u = rec.get("usage") or {}
        cost = rec.get("estimated_cost_usd")
        lines.append(
            f"{pad}├─ gen  {rec['name']}  model={rec['model']}  "
            f"tokens(in={u.get('input')}, out={u.get('output')}, total={u.get('total')})  "
            f"~${cost}"
        )
    return "\n".join(lines)


def _short(val, n=90):
    if val is None:
        return "None"
    s = val if isinstance(val, str) else json.dumps(val, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"
