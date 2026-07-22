# src/guardrail.py — post-generation grounding check for the agent's replies.
#
# The model can still fabricate: state an AQI it derived by mis-reading µg/m³,
# quote a WHO threshold from pretraining when retrieval returned only prose, or
# cite a GRAP stage that wasn't retrieved. verify_response checks the *provenance*
# of every number and named source in a reply against this turn's tool outputs —
# not the arithmetic or the reasoning, only "is this value/source actually in
# what we were handed". The agent regenerates once on any flag, then falls back
# to fallback_response (a grounded-only reply) rather than shipping the claim.
#
# Pure and self-contained: no network, no LLM, no dependency on the agent's
# orchestration — which is why it can be tested on its own without booting the
# agent or holding any API key.
import re
import json

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Numbers in these contexts are stripped before checking: they are not
# fabricated-measurement risk and a literal match would false-positive on good
# answers. Pollutant tokens (the digits in PM2.5/PM10), 4-digit years, multiples
# ("2.5 times" is arithmetic on grounded numbers, verified elsewhere or not at
# all), durations ("limit activity for 2 hours"), and particle-size descriptors
# ("2.5 micrometres" — the size that defines PM2.5, not a measured value; this
# one false-positived a correct "what is PM2.5" answer into a regeneration).
_PM_TOKEN_RE = re.compile(r"pm\s*(?:2\.5|10)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MULTIPLE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:times|x|×|-?fold)\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\d+(?:\.\d+)?[\s-]*(?:hours?|minutes?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:µm|μm|micromet(?:er|re)s?|microns?)\b", re.IGNORECASE
)

_SOURCE_FAMILIES = {
    "WHO": ("who", "world health organization"),
    "GRAP": ("grap", "graded response"),
    "CPCB": ("cpcb",),
    "EPA/NAAQS": ("epa", "naaqs"),
}
_STAGE_RE = re.compile(r"\bstage\s+(iv|iii|ii|i|[1-4])\b", re.IGNORECASE)
_DIGIT_TO_ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv"}


def _walk_numbers(obj, out: list) -> None:
    """Collect numbers from dict/list *values* (not keys, or the key
    'pm25_ugm3' would ground a spurious 25) and from any string content."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(x) for x in _NUMBER_RE.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_numbers(v, out)


def _grounded_numbers(tool_results: list[dict]) -> list[float]:
    out: list[float] = []
    _walk_numbers(tool_results, out)
    return out


def _is_grounded_number(candidate: float, grounded: list[float]) -> bool:
    for g in grounded:
        if abs(candidate - g) <= 0.5:
            return True
        if g != 0 and abs(candidate - g) / abs(g) <= 0.03:
            return True
    return False


def _response_numbers(text: str) -> list[float]:
    stripped = _PM_TOKEN_RE.sub(" ", text)
    stripped = _YEAR_RE.sub(" ", stripped)
    stripped = _MULTIPLE_RE.sub(" ", stripped)
    stripped = _DURATION_RE.sub(" ", stripped)
    stripped = _SIZE_RE.sub(" ", stripped)
    return [float(x) for x in _NUMBER_RE.findall(stripped)]


def verify_response(text: str, tool_results: list[dict]) -> list[str]:
    """Return a flag per unsupported claim; empty list means fully grounded."""
    if not text:
        return []

    flags: list[str] = []
    grounded_text = json.dumps(tool_results).lower()

    # With no tool outputs there is nothing to ground numbers against, and the
    # answer is definitional by construction (any standards/forecast query is
    # force-routed to a tool first). Checking numbers here only produces false
    # positives, e.g. "PM2.5 is < 2.5 micrometres" with no forecast to match.
    # The source/stage citation checks still run: asserting an agency you never
    # retrieved is a fabrication signal regardless of tool outputs.
    if tool_results:
        grounded = _grounded_numbers(tool_results)
        for candidate in _response_numbers(text):
            if not _is_grounded_number(candidate, grounded):
                flags.append(f"ungrounded number: {candidate:g} (not in tool outputs)")

    lower = text.lower()
    for label, aliases in _SOURCE_FAMILIES.items():
        cited = any(re.search(rf"\b{re.escape(a)}\b", lower) for a in aliases)
        if cited and not any(a in grounded_text for a in aliases):
            flags.append(f"cited source not retrieved: {label}")

    for m in _STAGE_RE.finditer(text):
        raw = m.group(1).lower()
        roman = _DIGIT_TO_ROMAN.get(raw, raw)
        if f"stage {roman}" not in grounded_text:
            flags.append(f"cited GRAP stage not retrieved: Stage {roman.upper()}")

    # De-duplicate while preserving order.
    seen = set()
    return [f for f in flags if not (f in seen or seen.add(f))]


def correction_message(flags: list[str]) -> str:
    return (
        "Your previous response contained claims not supported by the retrieved "
        "tool results:\n- " + "\n- ".join(flags) + "\n\n"
        "Every concentration (µg/m³), AQI value, and threshold you state must "
        "appear in the tool results, and you may only cite a source or GRAP "
        "stage that was actually retrieved this turn. Rewrite the answer using "
        "only supported values. If a value or source is not in the tool "
        "results, do not state it — say it could not be verified instead."
    )


def fallback_response(tool_results: list[dict], flags: list[str]) -> str:
    """Deterministic, grounded-only reply used when regeneration still fails.
    States only values pulled straight from the tool outputs, and names what
    was dropped — never ships the unverified claim."""
    lines = ["I can only report values I was able to verify against the data this turn.", ""]
    for result in tool_results:
        if "forecast" in result:
            lines.append("PM2.5 forecast:")
            for d in result["forecast"]:
                lines.append(
                    f"  Day {d['day']}: {d['pm25_ugm3']} µg/m³ "
                    f"(AQI {d['aqi']}, {d['aqi_category']})"
                )
        if "results" in result:
            sources = sorted(
                {f"{c['source_document']} — {c['section_title']}" for c in result["results"]}
            )
            lines.append("Retrieved guidance: " + "; ".join(sources))
        if "error" in result:
            lines.append(f"Tool error: {result['error']}")
    lines.append("")
    lines.append(
        "The following could not be verified against the retrieved data and were "
        "omitted: " + "; ".join(flags)
    )
    return "\n".join(lines)
