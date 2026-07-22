# scripts/test_guardrail.py — verify the post-generation guardrail catches the
# three historical grounding bugs from this session, without false-positiving on
# a legitimately-grounded answer (which is what makes it usable, not just noisy).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import the guardrail module directly, not the agent — verify_response is pure,
# so this test needs no API key and doesn't boot the model/TensorFlow stack.
import guardrail

WHO_DOC = "9789240034433-eng.pdf"
GRAP_DOC = "GRAP Schedule7e3bd726-7308-484f-995f-7cb3fa705b8d.pdf"


def _forecast(entries):
    return {
        "horizon": len(entries),
        "note": "pm25_ugm3 is a concentration; aqi is the CPCB sub-index.",
        "forecast": [
            {"day": i, "pm25_ugm3": p, "aqi": a, "aqi_category": c}
            for i, (p, a, c) in enumerate(entries, start=1)
        ],
    }


# --- Case 1: AQI-unit conflation ------------------------------------------
# Truth: 46.51 µg/m³ -> AQI 78. The buggy reply treats the concentration as if
# it were the index and invents "AQI 120".
CASE1 = (
    "AQI-unit conflation",
    "At 46.51 µg/m³ the air is Moderate, around AQI 120.",
    [_forecast([(46.51, 78, "Satisfactory")])],
    True,
)

# --- Case 2: fabricated WHO number (pre-hardcode retrieval state) ----------
# Retrieval returned only WHO methodology prose with no guideline numbers; the
# reply fills the gap with a threshold (40) from pretraining.
CASE2 = (
    "fabricated WHO number",
    "WHO's 24-hour safe limit is 40 µg/m³, so Delhi at 46.51 is only just above it.",
    [
        _forecast([(46.51, 78, "Satisfactory")]),
        {
            "results": [
                {
                    "text": "Methods used to develop the guidelines. An indicator "
                    "of airborne soot and combustion; systematic review of risk.",
                    "source_document": WHO_DOC,
                    "section_title": "Methods used to develop the guidelines",
                    "page_start": 7,
                    "page_end": 7,
                    "chunk_type": "text",
                }
            ]
        },
    ],
    True,
)

# --- Case 3: cited stage that wasn't retrieved -----------------------------
# Only Stage I was retrieved; the reply cites Stage III restrictions.
CASE3 = (
    "uncited GRAP stage",
    "This triggers GRAP Stage III restrictions, banning construction activity.",
    [
        _forecast([(46.51, 78, "Satisfactory")]),
        {
            "results": [
                {
                    "text": "Stage I - Poor (AQI 201-300). Ensure road dust "
                    "suppression and enforce dust-control at construction sites.",
                    "source_document": GRAP_DOC,
                    "section_title": "Stage I",
                    "page_start": 2,
                    "page_end": 2,
                    "chunk_type": "table",
                }
            ]
        },
    ],
    True,
)

# --- Case 4: clean answer (no false positive) ------------------------------
# The real good WHO-comparison answer, including a computed "2.5 times" multiple,
# "PM2.5" tokens, and the year 2021 — none of which should flag.
CASE4 = (
    "clean grounded answer",
    "The forecast shows Day 1: 38.26 µg/m³ (AQI 64, Satisfactory) up to Day 7: "
    "45.98 µg/m³ (AQI 77, Satisfactory). Per the WHO 2021 guidelines the PM2.5 "
    "24-hour guideline is 15 µg/m³ and the annual guideline is 5 µg/m³. Day 1's "
    "38.26 µg/m³ is about 2.5 times the WHO 24-hour guideline, so Delhi is worse "
    "than what WHO considers safe. Consider limiting outdoor activity for 2 hours.",
    [
        _forecast(
            [
                (38.26, 64, "Satisfactory"),
                (40.62, 68, "Satisfactory"),
                (42.22, 71, "Satisfactory"),
                (43.45, 73, "Satisfactory"),
                (44.42, 75, "Satisfactory"),
                (45.24, 76, "Satisfactory"),
                (45.98, 77, "Satisfactory"),
            ]
        ),
        {
            "results": [
                {
                    "text": "WHO 2021 Global Air Quality Guidelines - PM2.5: AQG "
                    "level 5 ug/m3 annual mean; 15 ug/m3 24-hour mean. Interim "
                    "targets annual 35 25 15 10; 24-hour 75 50 37.5 25.",
                    "source_document": WHO_DOC,
                    "section_title": "WHO 2021 PM2.5 guideline levels",
                    "page_start": 8,
                    "page_end": 8,
                    "chunk_type": "reference",
                }
            ]
        },
    ],
    False,
)

CASES = [CASE1, CASE2, CASE3, CASE4]


def main():
    all_ok = True
    for name, text, tool_results, expect_flags in CASES:
        flags = guardrail.verify_response(text, tool_results)
        got_flags = bool(flags)
        ok = got_flags == expect_flags
        all_ok = all_ok and ok
        print("=" * 78)
        print(f"CASE: {name}")
        print(f"  expected flags: {expect_flags}   got flags: {got_flags}   "
              f"[{'PASS' if ok else 'FAIL'}]")
        for f in flags:
            print(f"    - {f}")

    print("=" * 78)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
