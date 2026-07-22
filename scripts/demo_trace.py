# scripts/demo_trace.py — run a few queries and print the captured trace tree.
# Runs local-only when no LANGFUSE_* keys are set, so the trace structure can be
# inspected before wiring up the real dashboard.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent
from observability import tracer, render

QUERIES = [
    "is Delhi's air worse than what WHO considers safe?",
    "will next week's air be bad enough to trigger GRAP restrictions?",
    "what is PM2.5",
]


def main():
    print(f"Langfuse delivery enabled: {tracer.langfuse_enabled}  "
          f"(local capture always on)\n")
    for q in QUERIES:
        agent.answer(q)
        print("=" * 88)
        print(render(tracer.last_trace))
        print()


if __name__ == "__main__":
    main()
