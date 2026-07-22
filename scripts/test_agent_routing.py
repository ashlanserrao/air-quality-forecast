# scripts/test_agent_routing.py — verify deterministic pre-routing forces the
# right tools, and that a plain definitional query triggers neither.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent

QUERIES = [
    "will next week's air be bad enough to trigger GRAP restrictions?",
    "is Delhi's air worse than what WHO considers safe?",
    "Compare next week's Delhi pollution to international health limits",
    "what is PM2.5",
]


def main():
    original_run_tool = agent._run_tool

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        forced = agent._pre_route(query)
        print(f"pre-routed: {[(n, a) for n, a in forced] or 'none'}")

        invoked = []

        def traced(name, arguments, _orig=original_run_tool, _log=invoked):
            _log.append((name, arguments))
            return _orig(name, arguments)

        agent._run_tool = traced
        try:
            reply = agent.answer(query)
        finally:
            agent._run_tool = original_run_tool

        forced_names = [n for n, _ in forced]
        extra = invoked[len(forced):]
        print(f"actually invoked: {[n for n, _ in invoked] or 'none'}")
        print(f"  forced: {forced_names or 'none'}")
        print(f"  llm-chosen (beyond forced): {[n for n, _ in extra] or 'none'}")
        print(f"\nANSWER:\n{reply}\n")


if __name__ == "__main__":
    main()
