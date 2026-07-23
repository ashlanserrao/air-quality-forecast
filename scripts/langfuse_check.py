# scripts/langfuse_check.py — surface the REAL Langfuse connection status.
# Unlike observability.py (which swallows errors to stay non-fatal), this fails
# loudly so we can see exactly why traces aren't arriving. debug=True prints the
# actual ingestion HTTP batch request + response.
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langfuse").setLevel(logging.DEBUG)

pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
host = os.environ.get("LANGFUSE_HOST", "(unset -> SDK default https://cloud.langfuse.com)")

print("--- env as seen by Python ---")
print("PUBLIC_KEY:", (pk[:8] + "..." + pk[-4:]) if pk else "MISSING")
print("SECRET_KEY:", (sk[:8] + "..." + sk[-4:]) if sk else "MISSING")
print("HOST:      ", host)
print()

from langfuse import Langfuse

lf = Langfuse(debug=True)  # picks up the three env vars; debug prints HTTP batches

print("--- auth_check (pings the server with your keys) ---")
try:
    ok = lf.auth_check()
    print("auth_check() ->", ok)
except Exception as e:
    print("auth_check() RAISED:", type(e).__name__, e)

print()
print("--- sending one test trace (watch for the ingestion batch below) ---")
t = lf.trace(name="langfuse_check", input="connectivity test")
t.update(output="ok")
print("trace id:", t.id)
lf.flush()
print("flush() returned")
