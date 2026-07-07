"""Map which TrueFoundry gateway models this key can actually CALL (not just list).

Lists models, filters to the GPT/Claude/Gemini/Perplexity candidates, and probes
each with a tiny call, printing OK vs DENY(code). Never prints the key.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
c = OpenAI(api_key=os.environ["TFY_API_KEY"], base_url=os.environ["TFY_BASE_URL"], timeout=30, max_retries=0)
ids = [m.id for m in c.models.list().data]
kws = ["gpt-5", "gpt-4o", "claude", "gemini", "sonar", "perplexity"]
cand = [i for i in ids if "/" in i and any(k in i.lower() for k in kws)]
seen, probe = set(), []
for i in cand:
    if i not in seen:
        seen.add(i)
        probe.append(i)

print(f"probing {len(probe)} candidate models...\n")
ok = []
for m in probe:
    try:
        c.chat.completions.create(model=m, messages=[{"role": "user", "content": "ok"}], max_tokens=5)
        print("OK      ", m)
        ok.append(m)
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "status_code", "?")
        print(f"DENY {code} ", m)

print("\n=== ACCESSIBLE ===")
for label, kw in [("OpenAI", "gpt"), ("Claude", "claude"), ("Gemini", "gemini")]:
    hits = [m for m in ok if kw in m.lower()]
    print(f"  {label}: {hits}")
