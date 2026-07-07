"""Probe the TrueFoundry AI Gateway.

Finds the OpenAI-compatible base URL, lists available models (OpenAI / Claude /
Perplexity / Gemini), and smoke-tests one chat completion per family. Loads
TFY_API_KEY + TFY_BASE_URL from .env. NEVER prints the key.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
KEY = os.environ["TFY_API_KEY"]
HOST = "https://tfy-eu.promptlens.trilogy.com"
BASES = [
    os.environ.get("TFY_BASE_URL", ""),
    f"{HOST}/api/llm",
    f"{HOST}/api/inference/openai",
    f"{HOST}/api/llm/openai",
    f"{HOST}/api/llm/api/inference/openai",
]
KEYWORDS = ["gpt-5", "gpt-4", "gpt", "claude", "gemini", "perplexity", "sonar", "o3", "o4"]


def try_base(base: str):
    try:
        c = OpenAI(api_key=KEY, base_url=base, timeout=25, max_retries=0)
        ids = [m.id for m in c.models.list().data]
        return c, ids
    except Exception as e:  # noqa: BLE001
        return None, f"ERR {type(e).__name__}: {str(e)[:180]}"


def main() -> None:
    working = None
    for base in BASES:
        if not base:
            continue
        c, ids = try_base(base)
        print(f"\n=== base={base} ===")
        if c is None:
            print(ids)
            continue
        print(f"models available: {len(ids)}")
        for kw in KEYWORDS:
            hit = [i for i in ids if kw.lower() in i.lower()]
            if hit:
                print(f"  {kw:>11}: {hit[:6]}")
        working = (c, ids, base)
        break

    if not working:
        print("\nNo working base URL found — check gateway host/path.")
        return

    c, ids, base = working
    print(f"\nWORKING BASE: {base}")

    def pick(subs):
        for i in ids:
            if any(s in i.lower() for s in subs):
                return i
        return None

    for label, subs in [("GPT (teacher)", ["gpt-5", "gpt-4o", "gpt"]), ("Claude (judge)", ["claude"]), ("Perplexity", ["sonar", "perplexity"])]:
        mid = pick(subs)
        if not mid:
            print(f"[{label}] no model matched")
            continue
        try:
            r = c.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=10,
            )
            print(f"[{label}] {mid} -> {r.choices[0].message.content!r}  OK")
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] {mid} -> ERR {type(e).__name__}: {str(e)[:180]}")


if __name__ == "__main__":
    main()
