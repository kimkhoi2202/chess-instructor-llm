"""Pick the eval judge empirically. A good faithfulness judge must score a truthful
coaching text high and a fabricated one (invented mate/queen-win) at zero, using the
engine facts as ground truth. Prints faithful vs fabricated scores per candidate judge.
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
c = OpenAI(api_key=os.environ["TFY_API_KEY"], base_url=os.environ["TFY_BASE_URL"], timeout=60, max_retries=0)

FACTS = (
    "Ground-truth engine facts for this position (authoritative):\n"
    "- Recommended move: Nf3. It is sound (about +0.3), develops a knight, prepares castling.\n"
    "- There is NO forced checkmate. NO move wins the opponent's queen. NO fork or pin is present.\n"
)
FAITHFUL = ("I'd play Nf3. It develops your knight toward the center and gets you ready to castle, "
            "which is calmer and stronger than pushing the queen out early. Takeaway: develop and castle before attacking.")
FABRICATED = ("I'd play Nf3. It wins Black's queen by force and sets up an unstoppable mate in two. "
              "Takeaway: always look for the knight fork that traps the queen.")
RUBRIC = (
    "Score the coaching text's TRUTHFULNESS 0-2 using ONLY the engine facts as truth: "
    "2 = every claim matches the facts; 1 = minor imprecision; 0 = contains a claim that "
    "contradicts the facts (a mate/queen-win/fork the facts say does not exist). "
    'Return strict JSON only: {"truthfulness": <0|1|2>, "why": "<short>"}.'
)
JUDGES = [
    "claude-group/claude-opus-4-8",
    "claude-group/claude-sonnet-5",
    "claude-group/claude-fable-5",
    "gemini-group/gemini-3.1-pro",
    "claude-group/claude-opus-4-7",
]


def score(model: str, text: str):
    r = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict chess-coaching faithfulness judge."},
            {"role": "user", "content": FACTS + '\nCoaching text:\n"' + text + '"\n\n' + RUBRIC},
        ],
        max_tokens=400,
    )
    out = r.choices[0].message.content or ""
    try:
        return json.loads(out[out.find("{"): out.rfind("}") + 1]).get("truthfulness")
    except Exception:  # noqa: BLE001
        return f"parse-fail({out[:40]!r})"


for m in JUDGES:
    try:
        f, b = score(m, FAITHFUL), score(m, FABRICATED)
        verdict = "GOOD (2/0)" if f == 2 and b == 0 else "weak separation"
        print(f"{m:38} faithful={f}  fabricated={b}   {verdict}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:38} ERR {type(e).__name__} {str(e)[:90]}")
