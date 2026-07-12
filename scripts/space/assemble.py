"""Build the Chess-Coach Benchmark Space's index.html from the template.

The page is a fully STATIC, self-contained dashboard: every number is embedded
verbatim from the committed corrected-benchmark docs (``RESULTS_STAGE4_CORRECTED.md``
for the current v4 / v6-dpo / v6-dpo2 lineup on the 120 held-out TEST, and
``RESULTS_FULL_EVAL_803.md`` for the 803-position, 15-model field moat). There is no
runtime data injection, so "assembling" is a validate-and-copy: it guards against
leftover ``__PLACEHOLDER__`` tokens and writes the built file out.

Usage:
    python scripts/space/assemble.py [--out /tmp/space_bench/index.html]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "index.template.html"
DEFAULT_OUT = Path("/tmp/space_bench/index.html")

# any run of >=2 leading/trailing underscores around CAPS is a build placeholder
_PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")


def build(out: Path) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    leftover = sorted(set(_PLACEHOLDER.findall(html)))
    if leftover:
        raise SystemExit(f"template still has unsubstituted placeholders: {leftover}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[assemble] wrote {out} ({len(html):,} bytes)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output index.html path")
    args = p.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
