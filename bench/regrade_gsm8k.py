#!/usr/bin/env python3
"""Regrade an AIPerf GSM8K accuracy export with GLM-aware answer extraction.

The pinned AIPerf grader prefers the dataset's ``####`` marker and falls back
to the last number in the response. GLM-5.3 never emits ``####`` zero-shot,
so the fallback grabs whatever number happens to be last (units, restated
quantities), mis-grading a measurable fraction of correct answers. This tool
re-extracts the final answer with patterns matched to how GLM actually
answers, in priority order:

1. the last ``\\boxed{...}`` value;
2. the last bold ``**...**`` segment containing a number (GLM's final
   answer restatement is bold in practice);
3. the pinned grader's own last-number fallback.

It never modifies the export; it prints raw (pinned) and regraded accuracy
plus the disagreement count, so receipts can carry both numbers.
"""

from __future__ import annotations

import json
import re
import sys


NUM = r"-?\$?\d[\d,]*\.?\d*"


def _norm(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not re.fullmatch(r"-?\d+\.?\d*", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _nums(text: str) -> list[str]:
    return [n for n in (_norm(m) for m in re.findall(NUM, text)) if n is not None]


def extract(output: str) -> str | None:
    candidates: list[tuple[int, str]] = []

    def add(pattern: str, flags: int = 0) -> None:
        for m in re.finditer(pattern, output, flags):
            ns = _nums(m.group(1))
            if ns:
                candidates.append((m.end(), ns[-1]))

    add(r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}")
    add(r"\$\$((?:[^$]|\$(?!\$))+)\$\$")
    add(rf"answers? is[^.\n]*?({NUM})", re.IGNORECASE)
    for m in re.finditer(r"\*\*([^*]+)\*\*", output):
        if re.match(r"\s*Step\s*\d", m.group(1)):
            continue
        ns = _nums(m.group(1))
        if ns:
            candidates.append((m.end(), ns[-1]))
    if candidates:
        return max(candidates)[1]
    ns = _nums(output)
    return ns[-1] if ns else None


def main() -> None:
    path = sys.argv[1]
    raw_pass = regrade_pass = total = disagree = 0
    flipped_up, flipped_down = [], []
    for line in open(path):
        r = json.loads(line)
        total += 1
        raw = bool(r.get("passed"))
        raw_pass += raw
        expected = _norm(str(r.get("expected", "")))
        got = extract(r.get("model_output") or "")
        ok = expected is not None and got == expected
        regrade_pass += ok
        if ok != raw:
            disagree += 1
            (flipped_up if ok else flipped_down).append(r.get("conversation_id"))
    print(f"n={total}")
    print(f"pinned grader : {raw_pass}/{total} = {raw_pass/total:.1%}")
    print(f"regraded      : {regrade_pass}/{total} = {regrade_pass/total:.1%}")
    print(
        f"disagreements : {disagree} "
        f"(regrade-pass/pinned-fail {len(flipped_up)}, "
        f"regrade-fail/pinned-pass {len(flipped_down)})"
    )


if __name__ == "__main__":
    main()
