"""Re-score MATH-style reviews with the fixed numeric metric (no GPU).

Reads existing reviews jsonl (prediction already extracted by the adapter),
re-applies the CORRECTED scoring path (direct math_equal first, extraction as
fallback), and reports raw vs corrected accuracy plus every rescued row for
hand spot-checking. Monotonicity is asserted: the fix may only add credit.

Usage: .venv-chinda/bin/python scripts/rescore_math.py <reviews.jsonl> [...]
"""
import glob
import json
import sys

sys.path.insert(0, "/home/agicafet/Documents/ThaiLLM_Quantization/chinda-eval")
from evalscope.metrics.math_parser import extract_answer, math_equal, strip_answer_string  # noqa: E402


def corrected_score(pred: str, ref: str) -> float:
    ref_answer = strip_answer_string(ref)
    if math_equal(strip_answer_string(pred), ref_answer):
        return 1.0
    return float(math_equal(strip_answer_string(extract_answer(pred)), ref_answer))


for pattern in sys.argv[1:]:
    for path in sorted(glob.glob(pattern, recursive=True)):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        raw_sum, new_sum, rescued, regressed = 0.0, 0.0, [], []
        for r in rows:
            ss = r["sample_score"]["score"]
            raw = float(ss["value"].get("acc", 0.0))
            pred = str(ss.get("extracted_prediction") or "")
            gold = r.get("target")
            if gold is None:
                gold = (r.get("sample_metadata") or {}).get("answer", "")
            new = corrected_score(pred, str(gold))
            raw_sum += raw
            new_sum += new
            if new > raw:
                rescued.append((str(gold)[:30], pred[:30]))
            if new < raw:
                regressed.append((str(gold)[:30], pred[:30]))
        n = len(rows)
        assert not regressed, f"NON-MONOTONIC fix — regressions: {regressed[:5]}"
        print(f"{path.split('/otg72b/')[-1] if '/otg72b/' in path else path}")
        print(f"  n={n} raw={raw_sum/n:.4f} corrected={new_sum/n:.4f} rescued={len(rescued)}")
        for g, p in rescued[:10]:
            print(f"    rescued gold={g!r} pred={p!r}")
