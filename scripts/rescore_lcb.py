"""Offline re-score of live_code_bench-th with the FIXED adapter (no GPU).

Generation was greedy and is frozen in the reviews jsonl; only the scoring
changed (hidden-test decode + all-or-nothing + strict output equality), so we
re-execute the SAVED code against the FULL test set. Uses the patched
adapter's own run_code_test to avoid logic drift.

Run: .venv-chinda/bin/python scripts/rescore_lcb.py
"""
import base64
import glob
import json
import pickle
import sys
import zlib

sys.path.insert(0, "/home/agicafet/Documents/ThaiLLM_Quantization/chinda-eval")
import evalscope.benchmarks  # noqa: F401 - populates the registry
from evalscope.api.registry import BENCHMARK_REGISTRY


class _Mod:
    LiveCodeBenchThaiAdapter = BENCHMARK_REGISTRY["live_code_bench-th"].data_adapter


mod = _Mod()


def parse_tests(raw):
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(raw.encode())))
            parsed = json.loads(decoded) if isinstance(decoded, str) else decoded
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return []


from datasets import load_dataset

ds = load_dataset("iapp/code_generation_lite-th", split="test")
rows_by_q = {}
for row in ds:
    key = (row.get("question_content_th") or row.get("question_content") or "")[:200]
    rows_by_q[key] = row

adapter = object.__new__(mod.LiveCodeBenchThaiAdapter)
adapter.timeout = 10
adapter.debug = False

review_files = glob.glob(
    "/home/agicafet/Documents/ThaiLLM_Quantization/results/otg72b/live_code_bench-th/**/reviews/**/*.jsonl",
    recursive=True,
)
reviews = [json.loads(l) for f in review_files for l in open(f) if l.strip()]
print(f"reviews: {len(reviews)}, dataset rows: {len(ds)}")

strict_pass = 0
no_match = 0
results = []
for i, r in enumerate(reviews):
    ss = r["sample_score"]["score"]
    code = ss.get("extracted_prediction") or ""
    problem_text = str(r.get("input", ""))
    drow = None
    for key, row in rows_by_q.items():
        if key and key[:120] in problem_text:
            drow = row
            break
    if drow is None:
        no_match += 1
        results.append({"idx": r.get("index"), "status": "NO_DATASET_MATCH"})
        continue
    tests = parse_tests(drow.get("public_test_cases")) + parse_tests(drow.get("private_test_cases"))
    requires_stdin = ("อินพุตมาตรฐาน" in problem_text or "stdin" in problem_text.lower()
                      or "input()" in problem_text.lower())
    passed = 0
    for t in tests:
        if not isinstance(t, dict):
            continue
        if adapter.run_code_test(code, t.get("input", ""), str(t.get("output", "")).strip(), requires_stdin):
            passed += 1
    total = sum(1 for t in tests if isinstance(t, dict))
    ok = total > 0 and passed == total
    strict_pass += ok
    results.append({"idx": r.get("index"), "passed": passed, "total": total, "strict": ok,
                    "old_value": ss["value"]})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(reviews)} scored, strict so far {strict_pass}", flush=True)

n = len(reviews) - no_match
print(f"\nSTRICT ALL-TESTS (public+private) Pass@1: {strict_pass}/{n} = {strict_pass/n:.4f}")
print(f"unmatched rows: {no_match}")
out = "/home/agicafet/Documents/ThaiLLM_Quantization/results/otg72b/lcb_rescore_full_tests.json"
json.dump({"strict_pass": strict_pass, "n": n, "no_match": no_match, "rows": results},
          open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
