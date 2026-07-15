#!/usr/bin/env python3
"""Paired per-sample BF16-vs-NVFP4 analysis from lm_eval --log_samples output.

- McNemar exact test per task (same docs both runs -> paired flips).
- Matched-limit slices for cross-model tables: mmlu@first-10/subject,
  thai_wikipedia_ppl@first-200 docs.
Writes results/paired_analysis.json and prints a summary.
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent.parent
R = P / "results"


def load_samples(tag):
    """{task: {doc_id: row}} from all samples_*.jsonl under results/<tag>/."""
    out = defaultdict(dict)
    for f in glob.glob(str(R / tag / "**" / "samples_*.jsonl"), recursive=True):
        task = Path(f).name.split("samples_")[1].rsplit("_2026-", 1)[0]
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                out[task][int(row["doc_id"])] = row
    return out


def binom_two_sided_p(k, n):
    """Exact two-sided binomial test p-value for k successes of n at p=0.5."""
    if n == 0:
        return 1.0
    def logpmf(i):
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + n * math.log(0.5))
    lpk = logpmf(k)
    return min(1.0, sum(math.exp(lp) for i in range(n + 1)
                        if (lp := logpmf(i)) <= lpk + 1e-9))


def mcnemar(task_rows_b, task_rows_n, metric="acc"):
    n01 = n10 = both = neither = 0
    for did, rb in task_rows_b.items():
        rn = task_rows_n.get(did)
        if rn is None or metric not in rb or metric not in rn:
            continue
        b, n = rb[metric] > 0.5, rn[metric] > 0.5
        if b and n: both += 1
        elif b and not n: n01 += 1
        elif not b and n: n10 += 1
        else: neither += 1
    total = both + neither + n01 + n10
    return {
        "n": total, "both_right": both, "both_wrong": neither,
        "bf16_only_right": n01, "nvfp4_only_right": n10,
        "delta_acc": (n10 - n01) / total if total else None,
        "mcnemar_p": binom_two_sided_p(min(n01, n10), n01 + n10),
    }


def ppl_slice(tag, task, first_n):
    """bits_per_byte over first N docs from per-sample [ll, num_bytes]."""
    rows = load_samples(tag).get(task, {})
    ll = nb = 0.0
    for did in sorted(rows)[:first_n]:
        l, b = rows[did]["bits_per_byte"]
        ll += l; nb += b
    return -ll / (nb * math.log(2)) if nb else None


def main():
    sb, sn = load_samples("bf16"), load_samples("nvfp4")
    tasks = sorted(set(sb) & set(sn) - {"thai_wikipedia_ppl", "wikitext"})

    per_task, groups = {}, defaultdict(lambda: [0, 0, 0])  # group: [n01, n10, n]
    for t in tasks:
        res = mcnemar(sb[t], sn[t])
        per_task[t] = res
        if t.startswith("mmlu_"): g = "mmlu(all)"
        elif t.startswith("thai_exam_v2"): g = "thai_exam_v2(all)"
        elif t.startswith("thai_exam"): g = "thai_exam_v1(all)"
        elif t in ("belebele_tha_Thai", "xnli_th", "xcopa_th"): g = "thai_mc(all)"
        else: g = "english_mc(all)"
        groups[g][0] += res["bf16_only_right"]
        groups[g][1] += res["nvfp4_only_right"]
        groups[g][2] += res["n"]

    group_stats = {}
    for g, (n01, n10, n) in groups.items():
        group_stats[g] = {
            "n": n, "bf16_only_right": n01, "nvfp4_only_right": n10,
            "delta_acc": (n10 - n01) / n if n else None,
            "mcnemar_p": binom_two_sided_p(min(n01, n10), n01 + n10),
        }
    # grand total across every MC sample
    N01 = sum(v[0] for v in groups.values()); N10 = sum(v[1] for v in groups.values())
    N = sum(v[2] for v in groups.values())
    group_stats["ALL_MC"] = {
        "n": N, "bf16_only_right": N01, "nvfp4_only_right": N10,
        "delta_acc": (N10 - N01) / N,
        "mcnemar_p": binom_two_sided_p(min(N01, N10), N01 + N10),
    }

    # matched-limit slices for cross-model comparability
    matched = {}
    for tag in ["bf16", "nvfp4"]:
        s = load_samples(tag)
        mm_first10 = [r["acc"] for t, rows in s.items() if t.startswith("mmlu_")
                      for did, r in rows.items() if did < 10]
        matched[tag] = {
            "mmlu_first10_acc": sum(mm_first10) / len(mm_first10) if mm_first10 else None,
            "mmlu_first10_n": len(mm_first10),
            "thai_wiki_bpb_first200": ppl_slice(tag, "thai_wikipedia_ppl", 200),
        }

    out = {"per_task": per_task, "groups": group_stats, "matched_limits": matched}
    (R / "paired_analysis.json").write_text(json.dumps(out, indent=2))

    print(f"{'group':22} {'n':>6} {'b16only':>8} {'nv4only':>8} {'Δacc':>8} {'p':>8}")
    for g, v in sorted(group_stats.items()):
        print(f"{g:22} {v['n']:>6} {v['bf16_only_right']:>8} {v['nvfp4_only_right']:>8} "
              f"{v['delta_acc']:>8.4f} {v['mcnemar_p']:>8.4f}")
    print("\nmatched limits:", json.dumps(matched, indent=1))
    sig = {t: v for t, v in per_task.items() if v["mcnemar_p"] < 0.05}
    print("\nper-task significant (p<0.05):", list(sig) or "none")
    for t, v in sig.items():
        print(" ", t, v)


if __name__ == "__main__":
    main()
