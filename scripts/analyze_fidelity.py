#!/usr/bin/env python3
"""Token-level fidelity analysis: BF16 vs NVFP4 from usecases.json captures.

Outputs results/fidelity.json (metrics) and results/usecase_side_by_side.md.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent
R = P / "results"


def load(tag):
    return json.loads((R / tag / "usecases.json").read_text())


def teacher_forced_metrics(bf16, nvfp4):
    out = {}
    for lang in bf16["teacher_forced"]:
        agree = total = 0
        drift_sum = 0.0
        drift_n = 0
        for rb, rn in zip(bf16["teacher_forced"][lang], nvfp4["teacher_forced"][lang]):
            assert rb["idx"] == rn["idx"]
            n = min(len(rb["top_tokens"]), len(rn["top_tokens"]))
            # skip position 0 (no logprob for first token)
            for i in range(1, n):
                tb, tn = rb["top_tokens"][i], rn["top_tokens"][i]
                if tb is not None and tn is not None:
                    total += 1
                    agree += tb == tn
                lb, ln = rb["token_logprobs"][i], rn["token_logprobs"][i]
                if lb is not None and ln is not None:
                    drift_sum += abs(lb - ln)
                    drift_n += 1
        out[lang] = {
            "positions": total,
            "top1_agreement": agree / total if total else None,
            "mean_abs_logprob_drift": drift_sum / drift_n if drift_n else None,
        }
    return out


def generation_metrics(bf16, nvfp4):
    rows = {}
    for name in bf16["generations"]:
        gb, gn = bf16["generations"][name], nvfp4["generations"][name]
        tb, tn = gb.get("tokens") or [], gn.get("tokens") or []
        div = next((i for i, (a, b) in enumerate(zip(tb, tn)) if a != b),
                   min(len(tb), len(tn)))
        rows[name] = {
            "divergence_token": div,
            "bf16_tokens": len(tb),
            "nvfp4_tokens": len(tn),
            "identical": gb["text"] == gn["text"],
        }
    return rows


def main():
    bf16, nvfp4 = load("bf16"), load("nvfp4")
    fid = {
        "teacher_forced": teacher_forced_metrics(bf16, nvfp4),
        "generations": generation_metrics(bf16, nvfp4),
    }
    (R / "fidelity.json").write_text(json.dumps(fid, indent=2))

    lines = ["# Use-case side-by-side: BF16 vs NVFP4 (greedy, seed 0, 200 tokens)", ""]
    for name in bf16["generations"]:
        gb, gn = bf16["generations"][name], nvfp4["generations"][name]
        g = fid["generations"][name]
        lines += [f"## {name}",
                  f"*diverges at token {g['divergence_token']}"
                  f"{' — identical output' if g['identical'] else ''}*", "",
                  f"**Prompt:** `{gb['prompt'][:200]}`", "",
                  "**BF16:**", "```", gb["text"].strip()[:1200], "```", "",
                  "**NVFP4:**", "```", gn["text"].strip()[:1200], "```", ""]
    (R / "usecase_side_by_side.md").write_text("\n".join(lines))
    print(json.dumps(fid["teacher_forced"], indent=2))
    print("generation divergences:",
          {k: v["divergence_token"] for k, v in fid["generations"].items()})


if __name__ == "__main__":
    main()
