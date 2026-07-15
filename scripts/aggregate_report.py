#!/usr/bin/env python3
"""Aggregate lm_eval + vllm bench results into results/report.md.

Usage: aggregate_report.py  (expects results/{bf16,nvfp4}/ populated)
"""
import glob
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent.parent
R = P / "results"


def latest_results_json(run_dir: Path):
    """lm_eval writes <output_path>/<sanitized-model>/results_<ts>.json."""
    files = sorted(glob.glob(str(run_dir / "**" / "results_*.json"), recursive=True))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def collect_accuracy(tag: str):
    out = {}
    for sub in ["suite0", "thai_exam_v2", "mmlu", "ppl"]:
        data = latest_results_json(R / tag / sub)
        if not data:
            continue
        for task, metrics in data.get("results", {}).items():
            row = {}
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    continue
                base = k.split(",")[0]
                if base in ("acc", "acc_norm", "word_perplexity", "byte_perplexity",
                            "bits_per_byte"):
                    row[base] = v
                elif base in ("acc_stderr", "acc_norm_stderr"):
                    row[base] = v
            if row:
                out[task] = row
    return out


def collect_perf(tag: str):
    """Median over repeats per (in,out,concurrency) config."""
    runs = {}
    for f in sorted(glob.glob(str(R / tag / "perf" / "*.json"))):
        name = Path(f).stem  # e.g. 1024x128_c1_r2
        cfg = name.rsplit("_r", 1)[0]
        with open(f) as fh:
            d = json.load(fh)
        runs.setdefault(cfg, []).append(d)
    med = {}
    for cfg, reps in runs.items():
        keys = ["output_throughput", "total_token_throughput", "mean_ttft_ms",
                "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms",
                "median_itl_ms", "p99_itl_ms", "median_e2el_ms"]
        med[cfg] = {k: statistics.median([r[k] for r in reps if k in r])
                    for k in keys if any(k in r for r in reps)}
        med[cfg]["repeats"] = len(reps)
    return med


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    acc = {tag: collect_accuracy(tag) for tag in ["bf16", "nvfp4"]}
    perf = {tag: collect_perf(tag) for tag in ["bf16", "nvfp4"]}

    lines = ["# ThaiLLM-30B: BF16 vs NVFP4 (modelopt) on DGX Spark — comparison report", ""]

    lines += ["## Accuracy (identical lm_eval invocations, seed 0)", "",
              "| Task | Metric | BF16 | NVFP4 | Δ (NVFP4−BF16) | ±stderr(BF16) | Notes |",
              "|---|---|---|---|---|---|---|"]
    for task in sorted(set(acc["bf16"]) | set(acc["nvfp4"])):
        b, n = acc["bf16"].get(task, {}), acc["nvfp4"].get(task, {})
        for metric in ["acc", "acc_norm", "word_perplexity", "byte_perplexity", "bits_per_byte"]:
            if metric in b or metric in n:
                bv, nv = b.get(metric), n.get(metric)
                delta = (nv - bv) if (bv is not None and nv is not None) else None
                se = b.get(f"{metric}_stderr")
                note = ""
                if delta is not None and se:
                    note = "noise" if abs(delta) <= se else ("**signif.**" if abs(delta) > 2 * se else "~1-2σ")
                lines.append(
                    f"| {task} | {metric} | {fmt(bv)} | {fmt(nv)} | "
                    f"{fmt(delta) if delta is not None else '—'} | {fmt(se) if se else '—'} | {note} |")

    lines += ["", "## Performance (vllm bench serve, random dataset, ignore-eos, median of 3)", "",
              "| Config | Metric | BF16 | NVFP4 | Ratio (NVFP4/BF16) |", "|---|---|---|---|---|"]
    for cfg in sorted(set(perf["bf16"]) | set(perf["nvfp4"])):
        b, n = perf["bf16"].get(cfg, {}), perf["nvfp4"].get(cfg, {})
        for metric in ["output_throughput", "median_ttft_ms", "median_tpot_ms", "median_itl_ms"]:
            bv, nv = b.get(metric), n.get(metric)
            ratio = (nv / bv) if (bv and nv) else None
            lines.append(f"| {cfg} | {metric} | {fmt(bv, 2) if bv else '—'} | "
                         f"{fmt(nv, 2) if nv else '—'} | {fmt(ratio, 3) if ratio else '—'} |")

    lines += ["", "## Methodology", ""]
    meth = R / "methodology.md"
    if meth.exists():
        lines.append(meth.read_text())

    out = R / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
