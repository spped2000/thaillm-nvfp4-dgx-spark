#!/usr/bin/env python3
"""Consolidate every result into results/summary_data.json (report + charts source)."""
import glob
import json
import statistics
from pathlib import Path

P = Path(__file__).resolve().parent.parent
R = P / "results"


def latest_results(path):
    files = sorted(glob.glob(str(path / "**" / "results_*.json"), recursive=True))
    return json.load(open(files[-1]))["results"] if files else {}


def metric(r, task, key="acc,none"):
    return r.get(task, {}).get(key)


def primary_accuracy():
    out = {}
    for tag in ["bf16", "nvfp4"]:
        r = {}
        for sub in ["suite0", "thai_exam_v2", "mmlu", "ppl"]:
            r.update(latest_results(R / tag / sub))
        out[tag] = {
            "belebele_tha_Thai": (metric(r, "belebele_tha_Thai"), metric(r, "belebele_tha_Thai", "acc_stderr,none")),
            "xnli_th": (metric(r, "xnli_th"), metric(r, "xnli_th", "acc_stderr,none")),
            "xcopa_th": (metric(r, "xcopa_th"), metric(r, "xcopa_th", "acc_stderr,none")),
            "thai_exam_v2": (metric(r, "thai_exam_v2"), metric(r, "thai_exam_v2", "acc_stderr,none")),
            "thai_exam_v2_subsets": {s: metric(r, f"thai_exam_v2_{s}") for s in ["a_level", "ic", "onet", "tgat", "tpat1"]},
            "hellaswag": (metric(r, "hellaswag"), metric(r, "hellaswag", "acc_stderr,none")),
            "hellaswag_acc_norm": (metric(r, "hellaswag", "acc_norm,none"), metric(r, "hellaswag", "acc_norm_stderr,none")),
            "arc_challenge": (metric(r, "arc_challenge"), metric(r, "arc_challenge", "acc_stderr,none")),
            "winogrande": (metric(r, "winogrande"), metric(r, "winogrande", "acc_stderr,none")),
            "mmlu_5shot_50": (metric(r, "mmlu"), metric(r, "mmlu", "acc_stderr,none")),
            "thai_wiki_bpb": metric(r, "thai_wikipedia_ppl", "bits_per_byte,none"),
            "thai_wiki_byte_ppl": metric(r, "thai_wikipedia_ppl", "byte_perplexity,none"),
            "wikitext_bpb": metric(r, "wikitext", "bits_per_byte,none"),
            "wikitext_word_ppl": metric(r, "wikitext", "word_perplexity,none"),
        }
    return out


def perf():
    out = {}
    for tag in ["bf16", "nvfp4"]:
        runs = {}
        for f in sorted(glob.glob(str(R / tag / "perf" / "*.json"))):
            cfg = Path(f).stem.rsplit("_r", 1)[0]
            runs.setdefault(cfg, []).append(json.load(open(f)))
        out[tag] = {cfg: {k: statistics.median([r[k] for r in reps])
                          for k in ["output_throughput", "total_token_throughput",
                                    "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms",
                                    "p99_tpot_ms", "median_itl_ms", "median_e2el_ms"]}
                    for cfg, reps in runs.items()}
    return out


def references():
    out = {}
    for name in ["typhoon25", "sealion45", "qwen36-35b-nvfp4", "unsloth27b-nvfp4", "qwen3-8b-nvfp4"]:
        r = {}
        for sub in ["thai", "mmlu10", "ppl200"]:
            r.update(latest_results(R / f"ref_{name}" / sub))
        out[name] = {
            "belebele_tha_Thai": metric(r, "belebele_tha_Thai"),
            "xnli_th": metric(r, "xnli_th"),
            "xcopa_th": metric(r, "xcopa_th"),
            "thai_exam_v2": metric(r, "thai_exam_v2"),
            "mmlu_at10": metric(r, "mmlu"),
            "thai_bpb_at200": metric(r, "thai_wikipedia_ppl", "bits_per_byte,none"),
        }
    return out


def main():
    summary = {
        "primary_accuracy": primary_accuracy(),
        "paired": json.load(open(R / "paired_analysis.json")),
        "fidelity": json.load(open(R / "fidelity.json")),
        "perf": perf(),
        "footprint": {
            "bf16": {"disk_gb": 61.06, "load_s": 365.1, "weights_gib": 56.88},
            "nvfp4": {"disk_gb": 18.12, "load_s": 109.3, "weights_gib": 16.85},
        },
        "references": references(),
    }
    (R / "summary_data.json").write_text(json.dumps(summary, indent=2))
    print("wrote results/summary_data.json")
    refs = summary["references"]
    missing = [(n, k) for n, row in refs.items() for k, v in row.items() if v is None]
    print("still missing:", missing or "nothing")


if __name__ == "__main__":
    main()
