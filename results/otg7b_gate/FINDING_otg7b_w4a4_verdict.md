# VERDICT: REJECT — openthaigpt1.5-7b-instruct NVFP4 W4A4 fails the paired gate

Date: 2026-08-11 · Pre-registration: PREREGISTERED.md (written 2026-08-10 before
any measurement) · All numbers below are true paired BF16-vs-NVFP4 on one DGX
Spark GB10, identical serving flags (vllm/vllm-openai:v0.25.1, seed 0), same
llm-compressor 0.12.0.1 NVFP4 W4A4 recipe + half-Thai 512x512 calibration that
shipped ThaiLLM-30B-NVFP4 and the 72B.

## Ship criteria vs outcome (pre-registered, unchanged)

| gate | required | measured | verdict |
|---|---|---|---|
| every Thai group n.s. | p > 0.05 | lm-eval thai_mc -1.29 p=0.077 PASS; thai_exam_v2 -0.35 p=0.91 PASS; **generative: openthaieval -2.35 p=0.021, ifeval-th -10.23 p=0.0009, math_500-th -5.60 p=0.008 ALL FAIL** | **FAIL** |
| pooled MC >= -1.5 pt | >= -1.5 | ALL_MC -1.53 (n=19,786, p=3e-12) | **FAIL** (borderline) |
| Thai bpb increase < +0.03 | < +0.03 | 0.3949 -> 0.4125 = +0.0176 | PASS |
| Thai top-1 agreement > 88% | > 88% | 87.90% (32,424 positions) | **FAIL** (borderline) |

3 of 4 gates failed -> **REJECT**. The artifact stays local; not uploaded to HF.

## Full paired table (generative, chinda-eval fixed adapters, branch fix/thai-lcb-and-numeric-scoring)

| bench (n) | BF16 | NVFP4 | delta | McNemar p |
|---|---|---|---|---|
| openthaieval (1232) | 0.6396 | 0.6161 | -2.35 | 0.0215 SIG |
| ifeval-th prompt strict (215) | 0.6279 | 0.5256 | **-10.23** | 0.0009 SIG |
| math_500-th (500) | 0.5260 | 0.4700 | -5.60 | 0.0084 SIG |
| live_code_bench-th Pass@1 (111) | 0.1802 | 0.1622 | -1.80 | 0.75 n.s. |
| aime24-th (30) | 1/30 | 1/30 | 0 | tiny n, no claim |
| Thai tool conformance (12) | 8/8 | 8/8 | 0 | intact |

lm-eval logprob (matched limits): MMLU@10 0.774->0.730; significant per-task:
hellaswag -1.22 (p=2e-10), winogrande -3.08, mmlu conceptual_physics -16,
machine_learning -12, thai_exam_v2_ic -10.5.

Speed (single-stream 128->1024): BF16 12.4 tok/s (TPOT 80.9ms) -> NVFP4
40.3 tok/s (TPOT 24.8ms) = **3.26x**, TTFT 84->37ms. Direction and magnitude
as pre-registered. Footprint 15.24 GB -> 5.5 GB.

## The two findings that matter beyond this model

1. **W4A4 cost scales inversely with model size.** Same recipe, same box:
   30B -0.53 n.s. (July, paired) · 72B ~-2 vs card anchor (unpaired) ·
   7B -2.35..-10.2 SIGNIFICANT (paired). Small models cannot absorb 4-bit
   activations. For 7B-class use W4A16 (weight-only) or FP8, or don't quantize.
2. **Logprob MC metrics dramatically understate generative damage.**
   thai_exam_v2 moved -0.35 n.s. while ifeval-th collapsed -10.2 on the same
   checkpoint pair. Error accumulation over long decodes is invisible to
   single-token ranking. This retroactively QUALIFIES the July 30B study's
   "Thai not significantly degraded" conclusion (which was MC/logprob-first):
   its generative axes (usecases divergence review) passed, but future gates
   must weight generative benches as first-class ship criteria — the
   thai-eval-gate skill is updated accordingly.

Artifact quality checks performed before believing the bad numbers (audit-skill
rule cuts both ways): no repetition loops, prediction-length distributions
near-identical across sides (p50 536 vs 547 chars), storage layout identical,
extraction spot-checked. The degradation is real.

Artifacts: results/otg7b_gate/{gen_bf16,gen_nvfp4}/** (reviews per bench),
{bf16,nvfp4}/** (lm-eval samples, perf, usecases), paired_analysis.json,
fidelity.json, summary_data.json, gate scripts scripts/otg7b_gate*.sh.
