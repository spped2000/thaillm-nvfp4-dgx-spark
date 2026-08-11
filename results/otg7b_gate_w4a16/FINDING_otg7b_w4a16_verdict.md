# VERDICT ARM 2: W4A16 weight-only — still REJECT for the 7B, by one axis

Date: 2026-08-11 · Pre-registered as ARM 2 in PREREGISTERED.md (before
quantizing). BF16 side reused from arm 1 (identical contract, stated in
advance). Scheme NVFP4A16: FP4 weights group 16, activations BF16, data-free.

## Gate outcome

| gate | W4A4 (arm 1) | W4A16 (arm 2) | required |
|---|---|---|---|
| pooled ALL_MC | -1.53 FAIL | **-0.54 PASS** | >= -1.5 |
| thai_mc / thai_exam_v2 | -1.29 p=.08 / -0.35 | **-0.28 p=.65 / +0.88 p=.61 PASS** | n.s. |
| Thai bpb | +0.018 | **+0.011 PASS** | < +0.03 |
| Thai top-1 agreement | 87.9% FAIL | **90.3% PASS** | > 88% |
| openthaieval (gen) | -2.35 p=.02 FAIL | **-0.73 p=.40 PASS** | n.s. |
| ifeval-th (gen) | -10.2 p=.0009 FAIL | **-3.72 p=.20 PASS** (point est. still notable; n=215 underpowered for ±3pt) | n.s. |
| **math_500-th (gen)** | -5.60 p=.008 FAIL | **-5.40 p=.008 FAIL** | n.s. |
| lcb-th / aime / tools | n.s. / 1=1 / 8/8 | **0.00 p=1 / 1/30=1/30 / 8/8** | — |

Pre-registered decision rule: "if generative Thai still significant -> verdict
stays REJECT for 7B at 4-bit weights entirely." math_500-th is significant.
**VERDICT: REJECT** — both 4-bit arms, per the rule written before measurement.

Speed: 41.8 tok/s (TPOT 23.9ms) — slightly FASTER than W4A4 (40.3) and 3.4x
BF16. Marlin/compressed-tensors W4A16-FP4 on SM121/aarch64 works (the
pre-declared serve risk did not materialize). Footprint 5.5 GB.

## The mechanism, isolated by the two arms

- Activation quantization (A4) was responsible for the knowledge and
  instruction-following damage: openthaieval -2.35 -> -0.73, ifeval -10.2 ->
  -3.7 when activations went back to BF16.
- **The math damage lives in the FP4 weights themselves**: -5.6 vs -5.4,
  unchanged between arms. Long reasoning chains compound weight-precision
  error; short-form tasks absorb it. Mechanical bound held (W4A16 >= W4A4 on
  every axis) so this is not a harness artifact.
- Combined with July's 30B result (W4A4 ~free) and the 72B anchors (~-2):
  4-bit weights are safe at 30B+, marginal at 72B-class tasks, and
  math-destructive at 7B. Parameter count buys redundancy that absorbs
  weight-precision loss.

## What a release WOULD require

W4A16 passes every gate except Thai math. A use-case-scoped release ("chat /
instruction / RAG: gate-passing; math: degraded -5.4pt") is defensible but
requires a NEW pre-registration naming that scope - not a retroactive
weakening of this one. Owner's call; the checkpoint stays local meanwhile.

Artifacts: results/otg7b_gate_w4a16/ (lm suites, perf, usecases, gen reviews;
LCB as slim jsonl), paired_analysis.json, fidelity.json. BF16 baseline:
results/otg7b_gate/{bf16,gen_bf16}.

## Integrity note (2026-08-11)
run_perf.sh's container-side --result-dir still pointed at the July
results/{bf16,nvfp4}/perf paths (the host-side OUT was redirected but the
in-container path was missed), so the 7B perf JSONs transiently overwrote the
July 30B perf JSONs. Detected via git status before any commit; July files
restored byte-perfect from git; 7B JSONs preserved under the gate roots
(arm-1 bf16 + W4A4; arm-2 W4A16 JSONs were never saved - its full printed
metric tables live in results/otg7b_gate_w4a16/nvfp4/perf/*.log). Script
fixed to /work/$RELOUT/$TAG/perf. No published number was affected.
