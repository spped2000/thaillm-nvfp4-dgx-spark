# PRE-REGISTERED: openthaigpt1.5-7b-instruct BF16 vs NVFP4 paired gate

Written 2026-08-10, BEFORE any measurement of either side. Purpose: the first
TRUE paired A/B for the OpenThaiGPT family on this box, using the exact
recipe the (unpairable) 72B shipped with — llm-compressor 0.12.0.1 NVFP4 W4A4,
half-Thai 512x512 calibration, lm_head+embeddings BF16.

## Expected direction and range (from the ThaiLLM-30B paired study, same recipe family)

| axis | expectation | mechanical bound |
|---|---|---|
| pooled MC delta (lm-eval) | ~ -0.8 pt, accept to -1.5 | NVFP4 significantly BETTER on any axis = suspect the harness first, not the model |
| Thai MC groups (thai_exam_v2, belebele, xnli, xcopa) | not significant (McNemar p > 0.05) | — |
| Thai wiki bits/byte | increase < +0.03 | quantization can only lose byte fidelity |
| top-1 agreement (usecases) | > 88% Thai | — |
| decode speed | NVFP4 2-3x BF16 (weights 15.2 GB -> ~5 GB, bandwidth-bound) | NVFP4 slower than BF16 = config bug |
| load time | NVFP4 faster | — |

## Developer-card anchors (BF16, their runner - context only, never a delta)
OpenThaiEval 64.5 · MATH500-TH 24.2 · LiveCodeBench-TH 22.52 · AIME24-TH 0 · Language Accuracy 97.6

## Statistics, fixed in advance
- lm-eval: per-item pairing on doc_id, exact McNemar (scripts/paired_analysis.py).
- Generative: scripts/mcnemar_reviews.py on prompt-hash intersection;
  ifeval-th judged on --metric prompt_level_strict (binary), NOT inst_level fractions.
- n per bench: openthaieval 1232 / ifeval-th 215 / math_500-th 500 /
  live_code_bench-th 111 / aime24-th 30 / tool conformance 12.
- NO superiority claim on any bench with n <= 30 regardless of point estimate.
- Significance threshold p < 0.05, stated per-axis; 57-subject MMLU expects
  ~3 false positives - Bonferroni before declaring a subject regression.

## Grader/comparability rules
- chinda-eval MUST be on branch fix/thai-lcb-and-numeric-scoring (checked, is).
  LCB scored on ALL tests incl. hidden, all-or-nothing; MATH single-extraction.
- Fixed-grader numbers never share a column with buggy-grader or card numbers.
- Both sides serve on vllm/vllm-openai:v0.25.1 with byte-identical flags except
  the model path (compressed-tensors auto-detects; no --quantization flag).
  NOTE: image differs from the July 30B study (NGC 26.05) - within-pair
  consistency is the contract; cross-study speed numbers are not comparable.

## Ship criteria (from thai-eval-gate, unchanged)
Ship if: every Thai group n.s. AND pooled >= -1.5 pt AND Thai bpb < +0.03 AND
Thai top-1 agreement > 88%. Otherwise investigate per-subset before any verdict.

## Publication promise
Every number measured under this document gets published with its artifact
path, regardless of direction - including any result that embarrasses the
recipe, the 72B numbers, or this pre-registration.

---

# ARM 2 (pre-registered 2026-08-11, BEFORE quantizing): W4A16 weight-only

Same 7B, scheme NVFP4A16 (FP4 weights group 16, **activations BF16**, data-free
- calibration-independent by construction). BF16 side REUSED from arm 1
(identical serving contract: same image/flags/seed; pairing is per-item so
reuse is valid and stated).

Expectations, fixed in advance:
- Mechanical bound: every accuracy delta must be >= the W4A4 arm's (weight-only
  strictly dominates W4A4 in fidelity). W4A16 WORSE than W4A4 on any axis =
  harness bug, audit before believing.
- Target: ALL FOUR original gates pass (Thai axes n.s. incl. generative;
  pooled >= -1.5; bpb < +0.03; agreement > 88%). If generative Thai still
  significant -> verdict stays REJECT for 7B at 4-bit weights entirely.
- Speed: decode between BF16 (12.4 tok/s) and W4A4 (40.3); predict 2-3x BF16
  via Marlin W4A16. RISK pre-declared: compressed-tensors W4A16-FP4 kernels on
  SM121/aarch64 are UNVERIFIED on this box - a serve failure is a findable
  outcome, not a gate failure.
- Footprint: ~5.5 GB (same 4-bit weights).
- Same stats, same n, same no-superiority-at-tiny-n rules as arm 1.
