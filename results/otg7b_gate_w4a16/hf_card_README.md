---
license: other
license_name: qwen
license_link: LICENSE
language:
- th
- en
base_model: openthaigpt/openthaigpt1.5-7b-instruct
base_model_relation: quantized
tags:
- nvfp4
- fp4
- w4a16
- compressed-tensors
- vllm
- thai
- quantized
---

# OpenThaiGPT 1.5 7B Instruct — NVFP4 W4A16 (weight-only)

NVFP4 weight-only quantization of
[openthaigpt/openthaigpt1.5-7b-instruct](https://huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct)
by **AGICAFET LABS**. This checkpoint runs Thai chat, instruction-following,
RAG and tool-calling at **3.4x the speed and 1/2.8 the size** of the BF16
original — validated with true paired A/B tests on the same machine, the
first for the OpenThaiGPT family.

> โมเดล OpenThaiGPT 1.5 7B ฉบับ NVFP4 W4A16 โดย AGICAFET LABS — เร็วขึ้น 3.4 เท่า
> เล็กลง 2.8 เท่า วัดเทียบ BF16 แบบ paired จริงบนเครื่องเดียวกันทุกตัวเลข

**Quantized by AGICAFET LABS, not by the OpenThaiGPT team.** All credit for
the model itself belongs to [OpenThaiGPT](https://openthaigpt.aieat.or.th/)
(AIEAT and collaborators). Please cite them, not us, for the model.

## Serve with vLLM

```bash
vllm serve AGIcafet/openthaigpt1.5-7b-instruct-W4A16 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Verified on vLLM **v0.25.1**; on Blackwell SM12x it runs the Marlin
W4A16-FP4 path. The chat template is hermes-format for tool calls — do not
use `qwen3_xml`.

## How it was made

| | |
|---|---|
| Method | [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.12.0.1, `QuantizationModifier(targets="Linear", scheme="NVFP4A16")` |
| Scheme | **W4A16 weight-only** (FP4 weights, group 16, FP8 block scales; activations stay BF16), compressed-tensors format |
| Kept in BF16 | `lm_head`, embeddings, all activations |
| Calibration | none — NVFP4A16 is data-free by construction |
| Hardware | 1x NVIDIA DGX Spark (GB10, SM121, aarch64, 121 GB unified memory) |
| Date | 2026-08-11 |
| Size | **5.5 GB** (from 15.24 GB BF16 — 2.8x smaller) |

A W4A4 variant (quantized activations) was also built and measured — it
failed our release gate on instruction-following and knowledge and is **not
released**; its numbers appear below for transparency.

## Measured accuracy (true paired BF16-vs-quantized, one DGX Spark GB10)

Same machine, same serving flags, audited graders, paired per-item with
exact McNemar, pre-registered before measurement. All artifacts:
[study repo](https://github.com/spped2000/thaillm-nvfp4-dgx-spark)
(`results/otg7b_gate/`, `results/otg7b_gate_w4a16/`).

### Generative (chinda-eval, chat API)

| Benchmark (n) | BF16 | **W4A16 (this)** | W4A4 (not released) |
|---|---|---|---|
| OpenThaiEval (1,232) | 0.6396 | **0.6323** (−0.73, p=0.40) | 0.6161 (−2.35, p=0.02) |
| IFEval-TH prompt-strict (215) | 0.6279 | **0.5907** (−3.72, p=0.20) | 0.5256 (−10.2, p=0.0009) |
| MATH500-TH (500) | 0.5260 | **0.4720** (−5.40, p=0.008) | 0.4700 (−5.60, p=0.008) |
| LiveCodeBench-TH Pass@1, all 2,448 tests (111) | 0.1802 | **0.1802** (0.00, p=1) | 0.1622 (−1.80, p=0.75) |
| AIME24-TH (30) | 1/30 | **1/30** | 1/30 |
| Thai tool-calling conformance (12) | 8/8 | **8/8** | 8/8 |

### Log-likelihood (lm-eval, paired per item)

| Group / task | BF16 | **W4A16** | Δ (p) | W4A4 Δ (p) |
|---|---|---|---|---|
| Thai MC pooled (3,890) | — | — | **−0.28 (0.65)** | −1.29 (0.077) |
| ThaiExam v2 (565) | 0.4796 | **0.4885** | +0.88 (0.61) | −0.35 (0.91) |
| Belebele-TH (900) | 0.7856 | **0.7711** | −1.44 (0.098) | — |
| XNLI-TH (2,490) | 0.4506 | **0.4530** | +0.24 (0.80) | — |
| XCOPA-TH (500) | 0.5940 | **0.5860** | −0.80 (0.57) | — |
| ALL MC pooled (19,786) | — | — | −0.54 (0.003) | −1.53 (3e-12) |
| MMLU 5-shot @10/subj (570) | 0.7737 | **0.7491** | −2.5 | 0.7298 |
| HellaSwag (10,042) | 0.6004 | **0.5941** | −0.63 (1e-4) | −1.22 |
| Thai Wikipedia bits/byte @200 (lower better) | 0.3949 | **0.4055** (+0.011) | | 0.4125 (+0.018) |
| Thai teacher-forced top-1 agreement (32,424 pos.) | — | **90.3%** | | 87.9% |

### Speed & footprint (single-stream 128→1024, same box)

| | BF16 | **W4A16** | W4A4 |
|---|---|---|---|
| decode | 12.4 tok/s | **41.8 tok/s (3.4x)** | 40.3 tok/s |
| TTFT p50 | 84 ms | **50 ms** | 37 ms |
| disk | 15.24 GB | **5.5 GB** | 5.5 GB |

## Developer's published benchmarks (BF16 original — for comparison)

Reproduced from the
[developer's model card](https://huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct)
(their own runner, BF16). Cross-harness gaps are MEASURED for this model —
see the anchor-calibration section below before comparing columns.

| Benchmark | OpenThaiGPT 1.5 7b (developer) | our harness, BF16 | our harness, this W4A16 |
|---|---|---|---|
| OpenThaiEval | 64.5 | 63.96 | 63.23 |
| MATH500-TH | 24.2 | 52.60 | 47.20 |
| LiveCodeBench-TH | 22.52 | 18.02 | 18.02 |
| AIME24-TH | 0 | 3.33 (1/30) | 3.33 (1/30) |

## Anchor calibration — how big are cross-harness gaps, really?

Measuring the same BF16 model on both harnesses (a first for this family):
**OpenThaiEval +0.5 pt (tiny — card anchors on this axis are sound)** ·
LiveCodeBench-TH ~−4.5 pt (moderate — directional reads only) ·
**MATH500-TH +28 pt (huge — never compare MATH across these harnesses)**.
This calibration also applies to our
[72B NVFP4 card](https://huggingface.co/AGIcafet/openthaigpt-1.6-72b-instruct-NVFP4),
whose MATH-vs-card comparison is withdrawn accordingly.

## Honest limitations

- **Mathematical reasoning degrades and the gap is significant**: MATH500-TH
  −5.4 pt (p=0.008), unchanged from the W4A4 arm — the damage lives in the
  FP4 weights themselves, compounding over long reasoning chains. If your
  workload is math-heavy, use the BF16 original.
- Our all-axes release gate scored this artifact REJECT on that single axis;
  it is published as a **use-case-scoped release** (chat / instruction /
  RAG / tools) with the owner's sign-off — `RELEASE_SCOPE.md` in the study
  repo records the decision chain and the pre-registration.
- IFEval-TH's point estimate (−3.7 pt) is not statistically resolvable at
  n=215; we report it rather than hide it.
- ALL-MC pooled shows a small but real −0.54 pt (p=0.003, n=19,786) driven
  by English tasks; every Thai MC group is non-significant.

## Links

- Base model: https://huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct
- OpenThaiGPT project: https://openthaigpt.aieat.or.th/
- Study repo (pre-registrations, gate verdicts, all artifacts):
  https://github.com/spped2000/thaillm-nvfp4-dgx-spark
- AGICAFET LABS Thai LLM hub: https://github.com/AGICAFET-LABS/thai-vllm
