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

> ⚠️ **Scope: chat / instruction-following / RAG / tool-calling only.**
> On those tasks this checkpoint is statistically indistinguishable from BF16
> (true paired tests, below) at **3.4x the speed and 1/2.8 the size**.
> **Do NOT use it for mathematical reasoning**: MATH500-TH degrades
> **−5.4 pt (p=0.008, significant)** — FP4 weight error compounds over long
> reasoning chains. This limit is measured, not hypothetical.
>
> 🇹🇭 เหมาะกับงานแชท/ทำตามคำสั่ง/RAG/tool calling — วัดแบบ paired แล้วแยกจาก BF16
> ไม่ออกทางสถิติ ที่ความเร็ว 3.4 เท่า · **ห้ามใช้งานคณิตศาสตร์** (เสื่อมจริง −5.4pt)

Quantized by **AGICAFET LABS** from
[openthaigpt/openthaigpt1.5-7b-instruct](https://huggingface.co/openthaigpt/openthaigpt1.5-7b-instruct)
(credit for the model: [OpenThaiGPT](https://openthaigpt.aieat.or.th/) / AIEAT).
llm-compressor 0.12.0.1, scheme **NVFP4A16** (FP4 weights group 16, activations
BF16, data-free), `lm_head`+embeddings kept BF16. 15.24 GB → **5.5 GB**.

## Serve with vLLM

```bash
vllm serve AGIcafet/openthaigpt1.5-7b-instruct-W4A16 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Verified on vLLM v0.25.1; on Blackwell SM12x it runs the Marlin W4A16-FP4 path.

## Every benchmark we measured (true paired A/B, one DGX Spark GB10)

All three columns measured on the same machine, same serving flags, same
fixed-and-audited graders, paired per-item with exact McNemar. Pre-registered
before measurement; artifacts in
[the study repo](https://github.com/spped2000/thaillm-nvfp4-dgx-spark)
(`results/otg7b_gate/`, `results/otg7b_gate_w4a16/`). The W4A4 column is a
**rejected** artifact shown for transparency — it is not released.

### Generative (chinda-eval, chat API)

| Benchmark (n) | BF16 | **W4A16 (this)** | W4A4 (rejected) |
|---|---|---|---|
| OpenThaiEval (1,232) | 0.6396 | **0.6323** (−0.73, p=0.40 ✓) | 0.6161 (−2.35, p=0.02 ✗) |
| IFEval-TH prompt-strict (215) | 0.6279 | **0.5907** (−3.72, p=0.20 ✓\*) | 0.5256 (−10.2, p=0.0009 ✗) |
| MATH500-TH (500) | 0.5260 | **0.4720** (−5.40, p=0.008 ✗) | 0.4700 (−5.60, p=0.008 ✗) |
| LiveCodeBench-TH Pass@1, all 2,448 tests (111) | 0.1802 | **0.1802** (0.00, p=1 ✓) | 0.1622 (−1.80, p=0.75 ✓) |
| AIME24-TH (30) | 1/30 | **1/30** | 1/30 |
| Thai tool-calling conformance (12) | 8/8 | **8/8** | 8/8 |

\* point estimate −3.7 pt is worth knowing even though n=215 cannot resolve it;
we report it rather than hide it.

### Log-likelihood (lm-eval, paired per item)

| Group / task | BF16 | **W4A16** | Δ (p) | W4A4 Δ (p) |
|---|---|---|---|---|
| Thai MC pooled (3,890) | — | — | **−0.28 (0.65 ✓)** | −1.29 (0.077) |
| ThaiExam v2 (565) | 0.4796 | **0.4885** | +0.88 (0.61 ✓) | −0.35 (0.91) |
| Belebele-TH (900) | 0.7856 | **0.7711** | −1.44 (0.098 ✓) | — |
| XNLI-TH (2,490) | 0.4506 | **0.4530** | +0.24 (0.80 ✓) | — |
| XCOPA-TH (500) | 0.5940 | **0.5860** | −0.80 (0.57 ✓) | — |
| ALL MC pooled (19,786) | — | — | **−0.54 (0.003)** | −1.53 (3e-12) |
| MMLU 5-shot @10/subj (570) | 0.7737 | **0.7491** | −2.5 | 0.7298 |
| HellaSwag (10,042) | 0.6004 | **0.5941** | −0.63 (1e-4) | −1.22 |
| Thai Wikipedia bits/byte @200 (lower better) | 0.3949 | **0.4055** (+0.011 ✓) | | 0.4125 (+0.018) |
| Thai teacher-forced top-1 agreement (32,424 pos.) | — | **90.3% ✓** | | 87.9% ✗ |

### Speed & footprint (single-stream 128→1024, same box)

| | BF16 | **W4A16** | W4A4 |
|---|---|---|---|
| decode | 12.4 tok/s | **41.8 tok/s (3.4x)** | 40.3 tok/s |
| TTFT p50 | 84 ms | **50 ms** | 37 ms |
| disk | 15.24 GB | **5.5 GB** | 5.5 GB |

## Did our earlier big-model (72B) comparisons measure the right thing?

This 7B pair lets us CALIBRATE cross-harness anchoring for the first time —
our BF16 measurement vs the developer's own published number for the SAME model:

| bench | developer card (BF16) | our harness (BF16) | harness gap | verdict on 72B anchor use |
|---|---|---|---|---|
| OpenThaiEval | 64.5 | 63.96 | **+0.5 pt — tiny** | ✅ the 72B NVFP4 76.7-vs-78.7 reading (≈ −2 real quant cost) stands |
| LiveCodeBench-TH | 22.52 | 18.02 | ~−4.5 pt — moderate | ⚠️ 72B "not worse than card" reading OK; no superiority claims |
| MATH500-TH | 24.2 | 52.6 | **+28 pt — huge** | ❌ 72B MATH-vs-card comparison is WITHDRAWN as quantization evidence — the harness gap dominates everything |

And the model-size law, now on three same-recipe points (paired where possible):
W4A4 cost on OpenThaiEval-class tasks: **30B −0.53 n.s. · 72B ≈ −2 (anchored) ·
7B −2.35 significant** — parameter count buys redundancy that absorbs 4-bit error.

## Provenance & transparency

- Both arms pre-registered before measurement (`PREREGISTERED.md` in the study
  repo); the all-axes gate verdict for this artifact was REJECT — this is a
  **use-case-scoped release** authorized by the owner, with the failing axis
  stated in the warning block above (`RELEASE_SCOPE.md`).
- Graders audited (LCB hidden-test and MATH extraction defects found and fixed
  before any of these numbers were produced).
- Qwen LICENSE applies (see `LICENSE`); >100M MAU commercial use requires a
  separate license from Alibaba Cloud.
