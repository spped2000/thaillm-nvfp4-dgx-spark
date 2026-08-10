---
license: other
license_name: qwen
license_link: LICENSE
language:
- th
- en
base_model: openthaigpt/openthaigpt-1.6-72b-instruct
base_model_relation: quantized
tags:
- nvfp4
- fp4
- compressed-tensors
- vllm
- thai
- quantized
---

# OpenThaiGPT 1.6 72B Instruct — NVFP4 (W4A4)

NVFP4 quantization of
[openthaigpt/openthaigpt-1.6-72b-instruct](https://huggingface.co/openthaigpt/openthaigpt-1.6-72b-instruct)
by **AGICAFET LABS**. This checkpoint makes Thailand's largest open-weight
instruct model servable on a single 128 GB-class machine (e.g. NVIDIA DGX
Spark GB10), where the BF16 original (~145 GB) cannot even be loaded.

> โมเดล OpenThaiGPT 1.6 72B ฉบับ NVFP4 โดย AGICAFET LABS — ทำให้โมเดลไทยเปิดที่ใหญ่ที่สุด
> รันได้บนเครื่องเดี่ยวระดับ 128 GB (เช่น DGX Spark) ซึ่งไฟล์ BF16 ต้นฉบับ (~145 GB) โหลดไม่ได้เลย

**Quantized by AGICAFET LABS, not by the OpenThaiGPT team.** All credit for
the model itself belongs to [OpenThaiGPT](https://openthaigpt.aieat.or.th/)
(AIEAT / AIAT and collaborators). Please cite them, not us, for the model.

## Serve with vLLM

```bash
vllm serve AGIcafet/openthaigpt-1.6-72b-instruct-NVFP4 \
  --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Verified working on vLLM **v0.25.1**: on Blackwell SM12x the engine selects
`FlashInferCutlassNvFp4LinearKernel` (native FP4 GEMM). The chat template is
hermes-format for tool calls (`<tool_call>{json}</tool_call>`, no
`<function=`) — do **not** use `qwen3_xml`.

## How it was made

| | |
|---|---|
| Method | [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.12.0.1, `QuantizationModifier(targets="Linear", scheme="NVFP4")` |
| Scheme | NVFP4 **W4A4** (FP4 weights + FP4 activations, group 16, FP8 block scales), compressed-tensors format |
| Kept in BF16 | `lm_head`, embeddings |
| Calibration | 512 samples x 512 tokens, **half Thai**: 256 docs Thai Wikipedia (`wikimedia/wikipedia` 20231101.th) + 256 docs `abisee/cnn_dailymail` — the same set used for [AGIcafet/ThaiLLM-30B-NVFP4](https://huggingface.co/AGIcafet/ThaiLLM-30B-NVFP4), where Thai accuracy showed no significant degradation (paired McNemar, ThaiExam-v2 delta -0.53pt, n.s.) |
| Hardware | 1x NVIDIA DGX Spark (GB10, SM121, aarch64, 121 GB unified memory) |
| Date | 2026-08-01 |
| Size | **42 GB** single-file safetensors (from 145.4 GB BF16 — 3.45x smaller) |

Because the BF16 source is larger than the machine's RAM, the model was
loaded with accelerate **disk offload** (`device_map=auto`,
`max_memory={0: "55GiB", "cpu": "25GiB"}`, `offload_folder` on NVMe) and
calibrated layer-by-layer through llm-compressor's sequential pipeline —
the whole quantization ran on the same single DGX Spark that serves it.

(A first attempt via TensorRT Model Optimizer 0.43.0 `--low_memory_mode`
exported numerically broken NVFP4 — dequant cosine 0.76 vs BF16 — and was
abandoned; the finding is documented in the AGICAFET-LABS/thai-vllm hub.)

## Measured Thai accuracy (this NVFP4 checkpoint, 2026-08-02)

Two independent harnesses on one DGX Spark GB10, single protocol per harness,
all artifacts in [the study repo](https://github.com/spped2000/thaillm-nvfp4-dgx-spark):

| Generative (chinda-eval/EvalScope, chat API) | score | n |
|---|---|---|
| OpenThaiEval | **0.767** | 1,232 |
| IFEval-TH (instruction-level strict) | **0.757** | 215 |
| HellaSwag-TH | 0.590 | 300 |
| Code-switching (Thai language purity) | **0.992** | 500 |
| Thai tool-calling conformance (hermes) | 8/8 | 12 cases |
| MATH500-TH (measured 2026-08-05) | 0.492 | 500 |
| LiveCodeBench-TH all-tests strict (2026-08-05, fixed adapter) | 0.351 | 111 |
| AIME24-TH (2026-08-05) | 0.133 | 30 |

| Log-likelihood (lm-eval, completions API) | score |
|---|---|
| Belebele-TH | **0.879** |
| ThaiExam v2 (letter MC, 565 q) | 0.643 |
| XCOPA-TH | 0.638 |
| XNLI-TH | 0.473 |
| MMLU (5-shot, first 10/subject) | 0.868 |
| Thai Wikipedia bits/byte (200 docs, lower better) | 0.323 |

OpenThaiEval 0.767 and Belebele-TH 0.879 are the highest scores measured on
this machine across 8 Thai-capable models under the same protocols.

## BF16 vs this NVFP4 — direct Thai comparison

The base model's numbers are the developer's own published card values (BF16,
their runner); the NVFP4 column is our measurement of this checkpoint
(chinda-eval/EvalScope, DGX Spark GB10, 2026-08-02). **Different harnesses** —
treat small gaps as harness+precision combined, not as quantization loss alone.

| Benchmark | OpenThaiGPT 1.6 72b (BF16, developer) | **This NVFP4 (ours)** |
|---|---|---|
| OpenThaiEval | 78.7 | **76.7** (n=1,232) |
| Language accuracy / Thai purity* | 98.2 | **99.2*** (n=500) |
| AIME24-TH | 6.67 (2/30) | greedy 13.33 (4/30); **avg@8** (T=0.6) **8.33**, 95% CI [1.5, 15.2] — card value inside the CI |
| MATH500-TH | 43.2 (216/500) | **49.20** (246/500, same 500 items — gap NOT significant, p=0.066) |
| LiveCodeBench-TH | 32.43 (=36/111, all-or-nothing) | **35.14** (39/111, ALL 2,448 tests incl. hidden, all-or-nothing — statistically indistinguishable from the card) |
| IFEval-TH (inst strict) | not published | **75.7** (n=215) |
| Belebele-TH (lm-eval) | not published | **87.9** |
| HellaSwag-TH | not published | **59.0** (n=300) |
| Thai tool calling (hermes) | not published | **8/8** |

\* different tests with the same intent: the developer's "Language Accuracy"
vs our WangchanThaiInstruct code-switching purity — close in spirit, not the
same dataset, so compare loosely.

\* **LCB-TH scoring audit trail (2026-08-05):** both sides use the identical
111-problem `iapp/code_generation_lite-th` set (the card's 32.43 is exactly
36/111). The stock chinda-eval Thai adapter silently drops the 2,337 encoded
hidden tests and awards per-problem partial credit, which inflated our first
reading to 54.47. We fixed the adapter (hidden-test decode, all-or-nothing
scoring, strict output equality — fixes being offered upstream) and re-scored
the SAME frozen greedy generations against all 2,448 tests: **35.14** — the
number in the table, now genuinely on the card's scale. The intermediate
values (54.47 partial-credit/public-only; 45.05 strict/public-only) are kept
in the artifacts as the audit record. This is the expected physics: a
quantized model scores ≈ its BF16 original, never dramatically above it.

## Developer's published benchmarks (BF16 original — full table)

Reproduced verbatim from the
[developer's model card](https://huggingface.co/openthaigpt/openthaigpt-1.6-72b-instruct)
(technical report: [arXiv:2504.01789](https://arxiv.org/abs/2504.01789)).
These are **self-reported numbers for the BF16 original on the developer's own
runner** — not measurements of this NVFP4 checkpoint. The only overlapping
benchmark is OpenThaiEval: their BF16 78.7 vs our NVFP4 76.7 (chinda-eval) —
that gap mixes harness and precision effects, which cannot be separated on a
machine the BF16 model does not fit.

| **Benchmarks**        | **OpenThaiGPT 1.6 72b** | **OpenThaiGPT 1.5 7b** | **OpenThaiGPT 1.5 14b** | **OpenThaiGPT 1.5 72b** | **Typhoon2 Qwen2.5 7b** | **Typhoon2 Llama3.1 8b** | **Typhoon2 Llama3.1 70b** | **NECTEC Pathumma LLM Text 1.0.0 7b** |
|-----------------------|-------------------------|------------------------|-------------------------|-------------------------|-------------------------|--------------------------|---------------------------|------------------------------------|
| **AIME24-TH**         | 6.67                    | 0                      | 0                       | 6.67                    | 3.33                    | 3.33                     | **13.33**                     | 0                                  |
| **AIME24**            | **23.33**                   | 6.67                   | 10                      | **23.33**                   | 6.67                    | 3.33                     | 10                        | 0                                  |
| **MATH500-TH**        | 43.2                    | 24.2                   | 26.2                    | 62                      | 51.8                    | 31                       | **55.8**                      | 21.8                               |
| **MATH500**           | **82**                      | 40.4                   | 47.4                    | 83.2                    | 65.4                    | 49.6                     | 67.4                      | 42.8                               |
| **LiveCodeBench-TH**  | **32.43**                   | 22.52                  | 21.62                   | 12.61                   | 9.91                    | 8.11                     | 27.03                     | 0                                  |
| **LiveCodeBench**     | **54.21**                   | 31.12                  | 37.96                   | 46.38                   | 0.98                    | 5.87                     | 37.38                     | 0                                  |
| **OpenThaiEval**      | **78.7**                    | 64.5                   | 71.26                   | 77.16                   | 64.76                   | 56.63                    | 72.54                     | 65.27                              |
| **Language Accuracy** | 98.2                    | 97.6                   | 98.4                    | 99.4                    | 99.4                    | 98.6                     | **99.8**                      | 98.6                               |
| **AVERAGE**           | **52.34**            | 35.88                  | 39.11                   | 51.34                   | 37.78                   | 32.06                    | 47.91                     | 28.56                              |

## Honest limitations

- **No paired BF16-vs-NVFP4 comparison exists for this model on our
  hardware** — the BF16 original cannot load on the machine we own, which is
  the entire reason this checkpoint exists. The numbers above are absolute
  scores of THIS checkpoint; quantization loss cannot be isolated. The
  developer's own BF16 OpenThaiEval is 78.7 (their runner) vs our 76.7
  (chinda-eval) — harness and precision effects are entangled in that gap.
  (Our 30B experiment with the identical calibration recipe measured a
  -0.8pt overall MC delta, Thai n.s. — evidence about the *recipe*, not
  about *this model*.)
- Activations are quantized (W4A4); on pre-Blackwell GPUs vLLM falls back to
  Marlin W4A16 kernels — a different numeric path we have not measured.
- The original model's **Qwen LICENSE** applies unchanged (see `LICENSE`):
  attribution required; >100M MAU commercial use needs a separate license
  from Alibaba Cloud.

## Links

- Base model: https://huggingface.co/openthaigpt/openthaigpt-1.6-72b-instruct
- OpenThaiGPT project: https://openthaigpt.aieat.or.th/
- AGICAFET LABS Thai LLM hub (serve commands, evidence-backed benchmarks):
  https://github.com/AGICAFET-LABS/thai-vllm
