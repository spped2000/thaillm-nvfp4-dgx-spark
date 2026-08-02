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

## Honest limitations

- **No paired BF16-vs-NVFP4 accuracy comparison exists for this model on our
  hardware** — the BF16 original cannot load on the machine we own, which is
  the entire reason this checkpoint exists. Treat accuracy as unvalidated
  until independent numbers appear. (Our 30B experiment with the identical
  calibration recipe measured a -0.8pt overall MC delta, Thai n.s. — that is
  evidence about the *recipe*, not about *this model*.)
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
