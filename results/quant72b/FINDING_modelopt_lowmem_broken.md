# Finding: modelopt 0.43.0 `--low_memory_mode` exports numerically broken NVFP4

Date: 2026-08-01 · Box: DGX Spark GB10 (SM121, aarch64) · Container: nvcr.io/nvidia/vllm:26.05.post1-py3

## What was attempted
Rehearsal of the >RAM quantization path on Qwen/Qwen2.5-0.5B-Instruct (local dir)
with the exact 72B flags: `hf_ptq.py --qformat nvfp4 --kv_cache_qformat none
--calib_size 512 --calib_seq 512 --batch_size 1 --dataset thai_en_calib.jsonl
--attn_implementation sdpa --skip_generate --low_memory_mode`.

## Bugs hit in order (first two patched, see scripts/patch_modelopt_lowmem.py)
1. `patched_from_pretrained` forwards `attn_implementation` into accelerate's
   `load_checkpoint_and_dispatch()` → TypeError. Patched: route into `from_config`.
2. The path accepts only local checkpoint paths, not HF repo ids (ValueError from
   `load_checkpoint_in_model`). Not a bug for us — the 72B is a local dir anyway.
3. `patched_from_pretrained` never calls `model.tie_weights()` → tied
   `lm_head.weight` stays on meta → `dispatch_model`'s `.to()` raises
   "Cannot copy out of meta tensor". Patched: call `tie_weights()` before quantize.

## The fatal one (unpatchable post-hoc)
With all of the above fixed, hf_ptq completes and exports — but the export is wrong:
- `weight_scale` tensors are HALF the NVFP4 spec size: k_proj `[128, 28]` where
  block-16 over 896 in-features requires `[128, 56]`. vLLM 0.25.1 aborts with
  `start (0) + length (304) exceeds dimension size (152)`.
- 168 modelopt runtime-state tensors (`*.weight_quantizer._double_scale`) leak
  into the state dict (vLLM: "no module or parameter named ... weight_quantizer").
  Strippable (scripts/strip_quantizer_state.py) — but pointless, because:
- Dequantizing the exported FP4 against the BF16 source gives cosine similarity
  **0.756** (lo,hi nibble order; 0.015 hi,lo; 0.758 block-16-truncated). A healthy
  NVFP4 dequant is >0.99. The damage happens at quantization time — consistent
  with the quantizer calibrating on the already-compressed packed-uint8 view
  (448 bytes → 28 blocks of 16 BYTES = 32 logical values per scale).

## Verdict
`--low_memory_mode` (self-described "experimental") cannot produce a servable
NVFP4 checkpoint in modelopt 0.43.0. The 72B run pivots to llm-compressor
(compressed-tensors NVFP4 W4A4) with accelerate disk offload — both ends of that
path already proven separately on this box (35B quantized in 42 min; RedHatAI
compressed-tensors NVFP4 served under vLLM 0.25.1 in the a2 campaign).

Logs: quant_thai-quant-rehearsal.log (attempts 1–4), this directory.
