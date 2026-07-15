# Deployment Alternatives for ThaiLLM-30B (qwen3_moe) on DGX Spark GB10 — Research Findings

> **CORRECTION (post-audit 2026-07-15):** an earlier verification note in this file claimed, citing NVIDIA's 26.05 release notes, that the container is vLLM 0.20.1/FlashInfer 0.6.10 and that the main report should be amended. That is wrong for the `.post1` respin: `pip freeze` inside `nvcr.io/nvidia/vllm:26.05.post1-py3` shows **vLLM 0.21.0+2325b6f0.nvinternal.26.5.post1 / FlashInfer 0.6.11.post3** — the main report is correct; NVIDIA's release notes lag the .post1 image.

Verification note up front: the NGC `26.05` vLLM container ships **vLLM 0.20.1 on CUDA 13.2.1** per [NVIDIA's 26.05 release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-05.html) (flashinfer 0.6.10, transformers 5.6.0) — not 0.21 as commonly assumed; the `.post1` respin does not change the vLLM base. Adjust version claims in the report accordingly.

---

## 1. FP8 via ModelOpt

### Recipe (ModelOpt 0.43.0, same hf_ptq workflow already proven for NVFP4)

```bash
python examples/llm_ptq/hf_ptq.py \
  --pyt_ckpt_path /models/ThaiLLM-30B \
  --qformat fp8 \
  --kv_cache_qformat none \          # keep KV BF16 to match the NVFP4 checkpoint; default changed to fp8_cast (constant amax, no calib)
  --calib_size 512 --batch_size 4 \
  --export_path /models/ThaiLLM-30B-FP8 --export_fmt hf
```

- **Variants** ([ModelOpt changelog](https://nvidia.github.io/Model-Optimizer/reference/0_changelog.html), [vLLM modelopt docs](https://docs.vllm.ai/en/latest/features/quantization/modelopt/), [vllm#30938](https://github.com/vllm-project/vllm/issues/30938)): `fp8` = per-tensor static weight+activation scales (E4M3); `fp8_pc_pt` = per-channel weight + per-token dynamic activation; `fp8_pb_wo` = DeepSeek-style 128×128 block-scaled weight-only (exported as 4D `weight_scale [out_blk,1,in_blk,1]`). vLLM support for `fp8_pc_pt`/`fp8_pb_wo` landed only recently (tracked in #30938) — for NGC 26.05/vLLM 0.20.1, **plain per-tensor `fp8` is the safe recipe**. ModelOpt's MoE recipes exclude router gates and `lm_head` by default, matching your NVFP4 checkpoint. Qwen3 MoE PTQ is an officially supported ModelOpt path ([0.43.0 release](https://newreleases.io/project/github/NVIDIA/Model-Optimizer/release/0.43.0)).
- **Checkpoint size**: ~30.5B params × 1 B + BF16 embed/lm_head (~1.2 GB) + gates/norms ≈ **~31 GB** (half of BF16 61 GB); leaves ~85 GB for KV on the 121 GB pool.

### SM121 kernel reality — the important nuance

- **Plain FP8 (per-tensor / block-128) does NOT fall back to Marlin.** GB10 has native FP8 tensor cores. Dense linears run FlashInfer/CUTLASS FP8 natively; MoE experts run the **Triton fused_moe FP8 path** ("FLASHINFER options unavailable for SM121", MoE backend = TRITON) — confirmed by a real Qwen3.6-35B-A3B-FP8 (block-128) deployment on a GB10 in NGC 26.03 ([rikkarth benchmark](https://rikkarth.com/blog/2026-04-23-benchmark-results-for-qwen-qwen3-6-35b-a3b-fp8-nvidia-dgx-spark-gb10-serving-via-vllm)). Newer main-based builds also enable **Marlin FP8 MoE** on SM121 with good results ([495 tok/s at c32 on 2 Sparks, NVIDIA forums](https://forums.developer.nvidia.com/t/qwen3-5-35b-a3b-fp8-on-2x-dgx-spark-main-based-build-marlin-fp8-on-sm121-495-tok-s-at-c32/364842)). The [vlaicu DGX Spark playbook](https://vlaicu.io/posts/dgx-vllm/) concurs: "FP8 kernels function cleanly on sm121"; no `--moe-backend marlin` needed (unlike NVFP4).
- **The Marlin W8A16 dequant-to-BF16 fallback is specific to MXFP8** (block-32 MX format): `TrtLlmFp8ExpertsBase` gates on `is_device_capability_family(100)` (datacenter Blackwell only), so SM_121 MoE experts drop to Marlin W8A16 while dense layers use native `tcgen05.mma` — [vllm#43906](https://github.com/vllm-project/vllm/issues/43906), filed May 2026, unfixed. **Conclusion: use `--qformat fp8`, avoid `mxfp8`.**
- Serving flags (from [conselara SM121 gotchas](https://conselara.dev/notes/vllm-dgx-spark-sm121-gotchas/) + rikkarth): `--attention-backend TRITON_ATTN` (FlashInfer has confirmed FP8 accuracy bugs on Blackwell), `--gpu-memory-utilization 0.7–0.87`, never `--enforce-eager` (~55% throughput loss), stay on 580.x driver.

### Expected decode vs NVFP4 (273 GB/s budget)

- Bandwidth math: ~3.3 GB active weights/step in FP8 → ~80 tok/s ceiling; measured single-stream for the 35B-A3B FP8 sibling: **~21 tok/s at c1 (TPOT 48 ms), ~28–30 tok/s ceiling, 156 tok/s aggregate at c32** (rikkarth, 26.03). Your 30B should land slightly higher.
- NVFP4 halves active-weight bytes (~1.9 GB/step) → **roughly 1.5–1.8× FP8 decode at bs1** in principle; but note the [FP4 PSA thread](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069) measured NVFP4 *slower* than AWQ-4bit (Dec 2025) due to immature SM121 kernels; the gap has been closing through 2026. Frame FP8 as: **~0.55–0.65× NVFP4 decode speed, +1 bit fidelity, zero kernel drama.**

---

## 2. GGUF / llama.cpp

### Conversion + Thai imatrix + quantize

`Qwen3MoeForCausalLM` is cleanly supported by `convert_hf_to_gguf.py` (arch `qwen3moe`, in-tree since Qwen3 launch; [convert script](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py), [Qwen llama.cpp docs](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)):

```bash
python convert_hf_to_gguf.py /models/ThaiLLM-30B --outtype bf16 --outfile ThaiLLM-30B-BF16.gguf

# imatrix with Thai-heavy calibration (mix ~70/30 Thai/EN raw text, 1–5 MB; MoE needs enough
# chunks that all 128 experts receive activations — llama-quantize warns if some are missing)
./build/bin/llama-imatrix -m ThaiLLM-30B-BF16.gguf -f thai_en_calib.txt \
  -o ThaiLLM-30B.imatrix -ngl 99 -c 2048 --chunks 200

./build/bin/llama-quantize --imatrix ThaiLLM-30B.imatrix \
  ThaiLLM-30B-BF16.gguf ThaiLLM-30B-Q4_K_M.gguf Q4_K_M
```

- **Quant levels**: Q4_K_M ≈ 18.6 GB, Q5_K_M ≈ 21.7 GB, Q8_0 ≈ 32.5 GB. Unsloth-style **UD-Q4_K_XL** (Dynamic 2.0: per-layer bumps to Q5_K/Q6_K on sensitive tensors, imatrix from 1.5M+ curated tokens) is the quality Pareto winner in their KL-divergence tests ([Unsloth Dynamic 2.0](https://unsloth.ai/blog/dynamic-v2), [UD vs Q4_K_M discussion](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/6)) — for a Thai base model, reproduce the idea with your own Thai calibration set rather than reusing their EN chat corpus.
- **GB10 build**: `cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES="121a-real"` ([Arm learning path](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/2_gb10_llamacpp_gpu/), [NVIDIA dgx-spark-playbooks/llama-cpp](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/llama-cpp), [build.nvidia.com playbook](https://build.nvidia.com/spark/llama-cpp/instructions)). Docker: [docs/docker.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md) lists `server-cuda13` multi-arch, but community reports of missing/mismatched arm64 CUDA images persist ([forum thread](https://forums.developer.nvidia.com/t/building-llama-cpp-container-images-for-spark-gb10/353664), [cslev/llamacpp-cuda-arm64-docker](https://github.com/cslev/llamacpp-cuda-arm64-docker)) — safest is a self-built image on an `nvcr.io/nvidia/cuda:13.x` aarch64 base. Serve: `llama-server -m ThaiLLM-30B-Q4_K_M.gguf -ngl 99 -fa on -c 32768 --no-mmap`.
- **Measured speeds on Spark** (30B-A3B class): Q4_K_M **~89 tok/s decode @ short ctx, ~84 @ 2k, prefill ~2,500 t/s** ([DandinPower bench](https://github.com/DandinPower/llama.cpp_bench/blob/main/dgx_spark/report.md)); Q8_0 ~50 tok/s tg, ~1,800 t/s pp2048 ([official ggml-org DGX Spark thread #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)). **This is the fastest single-user decode path on Spark — roughly 2–3× vLLM FP8 at bs1** — the trade-off being weaker concurrent-request scaling than vLLM.

---

## 3. AWQ W4A16 via llm-compressor

- **Support status**: Qwen3 MoE AWQ landed in **llm-compressor 0.8.0** ([Red Hat announcement](https://developers.redhat.com/articles/2025/10/07/llm-compressor-080-extended-support-qwen3-and-more), [MoE example docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantizing_moe/)); working reference checkpoint: [nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq](https://huggingface.co/nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq). VL-MoE variants had open issues ([#1939](https://github.com/vllm-project/llm-compressor/issues/1939), [#2066](https://github.com/vllm-project/llm-compressor/issues/2066)) but text-only qwen3_moe is clean.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from datasets import load_dataset

ds = load_dataset("json", data_files="thai_calib.jsonl", split="train")  # raw Thai/EN text
recipe = AWQModifier(
    targets=["Linear"], scheme="W4A16",
    ignore=["lm_head", "re:.*mlp.gate$"],   # router gates stay BF16; regex anchors so gate_proj is NOT caught
)
oneshot(model="/models/ThaiLLM-30B", dataset=ds, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512,
        output_dir="/models/ThaiLLM-30B-W4A16-AWQ")
```

(Qwen3-30B-A3B has no shared expert, so no `shared_expert_gate` entry needed.) Checkpoint ≈ 17–18 GB.
- **vLLM on SM121**: served via **`awq_marlin`** — Marlin W4A16 is confirmed working on GB10 (conselara: GPTQ-Marlin functional; Machete requires SM90+ so it's Marlin on Spark). Empirically AWQ-int4 was **~32% faster than NVFP4 at c1 / 18% at high concurrency** on Spark in Dec 2025 ([PSA thread](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)) — i.e., AWQ is a serious rival to your NVFP4 checkpoint on this hardware, with W4A16 (BF16 activations) instead of W4A4, typically better Thai fidelity, at the cost of no FP4-native upside if SM121 kernels mature.

---

## 4. Speculative decoding for a BASE model (no MTP head)

vLLM 0.20.1 `--speculative-config` methods ([vLLM spec-decode docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)): `ngram`, `draft_model`, `eagle3`, `suffix`, `mtp` (unavailable — no MTP head in Qwen3-30B-A3B).

| Option | Command sketch | Realistic on Spark | Notes |
|---|---|---|---|
| **ngram** | `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'` | 1.1–1.3× on RAG/extractive/repetitive Thai; ~1.0× open-ended | Zero extra memory/bandwidth; **composes trivially with modelopt NVFP4** (proposer is tokenizer-level). Deploy-now option. |
| **Draft model Qwen3-0.6B** | `{"method":"draft_model","model":"Qwen3-0.6B","num_speculative_tokens":3}` — same tokenizer/vocab (151936) so no heterogeneous-vocab TLI needed | Likely ≤1.2×, **possibly a net loss** | A3B active weights are already tiny (~1.9–3.3 GB/step), so a 1.2 GB BF16 draft eats a large fraction of the bandwidth budget. The only public A3B+draft study (llama.cpp, RTX 3090, 19 configs incl. vocab-matched 0.8B draft) found **no variant achieved net speedup on A3B MoE** ([thc1006 benchmark repo](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)). Also: stock Qwen3-0.6B's distribution drifts from your Thai-continued base → low acceptance unless you also continue-pretrain the draft on Thai. |
| **EAGLE-3 head** | `--speculative-config '{"method":"eagle3","model":"<head>","num_speculative_tokens":3}'` (serving pattern per [RedHatAI Qwen3-30B-A3B speculator](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3)) | **1.5–2.5×** decode plausible (best option) | Off-the-shelf heads (RedHatAI, [lmsys SpecForge](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex)) target the **Instruct-2507** model and were trained on EN chat data (acceptance 2.1–3.1 at k=3) — hidden-state mismatch with a Thai-continued *base* model will crater acceptance. You must **train your own head with [SpecForge](https://www.lmsys.org/blog/2025-07-25-spec-forge/)** on Thai data (target served via SGLang, FSDP head training). Reported training costs range wildly: ~1.5 h on 1×H100 (small model, small data) → ~48 H200-hours (8–9B) → ~700 GPU-h (Qwen3-8B, large corpus); budget **low hundreds of GPU-hours** for 30B-A3B + Thai corpus. |

- **Composition with modelopt NVFP4 in NGC 26.05**: the docs impose no quantization restriction on spec decode, and the Spark community runs spec decode over NVFP4 checkpoints routinely (Qwen3.6-35B-A3B-**NVFP4** + MTP-3 hits **102–125 tok/s** single-stream on one Spark, [vlaicu playbook](https://vlaicu.io/posts/dgx-vllm/)) — that number is the existence proof that speculation is where the big Spark decode wins live, since verify passes amortize the 273 GB/s weight reads over 2–4 tokens. EAGLE-3 head loads in BF16 alongside modelopt weights. Flag as "verify on exact container tag" — this composition on 26.05.post1 specifically is community-evidenced, not NVIDIA-documented.

---

## 5. SGLang on GB10 with modelopt NVFP4

- **State (July 2026): community-fork territory, not the supported path.** Official playbook exists ([build.nvidia.com/spark/sglang](https://build.nvidia.com/spark/sglang)) with image **`lmsysorg/sglang:spark`** and documents `--quantization modelopt_fp4` for NVFP4 checkpoints; GB10 tracking issue is [sgl-project/sglang#11658](https://github.com/sgl-project/sglang/issues/11658). Stable non-spark tags fail to launch on sm_121; only CUDA-13 builds (`:spark`, `v0.5.10.post1-cu130`+, nightly `-cu13-` tags) work ([pre-built images forum thread](https://forums.developer.nvidia.com/t/new-pre-built-sglang-docker-images-for-nvidia-dgx-spark/360656), community images: [scitrera/dgx-spark-sglang](https://hub.docker.com/r/scitrera/dgx-spark-sglang)).
- **Known GB10 issues**: (a) **CUTLASS FP4 GEMM is broken on SM121** (silent garbage) → MoE must route through **Marlin W4A16**, same as vLLM; (b) **shared-memory ceiling**: GB10 allows 101,376 B/block but default SGLang MoE Triton configs request 147,456 B → `OutOfResources` at launch; needs tuned MoE kernel configs ([BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs)); (c) unified-memory tuning via `--mem-fraction-static`; run with `--gpus all --ipc=host --network host`.
- **Working reference**: the [scottgl9 GB10 fork](https://huggingface.co/scottgl/Qwen3.5-122B-A10B-NVFP4-GB10) (sglang-spark-gb10-optimizations) serves NVFP4 MoE on one Spark at **~46 tok/s decode (with NEXTN speculation; ~28 tok/s baseline)** for a 122B-A10B model, `SGLANG_QUANTIZE_LM_HEAD_FP8=0`, attention kept BF16. For your 30B-A3B **base** model there is no NEXTN/MTP head, so you'd get the unspeculated baseline class — plausibly ~40–60 tok/s, **unverified for this exact config**. Recommendation for the report: list SGLang as viable-but-frontier; vLLM NGC remains the vendor-supported NVFP4 route.

---

## One-table summary for the alternatives section

| Path | Checkpoint | bs=1 decode (measured, 30B-A3B class) | SM121 kernel truth | Effort |
|---|---|---|---|---|
| NVFP4 (yours) | 18.1 GB | ~30–45 t/s vLLM (Marlin MoE), 100+ with spec decode | MoE forced to Marlin W4A16; CUTLASS FP4 broken | done |
| FP8 modelopt | ~31 GB | ~21–30 t/s (c1), 156 t/s @c32 | Native FP8: Triton/Marlin-FP8 MoE fine; only **MXFP8** hits W8A16 dequant fallback (#43906) | low |
| GGUF Q4_K_M + Thai imatrix | ~18.6 GB | **~84–89 t/s** | CUDA 121a-real build; best bs=1 | low |
| AWQ W4A16 + Thai calib | ~17–18 GB | ≈ NVFP4 +18–32% (Dec-2025 data) | awq_marlin works on GB10 | low-med |
| EAGLE-3 spec (needs training) | +~1 GB head | 1.5–2.5× multiplier on any of the above | BF16 head, verify amortizes bandwidth | high (~10² GPU-h) |
| SGLang NVFP4 | 18.1 GB | ~28 t/s baseline class (fork-dependent) | fork + tuned MoE configs required | med-high |

Sources: [vllm#43906](https://github.com/vllm-project/vllm/issues/43906) · [rikkarth FP8 Spark bench](https://rikkarth.com/blog/2026-04-23-benchmark-results-for-qwen-qwen3-6-35b-a3b-fp8-nvidia-dgx-spark-gb10-serving-via-vllm) · [Marlin FP8 SM121 forum](https://forums.developer.nvidia.com/t/qwen3-5-35b-a3b-fp8-on-2x-dgx-spark-main-based-build-marlin-fp8-on-sm121-495-tok-s-at-c32/364842) · [conselara SM121 gotchas](https://conselara.dev/notes/vllm-dgx-spark-sm121-gotchas/) · [vlaicu Spark vLLM playbook](https://vlaicu.io/posts/dgx-vllm/) · [NGC 26.05 release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-05.html) · [vLLM modelopt docs](https://docs.vllm.ai/en/latest/features/quantization/modelopt/) · [vllm#30938](https://github.com/vllm-project/vllm/issues/30938) · [ggml-org Spark bench #16578](https://github.com/ggml-org/llama.cpp/discussions/16578) · [DandinPower Spark bench](https://github.com/DandinPower/llama.cpp_bench/blob/main/dgx_spark/report.md) · [Arm GB10 llama.cpp build](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/2_gb10_llamacpp_gpu/) · [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/llama-cpp) · [Unsloth Dynamic 2.0](https://unsloth.ai/blog/dynamic-v2) · [llm-compressor 0.8.0](https://developers.redhat.com/articles/2025/10/07/llm-compressor-080-extended-support-qwen3-and-more) · [llm-compressor MoE docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantizing_moe/) · [nm-testing AWQ checkpoint](https://huggingface.co/nm-testing/Qwen3-Coder-30B-A3B-Instruct-W4A16-awq) · [NVFP4 PSA thread](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069) · [vLLM spec-decode docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) · [RedHatAI EAGLE3 speculator](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3) · [SpecForge](https://www.lmsys.org/blog/2025-07-25-spec-forge/) · [A3B draft-decode negative result](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) · [build.nvidia.com/spark/sglang](https://build.nvidia.com/spark/sglang) · [sglang#11658](https://github.com/sgl-project/sglang/issues/11658) · [scottgl GB10 NVFP4](https://huggingface.co/scottgl/Qwen3.5-122B-A10B-NVFP4-GB10) · [GB10 MoE configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) · [sglang Spark images thread](https://forums.developer.nvidia.com/t/new-pre-built-sglang-docker-images-for-nvidia-dgx-spark/360656)