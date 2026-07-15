# Deploying ThaiLLM-30B on NVIDIA DGX Spark: Quantization & Inference-Stack Technical Report

*Compiled July 14, 2026. The DGX Spark / GB10 (sm_121) software ecosystem is moving very fast; treat every version-specific claim as a snapshot and re-verify against current NVIDIA docs, vLLM/SGLang/TensorRT-LLM release notes, and the model card before committing to production.*

---

## TL;DR

- **For a single DGX Spark, the winning combination for ThaiLLM-30B is a 4-bit MoE checkpoint served on a Blackwell-aware engine.** For maximum reliability and fastest time-to-working, use **GGUF Q4_K_XL (Unsloth dynamic) on llama.cpp/Ollama** via NVIDIA's official playbook. For maximum speed and multi-user concurrency, use **NVFP4 + a community SM121 vLLM build with FP8 KV cache, FlashInfer attention, and MTP speculative decoding**. Keep a **FP8 checkpoint** as the conservative, near-lossless fallback.
- **The Spark is bandwidth-bound, not compute-bound.** Its 273 GB/s LPDDR5x is the ceiling on decode speed; the MoE architecture (3B active of 30B total) is precisely what makes this model usable — expect roughly **50–100+ tok/s single-stream at 4-bit and ~28–52 tok/s at FP8**, versus only ~2.7 tok/s for a dense 70B on the same box.
- **ThaiLLM-30B is a *base* model** (continued pretraining of Qwen3-30B-A3B, no instruction tuning) — you must fine-tune it before serving. And because **Thai is a non-Latin, comparatively lower-resource language, quantization can degrade it more than English in ways automatic metrics hide** — use FP8 or 4-bit *with Thai calibration data*, and benchmark on Thai evals before shipping aggressive quantization.

---

## Key Findings

1. **Model identity.** ThaiLLM-30B (`ThaiLLM/ThaiLLM-30B`, Apache-2.0) is a `qwen3_moe` architecture, 31B params, BF16, **continued-pretrained from Qwen3-30B-A3B** on ~63B tokens: Fineweb2-ENG (24.0B), Fineweb2-TH (31.5B), and CuratedData (8.05B: research, news, law, business, medical, etc.). It improves on Qwen3-30B-Base for Thai (ThaiExam overall 0.6478 vs 0.5947; MMLU-Thai 0.7284 vs 0.7004; average 0.7511 vs 0.7378) at a slight cost on English MMLU (0.95 vs 0.96). The card states explicitly: **"This is a base model that requires instruction fine-tuning to align with specific user requirements"** — you need an SFT/DPO pass (the card recommends LLaMA-Factory) before it behaves as an assistant. For comparison, mature Thai instruction models built on similar bases include Typhoon (SCB10X), THaLLE (KBTG), OpenThaiGPT (AIEAT), and Pathumma (NECTEC).

2. **Architecture facts that drive every deployment decision** (from the Qwen3 Technical Report, arXiv:2505.09388, and the base model): 30.5B total / ~3.3B activated params, 128 experts with 8 activated per token, no shared experts, fine-grained expert segmentation, global-batch load-balancing loss, **48 layers**, GQA with **32 query heads / 4 KV heads**, head_dim 128, QK-Norm, 32K native context extensible to 128K–256K via YaRN + RoPE ABF, thinking/non-thinking modes via `/think` and `/no_think`. Qwen3 uses **GQA, not DeepSeek's MLA** — this is a critical clarification for technique transfer.

3. **Hardware reality.** GB10 pairs a 20-core ARM CPU (10× Cortex-X925 + 10× Cortex-A725) with a Blackwell GPU (6,144 CUDA cores, 192 fifth-gen Tensor Cores, 48 fourth-gen RT cores) over NVLink-C2C, sharing **128 GB unified LPDDR5x at 273 GB/s**. NVFP4 is native. Its compute capability is **sm_121** — *not* the sm_120 of consumer RTX 50-series nor the sm_100 of datacenter Blackwell — which is the root of most software friction.

4. **Software maturity ranking on GB10 (as of mid-2026):** llama.cpp/Ollama (most reliable, official playbooks) > vLLM (works via community SM121 builds, best throughput) ≈ SGLang (official playbook, good) > TensorRT-LLM (fragile on GB10, best via prebuilt NIM-style containers only). NIM microservices are available for some models; NeMo is training-focused; NVIDIA Dynamo is datacenter-scale and irrelevant to a single Spark.

---

## Hardware Analysis

### The bandwidth wall

LLM decode is memory-bandwidth-bound: each generated token requires reading the active weights plus the KV cache from memory, so **tokens/sec ≈ memory bandwidth ÷ bytes-read-per-token**. The Spark's LMSYS Org review is blunt: *"the only downside of this machine lies in memory bandwidth… offering up to 273 GB/s… this limited bandwidth is expected (and empirically shown) to be the key bottleneck in AI inference performance."* That is roughly an order of magnitude below datacenter HBM (H100 ~3.35 TB/s, B200 ~8 TB/s).

This is why the MoE design is the "killer app" for this box. A **dense** 30B/70B must stream nearly all weights every token; community/LMSYS measurements put a dense **Llama 3.1 70B at ~803 tok/s prefill / 2.7 tok/s decode** in FP8 — effectively unusable interactively. Qwen3-30B-A3B reads only its ~3B active params per token, moving far less data, and lands an order of magnitude higher.

The "1 PFLOP" headline is FP4 *with sparsity*; dense NVFP4 is ~500 TFLOPS. Compute is rarely the limiter here — bandwidth is.

### Realistic performance expectations (Qwen3-30B-A3B class, single stream)

| Config | Decode (single stream) | Notes |
|---|---|---|
| Dense 70B FP8 | ~2.7 tok/s | Why you must use MoE |
| MoE FP8 (stock vLLM) | ~28–30 tok/s | Turnkey but MoE experts fall back to Triton/Marlin on SM121 |
| MoE FP8 + MTP tuning | ~52–64 tok/s | Speculative decoding lift |
| MoE NVFP4 (optimized vLLM) | **~97–120 tok/s** | Best single-user; Tessera/Flowtivity measured 97–120 tok/s |
| MoE Q4_K_M (llama.cpp) | ~50–89 tok/s | Depends on flags; MTP/DFlash pushes higher |

Per Tessera (June 2026): Qwen 3.6 35B-A3B "decodes at roughly **28 to 30 tokens per second with a standard vLLM FP8 setup, and 97 to 120 tokens per second with NVFP4** quantization and an optimized engine." Numbers for the 30B-A3B base architecture will be very close.

**Prefill/TTFT is the weak spot.** Long prompts are slow to first token: an NVFP4 benchmark on Spark measured mean TTFT of ~42 s on an 8K-input prompt-heavy workload and ~168 s on a decode-heavy long run at 16 concurrent. For interactive chat with short prompts this is invisible; for 100K-token agentic/RAG workloads it is noticeable.

### Concurrency changes the economics

Single-stream tok/s is the wrong metric for multi-user serving. Because only 4 experts fire per token, effective bytes-per-token at decode drop sharply under batching, so **aggregate** throughput scales well: community reports cite ~322 tok/s aggregate at 8 concurrent streams (decode-only, warm prefix cache) for a 35B-A3B NVFP4 build, and one report reached ~695 tok/s aggregate at 256 concurrent streams on a gpt-oss-class MoE. Per-stream speed falls with concurrency (~97 tok/s at 1 user → ~42 tok/s each at 8) — **plan capacity by your concurrency budget, not the aggregate headline.**

---

## Quantization Deep-Dive: Quality / Memory / Speed Trade-offs

### Memory math (weights)

| Format | Bits/wt (effective) | ThaiLLM-30B weights | Fits 128GB with room for KV? |
|---|---|---|---|
| BF16 | 16 | ~61 GB | Yes, but tight for long context |
| FP8 (E4M3) | 8 | ~30–31 GB | Comfortable |
| INT4 / AWQ / GPTQ (W4A16) | ~4.5 | ~15–18 GB* | Very comfortable |
| NVFP4 | ~4.5 (E2M1 + FP8 block scale) | ~15–17 GB | Very comfortable |
| GGUF Q4_K_M / UD-Q4_K_XL | ~4.5 | ~17–19 GB | Very comfortable |
| GGUF Q6_K / Q8_0 | 6–8 | ~25–33 GB | Comfortable |

*4-bit sizes inflate when sensitive layers stay high-precision. Real example: a Qwen3.5-27B GPTQ-INT4 ended up 30.3 GB — nearly identical to its 30.9 GB FP8 counterpart — because attention/shared-expert layers were kept in 16-bit.

### KV cache math (Qwen3-30B-A3B: 48 layers, 4 KV heads, head_dim 128, GQA)

Per-token KV = 2 (K&V) × 48 layers × 4 KV heads × 128 × bytes = **49,152 × bytes/element**.

| Precision | Per token | 32K ctx | 128K ctx | 256K ctx |
|---|---|---|---|---|
| FP16 (2 B) | 96 KiB | 3.0 GiB (~3.22 GB) | 12.0 GiB (~12.88 GB) | 24.0 GiB (~25.8 GB) |
| FP8 (1 B) | 48 KiB | 1.5 GiB | 6.0 GiB | 12.0 GiB |

These formula-derived numbers match independent published Qwen3-30B-A3B figures exactly (98,304 B/token FP16; 3.22 GB @32K; 12.88 GB @128K). GQA's 4 KV heads (vs 32 query heads) already gives an 8× KV reduction versus MHA — a major reason long context is affordable on this box.

**Budget on 128 GB (usable ~119–120 GB):** with NVFP4 weights (~16 GB) + FP8 KV cache, even 256K context (12 GiB) leaves ~90 GB free — you can run **very long context, high concurrency, or a second model** simultaneously. With FP8 weights (~30 GB) + FP16 KV at 128K (12.9 GiB), you still have ample headroom.

### Quality by format

- **FP8 (E4M3) — conservative, recommended default for quality-sensitive Thai.** Near-baseline. Qwen ships official fine-grained (block-128) FP8 checkpoints; an enterprise agentic benchmark found FP8 "does not seem to harm performance for Qwen3 models from 14B to 235B." **Caveat on GB10:** MoE-expert FP8 kernels (FlashInfer-TRTLLM / CUTLASS / DeepGEMM) currently gate on datacenter Blackwell (family 100) and are unavailable on SM121, so vLLM falls back to a **Triton/Marlin W8A16** path (dequant experts to BF16) — correct output, but you lose the native FP8 throughput on the layers that matter most. Per one April 2026 vLLM benchmark of Qwen3.6-35B-A3B-FP8: *"MoE backend selected by vLLM: TRITON (FLASHINFER_TRTLLM/CUTLASS/DEEPGEMM all unavailable on SM121)."*
- **NVFP4 — fastest, best speed/quality on Spark.** Blackwell-native 4-bit (E2M1 values, FP8 E4M3 per-16-block scale, FP32 tensor scale; ~4.5 effective bits). Red Hat/NVIDIA report **~97–99% BF16 recovery for ~30B models** and near-baseline at 70B+; ~2.3× faster than INT4 because Blackwell tensor cores consume FP4 directly. Community: "20% faster than AWQ" on Spark. **Caveats:** SM121 lacks native FP4 MoE tensor kernels, so experts run through a Marlin FP4 fallback (still fast, weight-only compression); NVFP4 activation quant had an "illegal instruction" bug on SM121 (fixed via software E2M1 conversion PR #35947); ModelOpt keeps attention in BF16 by default (an intentional accuracy trade-off), so "NVFP4" as shipped is not uniformly 4-bit.
- **W4A16 AWQ / GPTQ.** ~15–18 GB. For Qwen, **AWQ is preferred** — Qwen's own docs flag GPTQ as having known issues, and AWQ's activation-aware rescaling is more robust to calibration choice. Independent study: GPTQ reliably introduces a modest quality drop vs FP16 while AWQ/SmoothQuant stay near full precision. On Blackwell, weight-only 4-bit still dequantizes to 16-bit for compute (slower than NVFP4).
- **GGUF (llama.cpp).** Q4_K_M / **Unsloth UD-Q4_K_XL** (dynamic, uses Q5/Q6 on important matrices, imatrix calibration) is the sweet spot; Q6_K/Q8_0 for higher fidelity. Unsloth Dynamic 2.0 reports ~99.9% KL-divergence retention and (on Aider Polyglot for the Coder variant) UD-Q4_K_XL scoring 60.9% vs 61.8% BF16. GGUF is the most portable and reliable path on GB10.
- **MoE-specific quantization pitfalls.** Expert weights dominate parameter count and tolerate 4-bit reasonably; **router/gate and attention are sensitive and should stay higher precision.** Do not quantize Mamba/SSM state layers in hybrid models. Keep the gate/router in FP16/BF16.
- **KV-cache quantization.** FP8 KV cache halves KV footprint at <1% accuracy loss and is well-supported (`--kv-cache-dtype fp8`). NVFP4 KV cache and turbo/INT8 KV variants exist for extreme long context.

---

## Per-Stack Sections

### vLLM
Qwen3-MoE is fully supported upstream, including FP8 (`--quantization fp8`, block-wise w8a8, or FP8-Marlin w8a16 on older archs) and AWQ. **On DGX Spark the stock images do not work**: vLLM's prebuilt PyTorch/CUDA kernels target up to sm_120 and CUDA 12, while GB10 needs **sm_121a + CUDA 13**, producing "no SM121 support," "SM 12.x requires CUDA ≥ 12.9," and FlashInfer/Marlin errors. Working paths:
- **Community containers** (e.g., `vllm/vllm-openai:cu130-nightly`, `hellohal2064/vllm-dgx-spark-gb10`, `eugr/spark-vllm-docker`) with SM121 patches, Triton built from main, and `TORCH_CUDA_ARCH_LIST=12.1a`.
- **Single-GPU config:** TP=1, **no expert parallelism** (EP and EPLB are for multi-GPU; EPLB's redundant experts even waste memory here). `--gpu-memory-utilization 0.8–0.85`.
- **Recommended NVFP4 launch (adapt from the community Qwen3.6-35B-A3B recipe):**
  ```
  vllm serve <your-NVFP4-checkpoint> \
    --tensor-parallel-size 1 --trust-remote-code \
    --quantization modelopt --kv-cache-dtype fp8 \
    --attention-backend flashinfer --moe-backend marlin \
    --gpu-memory-utilization 0.85 --max-model-len 65536 \
    --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  ```
- **CUDA graphs** improve decode ~20–30%; some SM121 paths (Mamba hybrids) still require `--enforce-eager`. Chunked prefill and prefix caching are essential for RAG/agent workloads.

### LMCache
LMCache extracts KV cache out of GPU memory into a tiered hierarchy (CPU RAM → local SSD/NVMe → remote/S3), with CacheGen (compression) and CacheBlend (non-prefix reuse), integrating with vLLM via the `LMCacheConnectorV1` connector; benchmarks cite 3–15× throughput/latency gains on multi-round QA and document analysis. **On the Spark's unified memory the classic "offload GPU→CPU RAM" value largely disappears** — GPU and CPU share the same physical LPDDR5x pool, so moving KV from "GPU" to "CPU" moves nothing physically and vLLM's built-in prefix caching already covers in-memory reuse. **Where LMCache still adds value on Spark:** (1) **persistence to NVMe** so KV survives restarts / model swaps; (2) **cross-session multi-turn caching** for long-lived assistants; (3) **RAG document-prefix caching** where the same documents are reused across many requests — cutting repeated prefill (the Spark's weakest phase). Skip it for single-user chat.

### TensorRT-LLM
Qwen3 MoE and NVFP4 (via TensorRT Model Optimizer PTQ) are first-class on datacenter Blackwell and are the production speed champion there. **On GB10 it is the most fragile option:** reports show `ptxas fatal: Value 'sm_121a' is not defined`, NVFP4 HF-export load failures specifically for **Qwen3-30B-A3B** (weight_scale mismatch), and trtllm-serve blockers for MoE on aarch64. NVIDIA publishes an official Spark TRT-LLM playbook (`build.nvidia.com/spark/trt-llm`, container `nvcr.io/nvidia/tensorrt-llm/release`), and it does list Qwen3 among supported models — but expect version-pinning pain. **Recommendation: only via NVIDIA's prebuilt container, and only if you need its in-flight batching / paged-KV performance enough to fight the toolchain.** Otherwise vLLM/SGLang give most of the benefit with far less friction.

### NeMo / NIM / Dynamo
- **NIM**: containerized, GPU-accelerated OpenAI-compatible microservices; a Spark playbook exists and Qwen3-32B (and Llama 3.1 8B) NIMs are available. NIM will be the cleanest "just works" enterprise path *if/when* a ThaiLLM/Qwen3-30B-A3B NIM is offered — otherwise you package your own. (For reference on NIM/Ollama perf envelope: gpt-oss-20B MXFP4 in Ollama on Spark hit ~2,053 tok/s prefill / 49.7 tok/s decode, ~4× slower than an RTX Pro 6000 Blackwell.)
- **NeMo Framework**: training/fine-tuning (there is a Spark NeMo fine-tune playbook) — use it for the required SFT step, not inference.
- **Dynamo**: NVIDIA's distributed multi-node inference framework — designed for datacenter clusters; **not relevant** to a single Spark.

### SGLang
RadixAttention prefix caching, strong MoE and NVFP4 support, official Spark playbook (`build.nvidia.com/spark/sglang`, container `lmsysorg/sglang:latest-cu130`, add `--quantization modelopt_fp4` for NVFP4). SM121 support is tracked (issue #11658) with community forks (e.g., `scottgl9/sglang-spark-gb10-optimizations`, `ubehera/sglang-spark`) that route NVFP4 around the broken CUTLASS FP4 path. SGLang reached ~52 tok/s decode on gpt-oss-120B MXFP4 on Spark. A solid alternative to vLLM, especially where RadixAttention helps shared-prefix workloads.

### llama.cpp / Ollama
The **most reliable turnkey path** on GB10, with official NVIDIA playbooks. Build with `-DGGML_CUDA=ON` for sm_121 (CUDA 13.x, `121a` for native FP4); serve GGUF via `llama-server` (OpenAI-compatible). Use **Unsloth UD-Q4_K_XL** dynamic quants. GGUF MoE support is good, startup is near-instant (vs vLLM's minutes), and it handles the 128 GB unified pool gracefully. Speculative decoding (**MTP** now merged; **DFlash** community fork) pushes 27B/35B-A3B to 30–40+ tok/s. Example:
```
./llama-server -hf unsloth/Qwen3-30B-A3B-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 --port 8080 -ngl 99 --ctx-size 131072 -fa 1 --jinja \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --min-p 0.0
```
(You will need to produce a ThaiLLM-30B GGUF after SFT, or quantize your fine-tuned checkpoint.)

---

## DeepSeek-Technique Applicability Analysis

| Technique | Applies to ThaiLLM-30B on Spark? | Why |
|---|---|---|
| **MLA (Multi-head Latent Attention)** | **No** | DeepSeek-specific; Qwen3 uses GQA (32 Q / 4 KV heads). Do not look for MLA KV compression here. |
| **MTP (multi-token prediction) speculative decoding** | **Yes — high value** | Native in Qwen3.5/3.6; MTP drafters reach ~68–83% acceptance on Spark and roughly double throughput. For the 30B-A3B base you'd need an MTP head or a small draft model. |
| **DeepGEMM / FP8 GEMM kernels** | **No** | Target datacenter Blackwell (SM100); SM121 lacks these paths (falls back to Marlin/Triton). |
| **EPLB (expert-parallel load balancing)** | **No** | Multi-GPU; on a single Spark it only wastes memory on redundant experts. |
| **DP-attention + EP serving (DeepSeek-V3 pattern)** | **No** | Cluster-scale; irrelevant to 1 GPU. Only relevant if you cluster two Sparks over ConnectX-7. |
| **EAGLE-3 / Medusa / draft-model speculative decoding** | **Yes** | General-purpose; a small Thai/Qwen draft model or EAGLE head can speed decode. DFlash (block-diffusion) is a strong llama.cpp option. |

Bottom line: on a single Spark, **speculative decoding (MTP/EAGLE/DFlash) is the one DeepSeek-adjacent lever that matters**; the rest are multi-GPU/datacenter-only.

---

## Qwen3-Paper-Informed Inference Settings

- **Sampling.** Thinking mode: **temp 0.6, top-p 0.95, top-k 20.** Non-thinking mode: **temp 0.7, top-p 0.8, top-k 20, presence_penalty 1.5.** (Note the ThaiLLM card's own base-model snippet used temp 0.4 / top-p 0.75 / top-k 40 / repetition_penalty 1.2 for raw completion — after SFT, adopt the Qwen3 instruct settings above.)
- **Presence/repetition penalty** (1.0–1.5) is especially useful to suppress the repetition that aggressive quantization can induce.
- **Thinking budget & mode switching.** Use `/think` and `/no_think` (or `chat_template_kwargs: {"enable_thinking": false}`) per turn; the thinking-budget mechanism can halt reasoning at a token threshold for latency control.
- **Long context (YaRN).** Native 32K; for up to 128K add RoPE scaling in `config.json` or via launch flag:
  `"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`
  (llama.cpp requires regenerating the GGUF after the change). Use only the factor you need — high YaRN factors degrade short-context quality.
- **Tool calling / structured output.** Qwen3 uses a Hermes-style tool parser; in vLLM add `--enable-auto-tool-choice --tool-call-parser hermes` (`qwen3_coder`/`qwen3_xml` for coder variants) and `--reasoning-parser qwen3`; use guided/JSON decoding for structured output.

---

## Practical Recommendation Matrix

| Scenario | Quant format | Stack + key flags | Est. memory (weights+KV) | Realistic performance |
|---|---|---|---|---|
| **Single-user, latency-first** | NVFP4 (attn BF16) | Community vLLM SM121: `--quantization modelopt --kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin` + MTP-3, `--max-num-seqs 1–2` | ~16 GB + ~2–6 GB KV | ~97–120 tok/s decode; TTFT seconds on long prompts |
| **Single-user, "just works"** | GGUF UD-Q4_K_XL | llama.cpp/Ollama, `-ngl 99 -fa 1`, MTP/DFlash draft | ~18 GB + KV | ~30–89 tok/s decode; instant startup |
| **Single-user, long-context (128K–256K)** | FP8 weights + FP8 KV cache | vLLM/SGLang, `--kv-cache-dtype fp8`, YaRN factor 4.0, chunked prefill | ~30 GB + 6–12 GB KV | Decode fine; watch multi-second/tens-of-seconds TTFT |
| **Multi-user throughput** | NVFP4 + FP8 KV | vLLM, `--max-num-seqs 8–64`, `--enable-chunked-prefill --enable-prefix-caching`, MTP | ~16 GB + large KV pool | ~250–320+ tok/s aggregate @8; ~42 tok/s/stream |
| **Multi-user RAG / shared prefixes** | NVFP4 or FP8 | SGLang (RadixAttention) **or** vLLM + **LMCache** (NVMe persistence + doc-prefix cache) + prefix caching | ~16–30 GB + KV + cache tier | High cache-hit TTFT reduction; big win on repeated docs |
| **Conservative / quality-critical Thai** | FP8 | vLLM/SGLang, benchmark on Thai evals first | ~30 GB + KV | ~28–52 tok/s; near-baseline quality |

**Rule of thumb:** start at FP8 to validate Thai quality, then move to NVFP4/Q4 for speed once your Thai eval gate confirms the loss is acceptable.

---

## Thai Language Quality Preservation

This is the single most important quality risk, and it is under-studied for Thai specifically.

- **Quantization hurts non-Latin, lower-resource languages more than English — and automatic metrics hide it.** Marchisio et al. ("How Does Quantization Affect Multilingual LLMs?", arXiv:2407.03211, EMNLP 2024 Findings) found *"harmful effects of quantization are apparent in human evaluation, which automatic metrics severely underestimate: a 1.7% average drop in Japanese across automatic tasks corresponds to a 16.0% drop reported by human evaluators,"* and that *"languages are disparately affected by quantization, with non-Latin script languages impacted worst,"* with *"challenging tasks like mathematical reasoning degrade fastest"* (up to −13.1% at 4-bit). The paper does **not test Thai** — use its non-Latin results (Arabic/Japanese/Korean/Chinese) as the proxy, and treat Thai (also non-Latin script, comparatively lower-resource) as at-risk.
- **A partial counterpoint:** a follow-up study (arXiv:2503.03592) found GGUF K-quant degradation for multilingual tasks generally under 3% and that non-English importance matrices didn't help much — so severity is method- and prompt-difficulty-dependent. Larger models (30B) are also more quantization-robust than small ones.
- **Use target-language calibration for AWQ/GPTQ.** "Calibrating Beyond English" (arXiv:2601.18306) shows *"non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines"* (up to 3.52 perplexity points), and that *"GPTQ is more sensitive to calibration-language shifts, while AWQ is more robust."* **Concrete action: build your AWQ/GPTQ (and NVFP4 PTQ) calibration set from Thai text** — ideally your own domain (the CuratedData categories: law, medical, business, news).
- **Best available Thai/SEA datapoint:** AI Singapore's SEA-LION v4 reports its **FP8-Dynamic and NVFP4 variants have "little degradation (<0.5% on average)"** on SEA-HELM (which explicitly scores Thai) versus BF16 — encouraging evidence that FP8/NVFP4 preserves SEA-language quality on ~27–32B models. Vendor claim; no independent Thai-only audit.
- **Open gap:** no rigorous FP16-vs-quantized *Thai* study exists from Typhoon/OpenThaiGPT as of this writing. **Therefore: benchmark your own quantized ThaiLLM-30B on Thai evals (ThaiExam, Belebele-TH, XNLI-TH, MMLU-Thai, XCOPA-TH) before committing** — and if possible do a small human eval, since automatic metrics under-report the damage.

---

## Suggested Step-by-Step Deployment Plan

**Phase 0 — Fine-tune (mandatory; ThaiLLM-30B is a base model).**
1. SFT with LLaMA-Factory (`--template qwen3`) on your Thai instruction data; optionally DPO. Use the Spark NeMo/LLaMA-Factory playbooks or a datacenter GPU for the training run.
2. Produce a BF16 instruction checkpoint; verify chat behavior and `/think`//no_think` handling.

**Phase 1 — Baseline in FP8 (validate quality first).**
```
# On the Spark, in an NGC container (e.g., nvcr.io/nvidia/pytorch:25.12-py3) or community SM121 vLLM image
vllm serve <your-ThaiLLM-30B-SFT-FP8> \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```
Run Thai evals (ThaiExam, Belebele-TH, MMLU-Thai) and record baseline. This is your quality gate.

**Phase 2 — Quantize for speed.**
- Produce **NVFP4** via NVIDIA TensorRT Model Optimizer (keep attention BF16 default; calibrate with Thai text), or **AWQ** via AutoAWQ/llm-compressor with a **Thai calibration set**, or **GGUF UD-Q4_K_XL** with Unsloth (imatrix on Thai/domain data).
- Re-run the Thai eval gate. Accept the quant only if degradation is within your tolerance (target <1–2% on automatic metrics *and* a spot human check).

**Phase 3 — Serve for the target scenario (see matrix).**
- Latency/concurrency: NVFP4 on community SM121 vLLM with FP8 KV, FlashInfer, Marlin MoE, MTP, prefix + chunked prefill.
- Turnkey/single-user: GGUF Q4_K_XL on llama.cpp with `-fa 1 -ngl 99` and MTP/DFlash.
- RAG/multi-user shared prefixes: add SGLang RadixAttention or vLLM + LMCache (NVMe persistence + document-prefix caching).

**Phase 4 — Long context & tools.**
- Add YaRN (`factor 4.0`) only if you need >32K; keep FP8 KV cache to fit 128K–256K.
- Wire Hermes tool parser + guided JSON for function calling / structured output.

**Phase 5 — Monitor & iterate.**
- Watch decode tok/s, TTFT, KV-cache utilization (DGX Dashboard / NVIDIA Sync). Tune `--max-num-seqs` to your real concurrency. Re-benchmark after every stack/driver upgrade — SM121 support changes rapidly.

---

## Caveats

- **Fast-moving ecosystem.** SM121 support in vLLM, SGLang, and TensorRT-LLM is actively evolving; kernels, container tags, and known bugs (NVFP4 MoE fallbacks, Mamba/Triton crashes, CUDA 13 wheel gaps) shift week to week. Every command here is a starting point to verify, not a guarantee.
- **ThaiLLM-30B is a base model.** Do not serve it directly as a chatbot — it needs instruction fine-tuning first. Confirm the exact base and license on the current model card.
- **GB10 ≠ datacenter Blackwell.** sm_121 diverges from sm_100; kernels advertised for "Blackwell" (FlashMLA, FlashInfer FP4 MoE, DeepGEMM, CUTLASS FP4 MoE) frequently do not run natively and fall back to Marlin/Triton, forfeiting peak FP4/FP8 throughput on the MoE layers that matter most.
- **Thai quality is the biggest unknown.** Published quantization-vs-Thai evidence is thin; the strongest signal (SEA-LION's <0.5% FP8/NVFP4 claim) is vendor-reported. Treat aggressive quantization as unproven for Thai until you measure it yourself, ideally with human evaluation.
- **Prefill/TTFT, not decode, is the practical bottleneck** for long-context/agentic Thai RAG on this hardware — design around it (prefix caching, LMCache document reuse, chunked prefill) rather than expecting API-like sub-second TTFT.
- **Numbers are for the Qwen3-30B-A3B *class***, drawn from Qwen3.5/3.6-35B-A3B and Qwen3-30B-A3B community benchmarks; your fine-tuned Thai checkpoint may differ, so re-measure on your own workload.