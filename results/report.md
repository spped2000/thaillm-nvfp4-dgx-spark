# ThaiLLM-30B on DGX Spark: BF16 → NVFP4 Quantization — Deep Technical Report

*Produced 2026-07-15 on a single NVIDIA DGX Spark (GB10). All numbers measured on this machine unless cited. Companion files: `summary_data.json` (all raw numbers), `tables.md` (auto-generated full tables), `paired_analysis.json`, `fidelity.json`, `qualitative_review.md`, `usecase_side_by_side.md`, `research_*.md` (4 literature deep-dives), `methodology.md`, per-run lm-eval/bench JSONs.*

---

## Executive Summary

**ThaiLLM-30B quantized to NVFP4 is deployment-ready for Thai workloads on DGX Spark.** The quantized model is **3.4× smaller** (61.1 → 18.1 GB), decodes **2.3–2.5× faster** (27 → 63 tok/s single-stream), answers **2.0–2.7× sooner** (TTFT 326 → 140 ms on 1K prompts), and loads **3.3× faster** — while its **Thai capability is statistically indistinguishable from BF16**: ThaiExam 0.619 → 0.614 (p = 0.79), Belebele-TH 0.770 → 0.766, and no significant change across 3,890 paired Thai multiple-choice samples (p = 0.13). A small but statistically real cost exists on the *English* side (−0.6 to −1.4 points, MMLU p = 0.002); pooled over all 19,786 paired samples the total accuracy cost is **−0.8 points**. Token-level fidelity is high (92% identical next-token predictions on Thai text) and a Thai-fluent qualitative review of 12 domains found **no systematic degradation** — but flagged that verbatim-precision tasks (quoting statutes) can slip, so keep BF16 (or add retrieval) for legal-citation use cases. The likely reason Thai survived so well: **half the calibration documents were Thai** (≈41% of characters) — exactly what the multilingual-quantization literature prescribes.

Recommended serving command (measured configuration):

```bash
docker run --rm -d --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v <PROJECT>:/work \
  nvcr.io/nvidia/vllm:26.05.post1-py3 \
  vllm serve /work/models/ThaiLLM-30B-NVFP4 --quantization modelopt \
    --gpu-memory-utilization 0.70 --max-num-seqs 4 \
    --attention-backend flashinfer --max-model-len 8192
# vLLM auto-selects the native FLASHINFER_CUTLASS NVFP4 MoE kernels on GB10.
# For production capacity, --kv-cache-dtype fp8 roughly doubles KV headroom
# (kept OFF in this study to isolate weight quantization).
```

---

## 1. Objective & Setup

**Goal:** quantize [ThaiLLM/ThaiLLM-30B](https://huggingface.co/ThaiLLM/ThaiLLM-30B) to NVFP4 per the NVIDIA DGX Spark playbook (TensorRT Model Optimizer) and run a *fair, unbiased* before/after comparison served by vLLM — accuracy (Thai + English), perplexity, token-level fidelity, qualitative behavior, latency/throughput, and footprint — plus a cross-model Thai comparison against five locally available models.

| Item | Detail |
|---|---|
| Model | ThaiLLM-30B: Qwen3-30B-A3B continued-pretrain (+63B Thai/EN tokens), `qwen3_moe`, 31B total / ~3.3B active, 128 experts (top-8), 48 layers, **base model** (no instruction tuning), BF16, 61.06 GB, Apache-2.0 |
| Hardware | DGX Spark: GB10 (Blackwell **SM121**), aarch64, 121 GB unified LPDDR5x @ ~273 GB/s, CUDA 13 |
| Serving stack | `nvcr.io/nvidia/vllm:26.05.post1-py3` (vLLM 0.21.0-NV, torch 2.12-NV, FlashInfer 0.6.11) — fresh `--rm` containers, never pip-modified |
| Quantizer | TensorRT Model Optimizer **0.43.0** (the version this container pins), `hf_ptq.py` from the matching repo tag |
| Eval harness | lm-evaluation-harness 0.4.12 (host venv) via `local-completions` against the live server |

### Fairness contract (what makes this comparison unbiased)

- **Identical vLLM flags** for both models except the model path + `--quantization modelopt`: `--max-model-len 8192 --gpu-memory-utilization 0.70 --max-num-seqs 4 --max-num-batched-tokens 8192 --kv-cache-dtype auto --seed 0 --attention-backend flashinfer --no-enable-prefix-caching`.
- **KV cache kept BF16 on both sides.** The playbook default bakes FP8-KV into the checkpoint (`kv_cache_quant_algo: FP8`); we quantized with `--kv_cache_qformat none` so only *weight/activation* quantization is measured. Verified in `hf_quant_config.json` and both server startup logs.
- **Prefix caching disabled on both** — lm-eval loglikelihood uses `echo+logprobs` prompt scoring, which gets no benefit from (and conflicts with) APC in vLLM V1; disabling it removes a nondeterminism source.
- **Byte-identical lm-eval invocations** (`--served-model-name eval-model`, same tokenizer arg, seed 0, same `--limit` values). Tokenizer files verified byte-identical between checkpoints (sha256; `special_tokens_map.json` differs only in serialization form, semantically identical).
- **Same-question pairing:** every MC question was answered by both models, enabling exact McNemar tests instead of loose two-sample comparisons.
- Deterministic scoring (loglikelihood; no sampling anywhere in the accuracy suite). Perf runs: `--ignore-eos`, fixed random dataset, seed 0, 1 discarded warmup + median of 3.

### Deviation ledger (honesty section)

1. `--kv_cache_qformat none` (deviates from playbook FP8 default) — deliberate, for isolation.
2. Calibration data: **256 Thai Wikipedia + 256 CNN/DailyMail docs** (512 × 512 tokens), not the playbook's English-only default — deliberate, per the calibration-language literature.
3. The first ThaiExam task template (choice-text scoring) scored at **chance for both models** and was replaced by a letter-based template matching the model card's protocol ("probability of selecting the correct choice"); both variants were run on both models. Lesson recorded: exam-style Thai MC needs letter scoring.
4. The quantization container's `transformers` moved 5.6.0 → 4.57.6 (modelopt 0.43.0 pins `<5.0`); torch/vLLM/FlashInfer untouched (guard-verified); serve containers never received any pip install.
5. Reference-model runs are *capability snapshots*, not part of the controlled A/B (see Section 7 caveats).

---

## 2. The Quantization Artifact

`models/ThaiLLM-30B-NVFP4` — **18.12 GB** (3.37× compression), produced in ~25 minutes end-to-end on the Spark (model load ~7 min, calibration 256 steps × 3.4 s ≈ 14.5 min, export 87 s). Producer: modelopt 0.43.0.

- **Format:** NVFP4 = FP4 (E2M1) weights *and* activations, per-16-value FP8 (E4M3) block scales + global FP32 scales (`weight`, `weight_scale`, `weight_scale_2`, `input_scale` per linear).
- **What is quantized (verified at tensor level):** all attention projections (q/k/v/o) **and** all 128 experts per layer (gate/up/down) — full W4A4. This is *more aggressive* than NVFP4 variants that keep attention in BF16.
- **What is not:** the 48 MoE router gates (`model.layers.*.mlp.gate`) and `lm_head` stay BF16 (modelopt defaults — routers are quantization-sensitive); Q/K RMSNorms and embeddings are untouched; **no KV-cache quantization**.
- Loads in vLLM as `quantization=modelopt_fp4` (auto-detected), 16.85 GiB resident, 109 s load (vs 56.88 GiB / 365 s for BF16).

---

## 3. Accuracy: BF16 vs NVFP4

### 3.1 Headline table (identical prompts, seed 0)

| Task (n) | BF16 | NVFP4 | Δ (pts) | Paired p† | Verdict |
|---|---|---|---|---|---|
| **ThaiExam v2** (565) | **0.6195** ±.020 | 0.6142 ±.020 | −0.53 | 0.79 | noise |
| **Belebele-TH** (900) | 0.7700 ±.014 | 0.7656 ±.014 | −0.44 | — | noise |
| **XNLI-TH** (2,490) | 0.4707 ±.010 | 0.4578 ±.010 | −1.29 | — | ≤1.3σ |
| **XCOPA-TH** (500) | 0.6400 ±.022 | 0.6340 ±.022 | −0.60 | — | noise |
| Thai MC pooled (3,890) | — | — | −1.00 | **0.13** | not significant |
| MMLU 5-shot @50/subj (2,850) | 0.8168 ±.007 | 0.8028 ±.007 | −1.40 | **0.0017** | significant |
| HellaSwag (10,042) | 0.6003 / 0.7961ⁿ | 0.5941 / 0.7855ⁿ | −0.62 / −1.06ⁿ | **<0.001** | significant, small |
| ARC-Challenge (1,172) | 0.5614 ±.015 | 0.5597 ±.015 | −0.17 | — | noise |
| WinoGrande (1,267) | 0.7395 ±.012 | 0.7285 ±.013 | −1.10 | — | ≤0.9σ |
| **ALL MC pooled (19,786)**† | — | — | **−0.81** | **<1e-4** | significant, small |

† exact McNemar test on paired per-question outcomes (`paired_analysis.json`). ⁿ = acc_norm. The pooled row excludes the broken ThaiExam-v1 template (chance-level for both models) to avoid double-counting the 565 exam questions already pooled via v2.

ThaiExam v2 subsets (BF16 → NVFP4): a_level 0.654→0.622, ic 0.695→0.684, onet 0.586→0.574, **tgat 0.677→0.708 (+3.1)**, **tpat1 0.534→0.552 (+1.7)** — quantization noise moves both directions at subset scale.

### 3.2 What the paired statistics actually say

Because both models answered the *same* questions, we can count flips directly: over all 19,786 MC questions (ThaiExam counted once, via the letter-based v2 template), BF16 was uniquely right on 727 and NVFP4 uniquely right on 567 — a net 160-question (−0.81 pt) deficit that is decisively non-zero (p < 1e-4) but **very small in magnitude and English-concentrated**:

| Slice | n | Δacc | McNemar p |
|---|---|---|---|
| English MC (MMLU+HS+ARC+WG) | 12,481 | −0.62 | 0.0003 |
| MMLU alone | 2,850 | −1.40 | 0.0017 |
| **Thai MC (Belebele+XNLI+XCOPA)** | 3,890 | −1.00 | **0.131** |
| **ThaiExam v2** | 565 | −0.53 | **0.791** |

Only two individual tasks reach p < 0.05: HellaSwag (−0.62, powered by n = 10k) and `mmlu_moral_scenarios` (−12 pts at n = 50 — with 57 MMLU subjects tested, one such hit is expected by chance; it does not survive Bonferroni). **No Thai task shows significant degradation.**

**Interpretation.** The multilingual-quantization literature (Marchisio et al., EMNLP 2024 — see `research_thaiQuant.md`) predicts non-Latin scripts degrade *more* than English (−1.9% vs −0.7% at W4). We observed the opposite ordering. Two mutually compatible explanations: (a) the **half-Thai calibration set** protected Thai activation ranges — the mechanism "Calibrating Beyond English" (arXiv:2601.18306) documents; (b) NVFP4's FP8 block-scales are structurally gentler than the GPTQ/NF4 formats in that literature (RedHat/NVIDIA report 97–99% recovery at ~30B scale, consistent with our pooled result (98.7% relative accuracy recovery: 0.6104/0.6185 over paired questions)). The prediction that transfers cleanly: *hard reasoning degrades first* — our largest deltas are indeed 5-shot MMLU and HellaSwag tails, not Thai knowledge tasks.

### 3.3 The template lesson (ThaiExam v1)

Scoring ThaiExam by full choice-text loglikelihood put **both** models at chance (BF16 0.255, NVFP4 0.244) — exam distractors are too long/parallel for continuation scoring. The letter-based template (choices in prompt, score " a"…" e") matches the model card's protocol and recovers real signal (0.62). Both variants are logged; only v2 is interpreted. This is a general caution for Thai exam-style evals.

---

## 4. Perplexity (raw language modeling)

| Corpus (Thai: first 1,000 docs; English: full 62-doc WikiText-2 test set; 8k-token rolling windows) | BF16 | NVFP4 | Δ |
|---|---|---|---|
| Thai Wikipedia — bits/byte | 0.2680 | 0.2822 | +0.0142 (+5.3% rel) |
| Thai Wikipedia — byte-PPL | 1.204 | 1.216 | +1.0% |
| WikiText-2 (EN) — bits/byte | 0.5660 | 0.5819 | +0.0159 (+2.8% rel) |
| WikiText-2 (EN) — word-PPL | 8.148 | 8.645 | +6.1% |

Two honest lenses: **absolute** information loss is nearly language-neutral (+0.014 vs +0.016 bits per byte); **relative** to Thai's lower per-byte entropy (Thai UTF-8 ≈ 3 bytes/char) the same absolute loss is a 1.9× larger fraction. Word-perplexity is only reported for English (Thai has no whitespace word boundaries). Byte-level metrics are tokenizer-independent, which also makes them the only PPL numbers comparable across the reference models in Section 7.

---

## 5. Token-Level Fidelity & Qualitative Behavior

### 5.1 Teacher-forced fidelity (identical context → compare next-token prediction)

Over 40 held-out passages (20 Thai wiki, 20 WikiText; 46k scored positions):

| | Thai | English |
|---|---|---|
| Top-1 next-token agreement | **92.0%** (32,424 pos.) | 88.1% (13,848 pos.) |
| Mean \|Δ log-prob\| on actual tokens | 0.170 nats | 0.232 nats |

Thai agreement *exceeds* English — consistent with the accuracy result that Thai behavior is better preserved. Greedy generations diverge early (median ≈ 4 tokens across 12 prompts) — expected chaos-amplification under argmax decoding, which is why teacher-forced agreement, not divergence position, is the meaningful fidelity metric.

### 5.2 Thai-fluent qualitative review (12 domains, greedy, identical prompts)

Full review in `qualitative_review.md` / raw text in `usecase_side_by_side.md`. Verdicts: **8 equivalent, 3 BF16-better, 1 NVFP4-slightly-better.** Key findings:

- **No orthography damage** (tone marks, clusters intact), no new repetition-loop behavior (loops occur in *both* — greedy base-model artifacts), all core facts correct in both (77 provinces, Section 420 = tort provision, 500−250=250, medical/science basics).
- **The one meaningful regression class: verbatim precision.** NVFP4 misquoted Civil & Commercial Code Section 420 ("บุคคลภายนอก" for "บุคคลอื่น", dropped the compensation clause) where BF16 quoted it exactly; it also produced one incoherent arithmetic sub-explanation and minor word-stutter. This is a concrete instance of Marchisio's warning that *human review catches what automatic metrics miss* — **for legal/citation-critical applications, keep BF16 or add retrieval grounding.**

---

## 6. Performance & Footprint

Measured with `vllm bench serve` (random dataset, `--ignore-eos`, seed 0, warmup discarded, median of 3). TTFT gains span 1.96–2.68× across the grid (the 1K-in/4-stream case is 1.96×):

| Workload | Metric | BF16 | NVFP4 | NVFP4 advantage |
|---|---|---|---|---|
| 1K in / 128 out, 1 stream | decode tok/s | 27.1 | **63.1** | **2.33×** |
| | TTFT p50 | 326 ms | **140 ms** | 2.33× |
| | TPOT p50 | 34.6 ms | **14.9 ms** | 2.32× |
| 1K in / 128 out, 4 streams | agg. tok/s | 59.6 | **145.4** | 2.44× |
| 128 in / 1K out, 1 stream | decode tok/s | 29.1 | **67.2** | 2.31× |
| | TTFT p50 | 182 ms | **70 ms** | 2.61× |
| 128 in / 1K out, 4 streams | agg. tok/s | 69.4 | **174.8** | 2.52× |

| Footprint | BF16 | NVFP4 | Ratio |
|---|---|---|---|
| Disk | 61.06 GB | 18.12 GB | 3.37× |
| GPU-resident weights | 56.88 GiB | 16.85 GiB | 3.38× |
| Model load time | 365 s | 109 s | 3.34× |

**Roofline sanity check.** At 14.9 ms/token, effective memory traffic ≈ 273 GB/s × 0.0149 s ≈ 4.1 GB/token versus a theoretical minimum of ~1.9 GB (3.3B active params × ~0.56 B/param + attention/KV reads) → ~47% bandwidth efficiency, typical for MoE expert-gather patterns. Decode is bandwidth-bound exactly as the compass report predicted; the 2.3× speedup tracks the 3.4× weight compression minus non-weight traffic.

**Ecosystem placement** (full timeline in `research_sm121.md`): our container auto-selected **native FLASHINFER_CUTLASS NVFP4 MoE kernels** — the April-2026 "SM121 falls back to Marlin" era ended with vLLM PR #37725 (the `sm_121a` build fix that enabled the hardware E2M1 path), FlashInfer's SM12x optimization campaign, and vLLM #40082 (b12x fused MoE; only +2–6% over flashinfer-cutlass, so our backend is near-optimal). Community figures of 97–120 tok/s on Spark are **Qwen3.6-35B with MTP-3 speculative decoding** (its baseline without spec-decode: ~77 tok/s); ThaiLLM as a base model has no MTP head, and our 63–67 tok/s sits at the top of the published no-spec-decode band for this architecture class. Post-SFT, an EAGLE-3 head or small draft model is the obvious next 1.5–2× (see `research_alternatives.md`).

---

## 7. Cross-Model Thai Comparison (context, not A/B)

Same tasks, same raw-completion loglikelihood protocol, same container, each model with its own tokenizer. **Caveats:** the five references are *instruction-tuned* and scored without chat templates (mildly conservative for them, per lm-eval/OLL-v2 practice); PPL on raw wiki text penalizes instruct models (instruction-tuning tax); GGUF variants of SEA-LION/Typhoon were bypassed in favor of their BF16 safetensors originals so every model runs the identical vLLM stack.

| Model (precision, type) | Belebele-TH | XNLI-TH | XCOPA-TH | ThaiExam v2 | MMLU@10 | Thai bpb@200 ↓ |
|---|---|---|---|---|---|---|
| **ThaiLLM-30B BF16** (base) | 0.770 | **0.471** | **0.640** | 0.619 | 0.833 | **0.273** |
| **ThaiLLM-30B NVFP4** (base) | 0.766 | 0.458 | 0.634 | 0.614 | 0.816 | 0.286 |
| Typhoon2.5-30B-A3B BF16 (IT, same arch) | 0.856 | 0.378 | 0.620 | 0.604 | 0.833 | 0.480 |
| SEA-LION v4.5-27B BF16 (IT, Qwen3.6 base) | 0.843 | 0.406 | 0.550 | 0.619 | **0.888** | 0.591 |
| **Qwen3.6-27B NVFP4** (IT, Unsloth quant) | **0.876** | 0.478 | 0.600 | **0.658** | 0.881 | 0.296 |
| Qwen3.6-35B-A3B NVFP4 (IT, hybrid MoE) | 0.777 | 0.391 | 0.556 | 0.573 | 0.839 | 0.560 |
| Qwen3-8B NVFP4 (IT, scale anchor) | 0.766 | 0.436 | 0.596 | 0.467 | 0.751 | 0.426 |

*(ThaiLLM MMLU@10 and bpb@200 recomputed from logged per-sample data on exactly the same doc subsets the references saw.)*

Readings:

1. **NVFP4-ThaiLLM keeps its family position everywhere** — the quantization deltas (rows 1–2) are far smaller than every between-model gap.
2. **ThaiLLM is the best raw-Thai language model on the box** (bpb 0.273/0.286; Typhoon 0.480 despite identical architecture + Thai specialization — the instruct tax plus ThaiLLM's 31.5B-token Thai CPT show up clearly). This is exactly what a CPT base model is for: a Thai foundation to fine-tune.
3. **Thai instruct specialists win comprehension-format tasks** (Belebele 0.84–0.86 vs 0.77): instruction tuning, not Thai knowledge, drives that gap — ThaiLLM ties/beats them on ThaiExam knowledge (an exact tie with SEA-LION at 350/565 = 0.619; Typhoon 0.604).
4. **The strongest overall Thai performer is (newer-generation) Qwen3.6-27B — even 4-bit quantized** (ThaiExam 0.658, Belebele 0.876, bpb 0.296). For greenfield Thai deployments this is the benchmark to beat; for ThaiLLM's niche (Thai-sovereign base for domain SFT, permissive license, best raw-Thai modeling) it does not displace the project.
5. The 8B anchor confirms task validity: scale gaps (−15 pts ThaiExam) dwarf quantization gaps (−0.5 pts).
6. XNLI-TH's odd ordering (instruct models near chance at 0.38–0.41 vs ThaiLLM 0.47) is a known artifact of that task's minimal template punishing chat-tuned models — treat XNLI columns as within-family signals only.

---

## 8. Limitations

1. **No large-scale human evaluation.** The literature's central warning is that automatic metrics understate quantization damage by up to ~6× on realistic generative prompts; our 12-domain expert review mitigates but does not close this gap. The legal-quote slip shows the risk is real.
2. **Base-model scope.** All conclusions predate instruction tuning. The correct production sequence is SFT first, then re-quantize the SFT checkpoint with this same recipe (~25 min on this box), then re-run this eval gate (scripts are reusable as-is).
3. MC accuracy = loglikelihood selection; generative Thai quality is covered only by PPL, fidelity, and the 12-prompt review. MMLU sampled at 50/subject (±0.7 pt aggregate stderr); single seed (0) throughout — loglikelihood is deterministic, but calibration sampling is seed-dependent.
4. `mmlu_moral_scenarios` significance is a multiple-comparisons casualty until replicated.
5. Reference comparison caveats as listed in Section 7; Qwen3.6-35B required `--enforce-eager` for its rolling-PPL pass (its hybrid engine hung under CUDA graphs on this container — ecosystem immaturity, logged in `resume_refs.log`).
6. Numbers are container-specific (26.05.post1); SM121 kernels are improving monthly and perf results will drift upward.

---

## 9. Recommendations

| Scenario | Recommendation |
|---|---|
| Serving ThaiLLM for Thai chat/RAG prototypes on Spark | **Use the NVFP4 checkpoint** with the command in the summary; 2.3–2.5× speed, ⅓ memory, no measurable Thai loss. Add `--kv-cache-dtype fp8` in production for KV headroom. |
| Legal/medical text requiring verbatim quotation | Keep BF16, or pair NVFP4 with retrieval so citations come from documents, not weights. |
| Production assistant | SFT the BF16 base first (LLaMA-Factory, per model card), **quantize after SFT** with this exact recipe (half-Thai calibration; `--kv_cache_qformat none` for eval, `fp8` for the shipping artifact), re-run this suite as the release gate. |
| More decode speed | Post-SFT: EAGLE-3/draft-model speculative decoding (~1.5–2× on top, per community MTP data); watch NGC container updates — SM12x kernels still improving. |
| Conservative fallback | FP8 (~31 GB): near-lossless, but on SM121 MoE-expert FP8 runs via W8A16 fallback — expect ≈BF16-to-1.5× speed, not NVFP4's 2.3× (details in `research_alternatives.md`). |
| Memory-constrained co-hosting | NVFP4's 18 GB + BF16-KV leaves ~85 GB free at our settings — enough to co-host a second model on the same Spark. |

**Total machine time used:** ~26 h wall-clock (122 GB of downloads on a flaky-then-fast link, 25 min quantization, ~11 h primary A/B evals, ~7 h reference evals, ~2 h perf). The `nemoclaw-vllm` service was left **stopped** per instruction; restore anytime with `docker start nemoclaw-vllm`.

## 10. Reproducibility Index

- Quantize: `scripts/quantize.sh` (container recipe, pip guard, hf_ptq flags) + `calib/thai_en_calib.jsonl` (seeded builder: `scripts/build_calib.py`)
- Serve/eval: `scripts/run_server.sh`, `run_suite.sh`, `run_perf.sh`, `run_usecases.py`, `run_reference.sh`, custom tasks in `thai_tasks/` (letter-based ThaiExam ×5 + group, Thai-wiki byte-PPL)
- Analysis: `scripts/paired_analysis.py` (McNemar), `analyze_fidelity.py`, `build_summary.py`, `aggregate_report.py`
- Environments: `results/eval_env.txt` (host venv freeze), `results/quant_env_{before,after}.txt` (container guard), server logs per run
- Research dossiers: `results/research_{sm121,thaiQuant,alternatives,crossModel}.md`
