<p align="center">
  <img src="agicafet-logo-new.png" alt="AGIcafet" width="120"><br>
  <b>AGIcafet Research</b>
</p>

# ThaiLLM-30B → NVFP4 on NVIDIA DGX Spark

A complete, reproducible quantization study: [ThaiLLM/ThaiLLM-30B](https://huggingface.co/ThaiLLM/ThaiLLM-30B) (Qwen3-MoE, Thai/English base model) quantized to **NVFP4** with NVIDIA TensorRT Model Optimizer and compared against BF16 under a strict fairness protocol on a single **DGX Spark (GB10, 121 GB unified memory)** — accuracy (Thai + English, paired statistics), perplexity, token-level fidelity, qualitative Thai review, serving performance, and a 7-model Thai benchmark context.

**📦 Quantized model:** [AGIcafet/ThaiLLM-30B-NVFP4](https://huggingface.co/AGIcafet/ThaiLLM-30B-NVFP4) ·
**📊 Interactive report:** [thaillm.agicafet.com](https://thaillm.agicafet.com) ·
**📄 Deep report:** [`results/report.md`](results/report.md)

## Headline results

| | BF16 | NVFP4 | Δ |
|---|---|---|---|
| Disk / GPU weights | 61.1 GB / 56.9 GiB | **18.1 GB / 16.9 GiB** | 3.4× smaller |
| Decode (1 stream) | 27.1 tok/s | **63.1 tok/s** | 2.33× faster |
| Decode (4 streams) | 69.4 tok/s | **174.8 tok/s** | 2.52× faster |
| TTFT p50 (1K prompt) | 326 ms | **140 ms** | 2.33× faster |
| **ThaiExam** (letter MC, 565 q) | 0.6195 | 0.6142 | −0.5 pt, p=0.79 (n.s.) |
| Belebele-TH | 0.7700 | 0.7656 | −0.4 pt (n.s.) |
| MMLU 5-shot @50/subj | 0.8168 | 0.8028 | −1.4 pt, p=0.002 |
| **All 19,786 paired MC questions** | — | — | **−0.81 pt**, p<1e-4 |
| Thai Wikipedia bits/byte | 0.2680 | 0.2822 | +0.014 |
| Thai top-1 token agreement | — | — | 92.0 % (32,424 positions) |

**Conclusion: NVFP4 is deployment-ready for Thai workloads** — Thai capability is statistically unchanged (the small, real cost concentrates on English reasoning tails), throughput gain is 2.3–2.5×. The half-Thai calibration set (256 Thai Wikipedia + 256 English news docs) is the likely reason Thai survived fully. One human-review caveat: verbatim quotation (legal statutes) can slip at 4-bit — keep BF16 or retrieval grounding for citation-critical work.

### Cross-model Thai context (same protocol, 7 models)

| Model | Belebele-TH | ThaiExam | XNLI-TH | Thai bpb ↓ |
|---|---|---|---|---|
| ThaiLLM-30B **BF16** (base) | 0.770 | 0.619 | **0.471** | **0.273** |
| ThaiLLM-30B **NVFP4** (base) | 0.766 | 0.614 | 0.458 | 0.286 |
| Qwen3.6-27B NVFP4 (IT) | **0.876** | **0.658** | 0.478 | 0.296 |
| SEA-LION v4.5-27B (IT) | 0.843 | 0.619 | 0.406 | 0.591 |
| Typhoon2.5-30B-A3B (IT) | 0.856 | 0.604 | 0.378 | 0.480 |
| Qwen3.6-35B-A3B NVFP4 (IT) | 0.777 | 0.573 | 0.391 | 0.560 |
| Qwen3-8B NVFP4 (IT) | 0.766 | 0.467 | 0.436 | 0.426 |

*(References are instruction-tuned, scored raw-completion style — mildly conservative for them. Full caveats in the report.)*

## Fairness protocol (why the comparison is trustworthy)

- Identical vLLM flags both sides except `--quantization modelopt`; **KV cache BF16 on both** (quantized with `--kv_cache_qformat none`), prefix caching off, seed 0, byte-identical lm-eval invocations.
- Every MC question answered by both models → **exact McNemar paired tests**, not loose two-sample comparisons.
- Tokenizer files verified identical (byte-identical except `special_tokens_map.json`, which is re-serialized with identical semantics); both server startup configs diffed.
- Full deviation ledger in [`results/report.md`](results/report.md), Section 1.

## Repository layout

```
scripts/          quantize.sh, run_server.sh, run_suite.sh, run_perf.sh,
                  run_usecases.py, run_reference.sh, paired_analysis.py,
                  analyze_fidelity.py, build_calib.py, build_summary.py
thai_tasks/       custom lm-eval tasks: letter-scored ThaiExam (x5 + group),
                  Thai-Wikipedia byte-perplexity
calib/            thai_en_calib.jsonl (256 TH + 256 EN, seeded)
results/          report.md (deep report) · report_visual.html (web report)
                  summary_data.json · paired_analysis.json · fidelity.json
                  qualitative_review.md · usecase_side_by_side.md
                  research_*.md (4 literature dossiers)
                  bf16/ nvfp4/ ref_*/ (lm-eval + vllm-bench outputs)
```

## Reproduce

Hardware: DGX Spark (GB10) or any Blackwell GPU with ≥80 GB for the BF16 side. Container: `nvcr.io/nvidia/vllm:26.05.post1-py3`.

```bash
# 1. calibration set + quantize (~25 min on GB10)
python scripts/build_calib.py
bash scripts/quantize.sh                      # -> models/ThaiLLM-30B-NVFP4

# 2. serve + evaluate each side (identical flags enforced by the scripts)
bash scripts/run_server.sh bf16   && bash scripts/run_suite.sh bf16  && bash scripts/run_perf.sh bf16
bash scripts/run_server.sh nvfp4  && bash scripts/run_suite.sh nvfp4 && bash scripts/run_perf.sh nvfp4

# 3. analysis
python scripts/paired_analysis.py && python scripts/analyze_fidelity.py && python scripts/build_summary.py
```

Serving the published checkpoint directly:

```bash
vllm serve AGIcafet/ThaiLLM-30B-NVFP4 --quantization modelopt \
  --gpu-memory-utilization 0.70 --attention-backend flashinfer
```

**Known issue (NGC 26.05.post1 only):** on this container (vLLM 0.21.0-NV), serving hybrid-architecture models (Qwen3.6 gated-delta-net family — not ThaiLLM-30B itself) stalled the engine permanently on the first 8k-token rolling prompt-logprobs request under CUDA graphs, with NCCL heartbeat errors ~1 h later; `--enforce-eager` avoided it for that workload. Not reproducible on upstream `vllm/vllm-openai:v0.25.1-aarch64` with CUDA graphs on (verified on the same machine, same model, same eval) — so prefer newer containers for hybrid models, and keep CUDA graphs on everywhere else.

## Roadmap — planned next phases

1. **Draft-model speculative decoding** — the Qwen3-30B-A3B base ships no MTP head (unlike Qwen3.6-gen models behind the community 97–120 tok/s figures), but a same-tokenizer drafter (e.g. Qwen3-0.6B) is expected to add ~1.3–1.6× decode speed, lossless w.r.t. greedy outputs (63 → ~80–100 tok/s).
2. **Thai instruction SFT** on the BF16 base → re-quantize with this exact recipe (~25 min) → re-run this suite as the release gate.
3. **EAGLE-3 head training** post-SFT (~1.5–2× additional decode, the best long-term path).
4. **FP8 KV cache** (`--kv-cache-dtype fp8`) in production serving — disabled in this study only to isolate weight quantization.

## License & attribution

Code and reports: **Apache-2.0** (see [LICENSE](LICENSE)). Base model: [ThaiLLM/ThaiLLM-30B](https://huggingface.co/ThaiLLM/ThaiLLM-30B) (Apache-2.0). Quantization: NVIDIA TensorRT Model Optimizer 0.43.0. Evaluation: EleutherAI lm-evaluation-harness 0.4.12.

A project by **[AGIcafet](https://agicafet.com)** · measurements 14–15 July 2026.
