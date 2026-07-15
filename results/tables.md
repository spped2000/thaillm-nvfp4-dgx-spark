# ThaiLLM-30B: BF16 vs NVFP4 (modelopt) on DGX Spark — comparison report

## Accuracy (identical lm_eval invocations, seed 0)

| Task | Metric | BF16 | NVFP4 | Δ (NVFP4−BF16) | ±stderr(BF16) | Notes |
|---|---|---|---|---|---|---|
| arc_challenge | acc | 0.5614 | 0.5597 | -0.0017 | 0.0145 | noise |
| arc_challenge | acc_norm | 0.5836 | 0.5870 | 0.0034 | 0.0144 | noise |
| belebele_tha_Thai | acc | 0.7700 | 0.7656 | -0.0044 | 0.0140 | noise |
| belebele_tha_Thai | acc_norm | 0.7700 | 0.7656 | -0.0044 | 0.0140 | noise |
| hellaswag | acc | 0.6003 | 0.5941 | -0.0062 | 0.0049 | ~1-2σ |
| hellaswag | acc_norm | 0.7961 | 0.7855 | -0.0106 | 0.0040 | **signif.** |
| mmlu | acc | 0.8168 | 0.8028 | -0.0140 | 0.0069 | **signif.** |
| mmlu_abstract_algebra | acc | 0.6000 | 0.6200 | 0.0200 | 0.0700 | noise |
| mmlu_anatomy | acc | 0.7600 | 0.7600 | 0.0000 | 0.0610 | noise |
| mmlu_astronomy | acc | 0.9200 | 0.9200 | 0.0000 | 0.0388 | noise |
| mmlu_business_ethics | acc | 0.8600 | 0.8600 | 0.0000 | 0.0496 | noise |
| mmlu_clinical_knowledge | acc | 0.8400 | 0.8600 | 0.0200 | 0.0524 | noise |
| mmlu_college_biology | acc | 0.9000 | 0.9000 | 0.0000 | 0.0429 | noise |
| mmlu_college_chemistry | acc | 0.5800 | 0.5600 | -0.0200 | 0.0705 | noise |
| mmlu_college_computer_science | acc | 0.7600 | 0.7400 | -0.0200 | 0.0610 | noise |
| mmlu_college_mathematics | acc | 0.6800 | 0.5600 | -0.1200 | 0.0666 | ~1-2σ |
| mmlu_college_medicine | acc | 0.8200 | 0.7600 | -0.0600 | 0.0549 | ~1-2σ |
| mmlu_college_physics | acc | 0.7800 | 0.7600 | -0.0200 | 0.0592 | noise |
| mmlu_computer_security | acc | 0.9200 | 0.8400 | -0.0800 | 0.0388 | **signif.** |
| mmlu_conceptual_physics | acc | 0.9600 | 0.9400 | -0.0200 | 0.0280 | noise |
| mmlu_econometrics | acc | 0.7000 | 0.6800 | -0.0200 | 0.0655 | noise |
| mmlu_electrical_engineering | acc | 0.8400 | 0.8000 | -0.0400 | 0.0524 | noise |
| mmlu_elementary_mathematics | acc | 0.8600 | 0.7800 | -0.0800 | 0.0496 | ~1-2σ |
| mmlu_formal_logic | acc | 0.6600 | 0.6600 | 0.0000 | 0.0677 | noise |
| mmlu_global_facts | acc | 0.5000 | 0.4600 | -0.0400 | 0.0714 | noise |
| mmlu_high_school_biology | acc | 0.9600 | 0.9600 | 0.0000 | 0.0280 | noise |
| mmlu_high_school_chemistry | acc | 0.8600 | 0.8800 | 0.0200 | 0.0496 | noise |
| mmlu_high_school_computer_science | acc | 0.8800 | 0.8600 | -0.0200 | 0.0464 | noise |
| mmlu_high_school_european_history | acc | 0.8400 | 0.8600 | 0.0200 | 0.0524 | noise |
| mmlu_high_school_geography | acc | 0.9200 | 0.9000 | -0.0200 | 0.0388 | noise |
| mmlu_high_school_government_and_politics | acc | 0.9600 | 0.9200 | -0.0400 | 0.0280 | ~1-2σ |
| mmlu_high_school_macroeconomics | acc | 0.8200 | 0.8200 | 0.0000 | 0.0549 | noise |
| mmlu_high_school_mathematics | acc | 0.7000 | 0.6000 | -0.1000 | 0.0655 | ~1-2σ |
| mmlu_high_school_microeconomics | acc | 0.9600 | 0.9800 | 0.0200 | 0.0280 | noise |
| mmlu_high_school_physics | acc | 0.8000 | 0.8000 | 0.0000 | 0.0571 | noise |
| mmlu_high_school_psychology | acc | 0.9800 | 0.9800 | 0.0000 | 0.0200 | noise |
| mmlu_high_school_statistics | acc | 0.6800 | 0.7800 | 0.1000 | 0.0666 | ~1-2σ |
| mmlu_high_school_us_history | acc | 0.9400 | 0.9000 | -0.0400 | 0.0339 | ~1-2σ |
| mmlu_high_school_world_history | acc | 0.9200 | 0.9000 | -0.0200 | 0.0388 | noise |
| mmlu_human_aging | acc | 0.7400 | 0.7200 | -0.0200 | 0.0627 | noise |
| mmlu_human_sexuality | acc | 0.9000 | 0.9000 | 0.0000 | 0.0429 | noise |
| mmlu_humanities | acc | 0.8108 | 0.8046 | -0.0062 | 0.0147 | noise |
| mmlu_international_law | acc | 0.9400 | 0.9400 | 0.0000 | 0.0339 | noise |
| mmlu_jurisprudence | acc | 0.8800 | 0.9000 | 0.0200 | 0.0464 | noise |
| mmlu_logical_fallacies | acc | 0.8400 | 0.8400 | 0.0000 | 0.0524 | noise |
| mmlu_machine_learning | acc | 0.7000 | 0.7200 | 0.0200 | 0.0655 | noise |
| mmlu_management | acc | 0.9400 | 0.9000 | -0.0400 | 0.0339 | ~1-2σ |
| mmlu_marketing | acc | 1.0000 | 0.9800 | -0.0200 | — |  |
| mmlu_medical_genetics | acc | 0.9200 | 0.9600 | 0.0400 | 0.0388 | ~1-2σ |
| mmlu_miscellaneous | acc | 0.9200 | 0.9400 | 0.0200 | 0.0388 | noise |
| mmlu_moral_disputes | acc | 0.7400 | 0.7800 | 0.0400 | 0.0627 | noise |
| mmlu_moral_scenarios | acc | 0.6000 | 0.4800 | -0.1200 | 0.0700 | ~1-2σ |
| mmlu_nutrition | acc | 0.8400 | 0.8600 | 0.0200 | 0.0524 | noise |
| mmlu_other | acc | 0.8077 | 0.7969 | -0.0108 | 0.0146 | noise |
| mmlu_philosophy | acc | 0.8800 | 0.8600 | -0.0200 | 0.0464 | noise |
| mmlu_prehistory | acc | 0.8600 | 0.8600 | 0.0000 | 0.0496 | noise |
| mmlu_professional_accounting | acc | 0.6400 | 0.6000 | -0.0400 | 0.0686 | noise |
| mmlu_professional_law | acc | 0.5600 | 0.5800 | 0.0200 | 0.0709 | noise |
| mmlu_professional_medicine | acc | 0.8800 | 0.8800 | 0.0000 | 0.0464 | noise |
| mmlu_professional_psychology | acc | 0.8200 | 0.7600 | -0.0600 | 0.0549 | ~1-2σ |
| mmlu_public_relations | acc | 0.6200 | 0.6600 | 0.0400 | 0.0693 | noise |
| mmlu_security_studies | acc | 0.8600 | 0.8000 | -0.0600 | 0.0496 | ~1-2σ |
| mmlu_social_sciences | acc | 0.8650 | 0.8467 | -0.0183 | 0.0134 | ~1-2σ |
| mmlu_sociology | acc | 0.8800 | 0.8600 | -0.0200 | 0.0464 | noise |
| mmlu_stem | acc | 0.7968 | 0.7779 | -0.0189 | 0.0126 | ~1-2σ |
| mmlu_us_foreign_policy | acc | 0.9600 | 0.9000 | -0.0600 | 0.0280 | **signif.** |
| mmlu_virology | acc | 0.6000 | 0.5800 | -0.0200 | 0.0700 | noise |
| mmlu_world_religions | acc | 0.8800 | 0.9000 | 0.0200 | 0.0464 | noise |
| thai_exam | acc | 0.2549 | 0.2442 | -0.0106 | 0.0184 | noise |
| thai_exam | acc_norm | 0.3115 | 0.3062 | -0.0053 | 0.0194 | noise |
| thai_exam_a_level | acc | 0.2283 | 0.2205 | -0.0079 | 0.0374 | noise |
| thai_exam_a_level | acc_norm | 0.2677 | 0.2913 | 0.0236 | 0.0394 | noise |
| thai_exam_ic | acc | 0.2000 | 0.2105 | 0.0105 | 0.0413 | noise |
| thai_exam_ic | acc_norm | 0.2421 | 0.2211 | -0.0211 | 0.0442 | noise |
| thai_exam_onet | acc | 0.2840 | 0.2531 | -0.0309 | 0.0355 | noise |
| thai_exam_onet | acc_norm | 0.3148 | 0.2963 | -0.0185 | 0.0366 | noise |
| thai_exam_tgat | acc | 0.2615 | 0.2462 | -0.0154 | 0.0549 | noise |
| thai_exam_tgat | acc_norm | 0.4154 | 0.3846 | -0.0308 | 0.0616 | noise |
| thai_exam_tpat1 | acc | 0.2845 | 0.2845 | 0.0000 | 0.0421 | noise |
| thai_exam_tpat1 | acc_norm | 0.3534 | 0.3621 | 0.0086 | 0.0446 | noise |
| thai_exam_v2 | acc | 0.6195 | 0.6142 | -0.0053 | 0.0204 | noise |
| thai_exam_v2_a_level | acc | 0.6535 | 0.6220 | -0.0315 | 0.0424 | noise |
| thai_exam_v2_ic | acc | 0.6947 | 0.6842 | -0.0105 | 0.0475 | noise |
| thai_exam_v2_onet | acc | 0.5864 | 0.5741 | -0.0123 | 0.0388 | noise |
| thai_exam_v2_tgat | acc | 0.6769 | 0.7077 | 0.0308 | 0.0585 | noise |
| thai_exam_v2_tpat1 | acc | 0.5345 | 0.5517 | 0.0172 | 0.0465 | noise |
| thai_wikipedia_ppl | byte_perplexity | 1.2042 | 1.2160 | 0.0118 | — |  |
| thai_wikipedia_ppl | bits_per_byte | 0.2680 | 0.2822 | 0.0141 | — |  |
| wikitext | word_perplexity | 8.1481 | 8.6453 | 0.4972 | — |  |
| wikitext | byte_perplexity | 1.4804 | 1.4969 | 0.0165 | — |  |
| wikitext | bits_per_byte | 0.5660 | 0.5819 | 0.0160 | — |  |
| winogrande | acc | 0.7395 | 0.7285 | -0.0110 | 0.0123 | noise |
| xcopa_th | acc | 0.6400 | 0.6340 | -0.0060 | 0.0215 | noise |
| xnli_th | acc | 0.4707 | 0.4578 | -0.0129 | 0.0100 | ~1-2σ |

## Performance (vllm bench serve, random dataset, ignore-eos, median of 3)

| Config | Metric | BF16 | NVFP4 | Ratio (NVFP4/BF16) |
|---|---|---|---|---|
| 1024x128_c1 | output_throughput | 27.15 | 63.05 | 2.323 |
| 1024x128_c1 | median_ttft_ms | 325.80 | 140.13 | 0.430 |
| 1024x128_c1 | median_tpot_ms | 34.60 | 14.94 | 0.432 |
| 1024x128_c1 | median_itl_ms | 34.68 | 14.85 | 0.428 |
| 1024x128_c4 | output_throughput | 59.60 | 145.41 | 2.440 |
| 1024x128_c4 | median_ttft_ms | 784.54 | 399.66 | 0.509 |
| 1024x128_c4 | median_tpot_ms | 61.34 | 24.66 | 0.402 |
| 1024x128_c4 | median_itl_ms | 59.70 | 24.24 | 0.406 |
| 128x1024_c1 | output_throughput | 29.06 | 67.22 | 2.314 |
| 128x1024_c1 | median_ttft_ms | 182.47 | 70.01 | 0.384 |
| 128x1024_c1 | median_tpot_ms | 34.25 | 14.83 | 0.433 |
| 128x1024_c1 | median_itl_ms | 34.35 | 14.84 | 0.432 |
| 128x1024_c4 | output_throughput | 69.37 | 174.77 | 2.519 |
| 128x1024_c4 | median_ttft_ms | 344.52 | 128.75 | 0.374 |
| 128x1024_c4 | median_tpot_ms | 57.16 | 22.69 | 0.397 |
| 128x1024_c4 | median_itl_ms | 57.25 | 22.76 | 0.397 |

## Methodology

### Methodology notes (fairness contract)

- **Hardware:** NVIDIA DGX Spark (GB10, SM121, aarch64), 121 GB unified memory, CUDA 13.
- **Serving:** both models served from `nvcr.io/nvidia/vllm:26.05.post1-py3` (vLLM 0.21.0-NV build) in fresh `--rm` containers, identical flags: `--max-model-len 8192 --gpu-memory-utilization 0.70 --max-num-seqs 4 --max-num-batched-tokens 8192 --kv-cache-dtype auto --seed 0 --attention-backend flashinfer --no-enable-prefix-caching`. NVFP4 side additionally: `--quantization modelopt --moe-backend <frozen at Phase 1.5>` (MoE-backend asymmetry vs BF16 is inherent to quantized-kernel serving; recorded in server logs).
- **Quantization:** TensorRT Model Optimizer 0.43.0 (`hf_ptq.py`, tag 0.43.0), `--qformat nvfp4`, **`--kv_cache_qformat none`** (KV cache stays BF16 on both sides so only weight/activation quantization is measured; deviates from playbook default fp8 KV deliberately). Calibration: 512 samples × 512 tokens — 256 Thai Wikipedia (20231101.th) + 256 CNN/DailyMail, seeded (0). Router gates (`mlp.gate`) and `lm_head` excluded per modelopt defaults. Quantization ran in a disposable container where transformers was moved 5.6.0→4.57.6 (modelopt 0.43.0 pins `<5.0`); torch/vllm/flashinfer untouched; serve containers never received pip installs.
- **Accuracy:** lm-eval 0.4.12 via `local-completions` against the live server, byte-identical invocations except output dir: `num_concurrent=8, tokenized_requests=True, max_length=8192, seed 0`. Tasks: belebele_tha_Thai, xnli_th, xcopa_th, thai_exam (custom YAMLs over scb10x/thai_exam, 5 subsets, choice-text loglikelihood, acc_norm reported), hellaswag, arc_challenge, winogrande (all 0-shot); mmlu 5-shot `--limit 50`/subject (deterministic first-50, identical docs both runs); wikitext + custom thai_wikipedia_ppl (`--limit 1000`, byte_perplexity/bits_per_byte — word-level PPL meaningless for Thai script). xnli_th/xcopa_th dataset paths patched to namespaced repos (facebook/xnli, cambridgeltl/xcopa) for datasets 5.0 compat.
- **Performance:** `vllm bench serve`, random dataset, `--ignore-eos` (forces identical output token counts), seed 0, grid (in,out,concurrency,prompts) = (1024,128,1,16), (1024,128,4,64), (128,1024,1,8), (128,1024,4,32); 1 discarded warmup + 3 scored repeats, median reported; 180 s cooldown after accuracy suite, 30 s between repeats.
- **Interpretation:** |Δacc| within ~1 stderr = noise. Same gpu-memory fraction gives NVFP4 a larger KV pool — a genuine deployment benefit, but not binding at max-num-seqs 4 / 8k ctx.
- **Tokenizer parity:** sha256 of tokenizer files verified identical between source and quantized checkpoint (`tokenizer_hashes_source.txt`).

