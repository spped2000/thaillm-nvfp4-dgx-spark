All six angles are now covered with cross-verified numbers. Synthesizing the final report.

# Quantization Effects on Thai / Non-Latin-Script Languages in LLMs — Research Report for Interpreting BF16 vs NVFP4 on ThaiLLM-30B (Qwen3-30B-A3B Thai CPT)

**Verification protocol:** Claims labeled **[V]** are verified against the primary source (paper full text or official model card, fetched directly). Claims labeled **[S]** come from search-result snippets of the primary source (high confidence, not independently re-fetched). Claims labeled **[I]** are my inference — clearly separated at the end of each section.

---

## 1. Marchisio et al. 2024, "How Does Quantization Affect Multilingual LLMs?" (arXiv:2407.03211, EMNLP 2024 Findings)

### Setup [V]
- **Models:** Command R+ (103B), Command R (35B), Aya 23 (35B and 8B) — all Cohere multilingual dense models.
- **Quantization:** weight-only **W8** (per-column INT8) and **W4-g** (group-wise GPTQ, group size 128); weight+activation **W8A8** and **W8A8-SmoothQuant**; for Aya models, bitsandbytes 8-bit and 4-bit (**NF4** quantile quantization).
- **Languages:** 10 primary (Arabic, French, German, English, Spanish, Italian, Portuguese, Korean, Japanese, Chinese), up to 23 in extended evals.
- **Automatic tasks:** mMMLU, MGSM, FLORES-200, Language Confusion, XWinograd, XCOPA, XStoryCloze, Belebele.
- **Human eval:** Spanish, French, Korean, Japanese; 150 prompts each from an internal challenging suite (human-translated) and Aya Dolly-200; pairwise FP16-vs-quantized win-rates.

### Key numbers [V]
- **Script gap:** "Across tasks, Latin-script languages scored −0.7% relative to FP16 for a 103B parameter model while non-Latin scripts scored −1.9%." (W4-g, aggregated over mMMLU/FLORES/Language Confusion.)
- **Automatic-vs-human gap:** "a 1.7% average drop in Japanese across automatic tasks corresponds to a 16.0% drop reported by human evaluators on realistic prompts." Per-language human-eval drops for W4-g on 103B (internal challenging set): Japanese −16.0%, French −16.6%, Spanish −4.6%, Korean −4.6%; average −10.5%. Note the gap is not script-specific — French (Latin) showed −16.6% human vs −0.3% automatic.
- **Task ordering — what dies first:** mathematical reasoning (MGSM), then challenging open-ended generation, then LLM-as-a-Judge: "Mathematical reasoning (−13.1%), performance on real-world challenging prompts (−10.5%), and LLM-as-a-Judge (−25.9%) results are severely degraded."
- **MGSM detail:** 35B W4-g averages **−13.1%** relative, worst **−17.3% in Chinese**; 103B W4-g only −2.9% on MGSM (i.e., −0.9% overall) — smaller models degrade far more on hard tasks.
- **Quantization sometimes helps:** "an average 1.3% boost across tasks for a 35B model quantized with W8A8."
- **Aya models:** MGSM and Belebele are the most degraded tasks under W4 (bnb NF4). [S]

### Applicability limits to NVFP4 [I]
- Their worst 4-bit results come from **INT4 GPTQ g128** (uniform integer grid, one FP16 scale per 128 weights) and **NF4**. NVFP4 is a materially finer format: E2M1 values in **blocks of 16** with **FP8 E4M3 block scales** plus an FP32 per-tensor scale — 8x smaller blocks and a non-uniform grid, which empirically tracks outliers better. Marchisio-scale degradation should be treated as a **loose upper bound** for NVFP4 weight error, not a prediction.
- However, NVFP4 (as deployed with modelopt NVFP4_DEFAULT_CFG) also quantizes **activations** to 4-bit, which Marchisio never tested (their lowest activation precision was A8). So the error composition is different in both directions; the paper's directional findings (non-Latin scripts and generative/reasoning tasks degrade first; automatic MC benchmarks understate generative harm) transfer better than its magnitudes.

---

## 2. GGUF K-quant multilingual study (arXiv:2503.03592, Borgersen & Goodwin, "English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance")

### Findings [V]
- **Setup:** Llama 3.3 70B, GGUF importance-matrix quantization at **Q4_K_S (~3.5x compression), Q3_K_S (~4.6x), Q2_K_S (~5.8x)** vs FP16; imatrix computed on **English, Norwegian (GPT-4o-translated), and Malayalam** text; evaluated on **MixEval in English and Norwegian**.
- **Result: null.** "No p-value is calculated to be lower than 0.2373 for the multiple choice dataset"; free-form differences that initially looked significant failed Bonferroni correction (adjusted alpha = 0.00417). Conclusion: imatrix language does not measurably shift English/Norwegian downstream scores, even at Q2/Q3.
- **Author-noted limitations:** single model, single quant family, machine-translated Norwegian, single-token MC responses "may artificially favor weaker quantized models," no human eval (they explicitly cite Marchisio's ~13% human-detected drop as what their protocol could miss).

### Relevance [I]
- This is calibration-**language** insensitivity for a llama.cpp-style importance weighting, on a Latin-script pair; it says little about Thai script and nothing about activation quantization. Its main value for you: it is the strongest published evidence that **weight-error weighting from an English-only proxy corpus is not automatically catastrophic for a second language**.

---

## 3. "Calibrating Beyond English" (arXiv:2601.18306, Chimoto, Elhoushi, Bassett — EACL 2026)

### Findings [V]
- **Setup:** Llama 3.1 8B and Qwen2.5 7B (+ BLOOMZ-7B1-MT robustness check); **W4A16 only, group size 128**, GPTQ and AWQ (Any4 as supplementary). 10 languages (English, French, Swahili, Chinese, isiXhosa, Sotho, Zulu, Yoruba, Igbo, Hausa); 8 calibration settings (5 monolingual + 3 multilingual mixes). **Thai is not included.**
- **Does it cover W4A4?** **No.** Weight-only quantization only; no activation quantization anywhere in the study.
- **Effect sizes:** non-English/multilingual calibration beats English-only; largest gain **−3.52 perplexity points** (multilingual mix, GPTQ, Llama 3.1 8B). Language-matched calibration gives the biggest per-language gains (e.g., isiXhosa calibration: −2.282 ppl on isiXhosa, with within-family transfer to Zulu/Sotho).
- **Method asymmetry:** **GPTQ is far more calibration-language-sensitive** (swings up to 3.52 ppl; Hessian second-order statistics depend on calibration data) vs **AWQ robust** (max ~0.35 ppl; activation-aware scaling keeps channel selection stable, only magnitudes shift).
- **Mechanism:** multilingual calibration sets have larger unique vocabularies and capture **broader activation tails**, reducing clipping error — degradation in bad language-quantizer pairs traced to activation-range distribution differences.
- **Downstream:** XNLI, XStoryCloze, Global MMLU confirm that evaluation-language-aligned calibration yields significant gains on that language.

### Relevance to your NVFP4 calibration [I]
- The mechanism that matters here (tail/outlier coverage of activations) is exactly the one your calibration touches — see Section 6. But the *magnitude* of sensitivity is method-dependent, and NVFP4 max-calibration sits at the **insensitive** end of the spectrum (far less data-dependent than GPTQ, less than AWQ). Your 256 Thai-wiki + 256 cnn_dailymail mix is directionally what this paper recommends (target-language + English mix).

---

## 4. SEA-LION quantization reports (AI Singapore) — closest published analog to your setup

### Verified numbers [V]
- **Gemma-SEA-LION-v4-27B-IT-NVFP4** (model card): "has little degradation (**<0.5%**) in performance compared to Gemma-SEA-LION-v4-27B-IT"; built with **vllm-project/llm-compressor**; on H100: **17.3 GB VRAM vs 51.4 GB** full precision, **62.43 vs 28.57 tok/s**. Same **<0.5%** claim on the **FP8-Dynamic** card (calibration-free dynamic activation scaling; no calibration set disclosed for either).
- **Qwen-SEA-LION-v4-32B-IT-4BIT** (model card): **GPTQ** 4-bit (8BIT = GPTQ 8-bit); "little degradation (**<0.3%** on average) in performance compared to Qwen-SEA-LION-v4-32B-IT."
- **Llama-SEA-LION-v3.5-70B-R-NVFP4** exists (13 languages incl. Thai) but the card publishes **no** degradation number or calibration details.
- **Per-language (Thai) deltas are NOT published in any card** — all cards defer to the SEA-HELM leaderboard (leaderboard.sea-lion.ai), which serves scores dynamically (not extractable here). The <0.5%/<0.3% figures are SEA-HELM *averages across SEA languages*, which include Thai in the eval set — but no Thai-specific quantization delta is public.
- **SEA-LION v4.5** (May 2026; Gemma-4-E2B-based E2B and Qwen3.6-27B-based 27B): GGUF variant confirmed (Gemma-SEA-LION-v4.5-E2B-IT-GGUF); **no NVFP4/FP8 SEA-HELM delta report for v4.5 found** — the <0.5% quantization claim in v4.5-adjacent search results refers back to the v4 27B model. [V for existence; absence of v4.5 NVFP4 numbers verified by fetching docs.sea-lion.ai/models/sea-lion-v4.5 and the sealion GitHub model doc, which contain no quantized-variant eval tables.]

### Relevance [I]
- This is the **best available precedent** for your exact scenario: a SEA-language (Thai included) instruction model, quantized to NVFP4 with the same general tool family, evaluated on a Thai-containing multilingual benchmark, showing **sub-1% average degradation**. Caveat: SEA-HELM is largely automatic/structured evaluation, precisely the kind Marchisio showed can understate generative degradation by ~10x.

---

## 5. Thai-specific quantization studies (2025–2026)

### What exists [V]
- **arXiv:2410.17145** ("Can General-Purpose LLMs Generalize to English-Thai Machine Translation?"): Llama-3 8B under **GPTQ at 8/4/3/2-bit** on English→Thai MT. At 4-bit: BLEU3 0.173 → 0.156, METEOR 0.371 → 0.349 (SCB dataset) — roughly a **6–10% relative drop**, with collapse at 3/2-bit; conclusion: "under more strict computational constraints, such as 4-bit quantization, LLMs fail to translate effectively" (relative to NLLB-600M, which used 10.81x less VRAM). This is the only peer-reviewed Thai-specific quantization-degradation measurement found.
- **OpenThaiGPT 1.5** (7B: 52.04% ThaiExam; 14B: 58.41% ThaiExam) publishes GGUF/community quantizations but **no ThaiExam deltas for the quantized variants**. [S]
- **Typhoon (SCB10X):** TheBloke GGUF/AWQ quants of Typhoon-7B exist; **no published Thai-benchmark deltas under quantization** from SCB10X found. Same for **KBTG THaLLE, NECTEC Pathumma/OpenThaiLLM, WangchanX**: no quantization-eval publications found. [V as absence, after 3 targeted searches]

### Bottom line [I]
- **There is no published ThaiExam/Belebele-TH-under-quantization study.** Your BF16-vs-NVFP4 ThaiLLM-30B comparison would be, to the best of this search, the first systematic Thai-centric NVFP4 evaluation — the nearest comparables are SEA-HELM aggregate deltas (Section 4) and the GPTQ-INT4 MT result above (which is a much cruder format on a much smaller dense model).

---

## 6. NVFP4 accuracy-recovery literature

### Format and recovery claims [V]
- **Format** (NVIDIA dev blog): E2M1 values, **16-value micro-blocks with FP8 E4M3 scales**, second-level **FP32 per-tensor scale**; ~3.5x memory vs FP16, 1.8x vs FP8. DeepSeek-R1-0528 FP8→NVFP4: MMLU-Pro 85→84, GPQA-Diamond 81→80, MATH-500 98→98, AIME-2024 89→91; claim of "1% or less accuracy degradation on key language modeling tasks."
- **Red Hat Developer (Feb 2026), "Accelerating LLMs with NVFP4":** large models (70B–235B) "consistently achieve ~99% recovery"; **mid-size (~30B): 97–99% recovery**; small (7B–14B): ~95–98% with task variability; **MoE models (Llama-4 Scout/Maverick, Qwen3-235B-A22B) exceed 99% recovery** — "exceptionally strong robustness due to NVFP4's expressive range." **The article contains no multilingual or non-English evaluation data.**
- **RedHatAI/Qwen3-30B-A3B-NVFP4 model card — your exact base architecture:** calibrated on **512 UltraChat samples (English), 2048 tokens, minmax observer**. BF16→NVFP4: MMLU 79.51→77.54 (97.52%), GSM8K 89.46→87.72 (98.05%), ARC-C 67.15→64.59 (96.19%), HellaSwag 77.55→76.74, WinoGrande 72.30→70.80, TruthfulQA 53.50→54.14 (101.2%); **OpenLLM-v1 average 73.25→71.92 = 98.19% recovery; OpenLLM-v2 average 52.78→48.99 = 92.81% recovery** (harder reasoning suites lose more — consistent with Marchisio's task ordering); HumanEval64 pass@2 93.62→91.13.
- **MoE-specific:** SharQ (arXiv:2606.26587) reports on Qwen3-30B-A3B that its method lifts FP4 average accuracy 68.40→69.17 and MMLU 77.72→78.79, i.e., plain 4-bit FP leaves ~1-point headroom on this MoE. [S] NVIDIA's QAD report (arXiv:2601.20088) shows PTQ alone can be insufficient for heavily post-trained (SFT/RL/merged) models, with quantization-aware **distillation** recovering near-BF16; robustness "to data quality and coverage." [V, abstract]
- **Calibration semantics of modelopt NVFP4_DEFAULT_CFG:** weight block scales are computed from the weights themselves; **activation per-block (16) scales are dynamic at runtime; calibration only pins the per-tensor FP32 global scales** via max/amax statistics. [S, consistent across modelopt docs/issues]

### Effect of Thai calibration on NVFP4 specifically [V as absence + I]
- **No published study of calibration-language effects on NVFP4 exists** (searched; nothing found). [V as absence]
- **[I]** Under max/amax calibration, the calibration set influences exactly **one scalar per activation tensor**. Your 256 Thai-wiki samples ensure Thai-token activation outliers are represented in those amax values — which is precisely the "activation tail coverage" mechanism Chimoto et al. identify as the failure mode of English-only calibration. Because NVFP4's per-block activation scales are dynamic anyway, residual calibration-language risk is confined to global-scale clipping, and your Thai+English 50/50 mix largely neutralizes it. Expect **much lower** calibration sensitivity than the GPTQ numbers in Section 3 (which involve Hessian estimation, absent here). The one asymmetry vs the RedHat card: your 512x512-token samples are shorter than their 2048-token samples; amax statistics are usually saturated by 512 samples, but long-context Thai activation outliers are the one thing your calibration would not have seen.

---

## Synthesis: what to expect / how to read your BF16-vs-NVFP4 delta [I — all inference]

1. **Expected magnitude:** For a Qwen3-30B-A3B derivative on automatic benchmarks, the literature centers on **~1–2 points absolute / 97–99% recovery** (RedHat card on the identical architecture: 98.2% on OpenLLM-v1, 92.8% on the harder v2), with MoE architectures at the robust end. Thai deltas up to ~2x the English delta would be consistent with Marchisio's non-Latin-script factor (−1.9% vs −0.7%); deltas beyond ~3–4 points on Thai MC benchmarks would be anomalous for NVFP4 and worth investigating (tokenizer-boundary effects, router sensitivity, or a calibration/global-scale problem).
2. **Task ordering to check:** Thai math/reasoning (MGSM-TH-style) and long-form generation should degrade first; logit-based MC (ThaiExam, Belebele-TH) last. If your MC deltas are near zero, that does **not** certify generative quality: the 1.7%-automatic vs 16%-human Japanese gap is the single most transferable finding in this literature. Add at least a small human or LLM-judged Thai generation eval before declaring parity.
3. **CPT caveat:** "Low-Bit Quantization Favors Undertrained LLMs" (arXiv:2411.17691) implies heavily-trained models suffer *more* quantization-induced degradation; Thai CPT further specializes weights toward Thai distributions that the NVFP4 grid must now represent — a mild reason to expect your Thai delta to exceed RedHatAI's English delta on the same architecture.
4. **Your calibration choice is defensible:** it matches the multilingual-mix recommendation of Chimoto et al., and NVFP4 max-calibration is the least calibration-sensitive mainstream PTQ path. If Thai degradation is nonetheless concentrated in generation quality, the literature's remedy is QAD/distillation (arXiv:2601.20088), not calibration-set tuning.

## Sources
- https://arxiv.org/abs/2407.03211 (+ ar5iv full text) — Marchisio et al., EMNLP 2024
- https://arxiv.org/abs/2503.03592 — Borgersen & Goodwin, GGUF K-quant
- https://arxiv.org/abs/2601.18306 — Chimoto et al., Calibrating Beyond English (EACL 2026)
- https://huggingface.co/aisingapore/Gemma-SEA-LION-v4-27B-IT-NVFP4 / -FP8-Dynamic / Qwen-SEA-LION-v4-32B-IT-4BIT / Llama-SEA-LION-v3.5-70B-R-NVFP4; https://docs.sea-lion.ai/models/sea-lion-v4.5
- https://arxiv.org/html/2410.17145v1 — English-Thai MT under GPTQ
- https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization; https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/; https://huggingface.co/RedHatAI/Qwen3-30B-A3B-NVFP4; https://arxiv.org/abs/2601.20088 (QAD); https://arxiv.org/pdf/2606.26587 (SharQ); https://arxiv.org/abs/2411.17691 (QiD scaling laws)