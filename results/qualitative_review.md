# Qualitative review: BF16 vs NVFP4 generations (Thai-fluent reviewer, 12 domains)

| # | Section | Verdict |
|---|---------|---------|
| 1 | th_factual_qa | Equivalent — both factually correct (77 provinces etc.) |
| 2 | th_news | Equivalent — both natural, coherent Thai central-bank prose |
| 3 | th_legal | **BF16 better** — BF16 quotes CCC §420 verbatim; NVFP4 misquotes "บุคคลอื่น" as "บุคคลภายนอก", drops compensation clause |
| 4 | th_medical | Equivalent — both medically sound |
| 5 | th_education | Equivalent — both correct photosynthesis |
| 6 | th_business | Equivalent (NVFP4 marginally more varied) |
| 7 | th_travel | Equivalent — BOTH degenerate into repetition loops (greedy base-model artifact) |
| 8 | th_math | **BF16 better** — both get 250 baht right; NVFP4's "teaching method" arithmetic incoherent (5×10+2×50≠250) |
| 9 | en_econ | NVFP4 slightly better — richer structure before looping |
| 10 | en_science | Equivalent — both high-quality |
| 11 | code_python | Equivalent — NVFP4 better code (canonical Fibonacci), BF16 no repetition |
| 12 | th_en_translate | **BF16 better** — NVFP4 shows word-level stutter ("ความยั่งยืน" หรือ "ความยั่งยืน") |

**Overall: no systematic quality degradation from NVFP4.** Thai orthography fully intact in both (no broken tone marks, no mojibake). Repetition loops and self-QA chains appear in BOTH versions — base-model + greedy artifacts, roughly balanced. NVFP4 shows a small cluster of localized coherence slips (statutory wording, arithmetic sub-explanation, word redundancy) that BF16 avoids, and is cleaner in others. Net: within normal greedy-divergence variation; caution advised for legal/precision-quoting use cases.

Notable: th_legal — BF16: "…ทำต่อบุคคลอื่นโดยผิดกฎหมาย…" (exact §420 wording); NVFP4: "…ทำต่อบุคคลภายนอก…" (wrong statutory term). Concrete instance of the Marchisio et al. finding that human review catches degradation automatic metrics miss.
