# Corrections (2026-10-28)

This file corrects statements in this repository as of commit `ab0d71f`
(2026-08-12).

- No data file was changed, renamed or deleted.
- Directory names and the Hugging Face repo name are kept, so existing links keep
  working.
- Where a false sentence could be fixed with a small edit, it was fixed in place
  and marked "corrected" or "withdrawn" with the date. The table under
  [In-place edits](#in-place-edits) quotes the original wording.
- Everything else is recorded only here.

Line numbers refer to the corrected files.

## 1. ThaiLLM-30B-NVFP4 was made with TensorRT Model Optimizer, not llm-compressor

ThaiLLM-30B-NVFP4 was quantized with NVIDIA TensorRT Model Optimizer 0.43.0
(`hf_ptq.py --qformat nvfp4`) and served through vLLM's ModelOpt path. Sources:
- `scripts/quantize.sh:2`, `:21-22` and `:42-51`;
- `results/quant.log:1`;
- `results/methodology.md:5`;
- `results/report.md:36` and `:60`;
- `results/nvfp4/server.log:13` ("Detected ModelOpt NVFP4 checkpoint").

Only the 7B arms and the 72B used llm-compressor 0.12.0.1
(`scripts/lc_nvfp4_quantize.py`).

So statements that the 7B used the same llm-compressor recipe that shipped
ThaiLLM-30B-NVFP4 are wrong. The 30B and 7B W4A4 runs share the NVFP4 W4A4
format, the calibration set (per the plan; see item 7) and the machine. They
differ in:
- the quantization tool;
- the model: Qwen3-MoE base vs dense Qwen2 instruct;
- the serving build: NGC vLLM 0.21.0-NV vs `vllm/vllm-openai:v0.25.1`.

What changed:
- **Fixed in place:** the header and finding 1 of
  `results/otg7b_gate/FINDING_otg7b_w4a4_verdict.md`.
- **Withdrawn, with a bracketed note in place:** the model-size conclusions in that
  finding 1 ("W4A4 cost scales inversely with model size", "For 7B-class use W4A16
  (weight-only) or FP8") and in
  `results/otg7b_gate_w4a16/FINDING_otg7b_w4a16_verdict.md:37-41` ("4-bit weights
  are safe at 30B+"). They compare one paired model per size across tools, model
  families and metrics: the 30B figure (-0.53 pt) is ThaiExam-v2 log-likelihood
  only (its pooled multiple-choice loss was -0.81 pt, p < 1e-4,
  `results/paired_analysis.json`), the 7B figures are generative, and the 72B is
  unpaired.
- **Not edited:** `PREREGISTERED.md:8` calls the 30B study the "same recipe
  family". It shares the NVFP4 W4A4 format and the calibration set, not the tool.
  The pre-registration files are left exactly as committed.
- llm-compressor PR #3025 repeated the error ("Same recipe (NVFP4 W4A4 via
  `QuantizationModifier`)").

## 2. The 7B "W4A16" arm is NVFP4A16, not llm-compressor's integer W4A16

Arm 2 of the 7B gate is labelled "W4A16" in:
- `results/otg7b_gate_w4a16/` and `scripts/otg7b_gate_w4a16.sh`;
- the FINDING and RELEASE_SCOPE files;
- the card copy `results/otg7b_gate_w4a16/hf_card_README.md`;
- the Hugging Face repo `AGIcafet/openthaigpt1.5-7b-instruct-W4A16`.

It is llm-compressor's `NVFP4A16` scheme:

- **Format.** FP4 weights in groups of 16 with FP8 (E4M3) scales; activations
  stay BF16. Sources:
  - `scripts/lc_nvfp4_quantize.py:49-56`;
  - `hf_card_README.md:51-52`;
  - the published `config.json`: weights `type: float`, `num_bits: 4`,
    `group_size: 16`; `input_activations: null`; format `nvfp4-pack-quantized`.
- **Data-free.** The log shows "Inferred `DataFreePipeline` for
  `QuantizationModifier`" (`results/otg7b_gate/lc_7b_w4a16.log:8`).
  - The script still tokenized the 512 calibration rows, but no calibration step
    ran on them.
  - `hf_card_README.md:54` says "Calibration: none".
- **Serving.** vLLM served it weight-only with `MarlinNvFp4LinearKernel`
  (`results/otg7b_gate_w4a16/nvfp4/server.log:23,32`).

In llm-compressor, `W4A16` names an integer 4-bit weight preset, which is a
different format (llm-compressor docs, `guides/compression_schemes.md`, "W4A16
and W8A16"). That preset was not tested here, and neither were the GPTQ/AWQ
algorithms usually paired with it.

Read "W4A16" in this repository as NVFP4A16. llm-compressor PR #3025 used the
same label.

## 3. The perf JSONs in `results/otg7b_gate/nvfp4/perf/` are NVFP4A16, not W4A4

All 12 `*.json` files in that W4A4 directory hold NVFP4A16 (arm 2) measurements:
- each matches the arm-2 log of the same name in
  `results/otg7b_gate_w4a16/nvfp4/perf/` on every printed latency statistic
  (mean, median, P90 and P99 of TTFT, TPOT, ITL and E2EL);
- their `date` fields fall in the arm-2 run (`20260811-034757` to
  `20260811-042145`).

The arm-1 W4A4 JSONs were not kept. They were never committed and cannot be
recovered from git. The `.log` files and `mem_*.txt` snapshots in the same
directory are genuine arm-1 W4A4 files. The BF16 JSONs in
`results/otg7b_gate/bf16/perf/` match their logs.

The integrity note in `FINDING_otg7b_w4a16_verdict.md` said the opposite: that
the W4A4 JSONs were preserved and the arm-2 JSONs never saved.
- That note is corrected in place.
- `results/otg7b_gate/nvfp4/perf/README.md` now explains the directory.

The published speed figures agree with the `.log` files and are unaffected:
- W4A4: 40.3 tok/s single-stream decode;
- NVFP4A16: 41.8 tok/s.

## 4. `scripts/mcnemar_reviews.py` is not in this repository

`PREREGISTERED.md:24` names `scripts/mcnemar_reviews.py` for the generative
McNemar tests, but that script was never committed here. The published generative
results can be reproduced from the per-item review files at
`<bench>/<run>/reviews/eval-model/*.jsonl` under these directories:
- `results/otg7b_gate/gen_bf16/`;
- `results/otg7b_gate/gen_nvfp4/`;
- `results/otg7b_gate_w4a16/gen_nvfp4/`.

1. Take the per-item score (0 or 1) from `sample_score.score.value`:
   - `acc` for openthaieval, math_500-th and aime24-th;
   - `prompt_level_strict` for ifeval-th;
   - `Pass@1` for live_code_bench-th.
2. Pair BF16 and quantized rows on the SHA-256 of `input`. This is the
   "prompt-hash intersection" named in `PREREGISTERED.md:24`, and `index` gives
   the same pairing. The live_code_bench-th `.slim.jsonl` files truncate `input`
   to 300 characters, so pair those on `index`.
3. Let b = items only BF16 got right and c = items only the quantized model got
   right. Then
   `p = min(1, 2 * sum(C(b+c, i) for i in 0..min(b, c)) / 2^(b+c))`,
   the exact two-sided McNemar test.

Recomputed this way with an independent script (not the original), this
reproduces the published values in `FINDING_otg7b_w4a4_verdict.md` and
`FINDING_otg7b_w4a16_verdict.md`:

| Benchmark (n) | NVFP4 W4A4: Δ pt, b/c, p | NVFP4A16: Δ pt, b/c, p |
|---|---|---|
| openthaieval (1,232) | −2.35, 89/60, 0.0215 | −0.73, 49/40, 0.397 |
| ifeval-th prompt-strict (215) | −10.23, 32/10, 0.00094 | −3.72, 19/11, 0.2005 |
| math_500-th (500) | −5.60, 67/39, 0.0084 | −5.40, 62/35, 0.0080 |
| live_code_bench-th Pass@1 (111) | −1.80, 6/4, 0.754 | 0.00, 5/5, 1 |
| aime24-th (30) | 0.00, 0/0, 1 | 0.00, 1/1, 1 |

## 5. The 7B pre-registration dates cannot be checked from git

`PREREGISTERED.md:3` says "Written 2026-08-10, BEFORE any measurement", and line
52 says arm 2 was pre-registered 2026-08-11 "BEFORE quantizing". The same claim
appears in:
- `FINDING_otg7b_w4a4_verdict.md:3`;
- `FINDING_otg7b_w4a16_verdict.md:3` and `:22`;
- `RELEASE_SCOPE.md:21`;
- `hf_card_README.md:66`;
- `scripts/otg7b_gate.sh:3`.

Git does not confirm these dates:
- `results/otg7b_gate/PREREGISTERED.md` first appears in commit `f9085ff`
  (2026-08-10 19:24 UTC), which also adds the arm-1 results and the REJECT
  verdict.
- The arm-2 section, and the identical copy
  `results/otg7b_gate_w4a16/PREREGISTERED.md`, first appear in `a2f89a0`
  (2026-08-11 06:49 UTC), together with the arm-2 results and verdict.
- Both commits come after the quantization runs they govern:
  - `results/otg7b_gate/lc_7b.log`, 2026-08-10T08:31;
  - `results/otg7b_gate/lc_7b_w4a16.log`, 2026-08-11T01:58.
- The arm-1 text was not changed after `f9085ff`; `a2f89a0` only appended the
  arm-2 section.

The dates are therefore self-reported. Read them as "decision criteria written
before the runs, committed together with the results".

Arm 2 was designed after arm 1 had failed: its section quotes arm-1 results
(`PREREGISTERED.md:66`).

## 6. The ThaiLLM-30B study was not pre-registered

No pre-registration exists for the July 2026 ThaiLLM-30B study (`README.md`,
`results/report.md`, `results/methodology.md`), and those documents do not
claim one. In the whole git history, the only pre-registration files are:
- `results/otg72b/PREREGISTERED.md`, for the 72B rescore round;
- the two 7B copies (item 5).

The 30B report has a fairness contract and a deviation ledger
(`results/report.md:39`, `:48`), but neither fixes decision criteria in
advance. The 7B pre-registration takes its expectations from the 30B results,
which does not make the 30B study pre-registered.

llm-compressor PR #3025 said "All runs were pre-registered before measurement".
That is wrong for the 30B. For the 7B it rests on the self-reported dates in
item 5.

## 7. Calibration data: the English half was CNN/DailyMail, and the language effect was not tested

The only calibration set in this repository is `calib/thai_en_calib.jsonl`,
built by `scripts/build_calib.py` with seed 0. It contains:
- 256 Thai Wikipedia documents (`wikimedia/wikipedia` `20231101.th`, train;
  `build_calib.py:21`);
- 256 English news articles from `abisee/cnn_dailymail` `3.0.0` (train;
  `build_calib.py:27`), not UltraChat.

Each document is at least 2,000 characters long, truncated to 4,000.

The file has 512 rows, 256 of which contain Thai script (checked when this file
was written; sha256
`af5b55f2bc80490b995e9cea2ee3298dad17850b7cdac968b070aaaa88d8ced2`).

The calibration snippet in llm-compressor PR #3025 used
`HuggingFaceH4/ultrachat_200k` for the English half. That is not what was
measured.

Only the W4A4 runs used this set; the 7B NVFP4A16 arm used no calibration data
(item 2). For the 7B W4A4 run:
- the `CALIB` path (a required variable) and `SEQ` (default 512) used for the run
  are not recorded in any committed script;
- `PREREGISTERED.md:6` ("half-Thai 512x512") and the 512/512 calibration steps in
  `results/otg7b_gate/lc_7b.log` are consistent with this file.

No other calibration set (English-only, Thai-only or a different mix) was ever
run, so the effect of calibration language was not measured. Read the
explanations in `README.md:31` ("the likely reason Thai survived fully") and
`results/report.md:103` as untested hypotheses.

## 8. 19 of the 215 IFEval-TH prompts were never graded

In every 7B ifeval-th run (BF16, NVFP4 W4A4 and NVFP4A16), evalscope logged "Error
in instruction following evaluation" with "Resource 'punkt_tab' not found" 19
times (`results/otg7b_gate*/gen_*/ifeval-th/<run>/logs/eval_log.log`). The 19
prompts, which hold 37 of the 306 instructions, are exactly those with a
`length_constraints:number_sentences` instruction. Their review rows have an
empty `strict_results` list and score 0 on every metric, for every model.

- Paired results are unaffected: these items are ties, so b/c and every McNemar
  p-value stay the same.
- Accuracies are understated by the same 19 items on every side. Over the 196
  graded prompts, prompt-level strict is 0.6888 (BF16), 0.5765 (W4A4) and 0.6480
  (NVFP4A16), and the deltas are -11.22 and -4.08 points instead of -10.23 and
  -3.72.
- Instruction-level figures taken from `strict_results` cover the 269 graded
  instructions only. evalscope's `mean_inst_level_strict` (69.30, 59.61 and 64.26)
  averages per prompt over all 215, counting the 19 as 0.
- The 72B ifeval-th run that has per-item reviews
  (`results/otg72b/ifeval-th/20260802_090238`) has the same 19 ungraded prompts.

## 9. Tool-calling conformance ran 8 calls per model, not 12

`PREREGISTERED.md:27`, `FINDING_otg7b_w4a4_verdict.md:31` and
`hf_card_README.md:79` list tool conformance as n = 12. The gate scripts ran
`scripts/thai_tool_conformance.py --modes auto,required`
(`scripts/otg7b_gate.sh:77`, `scripts/otg7b_gate_w4a16.sh:64`): four cases in two
modes, 8 calls per model, all passing. The script's third default mode, `named`,
was not run.

## 10. Per-item lm-eval samples are not published

`FINDING_otg7b_w4a4_verdict.md:67` lists "lm-eval samples" among the
artifacts, but `samples_*.jsonl` is excluded by `.gitignore:12-13`, so none were
committed. The log-likelihood McNemar counts can be checked only through
`paired_analysis.json`, whose per-task counts imply accuracies equal to the
committed `results_*.json`.

## In-place edits

| File | Original wording | Now |
|---|---|---|
| `results/otg7b_gate/FINDING_otg7b_w4a4_verdict.md`, header | "same llm-compressor 0.12.0.1 NVFP4 W4A4 recipe + half-Thai 512x512 calibration that shipped ThaiLLM-30B-NVFP4 and the 72B." | the llm-compressor recipe that shipped the 72B, with the same calibration set as ThaiLLM-30B-NVFP4, which was quantized with TensorRT Model Optimizer 0.43.0 |
| same file, finding 1 | "Same recipe, same box:" | "Same box and calibration set, but not the same tool (30B: TensorRT Model Optimizer 0.43.0; 72B and 7B: llm-compressor 0.12.0.1)" |
| same file, end of finding 1 | (none) | added "[Withdrawn ...: this size conclusion and the 7B-class advice are not supported by this data ...]" |
| `results/otg7b_gate_w4a16/FINDING_otg7b_w4a16_verdict.md`, mechanism bullet 3 | (none) | added "[Withdrawn ...: not supported by this data ...]" |
| `results/otg7b_gate_w4a16/FINDING_otg7b_w4a16_verdict.md`, integrity note | "7B JSONs preserved under the gate roots (arm-1 bf16 + W4A4; arm-2 W4A16 JSONs were never saved - its full printed metric tables live in results/otg7b_gate_w4a16/nvfp4/perf/*.log)" | the JSONs in `results/otg7b_gate/nvfp4/perf/` are arm-2 NVFP4A16; W4A4 survives as the `.log` files there |
| `README.md` | (none) | added a one-line pointer to this file |

New file: `results/otg7b_gate/nvfp4/perf/README.md`.
