# Finding: chinda-eval Thai adapter scoring defects (audit 2026-08-05)

Trigger: our NVFP4 scores exceeded the developer's BF16 card anchors — a red
flag the owner raised ("bias/hallucinated?"). Four-probe forensic audit
(adapter source, manual regrades, arXiv protocol, statistics). Verdict per gap:

## LiveCodeBench-TH (+22 pt) — measurement artifact, NOT model quality
- Card's 32.43 = EXACTLY 36/111 → both sides use the identical
  `iapp/code_generation_lite-th` 111-problem set (contest window
  2024-03-09..2024-05-25). Earlier "subset may differ" caveat was wrong.
- `live_code_bench_th_adapter.py:74-94`: json.loads-only test parsing SILENTLY
  DROPS all 2,337 base64+zlib+pickle hidden tests (mean 21.1/problem). The
  ENGLISH adapter in the same repo decodes them correctly
  (`live_code_bench/load_utils.py:31-36`) — a real Thai-adapter defect.
- Scoring is per-problem partial credit over ~2.5 public sample tests, with
  any-line lenient output match; official protocol is all-or-nothing.
- Honest numbers from our run: partial-credit mean 0.5447; STRICT
  all-public-tests-pass 50/111 = 0.4505 (now the headline). True
  hidden-test-inclusive score unknown until the adapter is fixed and re-run.
- Upstream fix candidates: mirror load_utils decode; all-pass scoring.
  (Candidate contribution to iapp/chinda-eval.)

## MATH500-TH (+6 pt) — legitimate comparison, NOT significant
- Same 500 items both sides (43.2 = 216/500 vs our 246/500). Developer's own
  chinda-eval scripts are greedy too. Fisher p = 0.066 → not significant.
- Grader is too STRICT, not lenient: metric.py:45 double-extraction voids
  72/500 symbolically-correct answers (\frac{14}{3} etc.). Both sides suffer
  the same bug if scored by the same family → comparison fair; absolute level
  understated for everyone. Manual regrade of 10 credited rows: all legitimate.
- Config deltas (our 4096 cap, thinking-off) bias OUR score DOWN.

## AIME24-TH (+6.7 pt) — pure noise
- 4/30 vs 2/30, Fisher p = 0.67. No claim possible at n=30.

## Card action taken (commit on HF)
LCB headline replaced with strict 45.05 + full scoring note; MATH500 marked
p=0.066 non-significant; AIME marked statistically indistinguishable.

Probes + synthesis JSON: workflow wf_808e3d46-095 (session transcript dir).
