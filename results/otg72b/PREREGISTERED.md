# Pre-registered expectations & pass criteria (written BEFORE measuring)

Registered 2026-08-05, before any of the runs below started. Every outcome
gets published regardless of direction; deviations from these expectations are
findings, not embarrassments to hide.

## 1. MATH500-TH re-score (fixed double-extraction, no new generation)
- Expectation: corrected accuracy = (246 + ~72) / 500 ≈ 0.636. Mechanically it
  can only be >= 0.492 (the fix only rescues false negatives).
- Sanity gates: corrected >= raw; rescued rows must have
  extracted_prediction textually/symbolically equal to gold; spot-check 10
  rescued rows by hand.
- Comparability rule: the corrected number is NOT comparable to the card's
  43.2 (scored by the buggy grader). The 49.2-vs-43.2 same-grader comparison
  remains the only fair one until the card is re-scored.

## 2. LCB-TH re-run (hidden tests decoded, all-or-nothing scoring)
- Expectation: result <= 0.4505 (adding 2,337 hidden tests and removing
  partial credit can only lower the score). If it comes out HIGHER, that is a
  bug in our fix, not a better model.
- No target: whatever it is, it is the number. Comparison to the card's
  36/111 becomes legitimate-with-caveat (their harness's hidden-test behavior
  is still unverified).

## 3. AIME24-TH avg@8 (temperature 0.6, seed set 0..7)
- Greedy is deterministic - re-running it "multiple times" is theater. The
  meaningful repeat is sampling avg@8 (community standard for AIME).
- Expectation: mean in the 0.05-0.20 band (greedy got 4/30). Report
  mean +- 95% CI; NO superiority claim vs the card at any outcome (n=30).

## 4. Speed levers (each vs the pre-registered control)
- Control: official bench 5.01 tok/s +-0.2% chat (evidence 20260803T045000Z).
- Protocol per arm: 2 boots, rep-1 discard, same vendored harness. Publish
  slowdowns as prominently as speedups.
- n-gram speculative decoding: adopt into the recipe only if chat or rag
  decode >= 1.15x control with boot spread < 5%. Expectation: rag benefits
  most (prompt-lookup hits), chat maybe none.
- KV cache fp8: adopt-candidate only if concurrent throughput (c4) >= 1.5x
  with single-stream regression < 5%. Accuracy flag stays
  "quality-pending-gate" until a Thai gate passes - speed alone ships nothing.
- Batch sweep (--max-num-seqs 4/8/16, c1..c16): descriptive, no gate.
  Expectation: near-linear aggregate scaling to ~8 then flattening.

## Anti-bias rules binding all of the above
- Thresholds above were fixed before any measurement; they do not move after.
- Every number ships with its artifact path; every surprising-GOOD number
  triggers the adapter/extraction audit rule before publication.
- One GPU job at a time; kill only by PID; monitors watch process liveness.
