# RELEASE SCOPE: AGIcafet/openthaigpt1.5-7b-instruct-W4A16

Authorized by the owner 2026-08-11 ("released ได้เลย เขียนผล benchmark ทั้งหมด").
This is a USE-CASE-SCOPED release of an artifact whose all-axes gate verdict
was REJECT (FINDING_otg7b_w4a16_verdict.md). No new measurement; every number
comes from the completed pre-registered paired gate.

## Scope
- **Released for**: Thai chat, instruction following, RAG, tool calling —
  every paired axis in these areas is statistically indistinguishable from
  BF16 (openthaieval -0.73 p=.40, ifeval-th -3.7 p=.20, lcb 0.00 p=1,
  thai_mc -0.28 p=.65, tools 8/8=8/8), at 3.4x the speed and 1/2.8 the size.
- **NOT for mathematical reasoning**: math_500-th -5.4pt, p=0.008,
  significant. Mechanism (isolated by the two-arm design): FP4 weight
  precision compounds over long reasoning chains. Stated as the first
  warning block on the card.
- The W4A4 arm is NOT released (failed instruction/knowledge axes too);
  its numbers appear on the card as reference so the choice is inspectable.

## Publication basis
Both arms fully pre-registered before measurement; every card number links
to artifacts in this repo (results/otg7b_gate/, results/otg7b_gate_w4a16/).
The publication promise in PREREGISTERED.md is honored: all outcomes shown,
including the rejected arm and the failed axis of the released one.
