# Perf files here: W4A4 logs, but NVFP4A16 JSONs

Added 2026-10-28. See also [`CORRECTIONS.md`](../../../../CORRECTIONS.md), item 3.

This directory belongs to arm 1 of the 7B gate (NVFP4 W4A4), but not every file
in it is W4A4. Nothing here has been renamed, moved or deleted, so existing
links and git history stay valid.

| Files | Contents | First committed in |
|---|---|---|
| `*.log` (12) | **W4A4 (arm 1).** The last 25 lines of each `vllm bench serve` run: mean, median, P90 and P99 of TTFT, TPOT, ITL and E2EL. The throughput lines were cut off by `tail -25`. | `f9085ff` (arm-1 commit, 2026-08-10 19:24 UTC) |
| `mem_*.txt` (4) | **W4A4 (arm 1).** `free -g` and `nvidia-smi` snapshots taken before each perf configuration. | `f9085ff` |
| `*.json` (12) | **NVFP4A16 (arm 2), not W4A4.** | `a2f89a0` (arm-2 commit, 2026-08-11 06:49 UTC) |

## Why the JSONs are NVFP4A16

- Each JSON matches the arm-2 log of the same name in
  [`results/otg7b_gate_w4a16/nvfp4/perf/`](../../../otg7b_gate_w4a16/nvfp4/perf/)
  on every printed latency statistic: mean, median (= P50), P90 and P99 of TTFT,
  TPOT, ITL and E2EL. Each also differs from the W4A4 log next to it in both
  median TTFT and median TPOT. Example, `128x1024_c1_r1`:
  - JSON: median TTFT 50.07 ms, median TPOT 23.92 ms;
  - arm-2 log: 50.07 ms and 23.92 ms;
  - W4A4 log here: 37.62 ms and 24.83 ms.
- Their `date` fields run from `20260811-034757` to `20260811-042145`, which is
  during the arm-2 run. The arm-1 BF16 JSONs in [`../../bf16/perf/`](../../bf16/perf/)
  are dated `20260810-161423` to `20260810-174245`.
- Most likely cause: until commit `a2f89a0`, `scripts/run_perf.sh` saved the
  JSONs inside the container to the July path `results/<tag>/perf/`, while the
  `.log` files went to the gate directory. Arm 2 ran only the `nvfp4` tag, so its
  JSONs replaced the arm-1 W4A4 JSONs in `results/nvfp4/perf/` before those were
  copied here.
- Arm 2 did not run BF16. That is why the BF16 JSONs are intact: all 12 match
  their logs on every printed statistic.

## What to use

- **W4A4 latency:** use the `.log` files here. The W4A4 JSONs were never
  committed and cannot be recovered. W4A4 throughput and the other JSON-only
  fields (run duration, token counts, standard deviations) were not kept.
- **NVFP4A16:** the JSONs here and the logs in `results/otg7b_gate_w4a16/nvfp4/perf/`
  describe the same runs.
- **The published speed figures are unaffected.** They agree with the `.log`
  files, not with these JSONs.
  - W4A4, single-stream 128→1024 decode: 40.3 tok/s = 1000 / 24.82 ms, the median
    TPOT of the three repeats.
  - These JSONs would give 41.8 tok/s instead, which is the NVFP4A16 figure.
