#!/usr/bin/env bash
# Performance benchmarks against the live eval server (run after run_suite.sh).
# Usage: run_perf.sh <bf16|nvfp4>
# Grid: (in,out,concurrency,prompts) x [1 discarded warmup + 3 scored repeats].
set -euo pipefail
source "$(dirname "$0")/common.sh"

TAG=${1:?usage: run_perf.sh <bf16|nvfp4>}
OUT=$P/results/$TAG/perf
mkdir -p "$OUT"

BENCH_BASE="vllm bench serve --backend openai --base-url http://127.0.0.1:$PORT \
  --model eval-model --tokenizer ThaiLLM/ThaiLLM-30B --dataset-name random \
  --request-rate inf --ignore-eos --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99"

echo "cooldown 180s before perf runs"; sleep 180

for cfg in "1024 128 1 16" "1024 128 4 64" "128 1024 1 8" "128 1024 4 32"; do
  read -r IN OUTLEN C N <<<"$cfg"
  echo "=== [$TAG] in=$IN out=$OUTLEN c=$C n=$N ==="
  free -g > "$OUT/mem_${IN}x${OUTLEN}_c${C}.txt"; nvidia-smi >> "$OUT/mem_${IN}x${OUTLEN}_c${C}.txt"
  # warmup (discarded)
  docker exec eval-vllm bash -c "$BENCH_BASE --random-input-len $IN --random-output-len $OUTLEN \
    --max-concurrency $C --num-prompts $(( N / 4 > 4 ? N / 4 : 4 ))" >/dev/null 2>&1 || true
  for r in 1 2 3; do
    docker exec eval-vllm bash -c "$BENCH_BASE --random-input-len $IN --random-output-len $OUTLEN \
      --max-concurrency $C --num-prompts $N \
      --save-result --result-dir /work/results/$TAG/perf \
      --result-filename ${IN}x${OUTLEN}_c${C}_r${r}.json" 2>&1 | tail -25 | tee "$OUT/${IN}x${OUTLEN}_c${C}_r${r}.log"
    sleep 30
  done
done
echo "=== [$TAG] perf done ==="
