#!/usr/bin/env bash
# Route A part 2: the two long-form generative benches, re-run with configs
# sized for a 4.86 tok/s model. Root cause of the first ifeval-th attempt dying:
# max_tokens 8192 -> a cap-runner takes ~28 min -> client timeout -> 5 retries
# -> 6h wall for 116/215 and evalscope only writes reviews at the END.
# Fixes: --timeout per request, lower max_tokens (no-think model; only
# pathological loop answers get truncated), batch 4 (= server max-num-seqs).
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
EVAL=$P/.venv-chinda/bin/evalscope
URL=http://127.0.0.1:8014/v1
MODEL=openthaigpt-1.6-72b-instruct-NVFP4
OUT=$P/results/otg72b
mkdir -p "$OUT"

run () {  # $1 dataset  $2 limit  $3 max_tokens  $4 timeout_s
  local ds=$1 lim=$2 mt=$3 to=$4
  local log=$OUT/${ds}.log
  echo "### $(date +%H:%M:%S) $ds (limit $lim, max_tokens $mt, timeout $to)"
  (cd "$P/chinda-eval" && timeout 18000 $EVAL eval \
     --model "$MODEL" --api-url "$URL/chat/completions" --api-key EMPTY \
     --eval-type openai_api --datasets "$ds" --dataset-hub huggingface \
     --limit "$lim" --eval-batch-size 4 --timeout "$to" \
     --generation-config "{\"max_tokens\":$mt,\"temperature\":0.0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
     --work-dir "$OUT/$ds" > "$log" 2>&1)
  local n
  n=$(find "$OUT/$ds" -path "*reviews*" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
  echo "STAGE_DONE $ds reviews_rows=$n"
}

run ifeval-th 215 4096 2400
run code_switching 500 2048 1200

echo "SUITE2_COMPLETE $(date +%H:%M:%S)"
