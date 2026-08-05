#!/usr/bin/env bash
# Route A part 3: the math/code fill-ins for the HF card comparison table.
# Same timeout discipline as suite2 (a ~5 tok/s model + default configs =
# retry spiral; evalscope writes reviews only at the END of a run).
# Datasets are NOT cached - they download from HF on first use.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
EVAL=$P/.venv-chinda/bin/evalscope
URL=http://127.0.0.1:8014/v1
MODEL=openthaigpt-1.6-72b-instruct-NVFP4
OUT=$P/results/otg72b
mkdir -p "$OUT"

run () {  # $1 dataset  $2 limit(or "full")  $3 max_tokens  $4 timeout_s  $5 wall_cap_s
  local ds=$1 lim=$2 mt=$3 to=$4 cap=$5
  local log=$OUT/${ds}.log
  local lim_args=()
  [ "$lim" != "full" ] && lim_args=(--limit "$lim")
  echo "### $(date +%H:%M:%S) $ds (limit $lim, max_tokens $mt, timeout $to)"
  (cd "$P/chinda-eval" && timeout "$cap" $EVAL eval \
     --model "$MODEL" --api-url "$URL/chat/completions" --api-key EMPTY \
     --eval-type openai_api --datasets "$ds" --dataset-hub huggingface \
     "${lim_args[@]}" --eval-batch-size 4 --timeout "$to" \
     --generation-config "{\"max_tokens\":$mt,\"temperature\":0.0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
     --work-dir "$OUT/$ds" > "$log" 2>&1)
  local n
  n=$(find "$OUT/$ds" -path "*reviews*" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
  echo "STAGE_DONE $ds reviews_rows=$n"
}

run aime24-th full 4096 2400 14400
run live_code_bench-th full 4096 2400 43200
run math_500-th 500 4096 2400 43200

echo "SUITE3_COMPLETE $(date +%H:%M:%S)"
