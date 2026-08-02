#!/usr/bin/env bash
# Route A: Thai-local generative suite for openthaigpt-1.6-72b-instruct-NVFP4.
# Attaches to the already-serving endpoint (serve-72b on :8014) like p4 does —
# deliberately NOT reusing p3/p5/c5 whose EXIT traps resurrect premize-vllm-chat-1.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
EVAL=$P/.venv-chinda/bin/evalscope
URL=http://127.0.0.1:8014/v1
MODEL=openthaigpt-1.6-72b-instruct-NVFP4
OUT=$P/results/otg72b
GEN='{"max_tokens":8192,"temperature":0.0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
mkdir -p "$OUT"

say () { echo ""; echo "### $(date +%H:%M:%S) $*"; }

curl -s -m 5 "$URL/models" | grep -q '"id"' || { echo "FATAL endpoint down"; exit 1; }

run () {  # $1 dataset  $2 limit
  local ds=$1 lim=$2
  local log=$OUT/${ds}.log
  say "$ds (limit $lim)"
  (cd "$P/chinda-eval" && timeout 21600 $EVAL eval \
     --model "$MODEL" --api-url "$URL/chat/completions" --api-key EMPTY \
     --eval-type openai_api --datasets "$ds" --dataset-hub huggingface \
     --limit "$lim" --eval-batch-size 2 --generation-config "$GEN" \
     --work-dir "$OUT/$ds" > "$log" 2>&1)
  local n
  n=$(find "$OUT/$ds" -path "*reviews*" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
  echo "STAGE_DONE $ds reviews_rows=$n"
}

run openthaieval 1232
run ifeval-th 215
run hellaswag-th 300
run code_switching 500

say "tool conformance (hermes, auto+required)"
$P/.venv-eval/bin/python $P/scripts/thai_tool_conformance.py \
  --url "$URL" --model "$MODEL" --modes auto,required \
  > "$OUT/tool_conformance.json" 2> "$OUT/tool_conformance.log" \
  || .venv-eval/bin/python $P/scripts/thai_tool_conformance.py \
       --url "$URL" --model "$MODEL" > "$OUT/tool_conformance.json" 2>> "$OUT/tool_conformance.log"
echo "STAGE_DONE tool_conformance"

echo "SUITE_COMPLETE $(date +%H:%M:%S)"
