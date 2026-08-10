#!/usr/bin/env bash
# Full paired BF16-vs-NVFP4 gate for openthaigpt1.5-7b-instruct.
# Pre-registered: results/otg7b_gate/PREREGISTERED.md (written before any run).
# Both sides: identical flags, image vllm/vllm-openai:v0.25.1, seed 0.
# Sequence per side: server -> lm-eval suite -> usecases -> perf -> down,
# then the generative chinda-eval set on a 32k chat profile.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
cd "$P"

export GATE_ROOT=$P/results/otg7b_gate
export IMG=vllm/vllm-openai:v0.25.1
export PORT=8001
export BF16_MODEL=/work/models/openthaigpt1.5-7b-instruct
export NVFP4_MODEL=/work/models/openthaigpt1.5-7b-instruct-NVFP4
export NVFP4_EXTRA=""                       # dense + compressed-tensors: no extra flags
export TOKENIZER=$P/models/openthaigpt1.5-7b-instruct   # host path for lm-eval

say () { echo "### $(date +%H:%M:%S) $*"; }

lm_side () {  # $1 = tag
  local tag=$1
  say "STAGE lm_server_$tag"
  bash scripts/run_server.sh "$tag" || { echo "STAGE_FAIL server_$tag"; return 1; }
  say "STAGE lm_suite_$tag"
  bash scripts/run_suite.sh "$tag" > "$GATE_ROOT/${tag}_suite_console.log" 2>&1 \
    || { echo "STAGE_FAIL suite_$tag"; return 1; }
  say "STAGE usecases_$tag"
  .venv-eval/bin/python scripts/run_usecases.py "$tag" > "$GATE_ROOT/$tag/usecases.log" 2>&1 \
    || echo "STAGE_WARN usecases_$tag"
  say "STAGE perf_$tag"
  PERF_TOKENIZER=${BF16_MODEL} bash scripts/run_perf.sh "$tag" > "$GATE_ROOT/${tag}_perf_console.log" 2>&1 \
    || echo "STAGE_WARN perf_$tag"
  docker rm -f eval-vllm >/dev/null 2>&1
  echo "STAGE_DONE lm_$tag"
}

gen_side () {  # $1 = tag, $2 = host model dir
  local tag=$1 mdir=$2
  say "STAGE gen_server_$tag"
  docker rm -f gen-vllm >/dev/null 2>&1
  docker run -d --name gen-vllm --gpus all --ipc=host -p 8014:8000 \
    -v "$mdir":/model:ro vllm/vllm-openai:v0.25.1 \
    --model /model --served-model-name eval-model \
    --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --seed 0 \
    --enable-auto-tool-choice --tool-call-parser hermes >/dev/null
  until curl -sf localhost:8014/health >/dev/null 2>&1; do
    sleep 10
    [ "$(docker inspect -f '{{.State.Status}}' gen-vllm 2>/dev/null)" = running ] \
      || { echo "STAGE_FAIL gen_server_$tag"; return 1; }
  done
  local EVAL=$P/.venv-chinda/bin/evalscope
  local OUT=$GATE_ROOT/gen_$tag
  mkdir -p "$OUT"
  run_bench () {  # dataset limit max_tokens timeout
    local ds=$1 lim=$2 mt=$3 to=$4
    local lim_args=()
    [ "$lim" != full ] && lim_args=(--limit "$lim")
    say "STAGE gen_${tag}_$ds"
    (cd "$P/chinda-eval" && timeout 14400 $EVAL eval \
       --model eval-model --api-url http://127.0.0.1:8014/v1/chat/completions \
       --api-key EMPTY --eval-type openai_api --datasets "$ds" \
       --dataset-hub huggingface "${lim_args[@]}" --eval-batch-size 4 --timeout "$to" \
       --generation-config "{\"max_tokens\":$mt,\"temperature\":0.0,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}" \
       --work-dir "$OUT/$ds" > "$OUT/$ds.log" 2>&1)
    local n
    n=$(find "$OUT/$ds" -path "*reviews*" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
    echo "STAGE_DONE gen_${tag}_$ds rows=$n"
  }
  run_bench openthaieval 1232 4096 1200
  run_bench ifeval-th 215 4096 1200
  run_bench math_500-th 500 4096 1200
  run_bench live_code_bench-th full 4096 1200
  run_bench aime24-th full 4096 1200
  .venv-eval/bin/python scripts/thai_tool_conformance.py \
    --url http://127.0.0.1:8014/v1 --model eval-model --modes auto,required \
    > "$OUT/tool_conformance.json" 2> "$OUT/tool_conformance.log" || echo "STAGE_WARN tools_$tag"
  docker rm -f gen-vllm >/dev/null 2>&1
  echo "STAGE_DONE gen_$tag"
}

lm_side bf16   || echo "SIDE_FAILED lm_bf16"
lm_side nvfp4  || echo "SIDE_FAILED lm_nvfp4"
gen_side bf16  "$P/models/openthaigpt1.5-7b-instruct"   || echo "SIDE_FAILED gen_bf16"
gen_side nvfp4 "$P/models/openthaigpt1.5-7b-instruct-NVFP4" || echo "SIDE_FAILED gen_nvfp4"

echo "GATE_COMPLETE $(date +%H:%M:%S)"
