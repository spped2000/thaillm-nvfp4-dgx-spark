#!/usr/bin/env bash
# ARM 2: W4A16 side only - BF16 side reused from arm 1 (identical contract,
# stated in PREREGISTERED.md). Same stages as otg7b_gate.sh nvfp4 side.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
cd "$P"
export GATE_ROOT=$P/results/otg7b_gate_w4a16
export IMG=vllm/vllm-openai:v0.25.1
export PORT=8001
export BF16_MODEL=/work/models/openthaigpt1.5-7b-instruct
export NVFP4_MODEL=/work/models/openthaigpt1.5-7b-instruct-W4A16
export NVFP4_EXTRA=""
export TOKENIZER=$P/models/openthaigpt1.5-7b-instruct

say () { echo "### $(date +%H:%M:%S) $*"; }

say "STAGE lm_server_nvfp4"
bash scripts/run_server.sh nvfp4 || { echo "STAGE_FAIL server_w4a16"; exit 1; }
say "STAGE lm_suite_nvfp4"
bash scripts/run_suite.sh nvfp4 > "$GATE_ROOT/nvfp4_suite_console.log" 2>&1 || echo "STAGE_FAIL suite_w4a16"
say "STAGE usecases_nvfp4"
.venv-eval/bin/python scripts/run_usecases.py nvfp4 > "$GATE_ROOT/nvfp4/usecases.log" 2>&1 || echo "STAGE_WARN usecases"
say "STAGE perf_nvfp4"
PERF_TOKENIZER=$BF16_MODEL bash scripts/run_perf.sh nvfp4 > "$GATE_ROOT/nvfp4_perf_console.log" 2>&1 || echo "STAGE_WARN perf"
docker rm -f eval-vllm >/dev/null 2>&1
echo "STAGE_DONE lm_w4a16"

say "STAGE gen_server_w4a16"
docker rm -f gen-vllm >/dev/null 2>&1
docker run -d --name gen-vllm --gpus all --ipc=host -p 8014:8000 \
  -v "$P/models/openthaigpt1.5-7b-instruct-W4A16":/model:ro vllm/vllm-openai:v0.25.1 \
  --model /model --served-model-name eval-model \
  --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --seed 0 \
  --enable-auto-tool-choice --tool-call-parser hermes >/dev/null
until curl -sf localhost:8014/health >/dev/null 2>&1; do
  sleep 10
  [ "$(docker inspect -f '{{.State.Status}}' gen-vllm 2>/dev/null)" = running ] || { echo "STAGE_FAIL gen_server_w4a16"; exit 1; }
done
EVAL=$P/.venv-chinda/bin/evalscope
OUT=$GATE_ROOT/gen_nvfp4
mkdir -p "$OUT"
run_bench () {
  local ds=$1 lim=$2
  local lim_args=()
  [ "$lim" != full ] && lim_args=(--limit "$lim")
  say "STAGE gen_w4a16_$ds"
  (cd "$P/chinda-eval" && timeout 14400 $EVAL eval \
     --model eval-model --api-url http://127.0.0.1:8014/v1/chat/completions \
     --api-key EMPTY --eval-type openai_api --datasets "$ds" \
     --dataset-hub huggingface "${lim_args[@]}" --eval-batch-size 4 --timeout 1200 \
     --generation-config '{"max_tokens":4096,"temperature":0.0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
     --work-dir "$OUT/$ds" > "$OUT/$ds.log" 2>&1)
  local n
  n=$(find "$OUT/$ds" -path "*reviews*" -name "*.jsonl" -exec cat {} + 2>/dev/null | wc -l)
  echo "STAGE_DONE gen_w4a16_$ds rows=$n"
}
run_bench openthaieval 1232
run_bench ifeval-th 215
run_bench math_500-th 500
run_bench live_code_bench-th full
run_bench aime24-th full
.venv-eval/bin/python scripts/thai_tool_conformance.py \
  --url http://127.0.0.1:8014/v1 --model eval-model --modes auto,required \
  > "$OUT/tool_conformance.json" 2> "$OUT/tool_conformance.log" || echo "STAGE_WARN tools"
docker rm -f gen-vllm >/dev/null 2>&1
echo "W4A16_GATE_COMPLETE $(date +%H:%M:%S)"
