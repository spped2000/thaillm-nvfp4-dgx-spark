#!/usr/bin/env bash
# lm-eval halves of the otg7b paired gate (rerun after the run_server.sh
# entrypoint fix; the generative halves completed in the first gate run).
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
cd "$P"
export GATE_ROOT=$P/results/otg7b_gate
export IMG=vllm/vllm-openai:v0.25.1
export PORT=8001
export BF16_MODEL=/work/models/openthaigpt1.5-7b-instruct
export NVFP4_MODEL=/work/models/openthaigpt1.5-7b-instruct-NVFP4
export NVFP4_EXTRA=""
export TOKENIZER=$P/models/openthaigpt1.5-7b-instruct

say () { echo "### $(date +%H:%M:%S) $*"; }
lm_side () {
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
lm_side bf16   || echo "SIDE_FAILED lm_bf16"
lm_side nvfp4  || echo "SIDE_FAILED lm_nvfp4"
echo "LM_GATE_COMPLETE $(date +%H:%M:%S)"
