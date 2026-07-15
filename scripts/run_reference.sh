#!/usr/bin/env bash
# Thai-capability reference run for a locally-cached model served with vLLM.
# Usage: run_reference.sh <name> <hf_id_or_path> [extra vllm args...]
# Reduced suite: Thai MC tasks + thai_exam_v2 + mmlu@10 + Thai byte-PPL@200.
# NOTE: capability snapshot, not a strict A/B — tokenizer is the model's own;
# instruct models are scored raw-completion style (documented in report).
set -euo pipefail
source "$(dirname "$0")/common.sh"

NAME=${1:?usage: run_reference.sh <name> <model> [extra vllm args]}
MODEL=${2:?}
shift 2
EXTRA="$*"
OUT=$P/results/ref_$NAME
mkdir -p "$OUT"

docker rm -f eval-vllm >/dev/null 2>&1 || true
docker run --rm -d --name eval-vllm --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HFC:/root/.cache/huggingface" -v "$P:/work" \
  -e HF_HUB_OFFLINE=1 "$IMG" \
  bash -c "vllm serve $MODEL $COMMON_SERVE_FLAGS --trust-remote-code $EXTRA 2>&1 | tee /work/results/ref_$NAME/server.log"

echo "waiting for server ($NAME) ..."
start=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  docker ps -q -f name=eval-vllm | grep -q . || { echo "SERVER DIED"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 10
  (( $(date +%s) - start > 1800 )) && { echo TIMEOUT; tail -30 "$OUT/server.log"; exit 1; }
done
echo "up in $(( $(date +%s) - start ))s"

export HF_HUB_OFFLINE=1
LM_EVAL=$VENV/bin/lm_eval
MODEL_ARGS="model=eval-model,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=$MODEL,num_concurrent=8,max_retries=3,tokenized_requests=True,max_length=8192"

$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks belebele_tha_Thai,xnli_th,xcopa_th,thai_exam_v2 \
  --include_path "$P/thai_tasks" --num_fewshot 0 --batch_size 16 --seed 0 \
  --output_path "$OUT/thai" --log_samples 2>&1 | tee "$OUT/thai.log" | tail -20

$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks mmlu --num_fewshot 5 --limit 10 --batch_size 16 --seed 0 \
  --output_path "$OUT/mmlu10" 2>&1 | tee "$OUT/mmlu10.log" | tail -4

$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks thai_wikipedia_ppl --include_path "$P/thai_tasks" \
  --limit 200 --batch_size 16 --seed 0 \
  --output_path "$OUT/ppl200" 2>&1 | tee "$OUT/ppl200.log" | tail -6

docker rm -f eval-vllm >/dev/null 2>&1 || true
echo "=== reference run $NAME done ==="
