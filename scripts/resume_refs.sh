#!/usr/bin/env bash
# Restart-proof completion of the remaining reference runs.
# Progress -> results/resume_refs.log ; final marker line: RESUME_REFS_DONE
set -uo pipefail
source "$(dirname "$0")/common.sh"
LOG=$P/results/resume_refs.log
exec >>"$LOG" 2>&1

ppl_only() {
  local name=$1 model=$2; shift 2
  echo "### ppl200 $name $(date +%H:%M) ###"
  docker rm -f eval-vllm >/dev/null 2>&1 || true
  docker run --rm -d --name eval-vllm --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HFC:/root/.cache/huggingface" -v "$P:/work" -e HF_HUB_OFFLINE=1 "$IMG" \
    bash -c "vllm serve $model $COMMON_SERVE_FLAGS --trust-remote-code $* 2>&1 | tee /work/results/ref_$name/server_ppl.log"
  local start=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    docker ps -q -f name=eval-vllm | grep -q . || { echo "$name PPL SERVER DIED"; return 1; }
    sleep 10
    (( $(date +%s) - start > 1800 )) && { echo "$name PPL TIMEOUT"; return 1; }
  done
  HF_HUB_OFFLINE=1 "$VENV/bin/lm_eval" --model local-completions \
    --model_args "model=eval-model,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=$model,num_concurrent=8,max_retries=3,tokenized_requests=True,max_length=8192" \
    --tasks thai_wikipedia_ppl --include_path "$P/thai_tasks" --limit 200 --batch_size 16 --seed 0 \
    --output_path "$P/results/ref_$name/ppl200" | tail -4
  local rc=$?
  docker rm -f eval-vllm >/dev/null 2>&1 || true
  return $rc
}

# 1) unsloth27b: only ppl200 missing
[ -z "$(ls "$P/results/ref_unsloth27b-nvfp4/ppl200" 2>/dev/null)" ] && \
  ppl_only unsloth27b-nvfp4 unsloth/Qwen3.6-27B-NVFP4 || echo "unsloth ppl already done/failed-tolerated"

# 2) qwen3-8b: full run
[ -z "$(ls "$P/results/ref_qwen3-8b-nvfp4/thai" 2>/dev/null)" ] && \
  ( bash "$P/scripts/run_reference.sh" qwen3-8b-nvfp4 nvidia/Qwen3-8B-NVFP4 --quantization modelopt \
    || bash "$P/scripts/run_reference.sh" qwen3-8b-nvfp4 nvidia/Qwen3-8B-NVFP4 --quantization modelopt --enforce-eager \
    || echo "qwen3-8b FAILED" )

# 3) qwen36-35b: ppl200 retry in eager mode (engine hung on rolling PPL before)
[ -z "$(ls "$P/results/ref_qwen36-35b-nvfp4/ppl200" 2>/dev/null)" ] && \
  ppl_only qwen36-35b-nvfp4 nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt --enforce-eager \
  || echo "35b ppl done or tolerated-fail"

docker rm -f eval-vllm >/dev/null 2>&1 || true
echo "RESUME_REFS_DONE $(date +%H:%M)"
