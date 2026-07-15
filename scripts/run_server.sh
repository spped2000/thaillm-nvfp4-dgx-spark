#!/usr/bin/env bash
# Start the eval vLLM server for one side of the comparison.
# Usage: run_server.sh <bf16|nvfp4>
# NVFP4 MoE backend is read from results/moe_backend.txt (frozen in Phase 1.5).
set -euo pipefail
source "$(dirname "$0")/common.sh"

TAG=${1:?usage: run_server.sh <bf16|nvfp4>}
mkdir -p "$P/results/$TAG"

if [[ $TAG == bf16 ]]; then
  MODEL=$BF16_MODEL
  EXTRA=""
else
  MODEL=$NVFP4_MODEL
  MOE_BACKEND=$(cat "$P/results/moe_backend.txt")
  EXTRA="--quantization modelopt --moe-backend $MOE_BACKEND"
fi

docker rm -f eval-vllm >/dev/null 2>&1 || true
docker run --rm -d --name eval-vllm --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HFC:/root/.cache/huggingface" -v "$P:/work" \
  -e HF_HUB_OFFLINE=1 "$IMG" \
  bash -c "vllm serve $MODEL $COMMON_SERVE_FLAGS $EXTRA 2>&1 | tee /work/results/$TAG/server.log"

echo "waiting for http://127.0.0.1:$PORT/v1/models ..."
start=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! docker ps -q -f name=eval-vllm | grep -q .; then
    echo "SERVER DIED — last log lines:"; tail -40 "$P/results/$TAG/server.log"; exit 1
  fi
  sleep 10
  if (( $(date +%s) - start > 2400 )); then
    echo "TIMEOUT waiting for server"; tail -40 "$P/results/$TAG/server.log"; exit 1
  fi
done
echo "server up in $(( $(date +%s) - start ))s"
free -g > "$P/results/$TAG/mem_after_load.txt"
nvidia-smi >> "$P/results/$TAG/mem_after_load.txt"
