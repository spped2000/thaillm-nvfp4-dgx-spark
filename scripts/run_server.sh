#!/usr/bin/env bash
# Start the eval vLLM server for one side of the comparison.
# Usage: run_server.sh <bf16|nvfp4>
# NVFP4 MoE backend is read from results/moe_backend.txt (frozen in Phase 1.5).
set -euo pipefail
source "$(dirname "$0")/common.sh"

TAG=${1:?usage: run_server.sh <bf16|nvfp4>}
RELOUT=${GATE_ROOT#"$P/"}   # container sees $P as /work
mkdir -p "$GATE_ROOT/$TAG"

if [[ $TAG == bf16 ]]; then
  MODEL=$BF16_MODEL
  EXTRA=""
else
  MODEL=$NVFP4_MODEL
  # NVFP4_EXTRA overridable: compressed-tensors checkpoints (llm-compressor)
  # auto-detect and take NO --quantization flag; --moe-backend is MoE-only.
  # Pass NVFP4_EXTRA="" for a dense compressed-tensors pair (otg7b gate).
  if [[ ${NVFP4_EXTRA+set} == set ]]; then
    EXTRA="$NVFP4_EXTRA"
  else
    MOE_BACKEND=$(cat "$P/results/moe_backend.txt")
    EXTRA="--quantization modelopt --moe-backend $MOE_BACKEND"
  fi
fi

docker rm -f eval-vllm >/dev/null 2>&1 || true
# --entrypoint bash: the stock vllm/vllm-openai image's entrypoint IS
# `vllm serve` (NGC's just banners+execs) — without the override, `bash -c`
# lands as arguments to vllm and the container dies before tee ever runs.
docker run --rm -d --name eval-vllm --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HFC:/root/.cache/huggingface" -v "$P:/work" \
  -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMG" \
  -c "vllm serve $MODEL $COMMON_SERVE_FLAGS $EXTRA 2>&1 | tee /work/$RELOUT/$TAG/server.log"

echo "waiting for http://127.0.0.1:$PORT/v1/models ..."
start=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! docker ps -q -f name=eval-vllm | grep -q .; then
    echo "SERVER DIED — last log lines:"; tail -40 "$GATE_ROOT/$TAG/server.log"; exit 1
  fi
  sleep 10
  if (( $(date +%s) - start > 2400 )); then
    echo "TIMEOUT waiting for server"; tail -40 "$GATE_ROOT/$TAG/server.log"; exit 1
  fi
done
echo "server up in $(( $(date +%s) - start ))s"
echo "$(( $(date +%s) - start ))" > "$GATE_ROOT/$TAG/load_seconds.txt"
free -g > "$GATE_ROOT/$TAG/mem_after_load.txt"
nvidia-smi >> "$GATE_ROOT/$TAG/mem_after_load.txt"
