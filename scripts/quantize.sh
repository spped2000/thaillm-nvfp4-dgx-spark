#!/usr/bin/env bash
# Phase 1: NVFP4 quantization of ThaiLLM-30B with TensorRT Model Optimizer 0.43.0.
# Runs in a disposable container from the NGC vLLM image. nemoclaw-vllm must be
# stopped first (memory). Expects wheels/ prefilled for offline pip fallback.
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "=== free memory before ==="; free -g | head -2

docker rm -f thai-quant >/dev/null 2>&1 || true
docker run -d --name thai-quant --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HFC:/root/.cache/huggingface" -v "$P:/work" \
  "$IMG" sleep infinity

docker exec thai-quant bash -c \
  'pip list 2>/dev/null | grep -iE "^(torch|vllm|transformers|flashinfer-python|accelerate|datasets) " > /work/results/quant_env_before.txt; cat /work/results/quant_env_before.txt'

# Try offline wheels first, then network. Never let pip touch torch/vllm.
docker exec thai-quant bash -c \
  'pip install -q --no-index --find-links /work/wheels "nvidia-modelopt[hf]==0.43.0" 2>/dev/null \
   || pip install -q "nvidia-modelopt[hf]==0.43.0"'

docker exec thai-quant bash -c \
  'pip list 2>/dev/null | grep -iE "^(torch|vllm|transformers|flashinfer-python|accelerate|datasets) " > /work/results/quant_env_after.txt; python -c "import modelopt; print(\"modelopt\", modelopt.__version__)"'

# Guard: torch/vllm/flashinfer binaries must be untouched. transformers is
# ALLOWED to move to 4.x here — modelopt 0.43.0 pins transformers<5.0 and this
# disposable container never serves; the eval/serve containers stay pristine.
for pkg in torch vllm flashinfer-python; do
  if ! diff <(grep "^$pkg " "$P/results/quant_env_before.txt") \
            <(grep "^$pkg " "$P/results/quant_env_after.txt") >/dev/null; then
    echo "FATAL: pip modified $pkg — aborting before quantization."
    exit 1
  fi
done
echo "guard OK: torch/vllm/flashinfer unchanged"

echo "=== launching hf_ptq (detached; expect 1-3h silence during calibration) ==="
docker exec -d thai-quant bash -c \
  'cd /work/Model-Optimizer/examples/llm_ptq && \
   python hf_ptq.py \
     --pyt_ckpt_path ThaiLLM/ThaiLLM-30B \
     --qformat nvfp4 \
     --kv_cache_qformat none \
     --calib_size 512 --calib_seq 512 --batch_size 0 \
     --dataset /work/calib/thai_en_calib.jsonl \
     --attn_implementation sdpa \
     --skip_generate \
     --export_path /work/models/ThaiLLM-30B-NVFP4 \
     > /work/results/quant.log 2>&1'
echo "quantization started — tail /work/results/quant.log and watch GPU util"
