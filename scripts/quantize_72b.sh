#!/usr/bin/env bash
# NVFP4 quantization of a model LARGER than RAM via modelopt --low_memory_mode.
# Parametrized so the exact same path can be rehearsed on a tiny model first:
#   MODEL_SRC=Qwen/Qwen2.5-0.5B-Instruct EXPORT_NAME=rehearsal-0.5B-NVFP4 \
#     CONTAINER=thai-quant-rehearsal bash scripts/quantize_72b.sh
# Real run (defaults):
#   bash scripts/quantize_72b.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL_SRC="${MODEL_SRC:-/work/models/openthaigpt-1.6-72b-instruct}"
EXPORT_NAME="${EXPORT_NAME:-openthaigpt-1.6-72b-instruct-NVFP4}"
CONTAINER="${CONTAINER:-thai-quant-72b}"
RESULTS="$P/results/quant72b"
mkdir -p "$RESULTS"

echo "=== free memory before ==="; free -g | head -2

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HFC:/root/.cache/huggingface" -v "$P:/work" \
  "$IMG" sleep infinity

docker exec "$CONTAINER" bash -c \
  "pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|flashinfer-python|accelerate|datasets) ' > /work/results/quant72b/env_before_${CONTAINER}.txt"

# Offline wheels first, then network. Never let pip touch torch/vllm.
docker exec "$CONTAINER" bash -c \
  'pip install -q --no-index --find-links /work/wheels "nvidia-modelopt[hf]==0.43.0" 2>/dev/null \
   || pip install -q "nvidia-modelopt[hf]==0.43.0"'

docker exec "$CONTAINER" bash -c \
  "pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|flashinfer-python|accelerate|datasets) ' > /work/results/quant72b/env_after_${CONTAINER}.txt; python -c 'import modelopt; print(\"modelopt\", modelopt.__version__)'"

# Guard: torch/vllm/flashinfer binaries must be untouched (transformers may
# legally move to 4.x — disposable container, never serves).
for pkg in torch vllm flashinfer-python; do
  if ! diff <(grep "^$pkg " "$RESULTS/env_before_${CONTAINER}.txt") \
            <(grep "^$pkg " "$RESULTS/env_after_${CONTAINER}.txt") >/dev/null; then
    echo "FATAL: pip modified $pkg — aborting before quantization."
    exit 1
  fi
done
echo "guard OK: torch/vllm/flashinfer unchanged"

# modelopt 0.43.0 --low_memory_mode forwards attn_implementation into
# accelerate's load_checkpoint_and_dispatch (TypeError). Route it into
# from_config instead — disposable-container patch, recorded in results/.
docker exec "$CONTAINER" python /work/scripts/patch_modelopt_lowmem.py \
  /usr/local/lib/python3.12/dist-packages/modelopt/torch/quantization/plugins/accelerate.py \
  | tee "$RESULTS/patch_${CONTAINER}.txt"

# --low_memory_mode: compress-while-loading; peak = NVFP4 output + one BF16
# layer, NOT full BF16 residency (the 72B is 145 GB > 121 GB RAM).
# --batch_size 1: hf_ptq's own --skip_generate help warns auto-probe (0)
# allocates speculatively on very large models.
echo "=== launching hf_ptq (detached; calibration is silently slow — MO issue #602) ==="
docker exec -d "$CONTAINER" bash -c \
  "cd /work/Model-Optimizer/examples/llm_ptq && \
   python hf_ptq.py \
     --pyt_ckpt_path '$MODEL_SRC' \
     --qformat nvfp4 \
     --kv_cache_qformat none \
     --calib_size 512 --calib_seq 512 --batch_size 1 \
     --dataset /work/calib/thai_en_calib.jsonl \
     --attn_implementation sdpa \
     --skip_generate \
     --low_memory_mode \
     --export_path /work/models/$EXPORT_NAME \
     > /work/results/quant72b/quant_${CONTAINER}.log 2>&1"
echo "quantization started — tail $RESULTS/quant_${CONTAINER}.log; do NOT trust NVML for memory, use free -g"
