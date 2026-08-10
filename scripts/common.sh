#!/usr/bin/env bash
# Shared config for the BF16-vs-NVFP4 comparison pipeline.
export P=/home/agicafet/Documents/ThaiLLM_Quantization
# IMG/PORT overridable: the study pins NGC 26.05, but compressed-tensors NVFP4
# checkpoints (e.g. the 72B) need stock v0.25.1 — pass IMG=... to override.
export IMG=${IMG:-nvcr.io/nvidia/vllm:26.05.post1-py3}
export HFC=$HOME/.cache/huggingface
export PORT=${PORT:-8001}
# Overridable since the otg7b paired gate (2026-08-10); defaults preserve the
# July ThaiLLM-30B study behaviour. GATE_ROOT redirects ALL gate outputs so a
# new pair can never overwrite the published 30B artifacts under results/.
export BF16_MODEL=${BF16_MODEL:-ThaiLLM/ThaiLLM-30B}
export NVFP4_MODEL=${NVFP4_MODEL:-/work/models/ThaiLLM-30B-NVFP4}
export GATE_ROOT=${GATE_ROOT:-$P/results}
export VENV=$P/.venv-eval

# Identical serving flags for both models (fairness contract).
# KV cache stays auto (BF16); prefix caching off (prompt-logprobs requests
# get no benefit and conflict with APC in vLLM V1).
export COMMON_SERVE_FLAGS="--host 0.0.0.0 --port $PORT --served-model-name eval-model \
  --max-model-len 8192 --gpu-memory-utilization 0.70 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --kv-cache-dtype auto --seed 0 \
  --attention-backend flashinfer --no-enable-prefix-caching"
