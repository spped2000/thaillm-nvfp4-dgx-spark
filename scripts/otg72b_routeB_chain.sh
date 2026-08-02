#!/usr/bin/env bash
# Waits for Route A part 2, then runs Route B (lm-eval reference suite) for the
# 72B NVFP4 under the study's fairness contract. Sequential because both need
# the whole GPU.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
cd "$P"

echo "waiting for SUITE2_COMPLETE..."
until grep -q "SUITE2_COMPLETE" results/otg72b_suite2.log 2>/dev/null; do sleep 60; done
echo "ROUTE_A_DONE $(date +%H:%M:%S)"

docker rm -f serve-72b >/dev/null 2>&1
sleep 5

# IMG: stock v0.25.1 (compressed-tensors NVFP4 needs it; NGC 26.05 is 0.21-NV).
# TOKENIZER: host path (lm-eval runs on host); MODEL: container path (/work).
IMG=vllm/vllm-openai:v0.25.1 \
TOKENIZER=$P/models/openthaigpt-1.6-72b-instruct-NVFP4 \
bash scripts/run_reference.sh otg72b-nvfp4 /work/models/openthaigpt-1.6-72b-instruct-NVFP4 \
  && echo "ROUTE_B_COMPLETE $(date +%H:%M:%S)" \
  || echo "ROUTE_B_FAILED rc=$?"
