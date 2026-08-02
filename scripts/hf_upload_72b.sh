#!/usr/bin/env bash
# Restart-proof upload of the 72B NVFP4 checkpoint. Log: results/quant72b/hf_upload_72b.log
set -u
cd "$(dirname "$0")/.."
export HF_TOKEN=$(head -1 .env)
LOG=results/quant72b/hf_upload_72b.log
exec >>"$LOG" 2>&1
ok=0
for i in $(seq 1 40); do
  echo "=== attempt $i $(date +%H:%M:%S) ==="
  if timeout 5400 .venv-eval/bin/hf upload AGIcafet/openthaigpt-1.6-72b-instruct-NVFP4 \
      models/openthaigpt-1.6-72b-instruct-NVFP4 . --token "$HF_TOKEN" \
      --commit-message "NVFP4 W4A4 (llm-compressor 0.12.0.1, half-Thai calibration, quantized on one DGX Spark)"; then
    ok=1; echo "HF_UPLOAD_COMPLETE $(date +%H:%M:%S)"; break
  fi
  echo "--- attempt $i failed/timed out; retrying in 30s"
  sleep 30
done
[ $ok -eq 1 ] || echo "HF_UPLOAD_FAILED_ALL_RETRIES"
