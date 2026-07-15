#!/usr/bin/env bash
# Run the accuracy suite against the live eval server.
# Usage: run_suite.sh <bf16|nvfp4>
# Invocations are byte-identical between tags except --output_path.
set -euo pipefail
source "$(dirname "$0")/common.sh"

TAG=${1:?usage: run_suite.sh <bf16|nvfp4>}
OUT=$P/results/$TAG
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
LM_EVAL=$VENV/bin/lm_eval
MODEL_ARGS="model=eval-model,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=ThaiLLM/ThaiLLM-30B,num_concurrent=8,max_retries=3,tokenized_requests=True,max_length=8192"

echo "=== [$TAG] 0-shot multiple-choice suite ==="
$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks belebele_tha_Thai,xnli_th,xcopa_th,thai_exam,hellaswag,arc_challenge,winogrande \
  --include_path "$P/thai_tasks" --num_fewshot 0 --batch_size 16 --seed 0 \
  --output_path "$OUT/suite0" --log_samples 2>&1 | tee "$OUT/suite0.log"

echo "=== [$TAG] MMLU 5-shot, limit 50/subject ==="
$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks mmlu --num_fewshot 5 --limit 50 --batch_size 16 --seed 0 \
  --output_path "$OUT/mmlu" --log_samples 2>&1 | tee "$OUT/mmlu.log"

echo "=== [$TAG] thai_exam_v2 (letter-based, model-card protocol) ==="
$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks thai_exam_v2 \
  --include_path "$P/thai_tasks" --num_fewshot 0 --batch_size 16 --seed 0 \
  --output_path "$OUT/thai_exam_v2" --log_samples 2>&1 | tee "$OUT/thai_exam_v2.log"

echo "=== [$TAG] perplexity (wikitext + thai_wikipedia_ppl, limit 1000) ==="
$LM_EVAL --model local-completions --model_args "$MODEL_ARGS" \
  --tasks wikitext,thai_wikipedia_ppl \
  --include_path "$P/thai_tasks" --limit 1000 --batch_size 16 --seed 0 \
  --output_path "$OUT/ppl" --log_samples 2>&1 | tee "$OUT/ppl.log"

echo "=== [$TAG] accuracy suite done ==="
