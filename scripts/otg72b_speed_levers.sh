#!/usr/bin/env bash
# Speed levers for the 72B NVFP4, per results/otg72b/PREREGISTERED.md §4.
# Three arms, each measured against the pre-registered control (official hub
# bench 5.01 tok/s +-0.2%). Single-stream arms use the hub's vendored harness
# via bench boots; concurrency uses `vllm bench serve` (run_perf.sh pattern).
# Every result is published, speedup or slowdown.
set -u
P=/home/agicafet/Documents/ThaiLLM_Quantization
TV=/home/agicafet/Documents/thai-vllm
MODEL_DIR=$P/models/openthaigpt-1.6-72b-instruct-NVFP4
OUT=$P/results/otg72b/levers
NAME=openthaigpt-1.6-72b-instruct-NVFP4
mkdir -p "$OUT"

say () { echo "### $(date +%H:%M:%S) $*"; }

launch () {  # $1 arm-name, rest = extra vllm args
  local arm=$1; shift
  docker rm -f serve-72b >/dev/null 2>&1
  docker run -d --name serve-72b --gpus all --ipc=host -p 8014:8000 \
    -v "$MODEL_DIR":/model:ro vllm/vllm-openai:v0.25.1 \
    --model /model --served-model-name "$NAME" \
    --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes \
    "$@" >/dev/null
  local t=0
  until curl -sf localhost:8014/health >/dev/null 2>&1; do
    sleep 15; t=$((t+15))
    docker inspect -f '{{.State.Status}}' serve-72b | grep -q running || { echo "ARM_DIED $arm"; docker logs serve-72b 2>&1 | grep -iE "error|invalid" | head -3 > "$OUT/${arm}_boot_error.txt"; return 1; }
    [ $t -ge 1500 ] && { echo "ARM_TIMEOUT $arm"; return 1; }
  done
  return 0
}

bench_boot () {  # $1 arm  $2 boot#
  OUTDIR=$OUT/$1 BOOT=$2 $TV/.venv/bin/python - <<'PY'
import json, os, sys
sys.path.insert(0, "/home/agicafet/Documents/thai-vllm/src")
from thai_vllm.bench.run import run_vendored
from pathlib import Path
outdir = Path(os.environ["OUTDIR"]); outdir.mkdir(parents=True, exist_ok=True)
data = run_vendored("http://localhost:8014/v1", "openthaigpt-1.6-72b-instruct-NVFP4",
                    outdir / f"boot{os.environ['BOOT']}.json")
print("BOOT_OK", os.environ["BOOT"])
PY
}

finalize_arm () {  # $1 arm
  OUTDIR=$OUT/$1 $TV/.venv/bin/python - <<'PY'
import json, os, sys
sys.path.insert(0, "/home/agicafet/Documents/thai-vllm/src")
from thai_vllm.bench.run import finalize
from pathlib import Path
outdir = Path(os.environ["OUTDIR"])
res = finalize(sorted(outdir.glob("boot*.json")))
(outdir / "final.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
chat = res.get("chat", {})
print(f"ARM_FINAL {outdir.name} chat={chat.get('decode_tok_s_median')} tok/s "
      f"+-{chat.get('decode_tok_s_median_boot_spread_pct')}% ttft={chat.get('ttft_p50_s')}")
PY
}

# ---- Arm A: n-gram speculative decoding (2 boots) ----
SPEC='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}'
for b in 1 2; do
  say "arm ngram boot $b"
  if launch ngram --speculative-config "$SPEC"; then bench_boot ngram $b; else echo "STAGE_DONE ngram FAILED_TO_BOOT"; break; fi
done
[ -f "$OUT/ngram/boot2.json" ] && finalize_arm ngram
echo "STAGE_DONE ngram"

# ---- Arm B: KV cache fp8 (2 boots single-stream + concurrency c4) ----
for b in 1 2; do
  say "arm kvfp8 boot $b"
  if launch kvfp8 --kv-cache-dtype fp8; then bench_boot kvfp8 $b; else echo "STAGE_DONE kvfp8 FAILED_TO_BOOT"; break; fi
done
[ -f "$OUT/kvfp8/boot2.json" ] && finalize_arm kvfp8
echo "STAGE_DONE kvfp8"

# ---- Arm C: concurrency sweep on BASELINE config vs kvfp8 (c1/c4/c8) ----
conc_sweep () {  # $1 tag  (server must be up)
  for C in 1 4 8; do
    say "conc $1 c$C"
    # vllm bench serve runs INSIDE the container (vllm installed there)
    docker exec serve-72b bash -c "vllm bench serve --backend openai --base-url http://127.0.0.1:8000 \
      --model /model --served-model-name $NAME --dataset-name random \
      --random-input-len 512 --random-output-len 256 \
      --max-concurrency $C --num-prompts $((C*8)) --seed 0" \
      > "$OUT/conc_$1_c$C.txt" 2>&1
    grep -E "Output token throughput|Request throughput" "$OUT/conc_$1_c$C.txt" | head -2
  done
}
# kvfp8 server is still up from arm B (if it booted) - sweep it, then baseline
docker ps -q -f name=serve-72b | grep -q . && conc_sweep kvfp8
say "baseline relaunch for conc sweep"
if launch baseline; then conc_sweep baseline; bench_boot baseline 1; fi
echo "STAGE_DONE conc"

echo "LEVERS_COMPLETE $(date +%H:%M:%S)"
