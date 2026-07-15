#!/usr/bin/env python3
"""Build the NVFP4 calibration set: 256 Thai Wikipedia + 256 CNN/DailyMail docs.

Deterministic (seed 0). Docs are filtered to >=2000 chars and truncated to
4000 chars — comfortably >=512 tokens for hf_ptq's --calib_seq 512. Output is
jsonl with a single "text" field (auto-detected by modelopt dataset_utils).
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "calib" / "thai_en_calib.jsonl"
N_PER_LANG = 256
MIN_CHARS = 2000
MAX_CHARS = 4000
rng = random.Random(0)

# Thai: cached full config (fetch_datasets.py ran first)
thai = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
pool_idx = [i for i, t in enumerate(thai["text"]) if len(t) >= MIN_CHARS]
thai_docs = [thai[i]["text"][:MAX_CHARS] for i in sorted(rng.sample(pool_idx, N_PER_LANG))]

# English: stream, no full download
en_pool = []
for row in load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True):
    if len(row["article"]) >= MIN_CHARS:
        en_pool.append(row["article"][:MAX_CHARS])
    if len(en_pool) >= 2000:
        break
en_docs = rng.sample(en_pool, N_PER_LANG)

docs = thai_docs + en_docs
rng.shuffle(docs)
with OUT.open("w") as f:
    for d in docs:
        f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")

th_count = sum(1 for d in docs if any("฀" <= c <= "๿" for c in d[:200]))
print(f"wrote {len(docs)} docs -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB), ~{th_count} Thai")
