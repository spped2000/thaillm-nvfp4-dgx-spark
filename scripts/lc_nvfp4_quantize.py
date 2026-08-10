"""NVFP4 (W4A4) quantization via llm-compressor for models LARGER than RAM.

The modelopt --low_memory_mode path exports numerically broken NVFP4
(weight_scale half-sized, dequant cosine 0.76 vs BF16 — see
results/quant72b/) so this is the working route: accelerate disk offload at
load (device_map=auto + max_memory + offload_folder) and llm-compressor's
sequential calibration, which onloads one layer at a time.

Env: SRC (local dir), OUT, CALIB (jsonl with "text"), NSAMP, SEQ,
     GPU_MEM / CPU_MEM (max_memory caps, e.g. "80GiB"; tiny caps in the
     rehearsal force the disk tier on purpose), OFFLOAD (folder).
"""
import json
import os

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SRC = os.environ["SRC"]
OUT = os.environ["OUT"]
CALIB = os.environ["CALIB"]
NSAMP = int(os.environ.get("NSAMP", "512"))
SEQ = int(os.environ.get("SEQ", "512"))
GPU_MEM = os.environ.get("GPU_MEM", "60GiB")
CPU_MEM = os.environ.get("CPU_MEM", "20GiB")
OFFLOAD = os.environ.get("OFFLOAD", "/work/offload")

tok = AutoTokenizer.from_pretrained(SRC)
rows = [json.loads(l)["text"] for l in open(CALIB, encoding="utf-8")][:NSAMP]
ds = Dataset.from_dict({"text": rows}).map(
    lambda b: tok(b["text"], truncation=True, max_length=SEQ),
    remove_columns=["text"],
)

model = AutoModelForCausalLM.from_pretrained(
    SRC,
    dtype="auto",
    device_map="auto",
    max_memory={0: GPU_MEM, "cpu": CPU_MEM},
    offload_folder=OFFLOAD,
)
# hf_device_map is absent when the whole model fit on one device (small
# models / big caps) — that IS the no-disk-tier signal, print it as such.
tiers = sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}) or ["single-device"]
print("hf_device_map tiers:", tiers)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*embed.*"],
)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=SEQ,
    num_calibration_samples=len(rows),
)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("SAVED", OUT)
