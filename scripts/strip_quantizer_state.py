"""Strip modelopt runtime quantizer-state tensors from an NVFP4 export.

The --low_memory_mode export path leaves `*.weight_quantizer._double_scale`
scalars (duplicates of weight_scale_2 under the runtime module name) in the
state dict; vLLM's loader then aborts with "no module or parameter named
'...weight_quantizer'". The full-residency path (ThaiLLM-30B) never wrote
them. Handles single-file and sharded checkpoints; rewrites the index.
Run inside the quant container: python strip_quantizer_state.py <export_dir>
"""
import json
import os
import sys

import torch  # noqa: F401 - safetensors "pt" framework needs it
from safetensors.torch import load_file, save_file

d = sys.argv[1]
index_path = os.path.join(d, "model.safetensors.index.json")

if os.path.exists(index_path):
    index = json.load(open(index_path))
    shards = sorted(set(index["weight_map"].values()))
else:
    index, shards = None, ["model.safetensors"]

dropped_total = 0
for shard in shards:
    path = os.path.join(d, shard)
    tensors = load_file(path)
    junk = [k for k in tensors if "quantizer" in k]
    if not junk:
        continue
    for k in junk:
        del tensors[k]
    save_file(tensors, path, metadata={"format": "pt"})
    dropped_total += len(junk)
    print(f"{shard}: dropped {len(junk)} quantizer-state tensors")

if index is not None:
    kept = {k: v for k, v in index["weight_map"].items() if "quantizer" not in k}
    index["weight_map"] = kept
    index.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(d, s)) for s in sorted(set(kept.values()))
    )
    json.dump(index, open(index_path, "w"), indent=2)
    print("index rewritten")

print(f"TOTAL dropped: {dropped_total}")
