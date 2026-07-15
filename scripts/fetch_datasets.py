#!/usr/bin/env python3
"""Predownload all eval datasets into the shared HF cache.

Dataset IDs mirror exactly what the installed lm_eval 0.4.12 task YAMLs
reference, so later runs work with HF_HUB_OFFLINE=1. cnn_dailymail is NOT
fetched here — build_calib.py streams it once and the calib jsonl is
self-contained.
"""
import sys

from datasets import get_dataset_config_names, load_dataset

TARGETS = [
    # (repo, config) — full DatasetDict (lm_eval loads all splits)
    ("facebook/belebele", "tha_Thai"),
    ("xnli", "th"),
    ("xcopa", "th"),
    ("scb10x/thai_exam", "a_level"),
    ("scb10x/thai_exam", "ic"),
    ("scb10x/thai_exam", "onet"),
    ("scb10x/thai_exam", "tgat"),
    ("scb10x/thai_exam", "tpat1"),
    ("Rowan/hellaswag", None),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/winogrande", "winogrande_xl"),
    ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1"),
    ("wikimedia/wikipedia", "20231101.th"),
]

MMLU_SUBJECTS = [s for s in get_dataset_config_names("cais/mmlu") if s not in ("all", "auxiliary_train")]
TARGETS += [("cais/mmlu", s) for s in MMLU_SUBJECTS]

failures = []
for repo, config in TARGETS:
    label = f"{repo}/{config or 'default'}"
    try:
        ds = load_dataset(repo, config)
        rows = {k: len(v) for k, v in ds.items()}
        print(f"OK   {label}  {rows}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {label}  {type(e).__name__}: {e}", flush=True)
        failures.append(label)

if failures:
    print("\nFAILED:", *failures, sep="\n  ")
    sys.exit(1)
print(f"\nAll {len(TARGETS)} dataset configs cached.")
