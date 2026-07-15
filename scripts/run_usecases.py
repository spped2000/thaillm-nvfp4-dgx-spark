#!/usr/bin/env python3
"""Use-case level BF16-vs-NVFP4 comparison against the live eval server.

Usage: run_usecases.py <bf16|nvfp4>
Part A: greedy generations for 12 domain prompts (Thai news/legal/medical/
        business/education/travel/QA + English + code), 200 tokens.
Part B: teacher-forced per-token fidelity — echo+logprobs over 40 held-out
        passages (20 Thai wiki, 20 wikitext) for top-token agreement analysis.
Writes results/<tag>/usecases.json — aggregate_report.py compares the two.
"""
import json
import sys
from pathlib import Path

import requests

TAG = sys.argv[1]
P = Path(__file__).resolve().parent.parent
OUT = P / "results" / TAG / "usecases.json"
URL = "http://127.0.0.1:8001/v1/completions"
MODEL = "eval-model"

PROMPTS = {
    "th_factual_qa": "คำถาม: ประเทศไทยมีทั้งหมดกี่จังหวัด\nคำตอบ:",
    "th_news": "กรุงเทพฯ – ธนาคารแห่งประเทศไทยแถลงวันนี้ว่า เศรษฐกิจไทยในไตรมาสที่ผ่านมา",
    "th_legal": "ตามประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 420 ผู้ใดจงใจหรือประมาทเลินเล่อ",
    "th_medical": "โรคเบาหวานชนิดที่ 2 มีสาเหตุหลักมาจาก",
    "th_education": "การสังเคราะห์ด้วยแสง (photosynthesis) คือกระบวนการที่",
    "th_business": "การวิเคราะห์งบการเงินของบริษัทประกอบด้วยขั้นตอนสำคัญดังนี้",
    "th_travel": "จังหวัดเชียงใหม่มีสถานที่ท่องเที่ยวที่มีชื่อเสียง ได้แก่",
    "th_math": "โจทย์: ซื้อของราคา 250 บาท จ่ายด้วยธนบัตร 500 บาท จะได้เงินทอนเท่าไร\nวิธีคิด:",
    "en_econ": "The main causes of inflation are",
    "en_science": "Photosynthesis is the process by which",
    "code_python": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "th_en_translate": "คำว่า \"ความยั่งยืน\" ในภาษาอังกฤษคือ",
}


def completions(payload):
    r = requests.post(URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def part_a():
    gens = {}
    for name, prompt in PROMPTS.items():
        d = completions({
            "model": MODEL, "prompt": prompt, "max_tokens": 200,
            "temperature": 0.0, "seed": 0, "logprobs": 1,
        })
        ch = d["choices"][0]
        gens[name] = {
            "prompt": prompt,
            "text": ch["text"],
            "tokens": ch.get("logprobs", {}).get("tokens"),
            "finish_reason": ch.get("finish_reason"),
        }
        print(f"gen {name}: {len(ch['text'])} chars")
    return gens


def passages():
    from datasets import load_dataset
    th = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
    th_docs, i = [], 0
    # skip the first ~1000 filtered docs (used by thai_wikipedia_ppl) — start deep
    seen_ok = 0
    for row in th:
        if len(row["text"]) >= 2000:
            seen_ok += 1
            if seen_ok > 3000 and len(th_docs) < 20:
                th_docs.append(row["text"][:3000])
        if len(th_docs) >= 20:
            break
    en = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")
    en_docs = [d["page"][:3000] for d in en if len(d["page"]) >= 2000][:20]
    return {"th": th_docs, "en": en_docs}


def part_b():
    out = {}
    for lang, docs in passages().items():
        recs = []
        for i, doc in enumerate(docs):
            d = completions({
                "model": MODEL, "prompt": doc, "max_tokens": 0,
                "temperature": 0.0, "echo": True, "logprobs": 1,
            })
            lp = d["choices"][0]["logprobs"]
            recs.append({
                "idx": i,
                "tokens": lp["tokens"],
                "token_logprobs": lp["token_logprobs"],
                # top_logprobs[k] = {predicted_top_token: logprob} at each position
                "top_tokens": [
                    (max(t, key=t.get) if t else None)
                    for t in (lp.get("top_logprobs") or [])
                ],
            })
            print(f"tf {lang} {i}: {len(lp['tokens'])} tokens")
        out[lang] = recs
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {"generations": part_a(), "teacher_forced": part_b()}
    OUT.write_text(json.dumps(result, ensure_ascii=False))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
