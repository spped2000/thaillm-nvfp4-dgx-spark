#!/usr/bin/env python3
"""Thai-aware latency benchmark for the agentic serving profile.

Measures what the Thai efficiency scorecard needs, in the shape the workload
actually has:

  * agentic  - a long, REUSED system prompt + tool schemas with a short varying
               user turn. This is the pattern prefix caching accelerates, so
               running it before/after `--enable-prefix-caching` shows the real
               TTFT win.
  * rag      - a long Thai document context + question (prefill-heavy).
  * chat     - a short Thai turn (decode-dominated).

Reports TTFT, decode tok/s and **chars/sec** (the honest Thai KPI: Thai costs
~2.3x more tokens per character than English, so tok/s alone flatters Thai).

Usage:
  python thai_latency_bench.py --url http://127.0.0.1:8000/v1 --model chat \
      --label before_prefix_cache --out results/phase0/latency_before.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request

THAI_RE = re.compile(r"[\u0e00-\u0e7f]")


def thai_chars(s: str) -> int:
    """Count only Thai codepoints.

    The honest Thai KPI: counting every character lets ASCII, markdown and code
    fences inflate chars/sec, which is why earlier runs reported 3.0-4.6 chars per
    token while the true Thai figure is ~1.7-2.1.
    """
    return len(THAI_RE.findall(s))

# A realistic reused agentic preamble (system + tool schemas), in Thai.
SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยอัจฉริยะขององค์กรภาครัฐไทย ทำหน้าที่ค้นหาและสรุปเอกสารราชการ "
    "ตอบเป็นภาษาไทยที่สุภาพ กระชับ และอ้างอิงแหล่งที่มาเสมอ "
    "เมื่อผู้ใช้ต้องการข้อมูลจากเอกสาร ให้เรียกใช้เครื่องมือที่เหมาะสม "
    "ห้ามเดาข้อมูลที่ไม่มีในเอกสาร หากไม่พบข้อมูลให้แจ้งผู้ใช้ตรงไปตรงมา "
    "รูปแบบการตอบ: สรุปสั้น 2-3 ประโยค ตามด้วยรายละเอียดเป็นข้อ ๆ และปิดท้ายด้วยแหล่งอ้างอิง "
) * 6  # ~long reused prefix, the thing prefix caching should cache

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "ค้นหาเอกสารราชการภายในระบบ",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "คำค้นหาภาษาไทย"},
                    "province": {"type": "string", "description": "ชื่อจังหวัด"},
                    "year": {"type": "integer", "description": "ปี พ.ศ."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "สรุปเอกสารตามรหัสเอกสาร",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "max_words": {"type": "integer"},
                },
                "required": ["doc_id"],
            },
        },
    },
]

THAI_DOC = (
    "ระเบียบกระทรวงมหาดไทยว่าด้วยการจัดทำแผนพัฒนาขององค์กรปกครองส่วนท้องถิ่น "
    "กำหนดให้องค์กรปกครองส่วนท้องถิ่นจัดทำแผนพัฒนาท้องถิ่นให้สอดคล้องกับแผนพัฒนาจังหวัด "
    "โดยต้องผ่านความเห็นชอบจากคณะกรรมการพัฒนาท้องถิ่นและประกาศใช้ภายในระยะเวลาที่กำหนด "
    "ทั้งนี้ให้คำนึงถึงการมีส่วนร่วมของประชาชนในพื้นที่เป็นสำคัญ "
) * 25

USER_TURNS = [
    "ขอเอกสารเกี่ยวกับการจัดทำแผนพัฒนาท้องถิ่นของจังหวัดเชียงใหม่ปี 2569",
    "ช่วยค้นหาระเบียบเรื่องการมีส่วนร่วมของประชาชนหน่อย",
    "สรุปขั้นตอนการประกาศใช้แผนพัฒนาให้หน่อยครับ",
    "มีเอกสารเกี่ยวกับคณะกรรมการพัฒนาท้องถิ่นของอุบลราชธานีไหม",
]


def stream_once(url: str, payload: dict, timeout: int = 300) -> dict:
    """POST with stream=true; measure TTFT and per-token timing."""
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                delta = ch.get("delta", {}) or {}
                # Reasoning models (Qwen3.6 + --reasoning-parser) stream thinking
                # tokens in `delta.reasoning` and the answer in `delta.content`.
                # The user waits for both, so both count toward latency; keep the
                # answer text separate for the chars/sec KPI.
                answer = delta.get("content")
                if isinstance(answer, list):
                    answer = "".join(p.get("text", "") or p.get("reasoning", "") for p in answer)
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if answer or reasoning:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
                if answer:
                    text_parts.append(answer)
                if reasoning:
                    reasoning_parts.append(reasoning)
    total = time.perf_counter() - t0
    text = "".join(text_parts)
    reasoning_text = "".join(reasoning_parts)
    out_tok = usage.get("completion_tokens") or usage.get("output_tokens") or n_chunks
    decode_s = max(total - (ttft or 0), 1e-6)
    all_chars = len(text) + len(reasoning_text)
    thai_all = thai_chars(text) + thai_chars(reasoning_text)
    return {
        "ttft_s": round(ttft or total, 4),
        "total_s": round(total, 4),
        "output_tokens": out_tok,
        "answer_chars": len(text),
        "reasoning_chars": len(reasoning_text),
        "thai_chars": thai_all,
        "decode_tok_s": round(out_tok / decode_s, 2) if out_tok else 0.0,
        # THE Thai KPI: Thai codepoints only. `decode_chars_s_all` is kept so
        # numbers measured before this fix stay comparable.
        "decode_chars_s": round(thai_all / decode_s, 2),
        "decode_chars_s_all": round(all_chars / decode_s, 2),
        "chars_per_token": round(thai_all / out_tok, 3) if out_tok else 0.0,
        "chars_per_token_all": round(all_chars / out_tok, 3) if out_tok else 0.0,
    }


def scenario_payload(kind: str, model: str, turn: str, max_tokens: int,
                     no_think: bool = False) -> dict:
    """Build one request.

    `no_think` matters for cross-model comparison. A server started WITHOUT
    `--reasoning-parser` returns the chain-of-thought inside `content` as raw `<think>`
    text, so at a small `max_tokens` the model spends the whole budget thinking in
    English and never reaches its Thai answer — which reads as "this model produces no
    Thai" when it produces plenty. Measured on Qwen3-8B here: Thai chars/s came out
    0.0-0.7 with `finish_reason: length`, purely from this. Disabling thinking in the
    request makes the comparison about the models rather than about which server flags
    each one happened to get.
    """
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_think else {}
    if kind == "agentic":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": turn},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": max_tokens,
            **extra,
        }
    if kind == "rag":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": "ตอบคำถามจากเอกสารที่ให้มาเท่านั้น"},
                {"role": "user", "content": f"เอกสาร:\n{THAI_DOC}\n\nคำถาม: {turn}"},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            **extra,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": turn}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="chat")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--scenarios", default="agentic,rag,chat")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the user turns")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false — required when the "
                         "server has no --reasoning-parser, or thinking lands in content and "
                         "eats the whole max_tokens budget")
    args = ap.parse_args()

    results: dict[str, list[dict]] = {}
    for kind in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        rows = []
        for rep in range(args.repeats):
            for turn in USER_TURNS:
                payload = scenario_payload(kind, args.model, turn, args.max_tokens, args.no_think)
                try:
                    m = stream_once(args.url, payload)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{kind}] request failed: {type(e).__name__}: {e}", flush=True)
                    continue
                m["rep"] = rep
                rows.append(m)
                print(
                    f"  [{kind}] rep{rep} ttft={m['ttft_s']:.3f}s "
                    f"tok/s={m['decode_tok_s']:.1f} chars/s={m['decode_chars_s']:.1f} "
                    f"ch/tok={m['chars_per_token']:.2f}",
                    flush=True,
                )
        results[kind] = rows

    print(f"\n=== SUMMARY [{args.label}] ===")
    print(f"{'scenario':10s} {'n':>3s} {'TTFT p50':>9s} {'TTFT p90':>9s} {'tok/s':>8s} {'chars/s':>9s} {'ch/tok':>7s}")
    summary = {}
    for kind, rows in results.items():
        if not rows:
            continue
        ttfts = sorted(r["ttft_s"] for r in rows)
        p50 = statistics.median(ttfts)
        p90 = ttfts[max(0, int(len(ttfts) * 0.9) - 1)]
        toks = statistics.median(r["decode_tok_s"] for r in rows)
        chs = statistics.median(r["decode_chars_s"] for r in rows)
        # A LEGITIMATE ZERO IS NOT MISSING DATA. The `if r["chars_per_token"]` filter drops
        # every 0.0, and when a scenario genuinely emits no Thai at all (Typhoon answering the
        # agentic prompt with thinking disabled produced 0.00 Thai chars on all 16 requests) the
        # generator was empty and statistics.median raised, killing the whole run — so the model
        # was recorded as "no artifact" when the honest answer was "zero Thai in this scenario".
        # Zero is a measurement. Report it, and flag it so it is never read as an error.
        cpts = [r["chars_per_token"] for r in rows if r["chars_per_token"]]
        cpt = statistics.median(cpts) if cpts else 0.0
        zero_thai = not cpts
        summary[kind] = {
            "n": len(rows), "ttft_p50_s": round(p50, 4), "ttft_p90_s": round(p90, 4),
            "decode_tok_s_median": round(toks, 2), "decode_chars_s_median": round(chs, 2),
            "chars_per_token_median": round(cpt, 3),
            "zero_thai_output": zero_thai,
        }
        print(f"{kind:10s} {len(rows):>3} {p50:>9.3f} {p90:>9.3f} {toks:>8.1f} {chs:>9.1f} {cpt:>7.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "summary": summary, "raw": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
