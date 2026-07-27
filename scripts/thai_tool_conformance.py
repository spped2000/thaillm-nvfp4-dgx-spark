#!/usr/bin/env python3
"""Thai tool-calling / structured-output conformance test.

Answers the questions the Thai-efficiency scorecard needs about tool calling:

  1. Does the model emit well-formed tool calls when the *arguments are Thai*
     (place names, Thai dates, Thai amounts)?
  2. Does it survive the three decoding modes that behave differently in vLLM?
       - tool_choice="auto"      -> no constrained decoding (Thai args safest)
       - tool_choice="required"  -> grammar-constrained (xgrammar non-ASCII risk)
       - guided/json_schema      -> grammar-constrained response_format
  3. Are Thai string values preserved byte-exact (no mojibake, no dropped
     combining marks, no U+FFFD) after a JSON round-trip?

Usage:
  python thai_tool_conformance.py --url http://127.0.0.1:8000/v1 --model chat
  python thai_tool_conformance.py ... --out results/phase0/tool_conformance.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

# Thai text hazards this test deliberately exercises:
#  - combining marks (tone marks / above-below vowels) inside JSON string values
#  - sara am U+0E33 (precomposed) which byte-level BPE splits mid-character
#  - digits in Thai numerals + Buddhist-era dates
#  - Thai place names with spaces absent (no word boundaries)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search internal documents. ค้นหาเอกสารภายในองค์กร",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "คำค้นหา (Thai search query)"},
                    "province": {"type": "string", "description": "ชื่อจังหวัด e.g. เชียงใหม่"},
                    "doc_type": {
                        "type": "string",
                        # ASCII enum values on purpose: the recommended pattern is
                        # ASCII keys/enums mapped to Thai in-app (xgrammar #418).
                        "enum": ["house_registration", "id_card", "land_deed", "invoice"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": "Create an appointment. สร้างนัดหมาย",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "ชื่อ-นามสกุลผู้ป่วย"},
                    "hospital": {"type": "string", "description": "ชื่อโรงพยาบาล"},
                    "date_th": {"type": "string", "description": "วันที่แบบไทย เช่น 15 มีนาคม 2569"},
                    "note": {"type": "string", "description": "หมายเหตุภาษาไทย"},
                },
                "required": ["patient_name", "hospital", "date_th"],
            },
        },
    },
]

CASES = [
    {
        "id": "search_province_combining_marks",
        "prompt": "ช่วยค้นหาเอกสารทะเบียนบ้านของจังหวัดเชียงใหม่ที่เกี่ยวกับน้ำประปาให้หน่อย",
        "expect_tool": "search_documents",
        # every one of these carries combining marks / sara am
        "expect_thai_in_args": ["เชียงใหม่"],
    },
    {
        "id": "search_sara_am_and_tone",
        "prompt": "ค้นหาใบแจ้งหนี้ค่าน้ำของจังหวัดอุบลราชธานี เรื่องค่าน้ำประปาเดือนนี้",
        "expect_tool": "search_documents",
        "expect_thai_in_args": ["อุบลราชธานี"],
    },
    {
        "id": "appointment_thai_name_date",
        "prompt": "จองนัดหมายให้คุณสมหญิง ศรีสุขใจ ที่โรงพยาบาลศิริราช วันที่ 15 มีนาคม 2569 หมายเหตุว่าผู้ป่วยแพ้ยาเพนนิซิลลิน",
        "expect_tool": "create_appointment",
        "expect_thai_in_args": ["สมหญิง", "ศิริราช"],
    },
    {
        "id": "appointment_northern_hospital",
        "prompt": "นัดหมายคุณธีร์ ณ เชียงใหม่ ที่โรงพยาบาลมหาราชนครเชียงใหม่ วันที่ 1 เมษายน 2569 หมายเหตุ ผู้ป่วยมีอาการไข้สูงและปวดศีรษะ",
        "expect_tool": "create_appointment",
        "expect_thai_in_args": ["เชียงใหม่"],
    },
]

THAI_RE = re.compile(r"[฀-๿]")


def has_thai(s: str) -> bool:
    return bool(THAI_RE.search(s))


def text_hazards(s: str) -> list[str]:
    """Detect corruption of Thai text after a JSON round-trip."""
    bad = []
    if "�" in s:
        bad.append("replacement_char_U+FFFD")
    # mojibake signature: Thai UTF-8 bytes decoded as latin-1
    if "à¸" in s or "à¹" in s:
        bad.append("mojibake_latin1")
    # a combining mark with no base character before it
    for i, ch in enumerate(s):
        if unicodedata.combining(ch) and (i == 0 or not ("฀" <= s[i - 1] <= "๿")):
            bad.append(f"orphan_combining_mark_at_{i}")
            break
    return bad


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_tool_calls(resp: dict) -> list[dict]:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        return []
    calls = msg.get("tool_calls") or []
    out = []
    for c in calls:
        fn = c.get("function", {})
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            parse_ok = True
        except json.JSONDecodeError:
            args, parse_ok = {}, False
        out.append({"name": fn.get("name"), "arguments": args, "raw": raw, "json_valid": parse_ok,
                    # keep the RAW string: json.loads silently turns \u0e41 back into แ, so a
                    # model that escaped Thai would still score "Thai preserved" while paying
                    # ~4.2x the tokens. Only the raw text can reveal it.
                    "raw_has_unicode_escape": bool(isinstance(raw, str) and re.search(r"\\u0[eE]", raw))})
    return out


def run_case(url: str, model: str, case: dict, mode: str, max_tokens: int,
             no_think: bool = False) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "คุณเป็นผู้ช่วยที่เรียกใช้เครื่องมือเมื่อจำเป็น ตอบเป็นภาษาไทย"},
            {"role": "user", "content": case["prompt"]},
        ],
        "tools": TOOLS,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if mode == "auto":
        payload["tool_choice"] = "auto"
    elif mode == "required":
        payload["tool_choice"] = "required"
    elif mode == "named":
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": case["expect_tool"]},
        }

    rec: dict = {"case": case["id"], "mode": mode, "thinking": not no_think}
    t0 = time.time()
    try:
        resp = post(f"{url}/chat/completions", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        rec.update(ok=False, error=f"HTTP {e.code}", detail=body, latency_s=round(time.time() - t0, 2))
        return rec
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=type(e).__name__, detail=str(e)[:300], latency_s=round(time.time() - t0, 2))
        return rec

    rec["latency_s"] = round(time.time() - t0, 2)
    calls = extract_tool_calls(resp)
    rec["n_tool_calls"] = len(calls)

    if not calls:
        content = (resp["choices"][0]["message"].get("content") or "")
        if isinstance(content, list):  # reasoning-style content parts
            content = " ".join(p.get("text", "") or p.get("reasoning", "") for p in content)
        rec.update(ok=False, error="no_tool_call", content_head=str(content)[:200])
        return rec

    call = calls[0]
    args_str = json.dumps(call["arguments"], ensure_ascii=False)
    hazards = text_hazards(args_str)
    if call.get("raw_has_unicode_escape"):
        hazards.append("thai_escaped_as_unicode_in_raw_output")
    thai_present = [t for t in case["expect_thai_in_args"] if t in args_str]

    rec.update(
        ok=call["json_valid"] and not hazards,
        tool_name=call["name"],
        tool_correct=call["name"] == case["expect_tool"],
        json_valid=call["json_valid"],
        raw_escaped_thai=bool(call.get("raw_has_unicode_escape")),
        raw_args=(call.get("raw") if isinstance(call.get("raw"), str) else None),
        args_have_thai=has_thai(args_str),
        thai_values_preserved=thai_present,
        thai_values_expected=case["expect_thai_in_args"],
        thai_preserved_all=len(thai_present) == len(case["expect_thai_in_args"]),
        hazards=hazards,
        args=call["arguments"],
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="chat")
    ap.add_argument("--modes", default="auto,required,named")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true",
                    help="disable reasoning (chat_template_kwargs.enable_thinking=false)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = []
    for mode in modes:
        for case in CASES:
            r = run_case(args.url, args.model, case, mode, args.max_tokens, args.no_think)
            results.append(r)
            flag = "PASS" if r.get("ok") else "FAIL"
            extra = r.get("error") or (
                f"tool={r.get('tool_name')} thai_ok={r.get('thai_preserved_all')} hazards={r.get('hazards')}"
            )
            print(f"[{flag}] {mode:9s} {r['case']:34s} {extra}", flush=True)

    summary = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            "n": len(rows),
            "tool_call_emitted": sum(1 for r in rows if r.get("n_tool_calls", 0) > 0),
            "correct_tool": sum(1 for r in rows if r.get("tool_correct")),
            "json_valid": sum(1 for r in rows if r.get("json_valid")),
            "thai_preserved_all": sum(1 for r in rows if r.get("thai_preserved_all")),
            "with_hazards": sum(1 for r in rows if r.get("hazards")),
            "errors": sorted({r["error"] for r in rows if r.get("error")}),
        }

    print("\n=== SUMMARY (Thai tool-calling conformance) ===")
    print(f"{'mode':10s} {'calls':>6s} {'tool_ok':>8s} {'json_ok':>8s} {'thai_ok':>8s} {'hazard':>7s}  errors")
    for mode, s in summary.items():
        print(
            f"{mode:10s} {s['tool_call_emitted']:>3}/{s['n']:<2} {s['correct_tool']:>7}  "
            f"{s['json_valid']:>7}  {s['thai_preserved_all']:>7}  {s['with_hazards']:>6}  {','.join(s['errors']) or '-'}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
