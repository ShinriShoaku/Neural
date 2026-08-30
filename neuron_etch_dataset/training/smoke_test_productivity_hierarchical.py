"""
smoke_test_productivity_hierarchical.py
==========================================
Smoke test 2-stage buat adapter productivity_core.

Cara pakai:
    python smoke_test_productivity_hierarchical.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/productivity_core_v1
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from productivity_chatml import (
    PRODUCTIVITY_STAGE1_PROMPT, STAGE2_PROMPTS, CATEGORIES,
    CATEGORY_TO_ACTION, CATEGORY_TO_INTENT, NO_STAGE2_CATEGORIES, AMBIGUOUS_CONSTANT_OUTPUT,
)

# (kalimat, expected_category, expected_payload_or_None)
TEST_CASES = [
    ("Bikin acara presentasi project besok jam 2 siang.", "calendar",
     {"target": {"type": "event", "value": "presentasi project"}, "parameters": {"time": "besok jam 2 siang"}}),
    ("Tolong ingetin aku jam 7 malam nanti kirim invoice.", "reminder",
     {"parameters": {"time": "jam 7 malam nanti", "content": "kirim invoice"}}),
    ("Catet todo follow up klien.", "todo",
     {"action": "create", "target": {"type": "todo", "value": "follow up klien"}}),
    ("Centang todo beli obat udah kelar.", "todo",
     {"action": "complete", "target": {"type": "todo", "value": "beli obat"}}),
    ("Set rutinitas jalan pagi tiap hari jam 7 pagi.", "schedule",
     {"target": {"type": "event", "value": "jalan pagi"}, "parameters": {"recurrence": "daily 07:00"}}),
    ("Alert aku kalau ada email baru masuk.", "notification",
     {"target": {"type": "notification", "value": "ada email baru masuk"}}),
    ("Kirim pesan 'oke ditunggu ya'.", "communication",
     {"target": {"type": "message", "value": "oke ditunggu ya"}}),
    ("Cancel todo antar paket ke kantor pos.", "update_delete",
     {"intent": "todo", "action": "delete", "target": {"type": "todo", "value": "antar paket ke kantor pos"}, "parameters": {}}),
    ("Geser aja deh.", "ambiguous", None),
    ("Cari tahu ibu kota Jepang.", "negative", None),
]


def _get_stop_token_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)
    return list(ids)


def generate(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = 120) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=_get_stop_token_ids(tokenizer),
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(raw_text: str):
    text = raw_text.strip()
    for marker in ("<|im_end|>", "\nuser\n", "\nassistant\n", "<|im_start|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def build_full_output(category: str, stage2_payload: dict) -> dict:
    """Gabungkan payload stage2 (bentuknya beda per kategori) jadi Task IR
    penuh, isi field yang deterministik dari lookup table."""
    if category == "todo":
        return {"intent": "todo", "action": stage2_payload.get("action"),
                "target": stage2_payload.get("target"), "parameters": {}}
    if category == "update_delete":
        return {"intent": stage2_payload.get("intent"), "action": stage2_payload.get("action"),
                "target": stage2_payload.get("target"), "parameters": stage2_payload.get("parameters", {})}
    # kategori dengan action/intent deterministik
    return {"intent": CATEGORY_TO_INTENT[category], "action": CATEGORY_TO_ACTION[category],
            "target": stage2_payload.get("target"), "parameters": stage2_payload.get("parameters", {})}


def run_pipeline(model, tokenizer, text: str, max_new_tokens: int) -> dict:
    problems = []
    raw1 = generate(model, tokenizer, PRODUCTIVITY_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    if not isinstance(parsed1, dict) or "category" not in parsed1:
        problems.append(f"stage1 JSON tidak valid: {raw1[:100]!r}")
        return {"category": None, "output": None, "raw1": raw1, "raw2": None, "problems": problems}

    category = parsed1["category"]
    if category not in CATEGORIES:
        problems.append(f"stage1 category tidak valid: {category!r}")
        return {"category": category, "output": None, "raw1": raw1, "raw2": None, "problems": problems}

    if category == "negative":
        return {"category": category, "output": None, "raw1": raw1, "raw2": None, "problems": problems}

    if category in NO_STAGE2_CATEGORIES:
        return {"category": category, "output": AMBIGUOUS_CONSTANT_OUTPUT,
                "raw1": raw1, "raw2": None, "problems": problems}

    raw2 = generate(model, tokenizer, STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict):
        problems.append(f"stage2 JSON tidak valid: {raw2[:150]!r}")
        return {"category": category, "output": None, "raw1": raw1, "raw2": raw2, "problems": problems}

    output = build_full_output(category, parsed2)
    return {"category": category, "output": output, "raw1": raw1, "raw2": raw2, "problems": problems}


def evaluate(result, expected_category, expected_payload):
    problems = list(result["problems"])
    if result["category"] != expected_category:
        problems.append(f"category salah: dapat {result['category']!r}, ekspektasi {expected_category!r}")
        return problems
    if expected_payload is None:
        return problems
    output = result["output"] or {}
    for key, exp_val in expected_payload.items():
        actual_val = output.get(key)
        if actual_val != exp_val:
            problems.append(f"{key} salah: dapat {actual_val!r}, ekspektasi {exp_val!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    print(f"Loading base model dari {args.base_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"Loading adapter dari {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    results = []
    for text, exp_cat, exp_payload in TEST_CASES:
        result = run_pipeline(model, tokenizer, text, args.max_new_tokens)
        problems = evaluate(result, exp_cat, exp_payload)
        status = "PASS" if not problems else "FAIL"
        results.append((status, exp_cat))
        print(f"\n[{status}] ({exp_cat}) {text}")
        print(f"    stage1 -> category={result['category']!r}   raw1={result['raw1'][:80]!r}")
        if result["raw2"] is not None:
            print(f"    stage2 -> output={result['output']}   raw2={result['raw2'][:150]!r}")
        else:
            print(f"    stage2 -> (di-skip) output={result['output']!r}")
        for p in problems:
            print(f"    - {p}")

    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_total = len(results)
    print(f"\n{'=' * 50}\nRINGKASAN: {n_pass}/{n_total} PASS ({n_pass/n_total*100:.0f}%)")

    by_cat: dict[str, list[str]] = {}
    for status, cat in results:
        by_cat.setdefault(cat, []).append(status)
    print("\nPer kategori:")
    for cat, statuses in by_cat.items():
        p = statuses.count("PASS")
        print(f"  {cat:16s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
