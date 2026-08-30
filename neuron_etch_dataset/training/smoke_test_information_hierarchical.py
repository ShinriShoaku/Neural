"""
smoke_test_information_hierarchical.py
=========================================
Smoke test 2-stage buat adapter information_core.

Cara pakai:
    python smoke_test_information_hierarchical.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/information_core_v1
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from information_chatml import (
    INFORMATION_STAGE1_PROMPT, STAGE2_PROMPTS, CATEGORIES,
    NO_STAGE2_CATEGORIES, AMBIGUOUS_CONSTANT_OUTPUT,
)

# (kalimat, expected_category, expected_target_or_None, expected_parameters_or_None)
# expected_*=None -> tidak dicek detail (cukup category benar)
TEST_CASES = [
    ("Cariin cara pindah kewarganegaraan.", "search", {"type": "web", "value": "cara pindah kewarganegaraan"}, {}),
    ("Gimana cuaca di Surabaya besok?", "weather", {"type": "weather", "value": "tomorrow"}, {"location": "Surabaya"}),
    ("Pukul berapa sekarang di London?", "time", {"type": "time", "value": "now"}, {"location": "London"}),
    ("Tolong artiin 'thank you very much' ke bahasa Jepang.", "translation", None, {"text": "thank you very much", "target_language": "ja"}),
    ("Cek siapa pendiri Amazon.", "lookup", {"type": "entity", "value": "pendiri Amazon"}, None),
    ("Hitung 25 tambah 17.", "calculation", None, {"expression": "25 + 17"}),
    ("Pilih mana, Grab atau Gojek?", "comparison", None, {"items": ["Grab", "Gojek"]}),
    ("Apa maksudnya bioteknologi?", "knowledge", {"type": "knowledge", "value": "bioteknologi"}, None),
    ("Cariin dulu deh.", "ambiguous", None, None),
    ("Simpen info kalau aku alergi seafood.", "negative", None, None),
]


def _get_stop_token_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)
    return list(ids)


def generate(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = 100) -> str:
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


def run_pipeline(model, tokenizer, text: str, max_new_tokens: int) -> dict:
    problems = []
    raw1 = generate(model, tokenizer, INFORMATION_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    if not isinstance(parsed1, dict) or "category" not in parsed1:
        problems.append(f"stage1 JSON tidak valid: {raw1[:100]!r}")
        return {"category": None, "target": None, "parameters": {}, "raw1": raw1, "raw2": None, "problems": problems}

    category = parsed1["category"]
    if category not in CATEGORIES:
        problems.append(f"stage1 category tidak valid: {category!r}")
        return {"category": category, "target": None, "parameters": {}, "raw1": raw1, "raw2": None, "problems": problems}

    if category == "negative":
        return {"category": category, "target": None, "parameters": {}, "raw1": raw1, "raw2": None, "problems": problems}

    if category in NO_STAGE2_CATEGORIES:
        return {"category": category, "target": AMBIGUOUS_CONSTANT_OUTPUT["target"],
                "parameters": AMBIGUOUS_CONSTANT_OUTPUT["parameters"],
                "raw1": raw1, "raw2": None, "problems": problems}

    raw2 = generate(model, tokenizer, STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict):
        problems.append(f"stage2 JSON tidak valid: {raw2[:150]!r}")
        return {"category": category, "target": None, "parameters": {}, "raw1": raw1, "raw2": raw2, "problems": problems}

    target = parsed2.get("target")
    parameters = parsed2.get("parameters", {})
    return {"category": category, "target": target, "parameters": parameters,
            "raw1": raw1, "raw2": raw2, "problems": problems}


def evaluate(result, expected_category, expected_target, expected_parameters):
    problems = list(result["problems"])
    if result["category"] != expected_category:
        problems.append(f"category salah: dapat {result['category']!r}, ekspektasi {expected_category!r}")
        return problems
    if expected_target is not None and result["target"] != expected_target:
        problems.append(f"target salah: dapat {result['target']!r}, ekspektasi {expected_target!r}")
    if expected_parameters is not None and result["parameters"] != expected_parameters:
        problems.append(f"parameters salah: dapat {result['parameters']!r}, ekspektasi {expected_parameters!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
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
    for text, exp_cat, exp_target, exp_params in TEST_CASES:
        result = run_pipeline(model, tokenizer, text, args.max_new_tokens)
        problems = evaluate(result, exp_cat, exp_target, exp_params)
        status = "PASS" if not problems else "FAIL"
        results.append((status, exp_cat))
        print(f"\n[{status}] ({exp_cat}) {text}")
        print(f"    stage1 -> category={result['category']!r}   raw1={result['raw1'][:80]!r}")
        if result["raw2"] is not None:
            print(f"    stage2 -> target={result['target']!r} params={result['parameters']}   raw2={result['raw2'][:150]!r}")
        else:
            print(f"    stage2 -> (di-skip) target={result['target']!r} params={result['parameters']!r}")
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
        print(f"  {cat:14s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
