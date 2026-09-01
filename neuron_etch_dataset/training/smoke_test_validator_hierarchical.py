"""
smoke_test_validator_hierarchical.py
=======================================
Smoke test 2-stage buat validator_core.

Cara pakai:
    python smoke_test_validator_hierarchical.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/validator_core_v1
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from validator_chatml import VALIDATOR_STAGE1_PROMPT, VALIDATOR_STAGE2_PROMPT, REASONS

# (original, generated_task_ir, expected_label, expected_reason_or_None)
TEST_CASES = [
    ("Buka spotify.",
     {"domain": "system", "intent": "application_control", "action": "launch",
      "target": {"type": "application", "value": "spotify"}, "parameters": {}},
     "valid", None),

    ("Putar lagu Tulus.",
     {"domain": "media", "intent": "playback", "action": "play",
      "target": {"type": "artist", "value": "Tulus"}, "parameters": {}},
     "valid", None),

    ("Buka firefox.",
     {"domain": "media", "intent": "application_control", "action": "launch",
      "target": {"type": "application", "value": "firefox"}, "parameters": {}},
     "invalid", "domain_mismatch"),

    ("Matiin bluetooth.",
     {"domain": "system", "intent": "hardware_control", "action": "enable_device",
      "target": {"type": "device", "value": "bluetooth"}, "parameters": {}},
     "invalid", "contradiction"),

    ("Cari lagu galau.",
     {"domain": "media", "intent": "search_media", "action": "search",
      "target": {"type": "song", "value": "workout"}, "parameters": {}},
     "invalid", "target_mismatch"),

    ("Debug NullPointerException di file main.py.",
     {"domain": "coding", "intent": "code_debugging", "action": "debug",
      "target": None, "parameters": {"error": "NullPointerException"}},
     "invalid", "missing_parameter"),

    ("Tutup discord.",
     {"domain": "system", "intent": "application_control", "action": "launch",
      "target": {"type": "application", "value": "discord"}, "parameters": {"priority": "urgent"}},
     "invalid", "hallucinated_parameter"),

    ("Tutup discord.",
     {"domain": "system", "intent": "application_control", "action": "restart_app",
      "target": {"type": "application", "value": "discord"}, "parameters": {}},
     "invalid", "intent_mismatch"),

    ("Set volume ke 50%.",
     {"domain": "coding", "intent": "code_generation", "action": "generate",
      "target": None, "parameters": {"language": "python"}},
     "invalid", "unsupported_action"),

    ("Buat script python buat validasi email.",
     {"domain": "coding", "intent": "code_generation", "action": "generate",
      "target": None, "parameters": {"language": "python", "requirements": "scraping website"}},
     "invalid", "parameter_mismatch"),

    ("Cek penggunaan RAM.",
     {"domain": "system", "intent": "information_query", "action": "check_resource",
      "target": {"type": "resource", "value": "itu"}, "parameters": {}},
     "invalid", "ambiguous"),
]


def _get_stop_token_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)
    return list(ids)


def generate(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = 40) -> str:
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


def run_pipeline(model, tokenizer, original: str, generated: dict, max_new_tokens: int) -> dict:
    problems = []
    user_content = f"Instruksi: \"{original}\"\nTask IR: {json.dumps(generated, ensure_ascii=False)}"

    raw1 = generate(model, tokenizer, VALIDATOR_STAGE1_PROMPT, user_content, max_new_tokens=20)
    parsed1, _ = parse_json(raw1)
    if not isinstance(parsed1, dict) or "label" not in parsed1:
        problems.append(f"stage1 JSON tidak valid: {raw1[:100]!r}")
        return {"label": None, "reason": None, "raw1": raw1, "raw2": None, "problems": problems}

    label = parsed1["label"]
    if label not in ("valid", "invalid"):
        problems.append(f"stage1 label tidak valid: {label!r}")
        return {"label": label, "reason": None, "raw1": raw1, "raw2": None, "problems": problems}

    if label == "valid":
        return {"label": label, "reason": None, "raw1": raw1, "raw2": None, "problems": problems}

    raw2 = generate(model, tokenizer, VALIDATOR_STAGE2_PROMPT, user_content, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict) or "reason" not in parsed2:
        problems.append(f"stage2 JSON tidak valid: {raw2[:100]!r}")
        return {"label": label, "reason": None, "raw1": raw1, "raw2": raw2, "problems": problems}

    reason = parsed2["reason"]
    if reason not in REASONS:
        problems.append(f"stage2 reason tidak valid: {reason!r}")

    return {"label": label, "reason": reason, "raw1": raw1, "raw2": raw2, "problems": problems}


def evaluate(result, expected_label, expected_reason):
    problems = list(result["problems"])
    if result["label"] != expected_label:
        problems.append(f"label salah: dapat {result['label']!r}, ekspektasi {expected_label!r}")
        return problems
    if expected_reason is not None and result["reason"] != expected_reason:
        problems.append(f"reason salah: dapat {result['reason']!r}, ekspektasi {expected_reason!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=40)
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
    for original, generated, exp_label, exp_reason in TEST_CASES:
        result = run_pipeline(model, tokenizer, original, generated, args.max_new_tokens)
        problems = evaluate(result, exp_label, exp_reason)
        status = "PASS" if not problems else "FAIL"
        results.append((status, exp_reason or exp_label))
        print(f"\n[{status}] ({exp_reason or exp_label}) {original}")
        print(f"    stage1 -> label={result['label']!r}   raw1={result['raw1'][:60]!r}")
        if result["raw2"] is not None:
            print(f"    stage2 -> reason={result['reason']!r}   raw2={result['raw2'][:80]!r}")
        else:
            print("    stage2 -> (di-skip, label=valid)")
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
        print(f"  {cat:24s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
