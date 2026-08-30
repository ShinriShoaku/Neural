"""
smoke_test_media_hierarchical.py
===================================
Smoke test 2-stage buat adapter media_core. Pola sama persis dengan
smoke_test_system_hierarchical.py.

Cara pakai:
    python smoke_test_media_hierarchical.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/media_core_v1
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from media_chatml import MEDIA_STAGE1_PROMPT, STAGE2_PROMPTS, CATEGORIES

# (kalimat, expected_category, expected_action, expected_target_value_or_None, expected_parameters)
TEST_CASES = [
    ("Puterin lagu Payung Teduh dong.", "playback", "play", "Payung Teduh", {}),
    ("Setel album Bumi ya.", "playback", "play", "Bumi", {}),
    ("Terusin lagunya.", "playback", "resume", None, {}),

    ("Temuin lagu galau di Joox.", "search", "search", "galau", {"player": "joox"}),

    ("Masukin lagu ini ke antrian dong.", "queue", "queue_add", "current", {}),
    ("Bersihin semua antrian.", "queue", "queue_clear", "current", {}),
    ("Kasih liat isi antrian dong.", "queue", "queue_list", "current", {}),

    ("Jeda dulu musiknya.", "player_control", "pause", "current", {}),
    ("Mundur satu lagu dong.", "player_control", "previous", "current", {}),
    ("Acakin urutan lagu.", "player_control", "shuffle", "current", {}),
    ("Berhentiin musik ini.", "player_control", "stop", "current", {}),

    ("Buka video review HP di Netflix.", "streaming", "play_stream", "current", {"player": "netflix"}),
    ("Load link video ini deh.", "streaming", "resolve", "url_placeholder", {}),

    ("Tau nggak nama artis lagu ini?", "metadata", "query_metadata", "current", {"field": "nama artis"}),

    ("Pindah ke yang lain.", "ambiguous", "play", None, {}),
    ("Ini kurang pas, ganti dong.", "ambiguous", "play", None, {}),

    ("Siapa member Coldplay?", "negative", None, None, {}),
    ("Kapan Tulus konser di Bandung?", "negative", None, None, {}),
]


def _get_stop_token_ids(tokenizer) -> list[int]:
    """<|im_end|> HARUS jadi stop token eksplisit -- kalau cuma andalin
    default eos_token_id, generate() kadang nggak berhenti di situ dan
    lanjut nge-halusinasi giliran percakapan baru."""
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
    # Jaring pengaman: potong di penanda giliran baru kalau model kelanjutan
    # nge-halusinasi turn berikutnya (harusnya sudah dicegah oleh eos_token_id
    # eksplisit di generate(), tapi ini tetap dijaga untuk adapter lama).
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
    raw1 = generate(model, tokenizer, MEDIA_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    if not isinstance(parsed1, dict) or "category" not in parsed1:
        problems.append(f"stage1 JSON tidak valid: {raw1[:100]!r}")
        return {"category": None, "action": None, "target_value": None, "parameters": {},
                "raw1": raw1, "raw2": None, "problems": problems}

    category = parsed1["category"]
    if category not in CATEGORIES:
        problems.append(f"stage1 category tidak valid: {category!r}")
        return {"category": category, "action": None, "target_value": None, "parameters": {},
                "raw1": raw1, "raw2": None, "problems": problems}

    if category == "negative":
        return {"category": category, "action": None, "target_value": None, "parameters": {},
                "raw1": raw1, "raw2": None, "problems": problems}

    raw2 = generate(model, tokenizer, STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict) or "action" not in parsed2:
        problems.append(f"stage2 JSON tidak valid: {raw2[:150]!r}")
        return {"category": category, "action": None, "target_value": None, "parameters": {},
                "raw1": raw1, "raw2": raw2, "problems": problems}

    action = parsed2.get("action")
    target = parsed2.get("target")
    target_value = target.get("value") if isinstance(target, dict) else None
    parameters = parsed2.get("parameters", {})
    return {"category": category, "action": action, "target_value": target_value,
            "parameters": parameters, "raw1": raw1, "raw2": raw2, "problems": problems}


def evaluate(result, expected_category, expected_action, expected_target_value, expected_parameters):
    problems = list(result["problems"])
    if result["category"] != expected_category:
        problems.append(f"category salah: dapat {result['category']!r}, ekspektasi {expected_category!r}")
        return problems
    if expected_action is None:
        return problems
    if result["action"] != expected_action:
        problems.append(f"action salah: dapat {result['action']!r}, ekspektasi {expected_action!r}")
    if expected_target_value is not None and result["target_value"] != expected_target_value:
        problems.append(f"target salah: dapat {result['target_value']!r}, ekspektasi {expected_target_value!r}")
    if expected_parameters and result["parameters"] != expected_parameters:
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
    for text, exp_cat, exp_action, exp_target, exp_params in TEST_CASES:
        result = run_pipeline(model, tokenizer, text, args.max_new_tokens)
        problems = evaluate(result, exp_cat, exp_action, exp_target, exp_params)
        status = "PASS" if not problems else "FAIL"
        results.append((status, exp_cat))
        print(f"\n[{status}] ({exp_cat}) {text}")
        print(f"    stage1 -> category={result['category']!r}   raw1={result['raw1'][:80]!r}")
        if result["raw2"] is not None:
            print(f"    stage2 -> action={result['action']!r} target={result['target_value']!r} "
                  f"params={result['parameters']}   raw2={result['raw2'][:120]!r}")
        else:
            print("    stage2 -> (di-skip)")
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
