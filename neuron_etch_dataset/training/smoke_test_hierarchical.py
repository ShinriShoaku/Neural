"""
smoke_test_hierarchical.py
============================
Smoke test buat adapter router hasil build_router_hierarchical.py.
BEDA dari smoke test sebelumnya: ini manggil model 2 KALI per input
(kecuali kalau stage1="ambiguous", cukup 1 kali panggilan karena stage2
di-skip, langsung domain=unknown):

    1. Stage 1 (ROUTER_STAGE1_PROMPT) -> dapat category
    2. Berdasarkan category:
       - "single_intent" / "implicit_intent" -> Stage 2a (domain tunggal)
       - "multi_intent"                       -> Stage 2b (segmentasi)
       - "ambiguous"                          -> langsung domain=unknown,
                                                  TIDAK panggil model lagi

Cara pakai:
    python smoke_test_hierarchical.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/router_core_v7
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from chatml_format import (
    ROUTER_STAGE1_PROMPT, ROUTER_STAGE2_SINGLE_PROMPT, ROUTER_STAGE2_MULTI_PROMPT,
)

VALID_DOMAINS = {"system", "media", "persona", "coding", "information",
                  "memory", "productivity", "unknown"}
VALID_CATEGORIES = {"single_intent", "multi_intent", "implicit_intent", "ambiguous"}

# SAMA PERSIS test set yang dipakai di percobaan sebelumnya, supaya adil.
TEST_CASES = [
    ("Nyalain wifi laptop aku dong.", "single_intent", ["system"]),
    ("Putarin lagu Payung Teduh yang galau.", "single_intent", ["media"]),
    ("Kirim chat ke Kaito nanya kabar.", "single_intent", ["persona"]),
    ("Perbaikin bug di file server.js ini.", "single_intent", ["coding"]),

    ("Buka spotify terus muter lagu Hindia yang paling baru.", "multi_intent", ["system", "media"]),
    ("Cek cuaca hari ini terus ingetin aku bawa payung nanti sore.", "multi_intent", ["information", "productivity"]),
    ("Tutup discord dan bikinin reminder meeting jam 2.", "multi_intent", ["system", "productivity"]),
    ("Cariin resep rendang terus simpen ke catatanku.", "multi_intent", ["information", "memory"]),

    ("Pause dulu musiknya terus lanjutin lagi abis ini.", "compound_command", ["media"]),
    ("Buka terminal terus langsung jalanin script deploy.", "compound_command", ["system"]),

    ("Matiin itu dong.", "ambiguous", ["unknown"]),
    ("Ulang yang tadi ya.", "ambiguous", ["unknown"]),
    ("Ganti aja deh yang ini.", "ambiguous", ["unknown"]),

    ("Menurutmu ujan itu romantis apa nyusahin sih?", "negative_unknown", ["unknown"]),
    ("asdkjaskjd wkwkwk apaan sih ini", "negative_unknown", ["unknown"]),
    ("Kamu bisa nyanyi nggak sih sebenernya?", "negative_unknown", ["unknown"]),

    ("Buka whatsapp.", "domain_overlap_disambiguation", ["system"]),
    ("Matiin suaranya, berisik banget.", "domain_overlap_disambiguation", ["system"]),
    ("Cariin alamat rumah nenek dong.", "domain_overlap_disambiguation", ["memory"]),

    ("Duh kipas laptop ini berisik banget dari tadi.", "implicit_intent", ["system"]),
    ("Kerjaan numpuk banget nih minggu ini, pusing.", "implicit_intent", ["productivity"]),
    ("Kok lagu ini kayaknya diputer mulu ya dari tadi.", "implicit_intent", ["media"]),

    ("Lanjutin lagi yang tadi ya.", "context_dependent", None),
    ("Balikin ke yang sebelumnya aja.", "context_dependent", None),
]


def generate(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = 150) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def run_pipeline(model, tokenizer, user_text: str, max_new_tokens: int) -> dict:
    """Return dict berisi 'category', 'domains' (list), 'raw_stage1', 'raw_stage2', 'problems'."""
    problems = []

    raw1 = generate(model, tokenizer, ROUTER_STAGE1_PROMPT, user_text, max_new_tokens=40)
    parsed1, err1 = parse_json(raw1)
    if not isinstance(parsed1, dict) or "category" not in parsed1:
        problems.append(f"stage1 JSON tidak valid: {raw1[:100]!r}")
        return {"category": None, "domains": [], "raw_stage1": raw1, "raw_stage2": None, "problems": problems}

    category = parsed1["category"]
    if category not in VALID_CATEGORIES:
        problems.append(f"stage1 category tidak valid: {category!r}")
        return {"category": category, "domains": [], "raw_stage1": raw1, "raw_stage2": None, "problems": problems}

    if category == "ambiguous":
        return {"category": category, "domains": ["unknown"], "raw_stage1": raw1, "raw_stage2": None, "problems": problems}

    if category in ("single_intent", "implicit_intent"):
        raw2 = generate(model, tokenizer, ROUTER_STAGE2_SINGLE_PROMPT, user_text, max_new_tokens=40)
        parsed2, err2 = parse_json(raw2)
        if not isinstance(parsed2, dict) or "domain" not in parsed2:
            problems.append(f"stage2-single JSON tidak valid: {raw2[:100]!r}")
            return {"category": category, "domains": [], "raw_stage1": raw1, "raw_stage2": raw2, "problems": problems}
        domain = parsed2["domain"]
        if domain not in VALID_DOMAINS:
            problems.append(f"stage2-single domain tidak valid: {domain!r}")
        return {"category": category, "domains": [domain], "raw_stage1": raw1, "raw_stage2": raw2, "problems": problems}

    if category == "multi_intent":
        raw2 = generate(model, tokenizer, ROUTER_STAGE2_MULTI_PROMPT, user_text, max_new_tokens=max_new_tokens)
        parsed2, err2 = parse_json(raw2)
        segments = None
        if isinstance(parsed2, dict):
            segments = parsed2.get("segments")
        elif isinstance(parsed2, list):
            segments = parsed2
            problems.append("stage2-multi: output list mentah, bukan {\"segments\": [...]}")
        if not isinstance(segments, list) or len(segments) == 0:
            problems.append(f"stage2-multi JSON/segments tidak valid: {raw2[:150]!r}")
            return {"category": category, "domains": [], "raw_stage1": raw1, "raw_stage2": raw2, "problems": problems}
        domains = [s.get("domain") if isinstance(s, dict) else None for s in segments]
        for d in domains:
            if d not in VALID_DOMAINS:
                problems.append(f"stage2-multi domain tidak valid: {d!r}")
        return {"category": category, "domains": domains, "raw_stage1": raw1, "raw_stage2": raw2, "problems": problems}

    problems.append("kondisi tidak terduga")
    return {"category": category, "domains": [], "raw_stage1": raw1, "raw_stage2": None, "problems": problems}


def evaluate(result: dict, expected_domains) -> list[str]:
    problems = list(result["problems"])
    if expected_domains is None:
        return problems
    actual = result["domains"]
    if len(actual) != len(expected_domains):
        problems.append(f"jumlah domain={len(actual)}, ekspektasi={len(expected_domains)}")
    elif actual != expected_domains:
        problems.append(f"domain salah: dapat {actual}, ekspektasi {expected_domains}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    print(f"Loading base model dari {args.base_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"Loading adapter dari {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    results = []
    for text, category_label, expected_domains in TEST_CASES:
        result = run_pipeline(model, tokenizer, text, args.max_new_tokens)
        problems = evaluate(result, expected_domains)
        status = "PASS" if not problems else "FAIL"
        results.append((status, category_label))
        print(f"\n[{status}] ({category_label}) {text}")
        print(f"    stage1 -> category={result['category']!r}   raw1={result['raw_stage1'][:80]!r}")
        if result["raw_stage2"] is not None:
            print(f"    stage2 -> domains={result['domains']}   raw2={result['raw_stage2'][:120]!r}")
        else:
            print(f"    stage2 -> (di-skip, langsung domains={result['domains']})")
        for p in problems:
            print(f"    - {p}")

    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_total = len(results)
    print(f"\n{'=' * 50}\nRINGKASAN: {n_pass}/{n_total} PASS ({n_pass/n_total*100:.0f}%)")

    by_cat: dict[str, list[str]] = {}
    for status, category_label in results:
        by_cat.setdefault(category_label, []).append(status)
    print("\nPer kategori:")
    for cat, statuses in by_cat.items():
        p = statuses.count("PASS")
        print(f"  {cat:32s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
