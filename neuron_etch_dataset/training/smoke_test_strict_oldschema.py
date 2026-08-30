"""
smoke_test_strict_oldschema.py
================================
Sama seperti smoke_test_simple.py (cek KETEPATAN domain, bukan cuma
validitas format), tapi buat adapter yang dilatih pakai skema LAMA
(build_router_messages -- ada id/confidence/overlap_hint). Dipakai buat
bandingin router_core_v3 (skema lama) vs router_core_v4 (skema simple)
secara adil, dengan kriteria yang SAMA PERSIS.

Cara pakai:
    python smoke_test_strict_oldschema.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/router_core_v3
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from chatml_format import ROUTER_SYSTEM_PROMPT

VALID_DOMAINS = {"system", "media", "persona", "coding", "information",
                  "memory", "productivity", "unknown"}

# SAMA PERSIS dengan TEST_CASES di smoke_test_simple.py -- supaya adil.
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


def build_prompt_messages(text: str) -> list[dict]:
    system_prompt = ROUTER_SYSTEM_PROMPT.format(
        last_domain="none", last_action="none", session_turn_count=1,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def generate(model, tokenizer, messages, max_new_tokens=250) -> str:
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


def evaluate_case(parsed, expected_domains):
    problems = []
    if parsed is None:
        return ["JSON tidak valid"]
    segments = parsed.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        return ["Field 'segments' kosong / bukan list"]
    actual_domains = [seg.get("domain") for seg in segments]
    for dom in actual_domains:
        if dom not in VALID_DOMAINS:
            problems.append(f"domain tidak valid: {dom!r}")
    if expected_domains is not None:
        if len(actual_domains) != len(expected_domains):
            problems.append(f"jumlah segment={len(actual_domains)}, ekspektasi={len(expected_domains)}")
        elif actual_domains != expected_domains:
            problems.append(f"domain salah: dapat {actual_domains}, ekspektasi {expected_domains}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=250)
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
    for text, category, expected_domains in TEST_CASES:
        messages = build_prompt_messages(text)
        raw = generate(model, tokenizer, messages, args.max_new_tokens)
        parsed, err = parse_json(raw)
        problems = evaluate_case(parsed, expected_domains)
        status = "PASS" if not problems else "FAIL"
        results.append((status, category))
        print(f"\n[{status}] ({category}) {text}")
        if problems:
            for p in problems:
                print(f"    - {p}")
        if parsed:
            doms = [s.get("domain") for s in parsed.get("segments", [])]
            print(f"    domains: {doms}")

    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_total = len(results)
    print(f"\n{'=' * 50}\nRINGKASAN: {n_pass}/{n_total} PASS ({n_pass/n_total*100:.0f}%)")

    by_cat: dict[str, list[str]] = {}
    for status, category in results:
        by_cat.setdefault(category, []).append(status)
    print("\nPer kategori:")
    for cat, statuses in by_cat.items():
        p = statuses.count("PASS")
        print(f"  {cat:32s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
