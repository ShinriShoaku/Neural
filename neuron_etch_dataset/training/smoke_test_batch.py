"""
smoke_test_batch.py
====================
Uji generalisasi adapter router pakai ~24 kalimat BARU (bukan turunan
persis dari generate_router_full.py) yang mencakup semua 8 task_category.
Load model SEKALI, jalanin semua kalimat, kasih ringkasan pass/fail
otomatis (bukan cuma print mentah kayak inference_smoke_test.py).

Cara pakai (dari folder neuron_etch_dataset/training/):
    python smoke_test_batch.py \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/router_core

Cek otomatis per kalimat:
  - Output harus JSON valid
  - Tiap segment.domain harus salah satu dari 8 domain + "unknown"
  - Jumlah segment harus masuk akal untuk kategori itu (single/compound
    -> 1 segment; multi_intent -> >=2 segment)
Ini BUKAN pengganti eval loss asli -- cuma sanity check cepat & manusiawi
buat lihat pola kegagalan sebelum putusin retrain atau tidak.
"""
from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from chatml_format import build_router_messages

VALID_DOMAINS = {"system", "media", "persona", "coding", "information",
                  "memory", "productivity", "unknown"}

# (kalimat, task_category_perkiraan, expected_min_segments, expected_max_segments)
# Semua kalimat di bawah SENGAJA ditulis baru, bukan hasil generate_router_full.py,
# supaya benar-benar tes generalisasi bukan hafalan.
TEST_CASES = [
    # single_intent
    ("Nyalain wifi laptop aku dong.", "single_intent", 1, 1),
    ("Putarin lagu Payung Teduh yang galau.", "single_intent", 1, 1),
    ("Kirim chat ke Kaito nanya kabar.", "single_intent", 1, 1),
    ("Perbaikin bug di file server.js ini.", "single_intent", 1, 1),

    # multi_intent (dua domain berbeda, HARUS split jadi >=2 segment)
    ("Buka spotify terus muter lagu Hindia yang paling baru.", "multi_intent", 2, 3),
    ("Cek cuaca hari ini terus ingetin aku bawa payung nanti sore.", "multi_intent", 2, 3),
    ("Tutup discord dan bikinin reminder meeting jam 2.", "multi_intent", 2, 3),
    ("Cariin resep rendang terus simpen ke catatanku.", "multi_intent", 2, 3),

    # compound_command (1 domain, 2 aksi berurutan -> boleh 1 segment gabungan)
    ("Pause dulu musiknya terus lanjutin lagi abis ini.", "compound_command", 1, 2),
    ("Buka terminal terus langsung jalanin script deploy.", "compound_command", 1, 2),

    # ambiguous (referent nggak jelas)
    ("Matiin itu dong.", "ambiguous", 1, 1),
    ("Ulang yang tadi ya.", "ambiguous", 1, 1),
    ("Ganti aja deh yang ini.", "ambiguous", 1, 1),

    # negative_unknown (di luar 7 domain / gibberish / chit-chat)
    ("Menurutmu ujan itu romantis apa nyusahin sih?", "negative_unknown", 1, 1),
    ("asdkjaskjd wkwkwk apaan sih ini", "negative_unknown", 1, 1),
    ("Kamu bisa nyanyi nggak sih sebenernya?", "negative_unknown", 1, 1),

    # domain_overlap_disambiguation
    ("Buka whatsapp.", "domain_overlap_disambiguation", 1, 1),
    ("Matiin suaranya, berisik banget.", "domain_overlap_disambiguation", 1, 1),
    ("Cariin alamat rumah nenek dong.", "domain_overlap_disambiguation", 1, 1),

    # implicit_intent
    ("Duh kipas laptop ini berisik banget dari tadi.", "implicit_intent", 1, 1),
    ("Kerjaan numpuk banget nih minggu ini, pusing.", "implicit_intent", 1, 1),
    ("Kok lagu ini kayaknya diputer mulu ya dari tadi.", "implicit_intent", 1, 1),

    # context_dependent
    ("Lanjutin lagi yang tadi ya.", "context_dependent", 1, 1),
    ("Balikin ke yang sebelumnya aja.", "context_dependent", 1, 1),
]


def build_prompt_messages(text: str) -> list[dict]:
    fake_row = {
        "input": text, "output": None, "label": "negative",
        "task_category": "smoke_test", "metadata": {},
        "segments": [{"text": text, "domain": "unknown"}],
    }
    return build_router_messages(fake_row)[:2]


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 250) -> str:
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


def evaluate_case(parsed, expected_min_seg, expected_max_seg):
    """Cek sederhana: valid JSON? domain valid? jumlah segment masuk akal?"""
    problems = []
    if parsed is None:
        return ["JSON tidak valid"]
    segments = parsed.get("segments")
    if not isinstance(segments, list) or len(segments) == 0:
        return ["Field 'segments' kosong / bukan list"]
    n = len(segments)
    if not (expected_min_seg <= n <= expected_max_seg):
        problems.append(f"jumlah segment={n}, ekspektasi {expected_min_seg}-{expected_max_seg}")
    for seg in segments:
        dom = seg.get("domain")
        if dom not in VALID_DOMAINS:
            problems.append(f"domain tidak valid: {dom!r}")
        conf = seg.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            problems.append(f"confidence tidak valid: {conf!r}")
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
    for text, category, emin, emax in TEST_CASES:
        messages = build_prompt_messages(text)
        raw = generate(model, tokenizer, messages, args.max_new_tokens)
        parsed, err = parse_json(raw)
        problems = evaluate_case(parsed, emin, emax)
        status = "PASS" if not problems else "FAIL"
        results.append((status, category, text, problems, parsed))
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
    for status, category, *_ in results:
        by_cat.setdefault(category, []).append(status)
    print("\nPer kategori:")
    for cat, statuses in by_cat.items():
        p = statuses.count("PASS")
        print(f"  {cat:32s} {p}/{len(statuses)}")


if __name__ == "__main__":
    main()
