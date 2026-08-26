"""
build_router_hierarchical.py
=============================
Bangun training data buat pendekatan Hierarchical Routing (2 stage):

Stage 1 -- Intent-Type Classifier: cuma nentuin kategori
    (single_intent / multi_intent / implicit_intent / ambiguous), TANPA
    perlu tahu domain apa. Task paling sederhana yang mungkin.

Stage 2a -- Single-domain classifier: dipanggil kalau stage1="single_intent"
    ATAU "implicit_intent". Cuma nentuin SATU domain buat seluruh
    kalimat, TANPA perlu mikir segmentasi.

Stage 2b -- Multi-segment splitter: dipanggil kalau stage1="multi_intent".
    Kalimat SUDAH DIPASTIKAN multi-domain, jadi model nggak perlu ragu
    "apa perlu dipecah" -- fokus penuh ke DI MANA titik pisahnya & domain
    apa masing-masing.

"ambiguous" tidak butuh model sama sekali di stage 2 -- langsung
domain="unknown" (dihemat, nggak usah training/inference buat itu).

Pemetaan 8 task_category asli -> 4 kategori stage1:
    single_intent, compound_command, domain_overlap_disambiguation
        -> "single_intent"  (semua cuma butuh 1 keputusan domain per kalimat)
    multi_intent          -> "multi_intent"
    implicit_intent       -> "implicit_intent"
    ambiguous, negative_unknown, context_dependent
        -> "ambiguous"      (tanpa histori percakapan, ini genuinely ambigu)

Ketiga jenis contoh (stage1, stage2a, stage2b) DIGABUNG jadi SATU file
training -- satu adapter LoRA, dibedakan lewat SYSTEM PROMPT yang beda
di tiap panggilan inference (bukan 2 model terpisah, biar hemat & simpel).

Cara pakai (dari folder training/):
    python build_router_hierarchical.py --val-ratio 0.1 --seed 42
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random

from chatml_format import (
    build_router_stage1_messages,
    build_router_stage2_single_messages,
    build_router_stage2_multi_messages,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

STAGE1_MAP = {
    "single_intent": "single_intent",
    "compound_command": "single_intent",
    "domain_overlap_disambiguation": "single_intent",
    "multi_intent": "multi_intent",
    "implicit_intent": "implicit_intent",
    "ambiguous": "ambiguous",
    "negative_unknown": "ambiguous",
    "context_dependent": "ambiguous",
}


def load_all_rows() -> list[dict]:
    rows = []
    pattern = os.path.join(OUTPUT_DIR, "router", "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_all_rows()
    print(f"Total baris mentah: {len(rows)}")

    stage1_examples = []
    stage2_single_examples = []
    stage2_multi_examples = []
    skipped_ambiguous_for_stage2 = 0

    for row in rows:
        orig_cat = row["task_category"]
        stage1_cat = STAGE1_MAP.get(orig_cat)
        if stage1_cat is None:
            continue

        stage1_examples.append({"messages": build_router_stage1_messages(row["input"], stage1_cat)})

        if stage1_cat == "multi_intent":
            stage2_multi_examples.append({"messages": build_router_stage2_multi_messages(row)})
        elif stage1_cat in ("single_intent", "implicit_intent"):
            # semua row non-multi punya tepat 1 segment -- ambil domain-nya
            domain = row["segments"][0]["domain"]
            stage2_single_examples.append(
                {"messages": build_router_stage2_single_messages(row["input"], domain)}
            )
        else:  # ambiguous -> tidak perlu stage2 (langsung domain=unknown di kode aplikasi)
            skipped_ambiguous_for_stage2 += 1
            # TAPI tetap masukin beberapa contoh "unknown" ke stage2_single supaya
            # classifier itu juga punya kelas "unknown" sebagai fallback yang valid
            # (jaga-jaga kalau stage1 salah klasifikasi jadi single_intent).
            stage2_single_examples.append(
                {"messages": build_router_stage2_single_messages(row["input"], "unknown")}
            )

    print(f"Stage1 examples       : {len(stage1_examples)}")
    print(f"Stage2-single examples: {len(stage2_single_examples)} "
          f"(termasuk {skipped_ambiguous_for_stage2} contoh 'unknown' dari kategori ambiguous)")
    print(f"Stage2-multi examples : {len(stage2_multi_examples)}")

    all_examples = stage1_examples + stage2_single_examples + stage2_multi_examples
    rng.shuffle(all_examples)
    print(f"TOTAL gabungan (1 adapter, multi-task): {len(all_examples)}")

    n_val = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    train_path = os.path.join(TRAINING_DATA_DIR, "router_hier.train.jsonl")
    val_path = os.path.join(TRAINING_DATA_DIR, "router_hier.val.jsonl")
    save_jsonl(train_path, train_examples)
    save_jsonl(val_path, val_examples)

    print(f"\nTRAIN: {len(train_examples)} -> {train_path}")
    print(f"VAL:   {len(val_examples)} -> {val_path}")
    print("\nContoh 1 baris tiap jenis:")
    print("--- stage1 ---")
    print(json.dumps(stage1_examples[0], indent=2, ensure_ascii=False))
    print("--- stage2-single ---")
    print(json.dumps(stage2_single_examples[0], indent=2, ensure_ascii=False))
    print("--- stage2-multi ---")
    print(json.dumps(stage2_multi_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
