"""
build_router_minimal.py
========================
Sama seperti build_router_simple.py, tapi pakai ROUTER_SYSTEM_PROMPT_MINIMAL
(~66 token, bukan ~376) -- system prompt dipangkas jadi cuma daftar domain +
1 kalimat instruksi, tanpa worked example / penjelasan overlap rule
panjang. Hipotesis: fine-tuning sendiri sudah cukup ngajarin format &
pola overlap dari ribuan contoh (system,user,assistant), jadi penjelasan
statis yang diulang di tiap prompt itu beban, bukan bantuan, buat model
sekecil 0.8B.

Cara pakai (dari folder training/):
    python build_router_minimal.py --val-ratio 0.1 --seed 42 --oversample-multi 1
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random

from chatml_format import build_router_messages_minimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


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
    parser.add_argument("--oversample-multi", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_all_rows()
    print(f"Total baris mentah: {len(rows)}")

    examples = [{"messages": build_router_messages_minimal(row)} for row in rows]
    rng.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    if args.oversample_multi > 0:
        def n_segments(ex):
            return ex["messages"][2]["content"].count('"domain"')

        multi = [e for e in train_examples if n_segments(e) >= 2]
        train_examples = train_examples + multi * args.oversample_multi
        rng.shuffle(train_examples)
        frac = sum(1 for e in train_examples if n_segments(e) >= 2) / len(train_examples)
        print(f"Oversample multi-segment {args.oversample_multi}x tambahan -> "
              f"{len(train_examples)} baris train, proporsi multi-segment: {frac:.1%}")

    train_path = os.path.join(TRAINING_DATA_DIR, "router_minimal.train.jsonl")
    val_path = os.path.join(TRAINING_DATA_DIR, "router_minimal.val.jsonl")
    save_jsonl(train_path, train_examples)
    save_jsonl(val_path, val_examples)

    print(f"TRAIN: {len(train_examples)} -> {train_path}")
    print(f"VAL:   {len(val_examples)} -> {val_path}")
    print("\nContoh 1 baris:")
    print(json.dumps(train_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
