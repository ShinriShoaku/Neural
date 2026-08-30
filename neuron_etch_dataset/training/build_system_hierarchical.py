"""
build_system_hierarchical.py
==============================
Bangun training data hierarchical (2 stage) buat specialist "system",
dari output/system/*.jsonl (hasil generate_system_full.py).

Stage 1 examples : semua 16,000 baris -> {"category": task_category}
Stage 2 examples : semua baris KECUALI ambiguous_negative (yang label
    positive, action != None) -> {"action", "target", "parameters"}
    lewat prompt yang sesuai task_category-nya (lihat system_chatml.py)

"ambiguous_negative" TIDAK butuh stage2 -- output langsung None kalau
stage1 bilang begitu (dihemat, mirip "ambiguous" di router_core).

Cara pakai (dari folder training/):
    python build_system_hierarchical.py --val-ratio 0.1 --seed 42
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random

from system_chatml import build_stage1_messages, build_stage2_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def load_all_rows() -> list[dict]:
    rows = []
    pattern = os.path.join(OUTPUT_DIR, "system", "*.jsonl")
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
    stage2_examples = []
    skipped_ambiguous = 0

    for row in rows:
        cat = row["task_category"]
        stage1_examples.append({"messages": build_stage1_messages(row["input"], cat)})

        if cat == "ambiguous_negative":
            skipped_ambiguous += 1
            continue

        out = row["output"]
        if out is None:
            continue  # jaga-jaga kalau ada positive category tapi output None (tidak seharusnya terjadi)

        stage2_examples.append({"messages": build_stage2_messages(
            row["input"], cat, out["action"], out["target"], out["parameters"] or {},
        )})

    print(f"Stage1 examples: {len(stage1_examples)}")
    print(f"Stage2 examples: {len(stage2_examples)} (skip {skipped_ambiguous} ambiguous_negative -- tidak butuh stage2)")

    all_examples = stage1_examples + stage2_examples
    rng.shuffle(all_examples)
    print(f"TOTAL gabungan (1 adapter, multi-task): {len(all_examples)}")

    n_val = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    train_path = os.path.join(TRAINING_DATA_DIR, "system_hier.train.jsonl")
    val_path = os.path.join(TRAINING_DATA_DIR, "system_hier.val.jsonl")
    save_jsonl(train_path, train_examples)
    save_jsonl(val_path, val_examples)

    print(f"\nTRAIN: {len(train_examples)} -> {train_path}")
    print(f"VAL:   {len(val_examples)} -> {val_path}")
    print("\nContoh 1 baris tiap jenis:")
    print("--- stage1 ---")
    print(json.dumps(stage1_examples[0], indent=2, ensure_ascii=False))
    print("--- stage2 ---")
    print(json.dumps(stage2_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
