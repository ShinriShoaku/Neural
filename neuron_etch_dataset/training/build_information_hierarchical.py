"""
build_information_hierarchical.py
====================================
Bangun training data hierarchical buat specialist "information".

Stage 1 : semua 8,000 baris -> {"category": task_category}
Stage 2 : 8 kategori extraction (search/weather/time/translation/lookup/
          calculation/comparison/knowledge) -> {"target"?, "parameters"?}
          "ambiguous" & "negative" TIDAK butuh stage2

Cara pakai (dari folder training/):
    python build_information_hierarchical.py --val-ratio 0.1 --seed 42
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random

from information_chatml import build_stage1_messages, build_stage2_messages, NO_STAGE2_CATEGORIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def load_all_rows() -> list[dict]:
    rows = []
    pattern = os.path.join(OUTPUT_DIR, "information", "*.jsonl")
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
    skipped = 0

    for row in rows:
        cat = row["task_category"]
        stage1_examples.append({"messages": build_stage1_messages(row["input"], cat)})

        if cat == "negative" or cat in NO_STAGE2_CATEGORIES:
            skipped += 1
            continue

        out = row["output"]
        if out is None:
            continue

        stage2_examples.append({"messages": build_stage2_messages(
            row["input"], cat, out["target"], out["parameters"] or {},
        )})

    print(f"Stage1 examples: {len(stage1_examples)}")
    print(f"Stage2 examples: {len(stage2_examples)} (skip {skipped} ambiguous/negative)")

    all_examples = stage1_examples + stage2_examples
    rng.shuffle(all_examples)
    print(f"TOTAL gabungan: {len(all_examples)}")

    n_val = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    train_path = os.path.join(TRAINING_DATA_DIR, "information_hier.train.jsonl")
    val_path = os.path.join(TRAINING_DATA_DIR, "information_hier.val.jsonl")
    save_jsonl(train_path, train_examples)
    save_jsonl(val_path, val_examples)

    print(f"\nTRAIN: {len(train_examples)} -> {train_path}")
    print(f"VAL:   {len(val_examples)} -> {val_path}")


if __name__ == "__main__":
    main()
