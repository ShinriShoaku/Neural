"""
build_productivity_hierarchical.py
=====================================
Bangun training data hierarchical buat specialist "productivity".

Payload stage2 BEDA bentuk per kategori (lihat productivity_chatml.py):
    - calendar/schedule/notification/communication: {"target":..., "parameters":...}
      (notification/communication cuma {"target":...})
    - reminder: {"parameters":...} saja
    - todo: {"action":..., "target":...}
    - update_delete: {"intent":..., "action":..., "target":..., "parameters":...}
    - ambiguous/negative: skip stage2

Cara pakai (dari folder training/):
    python build_productivity_hierarchical.py --val-ratio 0.1 --seed 42
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random

from productivity_chatml import build_stage1_messages, build_stage2_messages, NO_STAGE2_CATEGORIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def load_all_rows() -> list[dict]:
    rows = []
    pattern = os.path.join(OUTPUT_DIR, "productivity", "*.jsonl")
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


def build_payload(category: str, out: dict) -> dict:
    if category == "calendar":
        return {"target": out["target"], "parameters": out["parameters"] or {}}
    if category == "reminder":
        return {"parameters": out["parameters"] or {}}
    if category == "todo":
        return {"action": out["action"], "target": out["target"]}
    if category == "schedule":
        return {"target": out["target"], "parameters": out["parameters"] or {}}
    if category == "notification":
        return {"target": out["target"]}
    if category == "communication":
        return {"target": out["target"]}
    if category == "update_delete":
        return {"intent": out["intent"], "action": out["action"],
                "target": out["target"], "parameters": out["parameters"] or {}}
    raise ValueError(f"kategori tidak dikenal: {category}")


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

        payload = build_payload(cat, out)
        stage2_examples.append({"messages": build_stage2_messages(row["input"], cat, payload)})

    print(f"Stage1 examples: {len(stage1_examples)}")
    print(f"Stage2 examples: {len(stage2_examples)} (skip {skipped} ambiguous/negative)")

    all_examples = stage1_examples + stage2_examples
    rng.shuffle(all_examples)
    print(f"TOTAL gabungan: {len(all_examples)}")

    n_val = max(1, int(len(all_examples) * args.val_ratio))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    train_path = os.path.join(TRAINING_DATA_DIR, "productivity_hier.train.jsonl")
    val_path = os.path.join(TRAINING_DATA_DIR, "productivity_hier.val.jsonl")
    save_jsonl(train_path, train_examples)
    save_jsonl(val_path, val_examples)

    print(f"\nTRAIN: {len(train_examples)} -> {train_path}")
    print(f"VAL:   {len(val_examples)} -> {val_path}")


if __name__ == "__main__":
    main()
