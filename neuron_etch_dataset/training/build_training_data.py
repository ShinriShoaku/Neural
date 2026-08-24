"""
build_training_data.py
=======================
Baca dataset mentah dari ../output/<domain>/<task_category>.jsonl (hasil
main.py di root project), ubah jadi format ChatML messages (chatml_format.py),
lalu simpan ke training_data/<adapter_name>.train.jsonl dan .val.jsonl —
siap dipakai train_lora.py.

Cara pakai:
    python build_training_data.py                      # semua adapter
    python build_training_data.py --adapter system      # cuma satu
    python build_training_data.py --val-ratio 0.1        # default 10% val

Kalau kamu nambah data baru di ../output/<domain>/, tinggal jalankan lagi
script ini sebelum training — tidak nulis manual.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import random

from chatml_format import (
    build_router_messages,
    build_specialist_messages,
    build_validator_messages,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../neuron_etch_dataset
OUTPUT_DIR = os.path.join(ROOT, "output")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

SPECIALIST_DOMAINS = ["system", "media", "persona", "coding", "information", "memory", "productivity"]
ALL_ADAPTERS = ["router"] + SPECIALIST_DOMAINS + ["validator"]


def load_all_rows(domain_folder: str) -> list[dict]:
    rows = []
    pattern = os.path.join(OUTPUT_DIR, domain_folder, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_adapter_dataset(adapter_name: str) -> list[dict]:
    """Return list of {"messages": [...]} untuk satu adapter."""
    examples: list[dict] = []

    if adapter_name == "router":
        rows = load_all_rows("router")
        for row in rows:
            examples.append({"messages": build_router_messages(row)})

    elif adapter_name == "validator":
        rows = load_all_rows("validator")
        for row in rows:
            examples.append({"messages": build_validator_messages(row)})

    elif adapter_name in SPECIALIST_DOMAINS:
        rows = load_all_rows(adapter_name)
        for row in rows:
            examples.append({"messages": build_specialist_messages(adapter_name, row)})

    else:
        raise ValueError(f"Adapter tidak dikenal: {adapter_name}")

    return examples


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", choices=ALL_ADAPTERS, default=None,
                         help="Cuma build 1 adapter. Default: semua.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proporsi data buat validation split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples", type=int, default=10,
                         help="Warning kalau adapter punya sample di bawah angka ini (dataset masih terlalu kecil buat training serius)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    targets = [args.adapter] if args.adapter else ALL_ADAPTERS

    print(f"{'ADAPTER':<15} {'TOTAL':>7} {'TRAIN':>7} {'VAL':>7}")
    for adapter_name in targets:
        examples = build_adapter_dataset(adapter_name)
        rng.shuffle(examples)

        if len(examples) == 0:
            print(f"{adapter_name:<15} {'0':>7}  -> SKIP (belum ada data sama sekali di output/{adapter_name}/)")
            continue

        n_val = max(1, int(len(examples) * args.val_ratio)) if len(examples) >= 10 else 0
        val_examples = examples[:n_val]
        train_examples = examples[n_val:]

        train_path = os.path.join(TRAINING_DATA_DIR, f"{adapter_name}.train.jsonl")
        val_path = os.path.join(TRAINING_DATA_DIR, f"{adapter_name}.val.jsonl")
        save_jsonl(train_path, train_examples)
        save_jsonl(val_path, val_examples)

        warn = "  << dataset masih kecil, tambah data dulu sebelum training serius" \
            if len(examples) < args.min_examples else ""
        print(f"{adapter_name:<15} {len(examples):>7} {len(train_examples):>7} {len(val_examples):>7}{warn}")

    print(f"\nOutput di: {TRAINING_DATA_DIR}")
    print("\nContoh 1 baris training_data (format messages siap tokenizer.apply_chat_template):")
    sample_path = os.path.join(TRAINING_DATA_DIR, f"{targets[0]}.train.jsonl")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                print(json.dumps(json.loads(first_line), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
