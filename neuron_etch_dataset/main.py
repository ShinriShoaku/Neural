"""
main.py
=======
Entry point untuk generate SEMUA dataset Neuron-Etch vNext:
router, system, media, persona, coding, information, memory,
productivity, validator — masing-masing dipisah per task_category
sesuai §29-§38.

Cara pakai:
    python main.py

Output:
    output/router/<task_category>.jsonl
    output/system/<task_category>.jsonl
    output/media/<task_category>.jsonl
    output/persona/<task_category>.jsonl
    output/coding/<task_category>.jsonl
    output/information/<task_category>.jsonl
    output/memory/<task_category>.jsonl
    output/productivity/<task_category>.jsonl
    output/validator/<task_category>.jsonl

File yang task_category-nya belum diisi contoh di templates.py tetap
dibuat sebagai file kosong (0 baris) supaya struktur folder sudah
lengkap mengikuti tabel §30-§38 — tinggal isi templates.py.
"""

import json
import os

from generator import (
    generate_router_dataset,
    generate_system_dataset,
    generate_media_dataset,
    generate_persona_dataset,
    generate_coding_dataset,
    generate_information_dataset,
    generate_memory_dataset,
    generate_productivity_dataset,
    generate_validator_dataset,
)
from task_composition import TASK_TARGETS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_and_report(dataset_name: str, data: dict[str, list], out_subdir: str) -> tuple[int, int]:
    """
    Simpan tiap task_category ke file sendiri + cetak progress vs target
    dari task_composition.py. Return (total_generated, total_target).
    """
    targets = TASK_TARGETS.get(dataset_name, {})
    total_generated = 0
    total_target = sum(t for t in targets.values() if t) if targets else 0

    print(f"\n=== {dataset_name.upper()} ===")
    for task_category, samples in data.items():
        rows = [s.to_dict() for s in samples]
        path = os.path.join(OUTPUT_DIR, out_subdir, f"{task_category}.jsonl")
        save_jsonl(path, rows)

        target = targets.get(task_category)
        n = len(rows)
        total_generated += n

        if target:
            pct = (n / target * 100) if target else 0
            status = "OK" if n > 0 else "STUB (belum diisi)"
            print(f"  {task_category:<32} {n:>5} / {target:<6} ({pct:5.1f}%)  [{status}]")
        else:
            note = "(manual only, tidak ada di corruption pipeline §38.3)" \
                if dataset_name == "validator" and task_category in \
                   ("parameter_mismatch", "contradiction", "ambiguous") else \
                "(tidak ada target angka per-kategori di dokumen)"
            print(f"  {task_category:<32} {n:>5}  {note}")

    if total_target:
        print(f"  {'TOTAL':<32} {total_generated:>5} / {total_target:<6} "
              f"({total_generated/total_target*100:5.1f}%)")
    else:
        print(f"  {'TOTAL':<32} {total_generated:>5}")

    return total_generated, total_target


def main() -> None:
    grand_total_generated = 0
    grand_total_target = 0

    # --- Router ---
    router_data = generate_router_dataset()
    g, t = save_and_report("router", router_data, "router")
    grand_total_generated += g
    grand_total_target += t

    # --- 7 domain specialist ---
    specialist_generators = {
        "system": generate_system_dataset,
        "media": generate_media_dataset,
        "persona": generate_persona_dataset,
        "coding": generate_coding_dataset,
        "information": generate_information_dataset,
        "memory": generate_memory_dataset,
        "productivity": generate_productivity_dataset,
    }

    all_specialist_data: dict[str, dict[str, list]] = {}
    for domain_name, gen_fn in specialist_generators.items():
        data = gen_fn()
        all_specialist_data[domain_name] = data
        g, t = save_and_report(domain_name, data, domain_name)
        grand_total_generated += g
        grand_total_target += t

    # --- Validator (butuh hasil semua specialist di atas sebagai sumber) ---
    validator_data = generate_validator_dataset(all_specialist_data)
    g, t = save_and_report("validator", validator_data, "validator")
    grand_total_generated += g
    grand_total_target += t

    print(f"\n{'=' * 60}")
    print(f"GRAND TOTAL: {grand_total_generated} sample "
          f"(target dokumen §39: {grand_total_target}+ / 90,000 keseluruhan)")
    print(f"{'=' * 60}")

    print("\nContoh sample media (playback):")
    print(json.dumps(all_specialist_data["media"]["playback"][0].to_dict(), indent=2, ensure_ascii=False))

    print("\nContoh sample validator (target_mismatch, hasil corruption otomatis):")
    if validator_data["target_mismatch"]:
        print(json.dumps(validator_data["target_mismatch"][0].to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
