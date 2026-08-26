"""
Jalankan sekali: python patch_eval_oom.py
Menambahkan --eval-batch-size (default 1) + eval_accumulation_steps=1 ke
train_lora.py dan train_lora_unsloth.py, supaya eval tetap AKTIF tapi
tidak numpuk semua prediksi eval di VRAM sekaligus (itu penyebab asli OOM
sebelumnya, bukan eval-nya sendiri yang berat).
"""
import glob
import os

FILES = ["train_lora.py", "train_lora_unsloth.py"]

ARG_OLD = '    parser.add_argument("--batch-size", type=int, default=4)'
ARG_NEW = '''    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1,
                         help="Batch size khusus eval, dibikin kecil (default 1) supaya "
                              "tidak OOM -- prediksi eval numpuk di VRAM sebelum dihitung.")'''

TA_OLD = '''        eval_strategy="steps" if "validation" in tokenized_ds else "no",
        eval_steps=args.save_steps if "validation" in tokenized_ds else None,'''
TA_NEW = '''        eval_strategy="steps" if "validation" in tokenized_ds else "no",
        eval_steps=args.save_steps if "validation" in tokenized_ds else None,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=1,  # offload prediksi eval ke CPU tiap batch, cegah OOM'''


def patch_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    changed = False

    if "--eval-batch-size" in content:
        print(f"SKIP (arg sudah ada): {path}")
    elif ARG_OLD in content:
        content = content.replace(ARG_OLD, ARG_NEW, 1)
        changed = True
    else:
        print(f"WARNING: pola arg batch-size tidak ketemu persis di {path}, cek manual.")

    if "eval_accumulation_steps" in content:
        print(f"SKIP (eval_accumulation_steps sudah ada): {path}")
    elif TA_OLD in content:
        content = content.replace(TA_OLD, TA_NEW, 1)
        changed = True
    else:
        print(f"WARNING: pola TrainingArguments eval tidak ketemu persis di {path}, cek manual.")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"PATCHED: {path}")
    return changed


def main() -> None:
    home = os.path.expanduser("~/sandbox_workspace/Neural")
    any_found = False
    for fname in FILES:
        matches = glob.glob(os.path.join(home, "**", fname), recursive=True)
        for m in matches:
            any_found = True
            patch_file(m)
    if not any_found:
        print("Tidak ketemu train_lora.py / train_lora_unsloth.py di bawah", home)


if __name__ == "__main__":
    main()
