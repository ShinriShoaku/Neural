"""
patch_model_class.py
=====================
Fix bug: AutoModelForCausalLM meresolve arsitektur yang salah buat
Qwen3.5 (model multimodal-native) -- ganti ke class eksplisit
Qwen3_5ForConditionalGeneration. Juga fix deprecation torch_dtype= -> dtype=.

Berlaku ke SEMUA file yang load base model buat inference/training biasa
(bukan lewat Unsloth -- FastLanguageModel Unsloth sudah auto-detect
dengan benar dari awal, jadi train_lora_unsloth.py TIDAK disentuh):
    - train_lora.py               (training tanpa Unsloth)
    - inference_smoke_test.py
    - smoke_test_strict_oldschema.py
    - smoke_test_simple.py
    - smoke_test_minimal.py

Jalankan sekali: python patch_model_class.py
"""
import glob
import os
import re

FILES = [
    "train_lora.py",
    "inference_smoke_test.py",
    "smoke_test_strict_oldschema.py",
    "smoke_test_simple.py",
    "smoke_test_minimal.py",
]

NEW_IMPORT_LINE = (
    "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration"
)


def patch_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if "Qwen3_5ForConditionalGeneration" in content:
        print(f"SKIP (sudah dipatch): {path}")
        return False

    original = content

    # 1) Ganti pemakaian AutoModelForCausalLM.from_pretrained(...) jadi Qwen3_5ForConditionalGeneration
    content = content.replace(
        "AutoModelForCausalLM.from_pretrained(",
        "Qwen3_5ForConditionalGeneration.from_pretrained(",
    )

    # 2) Tambahkan import class barunya, taruh tepat setelah baris import transformers terakhir
    #    yang masih menyebut AutoModelForCausalLM di daftar impor (baik satu baris atau multi-baris).
    if "import AutoModelForCausalLM" in content or re.search(r"^\s*AutoModelForCausalLM,\s*$", content, re.MULTILINE):
        # Sisipkan baris import baru setelah baris "from transformers import ..." (satu baris)
        # ATAU setelah baris penutup ")" dari import multi-baris yang memuat AutoModelForCausalLM.
        lines = content.split("\n")
        out_lines = []
        inserted = False
        i = 0
        while i < len(lines):
            line = lines[i]
            out_lines.append(line)
            if not inserted and "AutoModelForCausalLM" in line:
                # Kasus satu baris: "from transformers import AutoModelForCausalLM, AutoTokenizer"
                if line.strip().startswith("from transformers import"):
                    out_lines.append(NEW_IMPORT_LINE)
                    inserted = True
                # Kasus multi-baris: baris ini cuma "    AutoModelForCausalLM," di dalam import(...)
                # -> cari baris ")" penutup, sisipkan setelah itu.
                elif line.strip().rstrip(",") == "AutoModelForCausalLM":
                    j = i + 1
                    while j < len(lines) and lines[j].strip() != ")":
                        out_lines.append(lines[j])
                        j += 1
                    if j < len(lines):
                        out_lines.append(lines[j])  # baris ")"
                        out_lines.append(NEW_IMPORT_LINE)
                        inserted = True
                        i = j
            i += 1
        content = "\n".join(out_lines)

    # 3) Fix deprecation: torch_dtype= -> dtype= (cuma di pemanggilan from_pretrained model)
    content = content.replace("torch_dtype=torch.bfloat16", "dtype=torch.bfloat16")
    content = content.replace("torch_dtype=dtype", "dtype=dtype")

    if content == original:
        print(f"WARNING: tidak ada perubahan terdeteksi di {path}, cek manual.")
        return False

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"PATCHED: {path}")
    return True


def main() -> None:
    home = os.path.expanduser("~/sandbox_workspace/Neural")
    any_found = False
    for fname in FILES:
        matches = glob.glob(os.path.join(home, "**", fname), recursive=True)
        for m in matches:
            any_found = True
            patch_file(m)
    if not any_found:
        print("Tidak ketemu file target di bawah", home)


if __name__ == "__main__":
    main()
