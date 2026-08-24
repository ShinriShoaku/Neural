"""
train_lora_unsloth.py
======================
Versi Unsloth dari train_lora.py — training LoRA yang sama persis secara
hasil (tokenisasi ChatML §65.2, assistant-only loss masking, penamaan
adapter §72.1), tapi loading & training-nya dipercepat lewat Unsloth
(patched kernel attention/MLP + opsi 4-bit QLoRA) yang biasanya
2-4x lebih cepat dan jauh lebih hemat VRAM dibanding transformers+peft biasa
— sangat kepakai untuk model sekecil 0.8B di GPU consumer.

PENTING soal instalasi Unsloth (TIDAK masuk requirements.txt biasa karena
command install-nya beda-beda tergantung versi CUDA/OS kamu):

    pip install unsloth

Kalau itu gagal / kamu butuh versi spesifik CUDA, ikuti instruksi resmi di
https://github.com/unslothai/unsloth#installation (cek dulu sebelum run,
karena command persisnya sering berubah mengikuti rilis PyTorch/CUDA baru).

Unsloth saat ini FOKUS ke GPU NVIDIA (CUDA) — tidak jalan di CPU-only atau
AMD ROCm. Kalau hardware kamu tidak didukung, pakai train_lora.py biasa.

Cara pakai — SAMA PERSIS argumennya dengan train_lora.py:

    python train_lora_unsloth.py \\
        --adapter system \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --train-file training_data/system.train.jsonl \\
        --val-file training_data/system.val.jsonl \\
        --output-dir ./adapters/system_specialist \\
        --epochs 3

Adapter hasil training TETAP format PEFT standar (adapter_model.safetensors
+ adapter_config.json) — artinya BISA di-load ulang pakai transformers+peft
biasa tanpa perlu Unsloth terinstall (mis. lewat inference_smoke_test.py
yang sudah ada, tidak perlu versi khusus).
"""

# Unsloth WAJIB di-import PALING AWAL (sebelum torch/transformers) supaya
# patch kernel-nya kepasang dengan benar — ini persyaratan resmi Unsloth,
# bukan gaya penulisan. Karena itu juga "from __future__" harus tetap jadi
# baris pertama secara sintaks Python, jadi ditaruh di atas import unsloth.
from __future__ import annotations

from unsloth import FastLanguageModel  # noqa: E402  -- harus sebelum torch/transformers

import argparse
import os

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from training_common import tokenize_example, PadCollator, save_versioned_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True,
                         choices=["router", "system", "media", "persona", "coding",
                                  "information", "memory", "productivity", "validator"])
    parser.add_argument("--base-model-dir", required=True,
                         help="Folder lokal hasil download_model.py (BUKAN nama repo HF)")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="1.0", help="§72.1 — major.minor, mis. 1.0")

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="QLoRA (4-bit) — paling hemat VRAM. Default OFF karena model "
                              "0.8B sudah kecil, biasanya lebih akurat pakai bf16 penuh. "
                              "Nyalakan kalau VRAM sangat terbatas.")
    args = parser.parse_args()

    print(f"=== Training adapter (Unsloth): {args.adapter} ===")
    print(f"Base model: {args.base_model_dir}")
    print(f"4-bit (QLoRA): {args.load_in_4bit}")

    # --- Load model + tokenizer lewat Unsloth (dari FOLDER LOKAL) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_dir,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect bf16/fp16 sesuai GPU
        load_in_4bit=args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Bungkus LoRA lewat Unsloth (dropout=0 & bias="none" WAJIB untuk
    #     kernel Unsloth yang sudah dioptimasi — lihat dokumentasi resmi) ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # wajib 0 untuk fast-path Unsloth
        bias="none",     # wajib "none" untuk fast-path Unsloth
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # versi checkpointing hemat memori dari Unsloth
        random_state=args.seed,
    )

    # --- Dataset (identik dengan train_lora.py, reuse training_common) ---
    data_files = {"train": args.train_file}
    if args.val_file and os.path.exists(args.val_file) and os.path.getsize(args.val_file) > 0:
        data_files["validation"] = args.val_file
    raw_ds = load_dataset("json", data_files=data_files)

    def _map_fn(example):
        return tokenize_example(tokenizer, example["messages"], args.max_seq_len)

    tokenized_ds = raw_ds.map(_map_fn, remove_columns=["messages"])
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
        eval_strategy="steps" if "validation" in tokenized_ds else "no",
        eval_steps=args.save_steps if "validation" in tokenized_ds else None,
        seed=args.seed,
        remove_unused_columns=False,
        # gradient_checkpointing TIDAK diset di sini — sudah ditangani
        # use_gradient_checkpointing="unsloth" di atas, kalau diset dobel
        # di TrainingArguments bisa konflik dengan patch Unsloth.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation"),
        data_collator=collator,
    )

    trainer.train()

    # --- Simpan adapter — SAMA seperti train_lora.py, format PEFT standar ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter (raw PEFT output) tersimpan di: {args.output_dir}")

    save_versioned_adapter(args.output_dir, args.adapter, args.version)

    print("\nSelesai. Smoke-test inference (tidak butuh Unsloth, cukup transformers+peft biasa):")
    print(f"  python inference_smoke_test.py --adapter {args.adapter} "
          f"--base-model-dir {args.base_model_dir} --adapter-dir {args.output_dir}")


if __name__ == "__main__":
    main()
