"""
train_lora.py
=============
Fine-tune SATU LoRA adapter (router / salah satu dari 7 specialist / validator)
di atas base model lokal (§65: Qwen3.5-0.8B, ChatML, thinking OFF wajib).

Loss HANYA dihitung di token assistant (JSON output) — system+user prompt
di-mask jadi -100 supaya model tidak "belajar" menulis ulang instruksi,
cuma belajar menghasilkan Task IR yang benar. Caranya: tokenize prefix
(system+user+assistant-header, dengan add_generation_prompt=True) secara
terpisah dari target (assistant content + <|im_end|>), baru gabung ID-nya
— jadi kita tahu persis di token ke berapa target dimulai, tanpa
tergantung template harus "sadar sendiri" mana bagian assistant.

Cara pakai (urutan training direkomendasikan §59: router -> system ->
persona -> media -> information -> coding -> memory -> productivity ->
validator):

    python train_lora.py \\
        --adapter system \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --train-file training_data/system.train.jsonl \\
        --val-file training_data/system.val.jsonl \\
        --output-dir ./adapters/system_specialist \\
        --epochs 3

Setelah training, adapter disimpan di --output-dir DAN disalin dengan nama
resmi sesuai §72.1 ke ./adapters_versioned/{name}.v{version}.safetensors
(default version v1.0 — naikkan --version kalau retrain dengan data
tambahan, atau major version kalau skema Task IR berubah, §72.1).
"""

from __future__ import annotations
import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from training_common import tokenize_example, PadCollator, save_versioned_adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Hyperparameter — default wajar untuk model sekecil 0.8B di dataset kecil-menengah
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true", help="Matikan bf16 (pakai fp32/fp16 biasa)")
    args = parser.parse_args()

    print(f"=== Training adapter: {args.adapter} ===")
    print(f"Base model: {args.base_model_dir}")

    # --- Load tokenizer & model dari FOLDER LOKAL (bukan nama repo) ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False  # wajib off saat training + gradient checkpointing

    # --- LoRA config (§25 — satu adapter per domain, base model dibagi) ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    data_files = {"train": args.train_file}
    if args.val_file and os.path.exists(args.val_file) and os.path.getsize(args.val_file) > 0:
        data_files["validation"] = args.val_file
    raw_ds = load_dataset("json", data_files=data_files)

    def _map_fn(example):
        return tokenize_example(tokenizer, example["messages"], args.max_seq_len)

    tokenized_ds = raw_ds.map(_map_fn, remove_columns=["messages"])

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    # --- TrainingArguments ---
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
        bf16=use_bf16,
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        report_to=[],
        eval_strategy="steps" if "validation" in tokenized_ds else "no",
        eval_steps=args.save_steps if "validation" in tokenized_ds else None,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation"),
        data_collator=collator,
    )

    trainer.train()

    # --- Simpan adapter (PEFT: adapter_model.safetensors + adapter_config.json) ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter (raw PEFT output) tersimpan di: {args.output_dir}")

    # --- Salin dengan nama resmi §72.1: {name}.v{version}.safetensors ---
    save_versioned_adapter(args.output_dir, args.adapter, args.version)

    print("\nSelesai. Smoke-test inference:")
    print(f"  python inference_smoke_test.py --adapter {args.adapter} "
          f"--base-model-dir {args.base_model_dir} --adapter-dir {args.output_dir}")


if __name__ == "__main__":
    main()
