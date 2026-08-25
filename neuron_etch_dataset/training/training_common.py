"""
training_common.py
===================
Kode yang dipakai BARENG oleh train_lora.py (training biasa via
transformers+peft) dan train_lora_unsloth.py (training dipercepat via
Unsloth) — supaya logic tokenisasi/masking dan penyimpanan adapter (§72.1)
selalu identik di kedua jalur, tidak ada duplikasi yang bisa divergen.
"""

from __future__ import annotations
import os
import shutil

import torch

# Nama file resmi sesuai §25 / §72.1
ADAPTER_FILE_BASENAME = {
    "router": "router_core",
    "validator": "validator_core",
}
# semua specialist domain lain -> "{domain}_specialist" (§25)


def resolve_basename(adapter_name: str) -> str:
    return ADAPTER_FILE_BASENAME.get(adapter_name, f"{adapter_name}_specialist")


# ---------------------------------------------------------------------------
# Tokenisasi dengan assistant-only loss masking (lihat train_lora.py untuk
# penjelasan lengkap kenapa ini dilakukan begini)
# ---------------------------------------------------------------------------

def _unwrap_ids(ids):
    """Beberapa Processor multimodal (mis. Qwen3-VL) mengembalikan input_ids
    dalam bentuk nested List[List[int]] walau input cuma 1 string, beda dari
    AutoTokenizer biasa yang langsung List[int]. Fungsi ini menyeragamkan
    keduanya jadi flat List[int]."""
    if len(ids) > 0 and isinstance(ids[0], list):
        return ids[0]
    return ids


def tokenize_example(tokenizer, messages: list[dict], max_seq_len: int) -> dict:
    prefix_messages = messages[:-1]  # system + user
    assistant_content = messages[-1]["content"]

    prefix_text = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prefix_ids = _unwrap_ids(tokenizer(text=prefix_text, add_special_tokens=False)["input_ids"])

    target_text = assistant_content + "<|im_end|>\n"
    target_ids = _unwrap_ids(tokenizer(text=target_text, add_special_tokens=False)["input_ids"])

    input_ids = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class PadCollator:
    """Dynamic padding untuk input_ids/attention_mask/labels hasil tokenize_example."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_attention, batch_labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Penyimpanan adapter dengan nama resmi §72.1
# ---------------------------------------------------------------------------

def save_versioned_adapter(output_dir: str, adapter_name: str, version: str) -> str | None:
    """
    Salin adapter_model.safetensors + adapter_config.json dari output_dir
    (raw hasil model.save_pretrained()) ke ../adapters_versioned/ dengan
    nama resmi {basename}.v{version}.safetensors (§72.1).

    Return path file .safetensors yang berhasil disalin, atau None kalau gagal.
    """
    basename = resolve_basename(adapter_name)
    versioned_dir = os.path.normpath(
        os.path.join(os.path.dirname(output_dir.rstrip("/")) or ".", "..", "adapters_versioned")
    )
    os.makedirs(versioned_dir, exist_ok=True)

    src_safetensors = os.path.join(output_dir, "adapter_model.safetensors")
    dst_name = f"{basename}.v{version}.safetensors"
    dst_path = os.path.join(versioned_dir, dst_name)

    if not os.path.exists(src_safetensors):
        print(f"PERINGATAN: {src_safetensors} tidak ditemukan, skip penamaan resmi.")
        return None

    shutil.copy2(src_safetensors, dst_path)
    # adapter_config.json WAJIB disimpan berdampingan — tanpa ini adapter
    # tidak bisa di-load ulang (PEFT butuh tahu r/alpha/target_modules)
    shutil.copy2(os.path.join(output_dir, "adapter_config.json"),
                 os.path.join(versioned_dir, f"{basename}.v{version}.adapter_config.json"))
    print(f"Adapter resmi (§72.1): {dst_path}")
    return dst_path
