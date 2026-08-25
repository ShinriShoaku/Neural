"""
Jalankan sekali: python patch_tokenize.py
Otomatis nyari & patch semua training_common.py di bawah folder Neural/.
"""
import glob
import os

OLD_FUNC = '''def tokenize_example(tokenizer, messages: list[dict], max_seq_len: int) -> dict:
    prefix_messages = messages[:-1]  # system + user
    assistant_content = messages[-1]["content"]

    prefix_text = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prefix_ids = tokenizer(text=prefix_text, add_special_tokens=False)["input_ids"]

    target_text = assistant_content + "<|im_end|>\\n"
    target_ids = tokenizer(text=target_text, add_special_tokens=False)["input_ids"]

    input_ids = prefix_ids + target_ids'''

NEW_FUNC = '''def _unwrap_ids(ids):
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

    target_text = assistant_content + "<|im_end|>\\n"
    target_ids = _unwrap_ids(tokenizer(text=target_text, add_special_tokens=False)["input_ids"])

    input_ids = prefix_ids + target_ids'''

def main():
    home = os.path.expanduser("~/sandbox_workspace/Neural")
    files = glob.glob(os.path.join(home, "**", "training_common.py"), recursive=True)
    if not files:
        print("Tidak ketemu training_common.py di bawah", home)
        return
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            content = f.read()
        if OLD_FUNC not in content:
            if "_unwrap_ids" in content:
                print(f"SKIP (sudah kepatch): {fp}")
            else:
                print(f"WARNING: pola lama tidak ketemu persis di {fp}, cek manual!")
            continue
        content = content.replace(OLD_FUNC, NEW_FUNC)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"PATCHED: {fp}")

if __name__ == "__main__":
    main()
