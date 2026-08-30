"""
patch_generation_stop.py
==========================
Fix bug: model.generate() tidak diberi tahu token <|im_end|> secara
eksplisit sebagai stop token, jadi kadang lanjut generate abis JSON
selesai (nge-halusinasi giliran percakapan baru: "\\nuser\\n...").

Dua lapis fix, diterapkan ke SEMUA smoke_test_*hierarchical.py:
  1. generate() sekarang cari token id "<|im_end|>" dan kasih ke
     model.generate(eos_token_id=[...]) secara eksplisit -- FIX UTAMA.
  2. parse_json() sekarang potong dulu raw text di penanda giliran baru
     ("<|im_end|>", "\\nuser\\n", "\\nassistant\\n") sebelum di-parse --
     JARING PENGAMAN kalau fix #1 belum cukup atau adapter lama dipakai.

Jalankan sekali: python patch_generation_stop.py
"""
import glob
import os

import re

FILES = [
    "smoke_test_hierarchical.py",
    "smoke_test_media_hierarchical.py",
    "smoke_test_persona_hierarchical.py",
    "smoke_test_system_hierarchical.py",
]

GENERATE_PATTERN = re.compile(
    r'def generate\(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = (\d+)\) -> str:\n'
    r'    messages = \[\{"role": "system", "content": system_prompt\}, \{"role": "user", "content": user_text\}\]\n'
    r'    prompt_text = tokenizer\.apply_chat_template\(\n'
    r'        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,\n'
    r'    \)\n'
    r'    inputs = tokenizer\(prompt_text, return_tensors="pt"\)\.to\(model\.device\)\n'
    r'    with torch\.no_grad\(\):\n'
    r'        output_ids = model\.generate\(\n'
    r'            \*\*inputs, max_new_tokens=max_new_tokens, do_sample=False,\n'
    r'            pad_token_id=tokenizer\.pad_token_id or tokenizer\.eos_token_id,\n'
    r'        \)\n'
    r'    new_tokens = output_ids\[0\]\[inputs\["input_ids"\]\.shape\[1\]:\]\n'
    r'    return tokenizer\.decode\(new_tokens, skip_special_tokens=True\)\.strip\(\)'
)

STOP_HELPER = '''def _get_stop_token_ids(tokenizer) -> list[int]:
    """<|im_end|> HARUS jadi stop token eksplisit -- kalau cuma andalin
    default eos_token_id, generate() kadang nggak berhenti di situ dan
    lanjut nge-halusinasi giliran percakapan baru."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)
    return list(ids)


'''

def _generate_replacement(m: "re.Match") -> str:
    max_new = m.group(1)
    return (
        STOP_HELPER +
        f'def generate(model, tokenizer, system_prompt: str, user_text: str, max_new_tokens: int = {max_new}) -> str:\n'
        '    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]\n'
        '    prompt_text = tokenizer.apply_chat_template(\n'
        '        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,\n'
        '    )\n'
        '    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)\n'
        '    with torch.no_grad():\n'
        '        output_ids = model.generate(\n'
        '            **inputs, max_new_tokens=max_new_tokens, do_sample=False,\n'
        '            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,\n'
        '            eos_token_id=_get_stop_token_ids(tokenizer),\n'
        '        )\n'
        '    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]\n'
        '    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()'
    )

OLD_PARSE = '''def parse_json(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)'''

NEW_PARSE = '''def parse_json(raw_text: str):
    text = raw_text.strip()
    # Jaring pengaman: potong di penanda giliran baru kalau model kelanjutan
    # nge-halusinasi turn berikutnya (harusnya sudah dicegah oleh eos_token_id
    # eksplisit di generate(), tapi ini tetap dijaga untuk adapter lama).
    for marker in ("<|im_end|>", "\\nuser\\n", "\\nassistant\\n", "<|im_start|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)'''


def patch_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if "_get_stop_token_ids" in content:
        print(f"SKIP (sudah kepatch): {path}")
        return False

    changed = False
    new_content, n_sub = GENERATE_PATTERN.subn(_generate_replacement, content, count=1)
    if n_sub == 1:
        content = new_content
        changed = True
    else:
        print(f"WARNING: pola generate() tidak ketemu persis di {path}, cek manual.")

    if OLD_PARSE in content:
        content = content.replace(OLD_PARSE, NEW_PARSE, 1)
        changed = True
    else:
        print(f"WARNING: pola parse_json() tidak ketemu persis di {path}, cek manual.")

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
        print("Tidak ketemu file target di bawah", home)


if __name__ == "__main__":
    main()
