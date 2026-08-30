"""
coding_chatml.py
==================
Builder ChatML hierarchical (2 stage) untuk specialist "coding".

Stage 1 -- Category classifier: 10-way (generate/debug/modify/review/
    explain/refactor/architecture/test/ambiguous/negative)

Stage 2 -- BEDA dari system/media/persona: karena `action` DETERMINISTIK
    dari category (lookup CATEGORY_TO_ACTION di generate_coding_full.py)
    untuk 8 dari 10 kategori, model TIDAK perlu memprediksi action sama
    sekali untuk kategori itu -- prompt-nya cukup minta ekstraksi
    parameter (language/requirements/error/file), action diisi kode
    aplikasi dari lookup table.

    Kategori yang PUNYA stage2 (extraction-only, 7 prompt beda bentuk
    parameter): generate, debug, modify, explain, refactor, architecture
    review & test TIDAK butuh stage2 sama sekali -- parameters selalu {}.

    "ambiguous" beda sendiri: TETAP butuh model prediksi action (karena
    tidak deterministik), target selalu null, parameters selalu {}.
    "negative" SKIP stage2 sepenuhnya (output None).
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["generate", "debug", "modify", "review", "explain", "refactor",
              "architecture", "test", "ambiguous", "negative"]

CATEGORY_TO_ACTION = {
    "generate": "generate", "debug": "debug", "modify": "modify",
    "review": "review", "explain": "explain", "refactor": "refactor",
    "architecture": "design", "test": "test",
}
CATEGORY_TO_INTENT = {
    "generate": "code_generation", "debug": "code_debugging", "modify": "code_modification",
    "review": "code_analysis", "explain": "code_explanation", "refactor": "refactoring",
    "architecture": "architecture", "test": "testing", "ambiguous": "code_modification",
}

# Kategori yang TIDAK butuh stage2 sama sekali -- parameters selalu {}
NO_STAGE2_CATEGORIES = {"review", "test"}

CODING_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori
perintah coding berikut (jangan jawab pertanyaannya, jangan proses lebih jauh):

- generate (bikin kode/script baru dari nol)
- debug (perbaiki error/bug, error-nya disebut jelas)
- modify (ubah/tambah fitur di kode yang sudah ada)
- review (minta feedback/cek kualitas kode)
- explain (minta dijelasin cara kerja kode)
- refactor (rapiin/optimize kode tanpa ubah fungsinya)
- architecture (minta desain/rancangan sistem level tinggi)
- test (minta dibikinin unit test)
- ambiguous (minta perbaiki/benerin TAPI tidak jelas apa masalahnya)
- negative (BUKAN soal coding sama sekali -- reminder, jadwal, dst)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Buat script python buat validasi email."
Output: {"category": "generate"}

Input: "Perbaiki kodenya dong."
Output: {"category": "ambiguous"}

Input: "Ingetin aku meeting jam 3."
Output: {"category": "negative"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CODING_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "generate": """Kalimat ini SUDAH DIPASTIKAN minta bikin kode baru. Ekstrak:
- language: bahasa pemrograman yang diminta (null kalau tidak disebut)
- requirements: deskripsi singkat apa yang mau dibuat

Output HANYA JSON murni: {"parameters": {"language": "..."|null, "requirements": "..."}}

Contoh:
Input: "Buat script python buat validasi email."
Output: {"parameters": {"language": "python", "requirements": "validasi email"}}""",

    "debug": """Kalimat ini SUDAH DIPASTIKAN minta perbaiki error/bug (error-nya disebut jelas). Ekstrak:
- error: nama/jenis error yang disebut
- file: nama file kalau disebut (JANGAN masukkan field ini kalau tidak disebut)

Output HANYA JSON murni: {"parameters": {"error": "...", "file": "..."}} atau {"parameters": {"error": "..."}}

Contoh:
Input: "Debug NullPointerException di file main.py."
Output: {"parameters": {"error": "NullPointerException", "file": "main.py"}}

Input: "Perbaiki bug IndexError ini."
Output: {"parameters": {"error": "IndexError"}}""",

    "modify": """Kalimat ini SUDAH DIPASTIKAN minta ubah/tambah fitur ke kode yang sudah ada. Ekstrak:
- requirements: perubahan spesifik yang diminta

Output HANYA JSON murni: {"parameters": {"requirements": "..."}}

Contoh:
Input: "Tambahin error handling di fungsi ini."
Output: {"parameters": {"requirements": "tambahin error handling"}}""",

    "explain": """Kalimat ini SUDAH DIPASTIKAN minta penjelasan cara kerja kode. Ekstrak:
- language: bahasa pemrograman kalau disebut (JANGAN masukkan field ini kalau tidak disebut)
- requirements: topik/task yang mau dijelasin kalau disebut (JANGAN masukkan field ini kalau tidak disebut)

Output HANYA JSON murni: {"parameters": {}} atau {"parameters": {"language": "...", "requirements": "..."}}

Contoh:
Input: "Jelasin cara kerja sorting data pakai python."
Output: {"parameters": {"language": "python", "requirements": "sorting data"}}

Input: "Jelasin kode ini."
Output: {"parameters": {}}""",

    "refactor": """Kalimat ini SUDAH DIPASTIKAN minta rapiin/optimize kode (tanpa ubah fungsinya). Ekstrak:
- requirements: tujuan spesifik refactor kalau disebut (mis. "biar lebih readable"),
  JANGAN masukkan field ini kalau tidak disebut tujuan spesifik

Output HANYA JSON murni: {"parameters": {}} atau {"parameters": {"requirements": "..."}}

Contoh:
Input: "Refactor function ini biar lebih readable."
Output: {"parameters": {"requirements": "biar lebih readable"}}

Input: "Rapiin kode ini."
Output: {"parameters": {}}""",

    "architecture": """Kalimat ini SUDAH DIPASTIKAN minta desain/rancangan sistem level tinggi. Ekstrak:
- requirements: sistem/komponen apa yang mau dirancang

Output HANYA JSON murni: {"parameters": {"requirements": "..."}}

Contoh:
Input: "Rancang microservices buat e-commerce."
Output: {"parameters": {"requirements": "microservices buat e-commerce"}}""",

    "ambiguous": """Kalimat ini minta perbaiki/benerin sesuatu TAPI tidak jelas apa masalahnya.
Tentukan:
- action: tebakan terbaik (modify | debug | explain | refactor)
- target: selalu null
- parameters: selalu {} (kosong, karena tidak jelas detailnya)

Output HANYA JSON murni, SELALU begini bentuknya: {"action": "...", "target": null, "parameters": {}}

Contoh:
Input: "Perbaiki kodenya dong."
Output: {"action": "modify", "target": null, "parameters": {}}""",
}


def build_stage2_extraction_messages(text: str, category: str, parameters: dict[str, Any]) -> list[dict[str, str]]:
    """Buat 7 kategori extraction-only (bukan ambiguous) -- model cuma ekstrak parameters,
    action/intent/target sudah pasti dari lookup table, TIDAK diprediksi model."""
    prompt = STAGE2_PROMPTS[category]
    assistant_json = json.dumps({"parameters": parameters}, ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]


def build_stage2_ambiguous_messages(text: str, action: str) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS["ambiguous"]
    assistant_json = json.dumps({"action": action, "target": None, "parameters": {}}, ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
