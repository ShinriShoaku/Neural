"""
memory_chatml.py
==================
Builder ChatML hierarchical (2 stage) untuk specialist "memory".

Stage 1 -- Category classifier: 9-way (remember/retrieve/update/forget/
    search/event_memory/preference_memory/ambiguous/negative)

Stage 2 -- dikelompokkan jadi 5 prompt (bukan 9):
    - "store"    <- remember, event_memory, preference_memory (semua action=remember)
    - "retrieve" <- retrieve
    - "update"   <- update
    - "forget"   <- forget
    - "search"   <- search
    - "ambiguous" -> output KONSTAN, skip stage2
    - "negative"  -> skip stage2 sepenuhnya
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["remember", "retrieve", "update", "forget", "search",
              "event_memory", "preference_memory", "ambiguous", "negative"]

CATEGORY_TO_ACTION = {
    "remember": "remember", "event_memory": "remember", "preference_memory": "remember",
    "retrieve": "retrieve", "update": "update", "forget": "forget", "search": "search",
}
CATEGORY_TO_INTENT = {
    "remember": "memory_store", "event_memory": "memory_store", "preference_memory": "memory_store",
    "retrieve": "memory_retrieve", "update": "memory_update", "forget": "memory_delete",
    "search": "memory_query", "ambiguous": "memory_store",
}
STAGE1_TO_STAGE2_GROUP = {
    "remember": "store", "event_memory": "store", "preference_memory": "store",
    "retrieve": "retrieve", "update": "update", "forget": "forget", "search": "search",
    "ambiguous": None, "negative": None,
}
AMBIGUOUS_CONSTANT_OUTPUT = {"parameters": {"content": None}}

MEMORY_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori berikut
(jangan jawab pertanyaannya, jangan proses lebih jauh):

- remember (minta diinget fakta identitas/data pribadi, nama/tempat/dll)
- preference_memory (minta diinget suka/nggak suka sesuatu)
- event_memory (minta diinget kejadian/rencana dengan penanda waktu)
- retrieve (nanya balik apakah kamu masih inget sesuatu)
- update (minta ganti/update info yang sudah pernah disimpan)
- forget (minta lupain/hapus ingatan soal sesuatu)
- search (minta cari ingatan/catatan soal topik tertentu)
- ambiguous (minta diinget/disimpen TAPI tidak jelas apa isinya)
- negative (BUKAN soal ingatan sama sekali -- pertanyaan info, dst)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Catet nama kucingku Milo."
Output: {"category": "remember"}

Input: "Kamu masih inget nama kucingku?"
Output: {"category": "retrieve"}

Input: "Cari tahu ibu kota Jepang."
Output: {"category": "negative"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MEMORY_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "store": """Kalimat ini SUDAH DIPASTIKAN minta simpen suatu fakta ke ingatan. Ekstrak:
- category: "identity" (data pribadi) | "preference" (suka/nggak suka) | "event" (kejadian/rencana)
- content: ringkasan singkat fakta yang mau disimpan, dalam bahasa Indonesia,
  format "user <fakta>" (mis. "user suka kopi hitam", "nama kucing user: Milo")

Output HANYA JSON murni: {"parameters": {"category": "...", "content": "..."}}

Contoh:
Input: "Catet nama kucingku Milo."
Output: {"parameters": {"category": "identity", "content": "nama kucing user: Milo"}}

Input: "Inget kalau aku suka kopi hitam."
Output: {"parameters": {"category": "preference", "content": "user suka kopi hitam"}}""",

    "retrieve": """Kalimat ini SUDAH DIPASTIKAN nanya balik apakah kamu masih inget sesuatu. Ekstrak:
- category: "identity" | "preference" | "event"
- query: topik singkat yang ditanyakan

Output HANYA JSON murni: {"parameters": {"category": "...", "query": "..."}}

Contoh:
Input: "Kamu masih inget nama kucingku aku?"
Output: {"parameters": {"category": "identity", "query": "nama kucing"}}""",

    "update": """Kalimat ini SUDAH DIPASTIKAN minta update/ganti info yang sudah pernah disimpan. Ekstrak:
- category: "identity" | "preference" | "event"
- content: ringkasan singkat info BARU-nya, format "user <fakta baru>"

Output HANYA JSON murni: {"parameters": {"category": "...", "content": "..."}}

Contoh:
Input: "Update, sekarang aku lebih suka kopi susu."
Output: {"parameters": {"category": "preference", "content": "user sekarang suka kopi susu"}}""",

    "forget": """Kalimat ini SUDAH DIPASTIKAN minta lupain/hapus ingatan soal sesuatu. Ekstrak:
- category: "identity" | "preference" | "event" (TANPA field content -- forget tidak butuh isi baru)

Output HANYA JSON murni: {"parameters": {"category": "..."}}

Contoh:
Input: "Lupain soal preferensi musik aku ya."
Output: {"parameters": {"category": "preference"}}""",

    "search": """Kalimat ini SUDAH DIPASTIKAN minta cari ingatan/catatan soal topik tertentu. Ekstrak:
- query: topik yang dicari (TANPA field category)

Output HANYA JSON murni: {"parameters": {"query": "..."}}

Contoh:
Input: "Cari memori soal rencana liburanku."
Output: {"parameters": {"query": "rencana liburanku"}}""",
}


def build_stage2_messages(text: str, group: str, parameters: dict[str, Any]) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS[group]
    assistant_json = json.dumps({"parameters": parameters}, ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
