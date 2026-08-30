"""
information_chatml.py
=======================
Builder ChatML hierarchical (2 stage) untuk specialist "information".

Stage 1 -- Category classifier: 10-way (search/weather/time/translation/
    lookup/calculation/comparison/knowledge/ambiguous/negative)

Stage 2 -- 8 prompt (extraction), `action` deterministik dari category
    (lookup CATEGORY_TO_ACTION). "ambiguous" outputnya KONSTAN (tidak
    butuh model sama sekali, mirip review/test di coding). "negative"
    skip stage2 (output None).
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["search", "weather", "time", "translation", "lookup",
              "calculation", "comparison", "knowledge", "ambiguous", "negative"]

CATEGORY_TO_ACTION = {
    "search": "search", "weather": "query", "time": "query",
    "translation": "translate", "lookup": "lookup", "calculation": "calculate",
    "comparison": "compare", "knowledge": "query",
}
CATEGORY_TO_INTENT = {
    "search": "search", "weather": "weather", "time": "time",
    "translation": "translation", "lookup": "lookup", "calculation": "calculation",
    "comparison": "comparison", "knowledge": "information_query", "ambiguous": "search",
}
NO_STAGE2_CATEGORIES = {"ambiguous"}
# Output konstan buat ambiguous -- diisi kode aplikasi, bukan model
AMBIGUOUS_CONSTANT_OUTPUT = {"target": {"type": "web", "value": None}, "parameters": {}}

INFORMATION_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori berikut
(jangan jawab pertanyaannya, jangan proses lebih jauh):

- search (cari info umum di internet, topik bebas)
- weather (nanya cuaca/suhu)
- time (nanya jam/waktu sekarang)
- translation (minta terjemahin kata/kalimat)
- lookup (nanya SIAPA seseorang/entitas tertentu)
- calculation (hitung matematika)
- comparison (bandingin dua hal, mana yang lebih baik)
- knowledge (nanya APA ITU suatu konsep/istilah)
- ambiguous (minta cari sesuatu tapi tidak jelas apa)
- negative (BUKAN soal informasi -- reminder, jadwal, dst)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Cari tahu ibu kota Prancis."
Output: {"category": "search"}

Input: "Apa itu fotosintesis?"
Output: {"category": "knowledge"}

Input: "Siapa presiden Indonesia?"
Output: {"category": "lookup"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INFORMATION_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "search": """Kalimat ini SUDAH DIPASTIKAN minta cari info umum. Ekstrak:
- target: {"type": "web", "value": "<topik yang dicari>"}

Output HANYA JSON murni: {"target": {...}}

Contoh:
Input: "Cari tahu ibu kota Prancis."
Output: {"target": {"type": "web", "value": "ibu kota Prancis"}}""",

    "weather": """Kalimat ini SUDAH DIPASTIKAN nanya cuaca. Ekstrak:
- target: {"type": "weather", "value": "today"|"tomorrow"|"day_after_tomorrow"} (default "today" kalau tidak disebut waktu)
- parameters: {"location": "<nama kota>"} HANYA kalau kota disebut, selain itu {}

Output HANYA JSON murni: {"target": {...}, "parameters": {...}}

Contoh:
Input: "Cuaca besok gimana?"
Output: {"target": {"type": "weather", "value": "tomorrow"}, "parameters": {}}

Input: "Cek cuaca di Tokyo."
Output: {"target": {"type": "weather", "value": "today"}, "parameters": {"location": "Tokyo"}}""",

    "time": """Kalimat ini SUDAH DIPASTIKAN nanya jam/waktu. Ekstrak:
- target: {"type": "time", "value": "now"}
- parameters: {"location": "<nama kota>"} HANYA kalau kota disebut, selain itu {}

Output HANYA JSON murni: {"target": {...}, "parameters": {...}}

Contoh:
Input: "Jam berapa sekarang?"
Output: {"target": {"type": "time", "value": "now"}, "parameters": {}}

Input: "Jam berapa di Tokyo?"
Output: {"target": {"type": "time", "value": "now"}, "parameters": {"location": "Tokyo"}}""",

    "translation": """Kalimat ini SUDAH DIPASTIKAN minta terjemahan. Ekstrak:
- parameters: {"text": "<teks yang diterjemahkan>", "target_language": "<kode bahasa: id/en/ja/ko/fr/es>"}

Output HANYA JSON murni: {"parameters": {...}}

Contoh:
Input: "Artiin 'good morning' ke bahasa Indonesia."
Output: {"parameters": {"text": "good morning", "target_language": "id"}}""",

    "lookup": """Kalimat ini SUDAH DIPASTIKAN nanya SIAPA suatu entitas/orang. Ekstrak:
- target: {"type": "entity", "value": "<entitas yang ditanyakan>"}

Output HANYA JSON murni: {"target": {...}}

Contoh:
Input: "Siapa presiden Indonesia?"
Output: {"target": {"type": "entity", "value": "presiden Indonesia"}}""",

    "calculation": """Kalimat ini SUDAH DIPASTIKAN minta hitung matematika. Ekstrak:
- parameters: {"expression": "<ekspresi matematika, pakai simbol + - * / **>"}

Output HANYA JSON murni: {"parameters": {"expression": "..."}}

Contoh:
Input: "Berapa hasil 12 kali 8?"
Output: {"parameters": {"expression": "12 * 8"}}""",

    "comparison": """Kalimat ini SUDAH DIPASTIKAN minta bandingin dua hal. Ekstrak:
- parameters: {"items": ["<hal pertama>", "<hal kedua>"]}

Output HANYA JSON murni: {"parameters": {"items": [...]}}

Contoh:
Input: "Lebih bagus mana, iPhone atau Samsung?"
Output: {"parameters": {"items": ["iPhone", "Samsung"]}}""",

    "knowledge": """Kalimat ini SUDAH DIPASTIKAN nanya APA ITU suatu konsep. Ekstrak:
- target: {"type": "knowledge", "value": "<konsep yang ditanyakan>"}

Output HANYA JSON murni: {"target": {...}}

Contoh:
Input: "Apa itu fotosintesis?"
Output: {"target": {"type": "knowledge", "value": "fotosintesis"}}""",
}


def build_stage2_messages(text: str, category: str, target: dict[str, Any] | None,
                           parameters: dict[str, Any]) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS[category]
    payload: dict[str, Any] = {}
    if target is not None:
        payload["target"] = target
    if category in ("weather", "time") or parameters:
        payload["parameters"] = parameters
    assistant_json = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
