"""
persona_chatml.py
===================
Builder ChatML hierarchical (2 stage) untuk specialist "persona".

Stage 1 -- Category classifier: 9-way (character_call/character_switch/
    conversation/topic_extraction/roleplay_intent/dialogue_control/
    context_dependency/ambiguous/negative)

Stage 2 -- di-KELOMPOKKAN jadi 5 prompt (bukan 9), karena beberapa
    kategori punya bentuk output IDENTIK:
    - "call_switch"      <- character_call, character_switch
    - "talk"              <- conversation, topic_extraction
    - "roleplay"          <- roleplay_intent
    - "dialogue_control"  <- dialogue_control
    - "unclear"           <- context_dependency, ambiguous (target selalu null)
    - "negative"          <- SKIP stage2 sepenuhnya
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["character_call", "character_switch", "conversation", "topic_extraction",
              "roleplay_intent", "dialogue_control", "context_dependency", "ambiguous", "negative"]

STAGE1_TO_STAGE2_GROUP = {
    "character_call": "call_switch",
    "character_switch": "call_switch",
    "conversation": "talk",
    "topic_extraction": "talk",
    "roleplay_intent": "roleplay",
    "dialogue_control": "dialogue_control",
    "context_dependency": "unclear",
    "ambiguous": "unclear",
    "negative": None,
}

PERSONA_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori berikut
(jangan jawab pertanyaannya, jangan proses lebih jauh):

- character_call (panggil karakter tertentu, nama disebut jelas)
- character_switch (ganti/pindah ke karakter lain, nama disebut jelas)
- conversation (minta karakter (nama jelas) bahas/tanya topik tertentu)
- topic_extraction (sama seperti conversation, kalimat lebih panjang/implicit)
- roleplay_intent (minta karakter (nama jelas) berperan jadi sesuatu)
- dialogue_control (lanjutin/balik ke obrolan, TANPA sebut nama karakter)
- context_dependency (referent karakter tidak jelas -- "dia", "nya", butuh histori)
- ambiguous (sama seperti context_dependency, referent tidak jelas)
- negative (pertanyaan teknis/faktual, BUKAN soal karakter/persona sama sekali)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Panggil Mailin."
Output: {"category": "character_call"}

Input: "Tanya dia soal itu lagi."
Output: {"category": "context_dependency"}

Input: "Jelasin rekursi dengan bahasa Python."
Output: {"category": "negative"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PERSONA_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "call_switch": """Kalimat ini SUDAH DIPASTIKAN minta panggil ATAU ganti ke karakter tertentu
(nama karakternya disebut jelas). Tentukan:
- action: call (kalau minta panggil/ajak ngobrol karakter itu) | switch (kalau minta GANTI/PINDAH ke karakter itu)
- target: {"type": "persona", "value": "<nama karakter>"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {}}

Contoh:
Input: "Panggil Mailin."
Output: {"action": "call", "target": {"type": "persona", "value": "Mailin"}, "parameters": {}}

Input: "Ganti karakter ke Aria."
Output: {"action": "switch", "target": {"type": "persona", "value": "Aria"}, "parameters": {}}""",

    "talk": """Kalimat ini SUDAH DIPASTIKAN minta ngobrol/nanya sesuatu ke karakter tertentu
(nama karakter DAN topik disebut jelas). Tentukan:
- action: talk (kalau minta bahas/cerita/ngobrolin topik) | ask (kalau bentuknya nanya)
- target: {"type": "persona", "value": "<nama karakter>"}
- parameters: {"topic": "<topik yang dibahas>"}

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {...}}

Contoh:
Input: "Suruh Mailin bahas rencana liburan."
Output: {"action": "talk", "target": {"type": "persona", "value": "Mailin"}, "parameters": {"topic": "rencana liburan"}}

Input: "Tanya Aria soal film terbaru."
Output: {"action": "ask", "target": {"type": "persona", "value": "Aria"}, "parameters": {"topic": "film terbaru"}}""",

    "roleplay": """Kalimat ini SUDAH DIPASTIKAN minta karakter tertentu berperan jadi sesuatu. Tentukan:
- action: selalu "roleplay"
- target: {"type": "persona", "value": "<nama karakter>"}
- parameters: {"role": "<peran yang diminta>"}

Output HANYA JSON murni: {"action": "roleplay", "target": {...}, "parameters": {...}}

Contoh:
Input: "Aria, jadi asisten pribadi aku ya."
Output: {"action": "roleplay", "target": {"type": "persona", "value": "Aria"}, "parameters": {"role": "asisten pribadi"}}""",

    "dialogue_control": """Kalimat ini SUDAH DIPASTIKAN minta lanjutin atau balik ke obrolan
(TANPA sebut nama karakter). Tentukan:
- action: continue (lanjutin obrolan yang lagi jalan) | resume_topic (balik ke topik SEBELUMNYA)
- target: {"type": "conversation", "value": "current"} untuk continue,
  {"type": "topic", "value": "previous"} untuk resume_topic
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {}}

Contoh:
Input: "Lanjutin obrolan tadi."
Output: {"action": "continue", "target": {"type": "conversation", "value": "current"}, "parameters": {}}

Input: "Balik ke topik sebelumnya."
Output: {"action": "resume_topic", "target": {"type": "topic", "value": "previous"}, "parameters": {}}""",

    "unclear": """Kalimat ini minta ngobrol/tanya ke suatu karakter, TAPI referent-nya tidak
jelas (pakai "dia"/"nya" tanpa nama). Tentukan:
- action: talk | ask | explain (sesuai kata kerja di kalimat)
- target: selalu null (karena referent tidak jelas)
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": null, "parameters": {}}

Contoh:
Input: "Tanya dia lagi soal itu."
Output: {"action": "ask", "target": null, "parameters": {}}

Input: "Suruh dia jelasin."
Output: {"action": "explain", "target": null, "parameters": {}}""",
}


def build_stage2_messages(text: str, group: str, action: str,
                           target: dict[str, Any] | None, parameters: dict[str, Any]) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS[group]
    assistant_json = json.dumps({"action": action, "target": target, "parameters": parameters},
                                 ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
