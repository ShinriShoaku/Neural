"""
productivity_chatml.py
========================
Builder ChatML hierarchical (2 stage) untuk specialist "productivity".

Stage 1 -- Category classifier: 9-way (calendar/reminder/todo/schedule/
    notification/communication/update_delete/ambiguous/negative)

Stage 2 -- 7 prompt beda:
    - calendar, reminder, schedule, notification, communication: action
      deterministik (lookup CATEGORY_TO_ACTION), model cuma ekstraksi
    - todo: action TIDAK deterministik (create|complete), model prediksi
    - update_delete: PALING kompleks -- intent, action (update|delete),
      target, parameters semua diprediksi model (cross-cutting)
    - ambiguous: output KONSTAN, skip stage2
    - negative: skip stage2 sepenuhnya
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["calendar", "reminder", "todo", "schedule", "notification",
              "communication", "update_delete", "ambiguous", "negative"]

CATEGORY_TO_INTENT = {
    "calendar": "calendar", "reminder": "reminder", "todo": "todo",
    "schedule": "schedule", "notification": "notification", "communication": "communication",
    "ambiguous": "calendar",
}
CATEGORY_TO_ACTION = {
    "calendar": "create", "reminder": "create", "schedule": "schedule",
    "notification": "notify", "communication": "send",
}
NO_STAGE2_CATEGORIES = {"ambiguous"}
AMBIGUOUS_CONSTANT_OUTPUT = {
    "intent": "calendar", "action": "update",
    "target": {"type": "event", "value": None}, "parameters": {},
}

PRODUCTIVITY_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori
berikut (jangan jawab pertanyaannya, jangan proses lebih jauh):

- calendar (bikin event/acara BARU dengan waktu tertentu)
- reminder (minta diingetin sesuatu di waktu tertentu)
- todo (tambah tugas ke daftar, ATAU tandain tugas selesai)
- schedule (bikin jadwal RUTIN/berulang, ada kata "tiap"/"setiap")
- notification (minta dikasih notifikasi kalau kondisi tertentu terjadi)
- communication (minta kirim/balas pesan)
- update_delete (minta UBAH atau HAPUS/batalin sesuatu yang sudah ada -- event/reminder/todo)
- ambiguous (minta ubah/hapus/geser TAPI tidak jelas yang mana)
- negative (BUKAN soal produktivitas -- ingatan, info, dst)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Bikin event meeting besok jam 10."
Output: {"category": "calendar"}

Input: "Hapus reminder beli galon."
Output: {"category": "update_delete"}

Input: "Jadwalin olahraga tiap pagi jam 6."
Output: {"category": "schedule"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PRODUCTIVITY_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "calendar": """Kalimat ini SUDAH DIPASTIKAN minta bikin event baru. Ekstrak:
- target: {"type": "event", "value": "<nama event>"}
- parameters: {"time": "<waktu, tulis persis seperti disebut user>"}

Output HANYA JSON murni: {"target": {...}, "parameters": {...}}

Contoh:
Input: "Bikin event meeting besok jam 10."
Output: {"target": {"type": "event", "value": "meeting"}, "parameters": {"time": "besok jam 10"}}""",

    "reminder": """Kalimat ini SUDAH DIPASTIKAN minta diingetin sesuatu. Ekstrak:
- parameters: {"time": "<waktu>", "content": "<hal yang mau diingetkan>"}
(target selalu null untuk reminder)

Output HANYA JSON murni: {"parameters": {"time": "...", "content": "..."}}

Contoh:
Input: "Ingetin aku besok jam 8 update project."
Output: {"parameters": {"time": "besok jam 8", "content": "update project"}}""",

    "todo": """Kalimat ini SUDAH DIPASTIKAN soal todo list. Tentukan:
- action: create (nambah todo baru) | complete (tandain sudah selesai)
- target: {"type": "todo", "value": "<nama tugas>"}

Output HANYA JSON murni: {"action": "...", "target": {...}}

Contoh:
Input: "Tambahin todo beli galon."
Output: {"action": "create", "target": {"type": "todo", "value": "beli galon"}}

Input: "Tandain todo beli galon udah selesai."
Output: {"action": "complete", "target": {"type": "todo", "value": "beli galon"}}""",

    "schedule": """Kalimat ini SUDAH DIPASTIKAN minta bikin jadwal rutin/berulang. Ekstrak:
- target: {"type": "event", "value": "<nama aktivitas>"}
- parameters: {"recurrence": "<pola, format: 'daily HH:MM' atau 'weekly <hari> HH:MM'>"}

Output HANYA JSON murni: {"target": {...}, "parameters": {...}}

Contoh:
Input: "Jadwalin olahraga tiap pagi jam 6."
Output: {"target": {"type": "event", "value": "olahraga"}, "parameters": {"recurrence": "daily 06:00"}}""",

    "notification": """Kalimat ini SUDAH DIPASTIKAN minta dikasih notifikasi kalau suatu kondisi terjadi. Ekstrak:
- target: {"type": "notification", "value": "<isi kondisi/pesan notifikasinya>"}

Output HANYA JSON murni: {"target": {...}}

Contoh:
Input: "Kirim notifikasi kalau meeting udah mulai."
Output: {"target": {"type": "notification", "value": "meeting udah mulai"}}""",

    "communication": """Kalimat ini SUDAH DIPASTIKAN minta kirim/balas pesan. Ekstrak:
- target: {"type": "message", "value": "<isi pesan persis>"}

Output HANYA JSON murni: {"target": {...}}

Contoh:
Input: "Balas pesan ini dengan 'oke siap'."
Output: {"target": {"type": "message", "value": "oke siap"}}""",

    "update_delete": """Kalimat ini SUDAH DIPASTIKAN minta UBAH atau HAPUS sesuatu yang sudah
ada (event/reminder/todo). Tentukan SEMUA field berikut:
- intent: "calendar" (event) | "reminder" | "todo"
- action: "update" (ubah/geser) | "delete" (hapus/batalin)
- target: {"type": "event"|"reminder"|"todo", "value": "<nama>"}
- parameters: {"time": "..."} HANYA kalau action=update DAN waktu baru disebut, selain itu {}

Output HANYA JSON murni: {"intent": "...", "action": "...", "target": {...}, "parameters": {...}}

Contoh:
Input: "Ubah jadwal meeting jadi jam 2 siang."
Output: {"intent": "calendar", "action": "update", "target": {"type": "event", "value": "meeting"}, "parameters": {"time": "jam 2 siang"}}

Input: "Hapus reminder beli galon."
Output: {"intent": "reminder", "action": "delete", "target": {"type": "reminder", "value": "beli galon"}, "parameters": {}}

Input: "Batalin meeting."
Output: {"intent": "calendar", "action": "delete", "target": {"type": "event", "value": "meeting"}, "parameters": {}}""",
}


def build_stage2_messages(text: str, category: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS[category]
    assistant_json = json.dumps(payload, ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
