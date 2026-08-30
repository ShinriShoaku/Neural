"""
media_chatml.py
=================
Builder ChatML hierarchical (2 stage) untuk specialist "media", pola
sama persis dengan system_chatml.py:

Stage 1 -- Category classifier: nentuin salah satu dari 8 task_category
    (playback/search/queue/player_control/streaming/metadata/ambiguous/negative)
Stage 2 -- SATU prompt beda per kategori, langsung hasilkan action +
    target + parameters.

"negative" TIDAK butuh stage2 (output langsung None).
"ambiguous" TETAP butuh stage2, tapi hasilnya selalu target=null.
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["playback", "search", "queue", "player_control", "streaming",
              "metadata", "ambiguous", "negative"]

MEDIA_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori
perintah media berikut (jangan jawab pertanyaannya, jangan proses lebih jauh):

- playback (putar lagu/album/playlist/genre, lanjutin lagu)
- search (cari lagu/playlist/artist di platform musik)
- queue (tambah/hapus/kosongin/liat antrian lagu)
- player_control (pause/skip/ulang/acak/stop lagu yang lagi diputer)
- streaming (putar video/stream di YouTube dkk, resolve link video)
- metadata (nanya judul/artis/lirik/durasi dari lagu yang lagi diputer)
- ambiguous (minta "ganti aja" / "yang lain" tanpa jelas kemana)
- negative (pertanyaan informasional soal musisi/lagu, BUKAN perintah kontrol media)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Putar lagu Noah."
Output: {"category": "playback"}

Input: "Pause musiknya."
Output: {"category": "player_control"}

Input: "Berapa harga tiket konser Noah?"
Output: {"category": "negative"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MEDIA_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


STAGE2_PROMPTS = {
    "playback": """Kalimat ini SUDAH DIPASTIKAN soal memutar musik. Tentukan:
- action: play (mulai putar sesuatu yang baru) | resume (lanjutin yang tadi)
- target: {"type": "artist"|"album"|"playlist"|"genre", "value": "<nama>"} (null kalau resume)
- parameters: {"player": "<nama app>"} HANYA kalau disebut eksplisit app-nya, selain itu {}

Output HANYA JSON murni: {"action": "...", "target": {...}|null, "parameters": {...}}

Contoh:
Input: "Putar lagu Noah."
Output: {"action": "play", "target": {"type": "artist", "value": "Noah"}, "parameters": {}}

Input: "Putar Tulus di Spotify."
Output: {"action": "play", "target": {"type": "artist", "value": "Tulus"}, "parameters": {"player": "spotify"}}

Input: "Lanjutin lagunya."
Output: {"action": "resume", "target": null, "parameters": {}}""",

    "search": """Kalimat ini SUDAH DIPASTIKAN minta cari sesuatu di platform musik. Tentukan:
- action: selalu "search"
- target: {"type": "song"|"playlist"|"artist"|"genre", "value": "<kata kunci>"}
- parameters: {"player": "<nama app>"} kalau disebut, selain itu {}

Output HANYA JSON murni: {"action": "search", "target": {...}, "parameters": {...}}

Contoh:
Input: "Cari lagu galau di Spotify."
Output: {"action": "search", "target": {"type": "song", "value": "galau"}, "parameters": {"player": "spotify"}}""",

    "queue": """Kalimat ini SUDAH DIPASTIKAN soal antrian lagu (queue). Tentukan:
- action: queue_add | queue_remove | queue_clear | queue_list
- target: {"type": "song"|"queue", "value": "current"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {}}

Contoh:
Input: "Tambahin lagu ini ke antrian."
Output: {"action": "queue_add", "target": {"type": "song", "value": "current"}, "parameters": {}}

Input: "Kosongin antrian lagu."
Output: {"action": "queue_clear", "target": {"type": "queue", "value": "current"}, "parameters": {}}""",

    "player_control": """Kalimat ini SUDAH DIPASTIKAN kontrol player musik yang lagi jalan. Tentukan:
- action: pause | next | previous | repeat | repeat_off | shuffle | shuffle_off | stop
- target: {"type": "player", "value": "current"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {"type": "player", "value": "current"}, "parameters": {}}

Contoh:
Input: "Pause musiknya."
Output: {"action": "pause", "target": {"type": "player", "value": "current"}, "parameters": {}}

Input: "Skip ke lagu berikutnya."
Output: {"action": "next", "target": {"type": "player", "value": "current"}, "parameters": {}}

Input: "Matikan musik."
Output: {"action": "stop", "target": {"type": "player", "value": "current"}, "parameters": {}}""",

    "streaming": """Kalimat ini SUDAH DIPASTIKAN soal streaming video (bukan musik). Tentukan:
- action: play_stream | resolve
- target: {"type": "video", "value": "current"} untuk play_stream,
  {"type": "stream", "value": "url_placeholder"} untuk resolve
- parameters: {"player": "<platform>"} kalau disebut, selain itu {}

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {...}}

Contoh:
Input: "Putar video ini di YouTube."
Output: {"action": "play_stream", "target": {"type": "video", "value": "current"}, "parameters": {"player": "youtube"}}""",

    "metadata": """Kalimat ini SUDAH DIPASTIKAN nanya info soal lagu yang lagi diputer. Tentukan:
- action: selalu "query_metadata"
- target: {"type": "song", "value": "current"}
- parameters: {"field": "<judul|nama artis|nama album|lirik|durasi|genre|tahun rilis>"}

Output HANYA JSON murni: {"action": "query_metadata", "target": {...}, "parameters": {...}}

Contoh:
Input: "Judul lagu ini apa?"
Output: {"action": "query_metadata", "target": {"type": "song", "value": "current"}, "parameters": {"field": "judul"}}""",

    "ambiguous": """Kalimat ini minta ganti sesuatu tapi TIDAK JELAS ganti apa (lagu, playlist,
atau player). Tentukan:
- action: selalu "play"
- target: selalu null (karena tidak jelas)
- parameters: {} (selalu kosong)

Output HANYA JSON murni, SELALU begini: {"action": "play", "target": null, "parameters": {}}""",
}


def build_stage2_messages(text: str, category: str, action: str,
                           target: dict[str, Any] | None, parameters: dict[str, Any]) -> list[dict[str, str]]:
    prompt = STAGE2_PROMPTS[category]
    assistant_json = json.dumps({"action": action, "target": target, "parameters": parameters},
                                 ensure_ascii=False)
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
        {"role": "assistant", "content": assistant_json},
    ]
