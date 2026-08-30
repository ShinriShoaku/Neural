"""
system_chatml.py
==================
Builder ChatML hierarchical (2 stage) untuk specialist "system":

Stage 1 -- Category classifier: nentuin salah satu dari 10 task_category
    (application/process/filesystem/shell/hardware/audio/display/network/
    system_query/ambiguous_negative). Output: {"category": "..."}

Stage 2 -- SATU prompt BEDA per kategori (10 prompt total), masing-masing
    cuma nyebut action yang relevan buat kategori itu (1-6 pilihan, bukan
    32 sekaligus). Langsung hasilkan action + target + parameters dalam
    1 langkah karena ruang keputusannya sudah sempit per kategori.
    Output: {"action": "...", "target": {"type":..,"value":..}|null,
             "parameters": {...}}

`domain` (selalu "system") dan `intent` (lookup dari action, lihat
ACTION_TO_INTENT di generate_system_full.py) TIDAK diprediksi model --
diisi oleh kode aplikasi sesudahnya.

"ambiguous_negative" di stage1 TIDAK butuh stage2 sama sekali -- label
negative/ambiguous, output langsung None.
"""
from __future__ import annotations
import json
from typing import Any

CATEGORIES = ["application", "process", "filesystem", "shell", "hardware",
              "audio", "display", "network", "system_query", "ambiguous_negative"]

SYSTEM_STAGE1_PROMPT = """Tugasmu HANYA mengklasifikasi kalimat user ke SALAH SATU kategori
perintah system berikut (jangan jawab pertanyaannya, jangan proses lebih jauh):

- application (buka/tutup/restart/cek status aplikasi)
- process (list/kill/cek proses yang berjalan)
- filesystem (bikin/hapus/pindah/cari file atau folder)
- shell (jalanin command/perintah terminal spesifik)
- hardware (nyalain/matiin bluetooth, kamera, mikrofon, dst)
- audio (mute/volume suara)
- display (brightness/resolusi layar)
- network (wifi: connect/disconnect/enable/disable/cek koneksi)
- system_query (cek penggunaan CPU/RAM/disk/baterai/dst)
- ambiguous_negative (bukan perintah system sama sekali, ATAU terlalu vague)

Output HANYA JSON murni, tanpa markdown fence: {"category": "..."}

Contoh:
Input: "Buka spotify."
Output: {"category": "application"}

Input: "Cek penggunaan RAM sekarang."
Output: {"category": "system_query"}

Input: "Ceritain dongeng tentang kucing."
Output: {"category": "ambiguous_negative"}"""


def build_stage1_messages(text: str, category: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_STAGE1_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": category}, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# Stage 2 -- satu prompt per kategori
# ---------------------------------------------------------------------------

STAGE2_PROMPTS = {
    "application": """Kalimat ini SUDAH DIPASTIKAN soal kontrol aplikasi. Tentukan:
- action: launch (buka) | close (tutup) | restart_app | check_app_status (nanya status)
- target: {"type": "application", "value": "<nama aplikasi>"}
- parameters: {} (selalu kosong buat kategori ini)

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {}}

Contoh:
Input: "Buka spotify."
Output: {"action": "launch", "target": {"type": "application", "value": "spotify"}, "parameters": {}}

Input: "Cek apakah discord lagi jalan?"
Output: {"action": "check_app_status", "target": {"type": "application", "value": "discord"}, "parameters": {}}""",

    "process": """Kalimat ini SUDAH DIPASTIKAN soal kontrol proses. Tentukan:
- action: list_processes (tampilkan daftar, TANPA target) | kill_process | check_process
- target: {"type": "process", "value": "<nama proses>"} (null kalau list_processes)
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}|null, "parameters": {}}

Contoh:
Input: "Kill process python3."
Output: {"action": "kill_process", "target": {"type": "process", "value": "python3"}, "parameters": {}}

Input: "Tampilin semua proses yang jalan."
Output: {"action": "list_processes", "target": null, "parameters": {}}""",

    "filesystem": """Kalimat ini SUDAH DIPASTIKAN soal file/folder. Tentukan:
- action: create_file | create_folder | delete_file | move_file | search_file | list_files
- target: {"type": "file"|"folder", "value": "<nama/path>"}
- parameters: {"destination": "<folder tujuan>"} HANYA untuk move_file, selain itu {}

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {...}}

Contoh:
Input: "Bikin file laporan.docx."
Output: {"action": "create_file", "target": {"type": "file", "value": "laporan.docx"}, "parameters": {}}

Input: "Pindahin foto_liburan.jpg ke folder Backup."
Output: {"action": "move_file", "target": {"type": "file", "value": "foto_liburan.jpg"}, "parameters": {"destination": "Backup"}}""",

    "shell": """Kalimat ini SUDAH DIPASTIKAN minta jalanin command shell. Tentukan:
- action: selalu "run_command"
- target: {"type": "command", "value": "<command persis>"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "run_command", "target": {...}, "parameters": {}}

Contoh:
Input: "Jalanin command `git status`."
Output: {"action": "run_command", "target": {"type": "command", "value": "git status"}, "parameters": {}}""",

    "hardware": """Kalimat ini SUDAH DIPASTIKAN soal perangkat keras (bukan audio/display). Tentukan:
- action: enable_device | disable_device | toggle_device
- target: {"type": "device", "value": "<nama device>"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}, "parameters": {}}

Contoh:
Input: "Nyalain bluetooth."
Output: {"action": "enable_device", "target": {"type": "device", "value": "bluetooth"}, "parameters": {}}

Input: "Matiin kamera."
Output: {"action": "disable_device", "target": {"type": "device", "value": "kamera"}, "parameters": {}}""",

    "audio": """Kalimat ini SUDAH DIPASTIKAN soal volume/suara. Tentukan:
- action: mute | unmute | volume_up | volume_down | set_volume
- target: selalu null (audio tidak punya target spesifik)
- parameters: {} untuk mute/unmute, {"amount": 10} untuk volume_up/down,
  {"level": N} untuk set_volume (N dari angka yang disebut user)

Output HANYA JSON murni: {"action": "...", "target": null, "parameters": {...}}

Contoh:
Input: "Matiin suara."
Output: {"action": "mute", "target": null, "parameters": {}}

Input: "Set volume ke 70%."
Output: {"action": "set_volume", "target": null, "parameters": {"level": 70}}

Input: "Naikin volume."
Output: {"action": "volume_up", "target": null, "parameters": {"amount": 10}}""",

    "display": """Kalimat ini SUDAH DIPASTIKAN soal layar (brightness/resolusi). Tentukan:
- action: brightness_up | brightness_down | set_brightness | set_resolution
- target: selalu null
- parameters: {} untuk brightness_up/down, {"level": N} untuk set_brightness,
  {"resolution": "WxH"} untuk set_resolution

Output HANYA JSON murni: {"action": "...", "target": null, "parameters": {...}}

Contoh:
Input: "Set kecerahan layar ke 50%."
Output: {"action": "set_brightness", "target": null, "parameters": {"level": 50}}

Input: "Ganti resolusi layar ke 1920x1080."
Output: {"action": "set_resolution", "target": null, "parameters": {"resolution": "1920x1080"}}""",

    "network": """Kalimat ini SUDAH DIPASTIKAN soal wifi/koneksi. Tentukan:
- action: connect_wifi | disconnect_wifi | enable_wifi | disable_wifi | check_connection
- target: {"type": "network", "value": "<nama wifi>"} HANYA untuk connect_wifi, selain itu null
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "...", "target": {...}|null, "parameters": {}}

Contoh:
Input: "Connect ke wifi Rumah_5G."
Output: {"action": "connect_wifi", "target": {"type": "network", "value": "Rumah_5G"}, "parameters": {}}

Input: "Matiin wifi."
Output: {"action": "disable_wifi", "target": null, "parameters": {}}""",

    "system_query": """Kalimat ini SUDAH DIPASTIKAN nanya status resource system. Tentukan:
- action: selalu "check_resource"
- target: {"type": "resource", "value": "<cpu|ram|disk|baterai|uptime|suhu|gpu|penyimpanan>"}
- parameters: {} (selalu kosong)

Output HANYA JSON murni: {"action": "check_resource", "target": {...}, "parameters": {}}

Contoh:
Input: "Cek penggunaan RAM."
Output: {"action": "check_resource", "target": {"type": "resource", "value": "ram"}, "parameters": {}}""",
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
