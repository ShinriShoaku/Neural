"""
generate_system_full.py
=========================
Generator dataset FULL untuk specialist "system" (§31), target 16,000
sample di 10 task_category:

    application      : 3000
    process          : 2000
    filesystem       : 3000
    shell            : 2000
    hardware         : 1500
    audio            : 1000
    display          :  500
    network          : 1000
    system_query     : 1000
    ambiguous_negative: 1000
    ------------------------
    TOTAL            : 16000

Desain buat mendukung pola HIERARCHICAL (2 stage) yang terbukti berhasil
di router_core (92% pass rate):
  - Tiap sample```` biasa punya field `output.action` yang berasal dari
    daftar ACTION tertutup (~30 nilai unik total lintas kategori) --
    dipakai buat Stage 1 (action classifier).
  - `output.intent` TIDAK perlu diprediksi model -- cukup lookup table
    statis ACTION_TO_INTENT (mirip confidence yang ditemukan hardcoded
    di router, ini secara sengaja membuatnya deterministic dari awal).
  - `output.target` + `output.parameters` adalah tugas Stage 2 (ekstraksi,
    dikondisikan pada action yang sudah diketahui dari Stage 1).

Vocab tiap kategori dibuat generous (200+ kombinasi unik) berdasar
pelajaran dari router_core: vocab tipis (<150 kombinasi) menyebabkan
model gagal generalisasi ke kata yang tak persis ada di training.
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260827)
OUT_DIR = Path(__file__).parent / "output" / "system"

_counter = itertools.count(1)
def new_id() -> str:
    return f"system_{next(_counter):06d}"


# ===========================================================================
# ACTION -> INTENT lookup (statis, TIDAK diprediksi model)
# ===========================================================================
ACTION_TO_INTENT = {
    # application
    "launch": "application_control", "close": "application_control",
    "restart_app": "application_control", "check_app_status": "application_control",
    # process
    "list_processes": "process_control", "kill_process": "process_control",
    "check_process": "process_control",
    # filesystem
    "create_file": "filesystem_control", "create_folder": "filesystem_control",
    "delete_file": "filesystem_control", "move_file": "filesystem_control",
    "copy_file": "filesystem_control", "rename_file": "filesystem_control",
    "search_file": "filesystem_control", "list_files": "filesystem_control",
    # shell
    "run_command": "shell_execution",
    # hardware
    "enable_device": "hardware_control", "disable_device": "hardware_control",
    "toggle_device": "hardware_control",
    # audio
    "mute": "audio_control", "unmute": "audio_control",
    "volume_up": "audio_control", "volume_down": "audio_control",
    "set_volume": "audio_control",
    # display
    "brightness_up": "display_control", "brightness_down": "display_control",
    "set_brightness": "display_control", "set_resolution": "display_control",
    # network
    "connect_wifi": "network_control", "disconnect_wifi": "network_control",
    "enable_wifi": "network_control", "disable_wifi": "network_control",
    "check_connection": "network_control",
    # system_query
    "check_resource": "information_query",
}


def wrap(v: str, o: str, w: str) -> str:
    return w.format(v=v, o=o, vl=v[0].lower() + v[1:] if v else v)


WRAPPERS = [
    "{v} {o}.", "{v} {o} dong.", "{v} {o} ya.", "{v} {o} sekarang.",
    "{v} {o} deh.", "Tolong {vl} {o}.", "Bisa {vl} {o} nggak?",
    "Coba {vl} {o}.", "{v} {o}, please.", "Eh, {vl} {o} dong.",
]
STATEMENT_WRAPPERS = [w for w in WRAPPERS if not w.rstrip().endswith("?")]


def combos(verbs, objects, wrappers=WRAPPERS):
    out = set()
    for v, o, w in itertools.product(verbs, objects, wrappers):
        out.add(wrap(v, o, w))
    out = list(out)
    RNG.shuffle(out)
    return out


def pad_to_target(pool: list, target: int) -> list:
    """Jamin selalu dapat PERSIS `target` item. Kalau pool unik nggak cukup,
    isi sisanya dengan mengulang pool yang sudah di-shuffle ulang (masih
    lebih baik daripada berhenti di tengah jalan / kurang dari target)."""
    if len(pool) >= target:
        return pool[:target]
    result = list(pool)
    while len(result) < target:
        extra = list(pool)
        RNG.shuffle(extra)
        result.extend(extra)
    return result[:target]


def sample_output(intent, action, target_type=None, target_value=None, **params):
    target = None
    if target_type is not None:
        target = {"type": target_type, "value": target_value}
    return {"domain": "system", "intent": intent, "action": action,
            "target": target, "parameters": params}


def make_sample(text, task_category, output, label="positive", difficulty="easy", note=None):
    meta = {"difficulty": difficulty}
    if note:
        meta["note"] = note
    return {
        "id": new_id(), "input": text, "output": output,
        "task_category": task_category, "context": {}, "label": label,
        "metadata": meta,
    }


# ===========================================================================
# 1. APPLICATION (target 3000)
# ===========================================================================
APPLICATIONS = [
    "foot", "firefox", "spotify", "discord", "vscode", "terminal", "slack",
    "telegram", "whatsapp", "obs studio", "steam", "blender", "gimp",
    "libreoffice", "file manager", "task manager", "kalkulator", "kamera",
    "thunderbird", "chromium", "brave", "vlc", "audacity", "inkscape",
    "krita", "postman", "docker desktop", "virtualbox", "zoom", "signal",
]
LAUNCH_VERBS = ["Buka", "Jalankan", "Nyalain", "Start", "Aktifin"]
CLOSE_VERBS = ["Tutup", "Matiin", "Keluar dari", "Stop", "Exit dari"]
RESTART_VERBS = ["Restart", "Restart aplikasi", "Reboot", "Muat ulang"]
CHECKAPP_VERBS = ["Cek apakah", "Apakah", "Cek status", "Masih jalan nggak", "Coba cek apakah"]


def gen_application(target: int = 3000) -> list[dict]:
    result = []
    launch = combos(LAUNCH_VERBS, APPLICATIONS)
    close = combos(CLOSE_VERBS, APPLICATIONS)
    restart = combos(RESTART_VERBS, APPLICATIONS)
    n_each = target // 4
    for text in launch[:n_each]:
        app = next(a for a in APPLICATIONS if a in text.lower())
        result.append(make_sample(text, "application",
                                   sample_output("application_control", "launch", "application", app)))
    for text in close[:n_each]:
        app = next(a for a in APPLICATIONS if a in text.lower())
        result.append(make_sample(text, "application",
                                   sample_output("application_control", "close", "application", app)))
    for text in restart[:n_each]:
        app = next(a for a in APPLICATIONS if a in text.lower())
        result.append(make_sample(text, "application",
                                   sample_output("application_control", "restart_app", "application", app)))
    remaining = target - 3 * n_each
    checkapp = []
    for v, a, suf in itertools.product(CHECKAPP_VERBS, APPLICATIONS, ["?", " sih?", " nggak?", " ya?"]):
        checkapp.append(f"{v} {a} lagi jalan{suf}")
    RNG.shuffle(checkapp)
    checkapp = pad_to_target(checkapp, remaining)
    for text in checkapp:
        app = next(a for a in APPLICATIONS if a in text.lower())
        result.append(make_sample(text, "application",
                                   sample_output("application_control", "check_app_status", "application", app)))
    return result[:target]


# ===========================================================================
# 2. PROCESS (target 2000)
# ===========================================================================
PROCESS_NAMES = [
    "python3", "node", "chrome-renderer", "spotify", "discord", "java",
    "postgres", "nginx", "docker", "ffmpeg", "rsync", "ssh-agent", "gpg-agent",
    "systemd", "pulseaudio", "pipewire", "Xorg", "electron", "code",
    "obs", "blender", "gimp", "firefox-bin", "steamwebhelper", "zoom",
    "mysql", "redis-server", "webpack", "gunicorn", "celery",
]
LIST_VERBS = ["Tampilin", "Liat", "Cek", "Kasih liat", "Kasih tau"]
KILL_VERBS = ["Kill", "Matiin proses", "Hentikan proses", "Bunuh proses", "Force stop",
              "Stop paksa", "Terminasi"]
CHECKPROC_VERBS = ["Cek apakah proses", "Apakah proses", "Cek status proses",
                    "Masih jalan nggak proses"]
LIST_OBJECTS = ["semua proses yang jalan", "proses yang makan CPU tinggi",
                 "daftar proses aktif", "proses yang lagi running",
                 "proses yang makan RAM banyak", "proses zombie",
                 "proses background", "semua yang lagi jalan sekarang"]


def gen_process(target: int = 2000) -> list[dict]:
    result = []
    n_each = target // 3
    list_texts = pad_to_target(combos(LIST_VERBS, LIST_OBJECTS, STATEMENT_WRAPPERS), n_each)
    for text in list_texts:
        result.append(make_sample(text, "process",
                                   sample_output("process_control", "list_processes", None, None)))
    kill_texts = pad_to_target(combos(KILL_VERBS, PROCESS_NAMES), n_each)
    for text in kill_texts:
        proc = next(p for p in PROCESS_NAMES if p.lower() in text.lower())
        result.append(make_sample(text, "process",
                                   sample_output("process_control", "kill_process", "process", proc)))
    remaining = target - 2 * n_each
    check_texts = []
    for v, p, suf in itertools.product(CHECKPROC_VERBS, PROCESS_NAMES,
                                        ["?", " sih?", " ya?", " nggak sih?", " nggak?", " deh?"]):
        check_texts.append(f"{v} {p} masih jalan{suf}")
    RNG.shuffle(check_texts)
    check_texts = pad_to_target(check_texts, remaining)
    for text in check_texts:
        proc = next(p for p in PROCESS_NAMES if p.lower() in text.lower())
        result.append(make_sample(text, "process",
                                   sample_output("process_control", "check_process", "process", proc)))
    return result[:target]


# ===========================================================================
# 3. FILESYSTEM (target 3000)
# ===========================================================================
FILE_NAMES = [
    "laporan.docx", "foto_liburan.jpg", "video_tutorial.mp4", "musik.mp3",
    "presentasi.pptx", "data.csv", "catatan.txt", "script.py", "backup.zip",
    "invoice.pdf", "desain.psd", "database.sql", "config.yaml", "notes.md",
    "resume.pdf", "screenshot.png", "recording.wav", "spreadsheet.xlsx",
]
FOLDER_NAMES = [
    "Dokumen", "Downloads", "Proyek Kuliah", "Foto Liburan", "Backup",
    "Musik", "Video", "Kerjaan", "Arsip Lama", "Desktop",
]
CREATE_FILE_VERBS = ["Bikin file", "Buat file", "Bikinin file baru namanya", "Generate file"]
CREATE_FOLDER_VERBS = ["Bikin folder", "Buat folder", "Bikinin folder baru namanya", "Generate folder"]
DELETE_VERBS = ["Hapus", "Delete", "Buang", "Musnahin", "Remove"]
SEARCH_VERBS = ["Cariin file", "Cari file", "Temuin file", "Search file"]
LISTFILES_VERBS = ["Tampilin isi folder", "Liat isi folder", "Cek isi folder", "Buka isi folder"]


def gen_filesystem(target: int = 3000) -> list[dict]:
    result = []
    n_each = target // 6
    for text in pad_to_target(combos(CREATE_FILE_VERBS, FILE_NAMES), n_each):
        f = next(x for x in FILE_NAMES if x.lower() in text.lower())
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "create_file", "file", f)))
    for text in pad_to_target(combos(CREATE_FOLDER_VERBS, FOLDER_NAMES), n_each):
        f = next(x for x in FOLDER_NAMES if x.lower() in text.lower())
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "create_folder", "folder", f)))
    for text in pad_to_target(combos(DELETE_VERBS, FILE_NAMES), n_each):
        f = next(x for x in FILE_NAMES if x.lower() in text.lower())
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "delete_file", "file", f)))
    move_verbs = ["Pindahin", "Move", "Geser"]
    move_texts = []
    for v, f, dest in itertools.product(move_verbs, FILE_NAMES, FOLDER_NAMES):
        move_texts.append((f"{v} {f} ke folder {dest}.", f, dest))
    RNG.shuffle(move_texts)
    move_texts = pad_to_target(move_texts, n_each)
    for text, f, dest in move_texts:
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "move_file", "file", f, destination=dest)))
    for text in pad_to_target(combos(SEARCH_VERBS, FILE_NAMES), n_each):
        f = next(x for x in FILE_NAMES if x.lower() in text.lower())
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "search_file", "file", f)))
    remaining = target - 5 * n_each
    listfiles_texts = pad_to_target(combos(LISTFILES_VERBS, FOLDER_NAMES, STATEMENT_WRAPPERS), remaining)
    for text in listfiles_texts:
        f = next(x for x in FOLDER_NAMES if x.lower() in text.lower())
        result.append(make_sample(text, "filesystem",
                                   sample_output("filesystem_control", "list_files", "folder", f)))
    return result[:target]


# ===========================================================================
# 4. SHELL (target 2000)
# ===========================================================================
SHELL_COMMANDS = [
    "ls -la", "ps aux", "df -h", "du -sh .", "git status", "git pull",
    "npm install", "pip install requests", "docker ps", "docker compose up",
    "systemctl status nginx", "journalctl -xe", "free -h", "top",
    "cat /etc/os-release", "whoami", "pwd", "history", "uptime",
    "netstat -tulpn", "chmod +x script.sh", "curl https://example.com",
    "grep -r 'error' logs/", "find . -name '*.py'", "tar -xzvf backup.tar.gz",
    "npm run build", "npm run dev", "yarn install", "pip freeze",
    "git log --oneline", "git diff", "git branch -a", "git checkout main",
    "docker images", "docker exec -it web bash", "kubectl get pods",
    "ssh user@server", "scp file.txt user@server:/tmp", "ping google.com",
    "traceroute google.com", "lsof -i :8080", "kill -9 1234",
    "mkdir new_project", "rm -rf node_modules", "npm test",
    "python3 manage.py migrate", "python3 manage.py runserver",
    "make build", "make clean", "cmake ..",
]
SHELL_VERBS = ["Jalanin command", "Run", "Eksekusi", "Tolong jalanin", "Coba jalanin perintah",
               "Execute", "Ketik dan jalanin"]
SHELL_SUFFIXES = ["", " dong", " ya", " please", " sekarang", " di terminal", " deh"]


def gen_shell(target: int = 2000) -> list[dict]:
    result = []
    combos_list = list(itertools.product(SHELL_VERBS, SHELL_COMMANDS, SHELL_SUFFIXES))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for verb, cmd, suf in combos_list:
        text = f"{verb} `{cmd}`{suf}."
        result.append(make_sample(text, "shell",
                                   sample_output("shell_execution", "run_command", "command", cmd)))
    return result[:target]


# ===========================================================================
# 5. HARDWARE (target 1500)
# ===========================================================================
DEVICES = [
    "bluetooth", "wifi adapter", "kamera", "webcam", "mikrofon", "touchpad",
    "keyboard backlight", "airplane mode", "monitor eksternal", "fingerprint reader",
    "trackpad", "speaker eksternal", "night light", "auto-rotate", "hotspot",
]
ENABLE_VERBS = ["Nyalain", "Aktifin", "Enable", "Hidupin", "Turn on"]
DISABLE_VERBS = ["Matiin", "Nonaktifin", "Disable", "Matikan", "Turn off"]
TOGGLE_VERBS = ["Toggle", "Ganti status", "Switch", "Ubah status"]


def gen_hardware(target: int = 1500) -> list[dict]:
    result = []
    n_each = target // 3
    for text in pad_to_target(combos(ENABLE_VERBS, DEVICES), n_each):
        d = next(x for x in DEVICES if x.lower() in text.lower())
        result.append(make_sample(text, "hardware",
                                   sample_output("hardware_control", "enable_device", "device", d)))
    for text in pad_to_target(combos(DISABLE_VERBS, DEVICES), n_each):
        d = next(x for x in DEVICES if x.lower() in text.lower())
        result.append(make_sample(text, "hardware",
                                   sample_output("hardware_control", "disable_device", "device", d)))
    remaining = target - 2 * n_each
    for text in pad_to_target(combos(TOGGLE_VERBS, DEVICES), remaining):
        d = next(x for x in DEVICES if x.lower() in text.lower())
        result.append(make_sample(text, "hardware",
                                   sample_output("hardware_control", "toggle_device", "device", d)))
    return result[:target]


# ===========================================================================
# 6. AUDIO (target 1000)
# ===========================================================================
AUDIO_OBJS = ["suara", "volume", "audio", "speaker", "sound", "suara laptop", "bunyi"]
MUTE_VERBS = ["Matiin", "Mute", "Silence", "Diemin"]
UNMUTE_VERBS = ["Nyalain lagi", "Unmute", "Aktifin lagi", "Bunyiin lagi"]
VOLUP_VERBS = ["Naikin", "Kerasin", "Volume up", "Gedein"]
VOLDOWN_VERBS = ["Turunin", "Kecilin", "Volume down", "Pelanin"]


def gen_audio(target: int = 1000) -> list[dict]:
    result = []
    n_each = target // 5
    for text in pad_to_target(combos(MUTE_VERBS, AUDIO_OBJS), n_each):
        result.append(make_sample(text, "audio", sample_output("audio_control", "mute")))
    for text in pad_to_target(combos(UNMUTE_VERBS, AUDIO_OBJS), n_each):
        result.append(make_sample(text, "audio", sample_output("audio_control", "unmute")))
    for text in pad_to_target(combos(VOLUP_VERBS, AUDIO_OBJS), n_each):
        result.append(make_sample(text, "audio", sample_output("audio_control", "volume_up", amount=10)))
    for text in pad_to_target(combos(VOLDOWN_VERBS, AUDIO_OBJS), n_each):
        result.append(make_sample(text, "audio", sample_output("audio_control", "volume_down", amount=10)))
    remaining = target - 4 * n_each
    setvol_texts = []
    for level in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        for phr in ["Set volume ke {}%.", "Volume-nya jadi {} persen dong.", "Atur volume ke {}.",
                    "Volume {} persen ya.", "Pasang volume di {}%."]:
            setvol_texts.append((phr.format(level), level))
    RNG.shuffle(setvol_texts)
    setvol_texts = pad_to_target(setvol_texts, remaining)
    for text, level in setvol_texts:
        result.append(make_sample(text, "audio", sample_output("audio_control", "set_volume", level=level)))
    return result[:target]


# ===========================================================================
# 7. DISPLAY (target 500)
# ===========================================================================
def gen_display(target: int = 500) -> list[dict]:
    result = []
    bright_objs = ["layar", "brightness", "kecerahan layar", "layar laptop", "monitor"]
    n_each = target // 4
    for text in pad_to_target(combos(["Naikin", "Terangin", "Naikkan"], bright_objs), n_each):
        result.append(make_sample(text, "display", sample_output("display_control", "brightness_up", amount=10)))
    for text in pad_to_target(combos(["Turunin", "Redupin", "Turunkan"], bright_objs), n_each):
        result.append(make_sample(text, "display", sample_output("display_control", "brightness_down", amount=10)))
    setb_texts = []
    for level in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        for phr in ["Set kecerahan layar ke {}%.", "Brightness-nya jadi {} persen.", "Atur kecerahan ke {}.",
                    "Kecerahan layar {} persen ya.", "Pasang brightness di {}%."]:
            setb_texts.append((phr.format(level), level))
    RNG.shuffle(setb_texts)
    setb_texts = pad_to_target(setb_texts, n_each)
    for text, level in setb_texts:
        result.append(make_sample(text, "display", sample_output("display_control", "set_brightness", level=level)))
    remaining = target - 3 * n_each
    res_texts = []
    resolutions = ["1920x1080", "2560x1440", "3840x2160", "1366x768", "1600x900",
                   "2560x1080", "1440x900", "1280x720", "3440x1440", "1920x1200"]
    phrasings = ["Ganti resolusi layar ke {}.", "Set resolusi ke {}.", "Ubah resolusi jadi {}.",
                 "Resolusi layar jadi {} dong.", "Pasang resolusi {} ya.", "Coba ganti resolusi ke {}."]
    for res, phr in itertools.product(resolutions, phrasings):
        res_texts.append((phr.format(res), res))
    RNG.shuffle(res_texts)
    res_texts = pad_to_target(res_texts, remaining)
    for text, res in res_texts:
        result.append(make_sample(text, "display", sample_output("display_control", "set_resolution", resolution=res)))
    return result[:target]


# ===========================================================================
# 8. NETWORK (target 1000)
# ===========================================================================
WIFI_NAMES = ["Rumah_5G", "Kantor-WiFi", "Kos_Elit", "Cafe_Sebelah", "TP-Link_XYZ",
              "Indihome-2.4G", "Starlink_Home", "MyRepublic_5G", "Tetangga_Sebelah",
              "Perpustakaan_Free", "Kampus-Hotspot", "Apartemen_501"]
CONNECT_VERBS = ["Connect ke wifi", "Sambungin ke wifi", "Konekin ke", "Hubungin ke wifi", "Login ke wifi"]
DISCONNECT_VERBS = ["Putusin koneksi wifi", "Disconnect wifi", "Matiin koneksi wifi", "Keluar dari wifi"]
ENABLEWIFI_VERBS = ["Nyalain wifi", "Aktifin wifi", "Enable wifi", "Hidupin wifi"]
DISABLEWIFI_VERBS = ["Matiin wifi", "Disable wifi", "Nonaktifin wifi", "Matikan wifi"]
CHECKCONN_VERBS = ["Cek koneksi internet", "Apakah internet nyala", "Cek status wifi", "Internet lagi nyambung nggak"]
FILLER_SUFFIX = ["sekarang", "deh", "ya", "dong", "please", ""]


def gen_network(target: int = 1000) -> list[dict]:
    result = []
    n_each = target // 5
    connect_texts = []
    for v, w, suf in itertools.product(CONNECT_VERBS, WIFI_NAMES, FILLER_SUFFIX):
        connect_texts.append((f"{v} {w} {suf}".strip() + ".", w))
    RNG.shuffle(connect_texts)
    connect_texts = pad_to_target(connect_texts, n_each)
    for text, w in connect_texts:
        result.append(make_sample(text, "network",
                                   sample_output("network_control", "connect_wifi", "network", w)))
    for text in pad_to_target(combos(DISCONNECT_VERBS, FILLER_SUFFIX, STATEMENT_WRAPPERS), n_each):
        result.append(make_sample(text, "network",
                                   sample_output("network_control", "disconnect_wifi", None, None)))
    for text in pad_to_target(combos(ENABLEWIFI_VERBS, FILLER_SUFFIX, STATEMENT_WRAPPERS), n_each):
        result.append(make_sample(text, "network",
                                   sample_output("network_control", "enable_wifi", None, None)))
    for text in pad_to_target(combos(DISABLEWIFI_VERBS, FILLER_SUFFIX, STATEMENT_WRAPPERS), n_each):
        result.append(make_sample(text, "network",
                                   sample_output("network_control", "disable_wifi", None, None)))
    remaining = target - 4 * n_each
    for text in pad_to_target(combos(CHECKCONN_VERBS, FILLER_SUFFIX, STATEMENT_WRAPPERS), remaining):
        result.append(make_sample(text, "network",
                                   sample_output("network_control", "check_connection", None, None)))
    return result[:target]


# ===========================================================================
# 9. SYSTEM_QUERY (target 1000)
# ===========================================================================
RESOURCES = ["cpu", "ram", "disk", "baterai", "uptime", "suhu", "gpu", "penyimpanan"]
QUERY_VERBS = ["Cek penggunaan", "Berapa persen", "Gimana status", "Tampilin info",
               "Cek sisa", "Kasih tau kondisi", "Berapa sih pemakaian", "Info penggunaan",
               "Lihat status", "Berapa persentase", "Cek kondisi", "Tunjukin penggunaan"]
def gen_system_query(target: int = 1000) -> list[dict]:
    result = []
    per_res = target // len(RESOURCES)
    remainder = target - per_res * len(RESOURCES)
    for i, r in enumerate(RESOURCES):
        n = per_res + (1 if i < remainder else 0)
        texts = pad_to_target(combos(QUERY_VERBS, [r], STATEMENT_WRAPPERS), n)
        for text in texts:
            result.append(make_sample(text, "system_query",
                                       sample_output("information_query", "check_resource", "resource", r)))
    return result[:target]


# ===========================================================================
# 10. AMBIGUOUS_NEGATIVE (target 1000)
# ===========================================================================
OFFTOPIC_TEMPLATES = [
    "Ceritain dongeng tentang {topic}.", "Menurutmu {topic} itu penting nggak?",
    "Kasih tau fakta random soal {topic} dong.", "Kamu suka {topic} nggak?",
    "Gimana pendapatmu soal {topic}?", "Kenapa ya orang suka {topic}?",
    "Buatin puisi tentang {topic}.", "Kalau kamu jadi {topic}, gimana rasanya?",
    "Pernah denger cerita soal {topic}?", "Menurutmu {topic} itu gimana sih?",
    "Kasih contoh soal {topic} dong.", "Kenapa {topic} penting buat orang?",
]
OFFTOPIC_TOPICS = ["kucing", "hujan", "cinta", "kopi", "liburan", "mimpi",
                    "musim panas", "laut", "gunung", "bintang", "persahabatan",
                    "waktu", "kebahagiaan", "kesepian", "keberanian", "senja",
                    "bulan purnama", "petualangan", "masa kecil", "warna favorit",
                    "anjing", "makanan pedas", "musik jazz", "buku", "film",
                    "olahraga", "sejarah", "seni lukis", "fotografi", "traveling"]
AMBIG_SYSTEM_ISH = [
    "Itu gimana ya caranya.", "Yang tadi itu gimana.", "Coba deh liat.",
    "Bantuin dong.", "Aku bingung nih.", "Kok gini ya.", "Gimana caranya ya.",
    "Bisa bantu nggak.", "Ini kenapa ya.", "Tolongin dong.",
    "Aduh gimana nih.", "Kok error terus ya.", "Nggak ngerti deh.",
    "Susah amat sih ini.", "Ini harusnya gimana.", "Kayaknya salah deh.",
    "Bingung banget nih aku.", "Coba cek deh.", "Hmm gimana ya enaknya.",
    "Ada yang aneh nih.", "Kok gitu ya.", "Ini bener nggak sih.",
    "Nggak jalan-jalan nih.", "Kayaknya ada yang salah.", "Duh ribet amat.",
    "Kok bisa gitu ya.", "Aneh banget deh.", "Nggak biasanya gini.",
    "Ini normal nggak sih.", "Kok jadi gini ya.", "Waduh kenapa nih.",
    "Gimana dong solusinya.", "Ini beneran kejadian.", "Kok bingung terus ya.",
    "Nggak paham aku.", "Susah dimengerti deh.",
]
AMBIG_SUFFIXES = ["", " sih", " banget", " nih", " deh", " beneran", " deh sekarang", " ya", " loh", " kok"]
def gen_ambiguous_negative(target: int = 1000) -> list[dict]:
    result = []
    offtopic = list({t.format(topic=topic) for t, topic in itertools.product(OFFTOPIC_TEMPLATES, OFFTOPIC_TOPICS)})
    RNG.shuffle(offtopic)
    n_off = target // 2
    offtopic = pad_to_target(offtopic, n_off)
    for text in offtopic:
        result.append(make_sample(text, "ambiguous_negative", None, label="negative",
                                   note="bukan domain system"))
    n_amb = target - n_off
    ambig_expanded = list({f"{p}{suf}" for p, suf in itertools.product(AMBIG_SYSTEM_ISH, AMBIG_SUFFIXES)})
    RNG.shuffle(ambig_expanded)
    ambig_pool = pad_to_target(ambig_expanded, n_amb)
    for text in ambig_pool:
        result.append(make_sample(text, "ambiguous_negative", None, label="ambiguous",
                                   note="terlalu vague buat resolve action system apa"))
    return result[:target]


# ===========================================================================
# MAIN
# ===========================================================================
def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    generators = {
        "application": (gen_application, 3000),
        "process": (gen_process, 2000),
        "filesystem": (gen_filesystem, 3000),
        "shell": (gen_shell, 2000),
        "hardware": (gen_hardware, 1500),
        "audio": (gen_audio, 1000),
        "display": (gen_display, 500),
        "network": (gen_network, 1000),
        "system_query": (gen_system_query, 1000),
        "ambiguous_negative": (gen_ambiguous_negative, 1000),
    }
    total = 0
    print(f"{'task_category':22s} {'target':>8s} {'actual':>8s}")
    print("-" * 42)
    for cat, (fn, tgt) in generators.items():
        samples = fn(tgt)
        save_jsonl(samples, OUT_DIR / f"{cat}.jsonl")
        print(f"{cat:22s} {tgt:8d} {len(samples):8d}")
        total += len(samples)
    print("-" * 42)
    print(f"{'TOTAL':22s} {16000:8d} {total:8d}")


if __name__ == "__main__":
    main()
