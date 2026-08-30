"""
generate_productivity_full.py
================================
Generator dataset FULL untuk specialist "productivity" (§37), target
8,000 sample di 9 task_category:

    calendar       : 1500
    reminder       : 1500
    todo           : 1000
    schedule       : 1000
    notification   :  750
    communication  :  750
    update_delete  :  750
    ambiguous      :  375
    negative       :  375
    ------------------
    TOTAL          : 8000

Beda dari domain lain: `update_delete` CROSS-CUTTING (bisa update/delete
event/reminder/todo/schedule apapun), jadi action DAN intent tidak
deterministik untuk kategori ini -- perlu diprediksi model. `todo` juga
punya 2 action (create/complete). Kategori lain tetap deterministik.

Waktu diekspresikan sebagai FRASA NATURAL ("besok jam 8"), bukan ISO
datetime -- lebih tractable buat data sintetis (nggak butuh tanggal
absolut yang berubah-ubah tergantung kapan digenerate).
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260902)
OUT_DIR = Path(__file__).parent / "output" / "productivity"

_counter = itertools.count(1)
def new_id() -> str:
    return f"productivity_{next(_counter):06d}"


CATEGORY_TO_INTENT = {
    "calendar": "calendar", "reminder": "reminder", "todo": "todo",
    "schedule": "schedule", "notification": "notification", "communication": "communication",
    "ambiguous": "calendar",
}
# action deterministik untuk 5 kategori ini (todo & update_delete beda, lihat bawah)
CATEGORY_TO_ACTION = {
    "calendar": "create", "reminder": "create", "schedule": "schedule",
    "notification": "notify", "communication": "send",
}
NO_STAGE2_CATEGORIES = {"ambiguous"}
AMBIGUOUS_CONSTANT_OUTPUT = {
    "intent": "calendar", "action": "update",
    "target": {"type": "event", "value": None}, "parameters": {},
}

WRAPPERS = [
    "{v} {o}.", "{v} {o} dong.", "{v} {o} ya.", "{v} {o} sekarang.",
    "{v} {o} deh.", "Tolong {vl} {o}.", "Bisa {vl} {o} nggak?",
    "Coba {vl} {o}.", "{v} {o}, please.", "Eh, {vl} {o} dong.",
]
STATEMENT_WRAPPERS = [w for w in WRAPPERS if not w.rstrip().endswith("?")]


def wrap(v: str, o: str, w: str) -> str:
    return w.format(v=v, o=o, vl=v[0].lower() + v[1:] if v else v)


def combos(verbs, objects, wrappers=WRAPPERS):
    out = set()
    for v, o, w in itertools.product(verbs, objects, wrappers):
        out.add(wrap(v, o, w))
    out = list(out)
    RNG.shuffle(out)
    return out


def pad_to_target(pool: list, target: int) -> list:
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
    return {"domain": "productivity", "intent": intent, "action": action,
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
# VOCAB
# ===========================================================================
EVENTS = [
    "meeting", "rapat tim", "presentasi project", "interview kerja",
    "checkup dokter", "arisan keluarga", "webinar karir", "konser musik",
    "wisuda", "acara ulang tahun", "rapat divisi", "sesi konsultasi",
    "kelas yoga", "sesi coaching", "demo produk", "training karyawan",
    "makan siang klien", "sidang skripsi", "medical checkup", "acara kantor",
]
TIME_PHRASES = [
    "besok jam 10", "besok jam 2 siang", "jam 3 sore ini", "minggu depan",
    "besok pagi jam 8", "lusa jam 9", "hari Senin jam 10", "jam 7 malam nanti",
    "besok siang jam 1", "jumat depan jam 2", "hari ini jam 4 sore",
    "minggu depan hari Rabu", "bulan depan tanggal 5", "besok jam 11 siang",
]
TODO_ITEMS = [
    "beli galon", "cuci baju", "kirim laporan bulanan", "beli tiket kereta",
    "bayar tagihan listrik", "isi bensin motor", "beli obat", "kirim invoice",
    "review dokumen kontrak", "update slide presentasi", "backup data project",
    "beli bahan masakan", "antar paket ke kantor pos", "cek email penting",
    "follow up klien", "siapin materi training", "renew domain website",
]
RECURRENCE_PHRASES = [
    ("daily 06:00", "tiap pagi jam 6"), ("daily 07:00", "tiap hari jam 7 pagi"),
    ("weekly monday 09:00", "tiap Senin jam 9"), ("weekly friday 17:00", "tiap Jumat jam 5 sore"),
    ("daily 20:00", "tiap malam jam 8"), ("weekly sunday 08:00", "tiap Minggu jam 8 pagi"),
    ("daily 12:00", "tiap siang jam 12"), ("weekly wednesday 15:00", "tiap Rabu jam 3 sore"),
]
NOTIFICATION_CONTENTS = [
    "meeting udah mulai", "deadline besok", "tugas udah selesai",
    "ada email baru masuk", "reminder rapat 5 menit lagi", "file udah keupload",
    "pesan baru dari klien", "sistem update selesai", "backup selesai dilakukan",
    "invoice udah dikirim", "laporan udah siap", "ada perubahan jadwal",
    "meeting online udah dimulai", "dokumen udah direview", "approval udah masuk",
    "pembayaran udah diterima", "jadwal besok berubah", "task baru ditambahkan",
]
MESSAGE_CONTENTS = [
    "oke siap", "baik, akan saya kerjakan", "terima kasih infonya",
    "noted, akan saya follow up", "sip, sampai jumpa besok",
    "mohon maaf telat balas", "sudah saya terima, terima kasih",
    "akan saya cek dan kabari lagi", "oke ditunggu ya", "siap laksanakan",
    "baik, dimengerti", "terima kasih atas waktunya", "oke, saya usahakan",
    "baik, akan segera saya proses", "sip, akan saya kabari",
]
CONTACTS = ["Pak Budi", "Bu Sari", "tim marketing", "klien", "atasan", "HRD"]


# ===========================================================================
# 1. CALENDAR (target 1500)
# ===========================================================================
CALENDAR_VERBS = ["Bikin event", "Buat acara", "Jadwalin", "Tambahin ke kalender",
                   "Bikinin acara", "Set jadwal"]
def gen_calendar(target: int = 1500) -> list[dict]:
    result = []
    texts = []
    for v, e, t in itertools.product(CALENDAR_VERBS, EVENTS, TIME_PHRASES):
        texts.append((f"{v} {e} {t}.", e, t))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, e, t in texts:
        result.append(make_sample(text, "calendar",
                                   sample_output("calendar", "create", "event", e, time=t)))
    return result[:target]


# ===========================================================================
# 2. REMINDER (target 1500)
# ===========================================================================
REMINDER_VERBS = ["Ingetin aku", "Ingatkan aku", "Reminder buat", "Tolong ingetin aku",
                   "Bikin reminder buat"]
def gen_reminder(target: int = 1500) -> list[dict]:
    result = []
    texts = []
    for v, t, item in itertools.product(REMINDER_VERBS, TIME_PHRASES, TODO_ITEMS):
        texts.append((f"{v} {t} {item}.", t, item))
        texts.append((f"{t.capitalize()}, {v.lower()} {item}.", t, item))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, t, item in texts:
        result.append(make_sample(text, "reminder",
                                   sample_output("reminder", "create", "reminder", None, time=t, content=item)))
    return result[:target]


# ===========================================================================
# 3. TODO (target 1000) -- action create ATAU complete
# ===========================================================================
TODO_CREATE_VERBS = ["Tambahin todo", "Bikin todo", "Catet todo", "Masukin ke todo list",
                      "Tambah ke daftar tugas", "Tulis todo"]
TODO_COMPLETE_VERBS = ["Tandain todo", "Selesain todo", "Centang todo", "Mark todo",
                        "Tandai todo", "Beresin todo"]
def gen_todo(target: int = 1000) -> list[dict]:
    result = []
    n_create = target // 2
    n_complete = target - n_create

    create_texts = pad_to_target(combos(TODO_CREATE_VERBS, TODO_ITEMS), n_create)
    for text in create_texts:
        item = next(x for x in TODO_ITEMS if x.lower() in text.lower())
        result.append(make_sample(text, "todo", sample_output("todo", "create", "todo", item)))

    complete_texts = []
    for v, item, suf in itertools.product(TODO_COMPLETE_VERBS, TODO_ITEMS,
                                            ["udah selesai", "beres", "udah kelar", "done", "udah rampung"]):
        complete_texts.append(f"{v} {item} {suf}.")
    RNG.shuffle(complete_texts)
    complete_texts = pad_to_target(complete_texts, n_complete)
    for text in complete_texts:
        item = next(x for x in TODO_ITEMS if x.lower() in text.lower())
        result.append(make_sample(text, "todo", sample_output("todo", "complete", "todo", item)))
    return result[:target]


# ===========================================================================
# 4. SCHEDULE (target 1000)
# ===========================================================================
SCHEDULE_VERBS = ["Jadwalin", "Bikin jadwal rutin buat", "Set rutinitas", "Atur jadwal buat",
                   "Bikinin jadwal buat", "Tolong jadwalin"]
SCHEDULE_EVENTS = ["olahraga", "meeting mingguan", "review project", "yoga",
                    "belajar bahasa Inggris", "meditasi", "cek email", "backup data",
                    "laporan mingguan", "standup tim", "jalan pagi", "baca buku",
                    "latihan gitar", "beres-beres rumah", "nulis jurnal", "belajar coding"]
def gen_schedule(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, e, (rec_code, rec_phrase) in itertools.product(SCHEDULE_VERBS, SCHEDULE_EVENTS, RECURRENCE_PHRASES):
        texts.append((f"{v} {e} {rec_phrase}.", e, rec_code))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, e, rec_code in texts:
        result.append(make_sample(text, "schedule",
                                   sample_output("schedule", "schedule", "event", e, recurrence=rec_code)))
    return result[:target]


# ===========================================================================
# 5. NOTIFICATION (target 750)
# ===========================================================================
NOTIF_VERBS = ["Kirim notifikasi kalau", "Notify aku kalau", "Kasih tau aku kalau",
               "Beritahu aku kalau", "Kirim notif kalau", "Alert aku kalau", "Info aku kalau"]
NOTIF_SUFFIXES = ["", " ya", " dong", " deh", " sekarang", " nanti", " ya nanti"]
def gen_notification(target: int = 750) -> list[dict]:
    result = []
    texts = []
    for v, c, suf in itertools.product(NOTIF_VERBS, NOTIFICATION_CONTENTS, NOTIF_SUFFIXES):
        texts.append(f"{v} {c}{suf}.")
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        c = next(x for x in NOTIFICATION_CONTENTS if x.lower() in text.lower())
        result.append(make_sample(text, "notification",
                                   sample_output("notification", "notify", "notification", c)))
    return result[:target]


# ===========================================================================
# 6. COMMUNICATION (target 750)
# ===========================================================================
COMM_VERBS = ["Balas pesan ini dengan", "Kirim pesan", "Reply dengan", "Balas dengan",
              "Kirim balasan", "Tolong balas dengan"]
COMM_SUFFIXES = ["", " ya", " dong", " deh", " sekarang", " aja"]
def gen_communication(target: int = 750) -> list[dict]:
    result = []
    texts = []
    for v, m, suf in itertools.product(COMM_VERBS, MESSAGE_CONTENTS, COMM_SUFFIXES):
        texts.append(f"{v} '{m}'{suf}.")
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        m = next(x for x in MESSAGE_CONTENTS if x.lower() in text.lower())
        result.append(make_sample(text, "communication",
                                   sample_output("communication", "send", "message", m)))
    return result[:target]


# ===========================================================================
# 7. UPDATE_DELETE (target 750) -- cross-cutting, action & intent diprediksi
# ===========================================================================
def gen_update_delete(target: int = 750) -> list[dict]:
    result = []
    n_each = target // 4

    # update event/calendar
    upd_cal_texts = []
    for e, t in itertools.product(EVENTS, TIME_PHRASES):
        upd_cal_texts.append((f"Ubah jadwal {e} jadi {t}.", e, t))
        upd_cal_texts.append((f"Geser {e} ke {t}.", e, t))
        upd_cal_texts.append((f"Pindahin {e} ke {t}.", e, t))
    RNG.shuffle(upd_cal_texts)
    upd_cal_texts = pad_to_target(upd_cal_texts, n_each)
    for text, e, t in upd_cal_texts:
        result.append(make_sample(text, "update_delete",
                                   sample_output("calendar", "update", "event", e, time=t)))

    # delete reminder
    del_rem_texts = []
    for item in TODO_ITEMS:
        del_rem_texts.append((f"Hapus reminder {item}.", item))
        del_rem_texts.append((f"Batalin reminder {item}.", item))
        del_rem_texts.append((f"Cancel reminder {item}.", item))
    RNG.shuffle(del_rem_texts)
    del_rem_texts = pad_to_target(del_rem_texts, n_each)
    for text, item in del_rem_texts:
        result.append(make_sample(text, "update_delete",
                                   sample_output("reminder", "delete", "reminder", item)))

    # delete event
    del_cal_texts = []
    for e in EVENTS:
        del_cal_texts.append((f"Batalin {e}.", e))
        del_cal_texts.append((f"Hapus event {e}.", e))
        del_cal_texts.append((f"Cancel {e}.", e))
        del_cal_texts.append((f"Batalin jadwal {e}.", e))
    RNG.shuffle(del_cal_texts)
    del_cal_texts = pad_to_target(del_cal_texts, n_each)
    for text, e in del_cal_texts:
        result.append(make_sample(text, "update_delete",
                                   sample_output("calendar", "delete", "event", e)))

    # delete todo
    remaining = target - 3 * n_each
    del_todo_texts = []
    for item in TODO_ITEMS:
        del_todo_texts.append((f"Hapus todo {item}.", item))
        del_todo_texts.append((f"Buang todo {item} dari list.", item))
        del_todo_texts.append((f"Cancel todo {item}.", item))
    RNG.shuffle(del_todo_texts)
    del_todo_texts = pad_to_target(del_todo_texts, remaining)
    for text, item in del_todo_texts:
        result.append(make_sample(text, "update_delete",
                                   sample_output("todo", "delete", "todo", item)))
    return result[:target]


# ===========================================================================
# 8. AMBIGUOUS (target 375) -- output KONSTAN, skip stage2
# ===========================================================================
AMBIG_PRODUCTIVITY_PHRASES = [
    "Ubah itu jadi besok aja.", "Geser dong.", "Ubah aja deh.", "Batalin yang itu.",
    "Hapus yang tadi.", "Ganti jadwalnya.", "Ubah waktunya.", "Geser ke lain waktu.",
    "Batalin aja.", "Hapus aja deh.", "Ubah yang ini.", "Ganti aja waktunya.",
    "Geser aja deh.", "Pindahin ke lain hari.", "Ganti jadwal yang itu.",
    "Ubah jadwal yang tadi.", "Batalin yang barusan.", "Hapus yang itu aja.",
    "Ganti waktunya deh.", "Pindahin waktunya.", "Ubah tanggalnya.",
    "Batalin yang tadi aja.", "Hapus itu deh.", "Geser waktu yang itu.",
]
AMBIG_SUFFIXES2 = ["", " dong", " ya", " deh", " sekarang", " aja", " nih", " beneran"]
def gen_ambiguous(target: int = 375) -> list[dict]:
    result = []
    combos_list = list(itertools.product(AMBIG_PRODUCTIVITY_PHRASES, AMBIG_SUFFIXES2))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for base, suf in combos_list:
        text = base.rstrip(".") + suf + "."
        result.append(make_sample(text, "ambiguous",
                                   sample_output("calendar", "update", "event", None),
                                   label="ambiguous", note="event mana yang diubah tidak jelas tanpa context"))
    return result[:target]


# ===========================================================================
# 9. NEGATIVE (target 375)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Ingat kalau {fact}.", "Cari tahu {topic}.", "Putar lagu {topic}.",
    "Apa itu {topic}?", "Jelasin {topic} dong.", "Cek {topic} dong.",
    "Berapa {topic}?", "Tolong cariin {topic}.",
]
NEGATIVE_FACTS = ["aku suka kopi hitam", "nama kucingku Milo", "aku alergi seafood",
                   "aku kerja di startup fintech", "aku tinggal di Bandung",
                   "aku kuliah di ITB", "golongan darahku O", "aku suka musik jazz"]
NEGATIVE_TOPICS = ["ibu kota Jepang", "fotosintesis", "harga bitcoin",
                    "resep rendang", "cuaca hari ini", "lagu Noah",
                    "arti kata algoritma", "berita politik", "harga emas",
                    "populasi Indonesia", "review HP terbaru", "tempat wisata Bali",
                    "jarak Jakarta ke Bandung", "definisi machine learning"]
def gen_negative(target: int = 375) -> list[dict]:
    result = []
    texts = []
    for fact in NEGATIVE_FACTS:
        texts.append(f"Ingat kalau {fact}.")
    for t, topic in itertools.product(NEGATIVE_TEMPLATES[1:], NEGATIVE_TOPICS):
        texts.append(t.format(topic=topic))
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        result.append(make_sample(text, "negative", None, label="negative",
                                   note="domain memory/information/media, bukan productivity"))
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
        "calendar": (gen_calendar, 1500),
        "reminder": (gen_reminder, 1500),
        "todo": (gen_todo, 1000),
        "schedule": (gen_schedule, 1000),
        "notification": (gen_notification, 750),
        "communication": (gen_communication, 750),
        "update_delete": (gen_update_delete, 750),
        "ambiguous": (gen_ambiguous, 375),
        "negative": (gen_negative, 375),
    }
    total = 0
    print(f"{'task_category':16s} {'target':>8s} {'actual':>8s}")
    print("-" * 36)
    for cat, (fn, tgt) in generators.items():
        samples = fn(tgt)
        save_jsonl(samples, OUT_DIR / f"{cat}.jsonl")
        print(f"{cat:16s} {tgt:8d} {len(samples):8d}")
        total += len(samples)
    print("-" * 36)
    print(f"{'TOTAL':16s} {8000:8d} {total:8d}")


if __name__ == "__main__":
    main()
