"""
generate_memory_full.py
=========================
Generator dataset FULL untuk specialist "memory" (§36), target 7,000
sample di 9 task_category:

    remember           : 1500
    retrieve           : 1000
    update             : 1000
    forget             :  750
    search             :  750
    event_memory       :  750
    preference_memory  :  750
    ambiguous          :  250
    negative           :  250
    ------------------------
    TOTAL              : 7000

`target` SELALU null. `action` deterministik dari category. Karena
remember/event_memory/preference_memory SAMA-SAMA action="remember"
(cuma beda "category" di parameters), Stage 2 dikelompokkan jadi 5
prompt (bukan 9): "store" (gabung 3 kategori itu), "retrieve",
"update", "forget", "search". "ambiguous" outputnya konstan (skip
stage2). "negative" skip sepenuhnya.
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260901)
OUT_DIR = Path(__file__).parent / "output" / "memory"

_counter = itertools.count(1)
def new_id() -> str:
    return f"memory_{next(_counter):06d}"


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


def sample_output(intent, action, **params):
    return {"domain": "memory", "intent": intent, "action": action, "target": None, "parameters": params}


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
PREFERENCE_ITEMS = [
    "kopi hitam", "makanan pedas", "musik jazz", "kucing daripada anjing",
    "warna biru", "nonton film daripada baca buku", "kerja pagi daripada malam",
    "kopi daripada teh", "liburan ke pantai daripada gunung", "masakan Padang",
    "olahraga lari", "baca buku fiksi", "nonton anime", "dengerin podcast",
    "makan sayur", "musik akustik", "film horor", "kerja remote",
    "belanja online", "masak sendiri daripada beli",
]
PET_NAMES = ["Milo", "Kitty", "Coco", "Bobby", "Luna", "Simba", "Momo", "Oyen"]
WORKPLACES = ["startup fintech", "perusahaan konsultan", "bank BUMN", "agency digital",
               "perusahaan logistik", "rumah sakit", "sekolah internasional", "kantor pemerintah"]
CITIES = ["Bandung", "Surabaya", "Yogyakarta", "Semarang", "Medan", "Makassar", "Bali", "Malang"]
ALLERGENS = ["seafood", "kacang", "susu sapi", "gluten", "telur", "debu", "udang", "durian"]
UNIVERSITIES = ["ITB", "UI", "UGM", "ITS", "Binus", "Unpad", "Undip", "Telkom University"]
EVENT_TOPICS = [
    "wawancara kerja", "reuni SMA", "presentasi proyek", "acara keluarga",
    "trip ke Bali", "konser musik", "ujian skripsi", "acara nikahan temen",
    "medical checkup", "rapat penting", "kompetisi coding", "webinar karir",
]

# (category, template_dengan_slot, content_template_dengan_slot, values)
FACT_BANK_IDENTITY = [
    ("identity", "nama kucingku {v}", "nama kucing user: {v}", PET_NAMES),
    ("identity", "kerja di {v}", "user kerja di {v}", WORKPLACES),
    ("identity", "tinggal di {v}", "user tinggal di {v}", CITIES),
    ("identity", "alergi {v}", "user alergi {v}", ALLERGENS),
    ("identity", "kuliah di {v}", "user kuliah di {v}", UNIVERSITIES),
]
FACT_BANK_PREFERENCE = [
    ("preference", "suka {v}", "user suka {v}", PREFERENCE_ITEMS),
    ("preference", "nggak suka {v}", "user nggak suka {v}", PREFERENCE_ITEMS),
]
FACT_BANK_EVENT = [
    ("event", "pernah cerita soal {v} minggu lalu", "user cerita soal {v} minggu lalu", EVENT_TOPICS),
    ("event", "abis {v} kemarin", "user abis {v} kemarin", EVENT_TOPICS),
    ("event", "mau {v} bulan depan", "user mau {v} bulan depan", EVENT_TOPICS),
]

REMEMBER_VERBS = ["Inget", "Inget ya", "Catet", "Simpen info kalau", "Tolong inget",
                   "Inget-inget ya", "Simpen ya kalau", "Tolong catet kalau"]
FORGET_VERBS = ["Lupain soal", "Hapus ingatan soal", "Hapus info soal", "Lupain aja soal",
                "Buang ingatan soal", "Tolong lupain soal"]
RETRIEVE_VERBS = ["Kamu masih inget", "Inget nggak", "Tau nggak", "Kamu tau nggak",
                   "Kamu inget nggak", "Masih inget nggak"]
UPDATE_VERBS = ["Update,", "Ganti info,", "Sekarang,", "Update dong,", "Ganti ya,", "Revisi,"]
SEARCH_VERBS = ["Cari memori soal", "Cariin ingatan soal", "Cari info yang aku simpen soal",
                 "Cariin yang aku pernah bilang soal", "Cari catatan soal"]


# ===========================================================================
# helper: bangun teks + content dari FACT_BANK
# ===========================================================================
def _build_fact_texts(fact_bank, verbs, target):
    texts = []
    for cat, tmpl, content_tmpl, values in fact_bank:
        for v in values:
            phrase = tmpl.format(v=v)
            content = content_tmpl.format(v=v)
            for verb in verbs:
                for suf in ["", " ya", " dong"]:
                    if verb.endswith(","):
                        text = f"{verb} aku {phrase}{suf}."
                    else:
                        text = f"{verb} aku {phrase}{suf}."
                    texts.append((text, cat, content))
    RNG.shuffle(texts)
    return pad_to_target(texts, target)


# ===========================================================================
# 1. REMEMBER (target 1500) -- identity facts
# ===========================================================================
def gen_remember(target: int = 1500) -> list[dict]:
    result = []
    texts = _build_fact_texts(FACT_BANK_IDENTITY, REMEMBER_VERBS, target)
    for text, cat, content in texts:
        result.append(make_sample(text, "remember",
                                   sample_output("memory_store", "remember", category=cat, content=content)))
    return result[:target]


# ===========================================================================
# 2. PREFERENCE_MEMORY (target 750)
# ===========================================================================
def gen_preference_memory(target: int = 750) -> list[dict]:
    result = []
    texts = _build_fact_texts(FACT_BANK_PREFERENCE, REMEMBER_VERBS, target)
    for text, cat, content in texts:
        result.append(make_sample(text, "preference_memory",
                                   sample_output("memory_store", "remember", category=cat, content=content)))
    return result[:target]


# ===========================================================================
# 3. EVENT_MEMORY (target 750)
# ===========================================================================
def gen_event_memory(target: int = 750) -> list[dict]:
    result = []
    texts = _build_fact_texts(FACT_BANK_EVENT, REMEMBER_VERBS, target)
    for text, cat, content in texts:
        result.append(make_sample(text, "event_memory",
                                   sample_output("memory_store", "remember", category=cat, content=content)))
    return result[:target]


# ===========================================================================
# 4. UPDATE (target 1000) -- reuse preference + identity fact bank
# ===========================================================================
def gen_update(target: int = 1000) -> list[dict]:
    result = []
    combined_bank = FACT_BANK_PREFERENCE + FACT_BANK_IDENTITY
    texts = []
    for cat, tmpl, content_tmpl, values in combined_bank:
        for v in values:
            phrase = tmpl.format(v=v)
            content = content_tmpl.format(v=v)
            for verb in UPDATE_VERBS:
                for suf in ["", " ya"]:
                    text = f"{verb} sekarang aku {phrase}{suf}."
                    texts.append((text, cat, content))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, cat, content in texts:
        result.append(make_sample(text, "update",
                                   sample_output("memory_update", "update", category=cat, content=content)))
    return result[:target]


# ===========================================================================
# 5. RETRIEVE (target 1000)
# ===========================================================================
RETRIEVE_QUERIES = [
    ("identity", "nama kucing"), ("identity", "tempat kerja"), ("identity", "tempat tinggal"),
    ("identity", "alergi apa"), ("identity", "kuliah dimana"),
    ("preference", "makanan favorit"), ("preference", "musik favorit"), ("preference", "warna favorit"),
    ("event", "cerita minggu lalu"), ("event", "rencana bulan depan"), ("event", "wawancara kerja"),
    ("identity", "golongan darah"), ("preference", "film favorit"), ("preference", "olahraga favorit"),
    ("event", "cerita reuni SMA"), ("identity", "nomor plat motor"),
]
RETRIEVE_SUFFIXES = ["", " sih", " ya", " deh", " nggak sih"]
def gen_retrieve(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, (cat, q), suf in itertools.product(RETRIEVE_VERBS, RETRIEVE_QUERIES, RETRIEVE_SUFFIXES):
        texts.append((f"{v} {q} aku{suf}?", cat, q))
        texts.append((f"{v} soal {q} aku{suf}?", cat, q))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, cat, q in texts:
        result.append(make_sample(text, "retrieve",
                                   sample_output("memory_retrieve", "retrieve", category=cat, query=q)))
    return result[:target]


# ===========================================================================
# 6. FORGET (target 750) -- hanya category, tanpa content
# ===========================================================================
FORGET_TOPICS = [
    ("preference", "preferensi musik aku"), ("preference", "makanan favoritku"),
    ("identity", "data pekerjaan aku"), ("identity", "tempat tinggalku"),
    ("identity", "info alergiku"), ("event", "cerita minggu lalu"),
    ("event", "rencana liburanku"), ("preference", "preferensi film aku"),
    ("identity", "data kuliahku"), ("identity", "info kucingku"),
    ("preference", "preferensi warna aku"), ("event", "cerita wawancara kerjaku"),
    ("event", "cerita reuni SMA"), ("preference", "preferensi olahraga aku"),
]
FORGET_SUFFIXES = ["", " ya", " dong", " deh", " sekarang", " aja"]
def gen_forget(target: int = 750) -> list[dict]:
    result = []
    texts = pad_to_target(list(itertools.product(FORGET_VERBS, FORGET_TOPICS, FORGET_SUFFIXES)), target)
    for verb, (cat, topic), suf in texts:
        text = f"{verb} {topic}{suf}."
        result.append(make_sample(text, "forget",
                                   sample_output("memory_delete", "forget", category=cat)))
    return result[:target]


# ===========================================================================
# 7. SEARCH (target 750) -- free-text query, tanpa category
# ===========================================================================
SEARCH_QUERIES = [
    "rencana liburanku", "cerita soal kerjaan", "preferensi makanan aku",
    "info soal kucingku", "cerita minggu lalu", "rencana weekend",
    "obrolan soal film", "info alergi aku", "cerita soal keluarga",
    "rencana kuliah", "obrolan soal kerjaan baru", "cerita soal mantan",
]
def gen_search(target: int = 750) -> list[dict]:
    result = []
    texts = pad_to_target(combos(SEARCH_VERBS, SEARCH_QUERIES, STATEMENT_WRAPPERS), target)
    for text in texts:
        q = next(x for x in SEARCH_QUERIES if x.lower() in text.lower())
        result.append(make_sample(text, "search", sample_output("memory_query", "search", query=q)))
    return result[:target]


# ===========================================================================
# 8. AMBIGUOUS (target 250) -- output KONSTAN, skip stage2
# ===========================================================================
AMBIG_MEMORY_PHRASES = [
    "Inget yang tadi ya.", "Simpen itu dong.", "Catet yang barusan.",
    "Inget itu ya.", "Simpen deh.", "Catet aja.", "Inget yang ini.",
    "Simpen yang tadi.", "Catet yang itu.", "Inget dong yang barusan.",
    "Simpen info tadi.", "Catet ya yang tadi.", "Inget deh yang itu.",
    "Simpen yang barusan aja.", "Catet dulu deh.", "Inget-inget deh.",
    "Simpen yang ini dong.", "Catet dong ya.",
]
def gen_ambiguous(target: int = 250) -> list[dict]:
    result = []
    combos_list = list(itertools.product(AMBIG_MEMORY_PHRASES, ["", " dong", " ya", " deh", " sekarang", " aja"]))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for base, suf in combos_list:
        text = base.rstrip(".") + suf + "."
        result.append(make_sample(text, "ambiguous",
                                   sample_output("memory_store", "remember", content=None),
                                   label="ambiguous", note="konten yang mau diingat tidak jelas tanpa context"))
    return result[:target]


# ===========================================================================
# 9. NEGATIVE (target 250)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Cari tahu {topic}.", "Berapa {topic}?", "Apa itu {topic}?",
    "Putar musik soal {topic}.", "Jelasin {topic} dong.",
    "Cek {topic} dong.", "Tolong cariin {topic}.", "Info soal {topic} dong.",
]
NEGATIVE_TOPICS = ["ibu kota Jepang", "hasil 25 kali 4", "fotosintesis",
                    "lagu terbaru", "harga bitcoin", "cuaca hari ini",
                    "resep rendang", "jarak Jakarta ke Bandung", "arti kata algoritma",
                    "populasi Indonesia", "harga emas", "berita politik",
                    "tempat wisata Bali", "review HP terbaru"]
def gen_negative(target: int = 250) -> list[dict]:
    result = []
    texts = []
    for t, topic in itertools.product(NEGATIVE_TEMPLATES, NEGATIVE_TOPICS):
        texts.append(t.format(topic=topic))
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        result.append(make_sample(text, "negative", None, label="negative",
                                   note="domain information/media, bukan memory"))
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
        "remember": (gen_remember, 1500),
        "retrieve": (gen_retrieve, 1000),
        "update": (gen_update, 1000),
        "forget": (gen_forget, 750),
        "search": (gen_search, 750),
        "event_memory": (gen_event_memory, 750),
        "preference_memory": (gen_preference_memory, 750),
        "ambiguous": (gen_ambiguous, 250),
        "negative": (gen_negative, 250),
    }
    total = 0
    print(f"{'task_category':20s} {'target':>8s} {'actual':>8s}")
    print("-" * 40)
    for cat, (fn, tgt) in generators.items():
        samples = fn(tgt)
        save_jsonl(samples, OUT_DIR / f"{cat}.jsonl")
        print(f"{cat:20s} {tgt:8d} {len(samples):8d}")
        total += len(samples)
    print("-" * 40)
    print(f"{'TOTAL':20s} {7000:8d} {total:8d}")


if __name__ == "__main__":
    main()
