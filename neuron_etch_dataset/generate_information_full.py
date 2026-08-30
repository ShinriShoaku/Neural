"""
generate_information_full.py
==============================
Generator dataset FULL untuk specialist "information" (§35), target
8,000 sample di 10 task_category:

    search       : 1500
    weather      : 1000
    time         :  500
    translation  : 1000
    lookup       : 1000
    calculation  :  500
    comparison   : 1000
    knowledge    : 1000
    ambiguous    :  250
    negative     :  250
    ------------------
    TOTAL        : 8000

`action` deterministik dari category (lookup CATEGORY_TO_ACTION).
"ambiguous" outputnya KONSTAN (sama persis apapun kalimatnya) jadi
TIDAK butuh stage2 sama sekali (mirip review/test di coding).
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260831)
OUT_DIR = Path(__file__).parent / "output" / "information"

_counter = itertools.count(1)
def new_id() -> str:
    return f"information_{next(_counter):06d}"


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
    return {"domain": "information", "intent": intent, "action": action,
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
SEARCH_QUERIES = [
    "ibu kota Prancis", "resep rendang", "cara bikin CV yang bagus",
    "tempat wisata di Bali", "harga tiket pesawat ke Jepang", "berita politik terkini",
    "tren fashion 2026", "review HP terbaru", "lowongan kerja remote",
    "cara investasi saham", "tips diet sehat", "jadwal film bioskop",
    "hotel murah di Bandung", "kursus online gratis", "startup unicorn Indonesia",
    "cara daftar SIM online", "syarat visa Jepang", "harga emas hari ini",
    "cara membuat website", "aplikasi belajar bahasa Inggris",
    "cara pindah kewarganegaraan", "tips wawancara kerja", "harga rumah subsidi",
    "cara buka rekening bank online", "review laptop gaming", "tempat kuliner enak",
    "cara ternak lele", "harga mobil bekas", "tips menabung", "cara bikin NPWP",
]
WEATHER_TIMES = ["today", "tomorrow", "day_after_tomorrow"]
WEATHER_TIME_PHRASES = {"today": ["hari ini", "sekarang", "saat ini"],
                          "tomorrow": ["besok", "besok pagi", "besok siang"],
                          "day_after_tomorrow": ["lusa", "dua hari lagi"]}
CITIES = ["Jakarta", "Bandung", "Surabaya", "Tokyo", "Singapura", "Bangkok",
          "Kuala Lumpur", "Sydney", "London", "New York", "Paris", "Seoul"]
ENTITIES = [
    "presiden Indonesia", "CEO Tesla", "penemu telepon", "pendiri Microsoft",
    "gubernur Jakarta", "juara piala dunia terakhir", "penulis novel Laskar Pelangi",
    "sutradara film Parasite", "pencetak gol terbanyak Liga Inggris",
    "menteri keuangan Indonesia", "presiden AS saat ini", "raja Inggris",
    "CEO Apple", "pendiri Facebook", "juara Ballon d'Or terakhir",
    "gubernur Bank Indonesia", "ketua umum PSSI", "pemenang Oscar terbaru",
    "penemu lampu pijar", "pendiri Amazon", "CEO Google", "penulis Harry Potter",
    "penemu vaksin polio", "arsitek Menara Eiffel", "pelukis Mona Lisa",
    "penemu telepon genggam", "pendiri Alibaba", "juara MotoGP terakhir",
]
NUM_WORDS_A = list(range(2, 50))
NUM_WORDS_B = list(range(2, 20))
CALC_OPS = [("kali", "*"), ("tambah", "+"), ("kurang", "-"), ("dibagi", "/"), ("pangkat", "**")]
COMPARISON_PAIRS = [
    ("iPhone", "Samsung"), ("Python", "Java"), ("Jakarta", "Surabaya"),
    ("Netflix", "Disney+"), ("kopi", "teh"), ("kucing", "anjing"),
    ("gym", "yoga"), ("nasi goreng", "mie goreng"), ("Honda", "Yamaha"),
    ("PS5", "Xbox Series X"), ("MacBook", "laptop Windows"), ("kereta", "pesawat"),
    ("kuliah online", "kuliah offline"), ("Instagram", "TikTok"),
    ("saham", "reksadana"), ("apartemen", "rumah"),
    ("Grab", "Gojek"), ("YouTube", "TikTok"), ("Toyota", "Honda"),
    ("Spotify", "Apple Music"), ("kerja kantoran", "freelance"),
    ("motor matic", "motor manual"), ("emas", "saham"), ("tabungan", "deposito"),
    ("kos", "kontrakan"), ("WFH", "WFO"), ("laptop", "PC desktop"),
    ("BCA", "Mandiri"), ("indomie goreng", "indomie kuah"), ("Bandung", "Yogyakarta"),
]
KNOWLEDGE_TOPICS = [
    "fotosintesis", "gravitasi", "blockchain", "machine learning", "inflasi",
    "demokrasi", "globalisasi", "revolusi industri", "teori evolusi",
    "efek rumah kaca", "energi terbarukan", "kecerdasan buatan",
    "sistem tata surya", "hukum newton", "teori relativitas", "DNA",
    "vaksin", "resesi ekonomi", "hak asasi manusia", "otonomi daerah",
    "cryptocurrency", "resesi global", "krisis iklim", "urbanisasi",
    "bioteknologi", "nanoteknologi", "energi nuklir", "reboisasi",
    "meritokrasi", "desentralisasi", "hiperinflasi", "stagflasi",
]
TRANSLATE_PHRASES = [
    "good morning", "thank you very much", "how are you", "nice to meet you",
    "see you later", "I love you", "happy birthday", "good luck",
    "excuse me", "welcome", "congratulations", "take care",
]
LANG_TARGETS = [("id", "bahasa Indonesia"), ("en", "bahasa Inggris"),
                 ("ja", "bahasa Jepang"), ("ko", "bahasa Korea"),
                 ("fr", "bahasa Perancis"), ("es", "bahasa Spanyol")]


# ===========================================================================
# 1. SEARCH (target 1500)
# ===========================================================================
SEARCH_VERBS = ["Cari tahu", "Cariin", "Cari", "Search", "Temuin info soal"]
def gen_search(target: int = 1500) -> list[dict]:
    result = []
    for text in pad_to_target(combos(SEARCH_VERBS, SEARCH_QUERIES), target):
        q = next(x for x in SEARCH_QUERIES if x.lower() in text.lower())
        result.append(make_sample(text, "search", sample_output("search", "search", "web", q)))
    return result[:target]


# ===========================================================================
# 2. WEATHER (target 1000)
# ===========================================================================
WEATHER_VERBS = ["Cek cuaca", "Gimana cuaca", "Cuaca", "Berapa suhu", "Cek suhu",
                  "Kasih tau cuaca", "Info cuaca", "Cek prakiraan cuaca"]
def gen_weather(target: int = 1000) -> list[dict]:
    result = []
    n_notime = int(target * 0.3)
    n_time = int(target * 0.4)
    n_city = target - n_notime - n_time

    for text in pad_to_target(combos(WEATHER_VERBS, ["", "hari ini"], STATEMENT_WRAPPERS), n_notime):
        result.append(make_sample(text, "weather", sample_output("weather", "query", "weather", "today")))

    time_texts = []
    for v, key in itertools.product(WEATHER_VERBS, WEATHER_TIMES):
        for phrase in WEATHER_TIME_PHRASES[key]:
            time_texts.append((f"{v} {phrase}?", key))
            time_texts.append((f"{v} {phrase} dong.", key))
    RNG.shuffle(time_texts)
    time_texts = pad_to_target(time_texts, n_time)
    for text, key in time_texts:
        result.append(make_sample(text, "weather", sample_output("weather", "query", "weather", key)))

    city_texts = []
    for v, c in itertools.product(WEATHER_VERBS, CITIES):
        city_texts.append((f"{v} di {c}.", c))
        city_texts.append((f"{v} di {c} dong.", c))
    RNG.shuffle(city_texts)
    city_texts = pad_to_target(city_texts, n_city)
    for text, c in city_texts:
        result.append(make_sample(text, "weather", sample_output("weather", "query", "weather", "today", location=c)))
    return result[:target]


# ===========================================================================
# 3. TIME (target 500)
# ===========================================================================
TIME_VERBS = ["Jam berapa", "Sekarang jam berapa", "Cek jam", "Jam berapa sekarang",
              "Kasih tau jam berapa", "Pukul berapa sekarang", "Cek waktu"]
def gen_time(target: int = 500) -> list[dict]:
    result = []
    n_now = int(target * 0.5)
    n_city = target - n_now

    now_texts = pad_to_target(combos(TIME_VERBS, [""], STATEMENT_WRAPPERS), n_now)
    for text in now_texts:
        result.append(make_sample(text, "time", sample_output("time", "query", "time", "now")))

    city_texts = []
    for v, c in itertools.product(TIME_VERBS, CITIES):
        city_texts.append((f"{v} di {c}?", c))
        city_texts.append((f"{v} sekarang di {c}?", c))
    RNG.shuffle(city_texts)
    city_texts = pad_to_target(city_texts, n_city)
    for text, c in city_texts:
        result.append(make_sample(text, "time", sample_output("time", "query", "time", "now", location=c)))
    return result[:target]


# ===========================================================================
# 4. TRANSLATION (target 1000)
# ===========================================================================
TRANSLATE_VERBS = ["Artiin", "Terjemahin", "Translate", "Apa artinya", "Tolong artiin",
                    "Bisa artiin", "Coba terjemahin"]
def gen_translation(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, phrase, (code, lang_name) in itertools.product(TRANSLATE_VERBS, TRANSLATE_PHRASES, LANG_TARGETS):
        texts.append((f"{v} '{phrase}' ke {lang_name}.", phrase, code))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, phrase, code in texts:
        result.append(make_sample(text, "translation",
                                   sample_output("translation", "translate", text=phrase, target_language=code)))
    return result[:target]


# ===========================================================================
# 5. LOOKUP (target 1000)
# ===========================================================================
LOOKUP_VERBS = ["Siapa", "Cari tahu siapa", "Cek siapa", "Tau nggak siapa", "Kasih tau siapa"]
def gen_lookup(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, e, suf in itertools.product(LOOKUP_VERBS, ENTITIES, ["?", " sih?", " dong?", " ya?"]):
        texts.append((f"{v} {e}{suf}", e))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, e in texts:
        result.append(make_sample(text, "lookup", sample_output("lookup", "lookup", "entity", e)))
    return result[:target]


# ===========================================================================
# 6. CALCULATION (target 500)
# ===========================================================================
def gen_calculation(target: int = 500) -> list[dict]:
    result = []
    combos_list = []
    for a in NUM_WORDS_A:
        for b in NUM_WORDS_B:
            for op_word, op_sym in CALC_OPS:
                combos_list.append((a, b, op_word, op_sym))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    verbs = ["Berapa hasil", "Hitung", "Berapa"]
    for a, b, op_word, op_sym in combos_list:
        v = RNG.choice(verbs)
        text = f"{v} {a} {op_word} {b}?"
        expr = f"{a} {op_sym} {b}"
        result.append(make_sample(text, "calculation",
                                   sample_output("calculation", "calculate", expression=expr)))
    return result[:target]


# ===========================================================================
# 7. COMPARISON (target 1000)
# ===========================================================================
COMPARE_VERBS = ["Lebih bagus mana,", "Mendingan mana,", "Bandingin", "Beda", "Mana yang lebih oke,",
                  "Menurutmu bagusan mana,", "Pilih mana,"]
COMPARE_SUFFIXES = ["", " ya", " dong", " sih", " menurutmu"]
def gen_comparison(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, (a, b), suf in itertools.product(COMPARE_VERBS, COMPARISON_PAIRS, COMPARE_SUFFIXES):
        if v.endswith(","):
            texts.append((f"{v} {a} atau {b}{suf}?", a, b))
        elif v == "Bandingin":
            texts.append((f"{v} {a} sama {b}{suf}.", a, b))
        else:  # Beda
            texts.append((f"{v} {a} sama {b} apa{suf}?", a, b))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, a, b in texts:
        result.append(make_sample(text, "comparison",
                                   sample_output("comparison", "compare", items=[a, b])))
    return result[:target]


# ===========================================================================
# 8. KNOWLEDGE (target 1000)
# ===========================================================================
KNOWLEDGE_VERBS = ["Apa itu", "Jelasin soal", "Apa yang dimaksud dengan", "Definisi",
                    "Coba jelasin", "Bisa jelasin", "Apa maksudnya"]
KNOWLEDGE_SUFFIXES = ["", " dong", " ya", " sih", " deh"]
def gen_knowledge(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, t, suf in itertools.product(KNOWLEDGE_VERBS, KNOWLEDGE_TOPICS, KNOWLEDGE_SUFFIXES):
        if v.startswith("Apa"):
            texts.append((f"{v} {t}{suf}?", t))
        else:
            texts.append((f"{v} {t}{suf}.", t))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, t in texts:
        result.append(make_sample(text, "knowledge", sample_output("information_query", "query", "knowledge", t)))
    return result[:target]


# ===========================================================================
# 9. AMBIGUOUS (target 250) -- output KONSTAN, tidak butuh stage2
# ===========================================================================
AMBIG_INFO_PHRASES = [
    "Cari itu deh.", "Cariin dong.", "Search yang tadi.", "Cari yang itu.",
    "Coba cariin.", "Cari yang kemarin.", "Cariin lagi deh.", "Search dong.",
    "Cari yang tadi aja.", "Coba search deh.", "Cariin yang itu dong.",
    "Cari lagi ya.", "Cariin dulu deh.", "Coba cari lagi.", "Search itu dong.",
    "Cari yang barusan.", "Cariin yang tadi disebut.", "Coba deh cariin.",
]
def gen_ambiguous(target: int = 250) -> list[dict]:
    result = []
    combos_list = list(itertools.product(AMBIG_INFO_PHRASES, ["", " dong", " ya", " deh", " sekarang", " aja"]))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for base, suf in combos_list:
        text = base.rstrip(".") + suf + "."
        result.append(make_sample(text, "ambiguous",
                                   sample_output("search", "search", "web", None),
                                   label="ambiguous", note="objek pencarian tidak jelas tanpa context"))
    return result[:target]


# ===========================================================================
# 10. NEGATIVE (target 250)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Ingetin aku soal {topic}.", "Simpen info kalau {topic}.", "Jadwalin {topic} dong.",
    "Kamu masih inget soal {topic}?", "Tambahin todo buat {topic}.",
    "Catet dong soal {topic}.", "Update jadwal soal {topic}.", "Hapus reminder soal {topic}.",
]
NEGATIVE_TOPICS = ["meeting jam 3", "beli galon", "alergi seafood aku", "ulang tahun adik",
                    "deadline project", "jadwal dokter", "reminder minum obat",
                    "preferensi makanan aku", "nomor plat motorku", "agenda weekend",
                    "checkup kesehatan", "renew paspor", "interview kerja",
                    "kumpul keluarga", "servis motor", "bayar tagihan listrik"]
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
                                   note="domain productivity/memory, bukan information"))
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
        "search": (gen_search, 1500),
        "weather": (gen_weather, 1000),
        "time": (gen_time, 500),
        "translation": (gen_translation, 1000),
        "lookup": (gen_lookup, 1000),
        "calculation": (gen_calculation, 500),
        "comparison": (gen_comparison, 1000),
        "knowledge": (gen_knowledge, 1000),
        "ambiguous": (gen_ambiguous, 250),
        "negative": (gen_negative, 250),
    }
    total = 0
    print(f"{'task_category':14s} {'target':>8s} {'actual':>8s}")
    print("-" * 32)
    for cat, (fn, tgt) in generators.items():
        samples = fn(tgt)
        save_jsonl(samples, OUT_DIR / f"{cat}.jsonl")
        print(f"{cat:14s} {tgt:8d} {len(samples):8d}")
        total += len(samples)
    print("-" * 32)
    print(f"{'TOTAL':14s} {8000:8d} {total:8d}")


if __name__ == "__main__":
    main()
