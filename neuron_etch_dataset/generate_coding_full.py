"""
generate_coding_full.py
=========================
Generator dataset FULL untuk specialist "coding" (§34), target 10,000
sample di 10 task_category:

    generate     : 2000
    debug        : 1500
    modify       : 1500
    review       : 1000
    explain      : 1000
    refactor     : 1000
    architecture : 1000
    test         :  500
    ambiguous    :  500
    negative     :  500
    ------------------
    TOTAL        : 10000

Beda penting dari system/media/persona: `target` SELALU null (kode
tidak punya "objek bernama" seperti app/device). Semua informasi ada
di `parameters` (language, requirements, error, file). Karena itu,
`action` untuk 8 dari 10 kategori DETERMINISTIK dari task_category
(lookup CATEGORY_TO_ACTION) -- model tidak perlu memprediksinya sama
sekali. Stage 2 murni tugas EKSTRAKSI parameter.
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260830)
OUT_DIR = Path(__file__).parent / "output" / "coding"

_counter = itertools.count(1)
def new_id() -> str:
    return f"coding_{next(_counter):06d}"


CATEGORY_TO_ACTION = {
    "generate": "generate", "debug": "debug", "modify": "modify",
    "review": "review", "explain": "explain", "refactor": "refactor",
    "architecture": "design", "test": "test",
}
CATEGORY_TO_INTENT = {
    "generate": "code_generation", "debug": "code_debugging", "modify": "code_modification",
    "review": "code_analysis", "explain": "code_explanation", "refactor": "refactoring",
    "architecture": "architecture", "test": "testing", "ambiguous": "code_modification",
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


def sample_output(intent, action, **params):
    return {"domain": "coding", "intent": intent, "action": action, "target": None, "parameters": params}


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
LANGUAGES = ["python", "javascript", "bash", "java", "go", "rust", "c++",
             "php", "typescript", "sql", "ruby", "swift", "kotlin", "c#", "r"]
TASKS = [
    "cari file duplikat", "validasi email", "sorting data", "koneksi ke database",
    "kirim email otomatis", "scraping website", "generate laporan PDF",
    "enkripsi password", "kompresi gambar", "parsing JSON", "REST API endpoint",
    "autentikasi user", "upload file ke S3", "generate QR code",
    "konversi CSV ke Excel", "caching dengan Redis", "rate limiting API",
    "pagination data", "search dengan Elasticsearch", "background job dengan Celery",
    "web scraper", "chatbot sederhana", "kalkulator", "todo list app",
    "sistem login", "payment gateway integration", "notifikasi push",
    "image resizing", "video compression", "text-to-speech",
]
ERRORS = ["NullPointerException", "IndexError", "TypeError", "SyntaxError",
          "infinite loop", "memory leak", "race condition", "deadlock",
          "segfault", "timeout error", "connection refused", "404 error",
          "500 error", "stack overflow", "undefined variable",
          "division by zero", "key error", "import error", "permission denied",
          "out of memory"]
FILES = ["main.py", "app.js", "server.py", "index.html", "utils.py",
         "models.py", "views.py", "config.yaml", "database.sql", "api.py",
         "auth.js", "handler.go", "service.rs", "controller.php", "App.tsx"]
MODIFY_REQS = [
    "tambahin error handling", "ubah biar support async", "tambahin logging",
    "ganti jadi pakai class", "tambahin validasi input", "ubah ke pakai type hints",
    "tambahin dokumentasi", "ganti library-nya ke yang lebih baru",
    "tambahin retry logic", "ubah biar lebih cepat", "tambahin caching",
    "ganti struktur data-nya", "tambahin unit test coverage",
    "ubah biar thread-safe", "tambahin rate limiting", "tambahin pagination",
    "ubah biar bisa handle concurrent request", "tambahin fitur export CSV",
    "ganti ke pakai environment variable", "tambahin fitur search",
]
ARCH_REQS = [
    "sistem antrian buat aplikasi chat", "microservices buat e-commerce",
    "database schema buat social media", "caching layer buat API",
    "sistem notifikasi real-time", "arsitektur buat load balancing",
    "sistem autentikasi terdistribusi", "pipeline data buat analytics",
    "arsitektur event-driven", "sistem file storage terdistribusi",
    "arsitektur buat rekomendasi produk", "sistem logging terpusat",
    "arsitektur buat search engine", "sistem payment processing",
    "arsitektur multi-tenant SaaS", "sistem CDN buat static assets",
    "arsitektur buat real-time collaboration", "sistem job scheduler",
    "arsitektur buat data warehouse", "sistem rate limiter terdistribusi",
]


# ===========================================================================
# 1. GENERATE (target 2000)
# ===========================================================================
GENERATE_VERBS = ["Buat", "Bikin", "Bikinin", "Generate", "Tulisin"]
def gen_generate(target: int = 2000) -> list[dict]:
    result = []
    texts = []
    for v, lang, task in itertools.product(GENERATE_VERBS, LANGUAGES, TASKS):
        texts.append((f"{v} script {lang} buat {task}.", lang, task))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, lang, task in texts:
        result.append(make_sample(text, "generate",
                                   sample_output("code_generation", "generate", language=lang, requirements=task)))
    return result[:target]


# ===========================================================================
# 2. DEBUG (target 1500)
# ===========================================================================
DEBUG_VERBS = ["Debug", "Perbaiki bug", "Cek kenapa error", "Benerin error", "Fix"]
def gen_debug(target: int = 1500) -> list[dict]:
    result = []
    n_with_file = int(target * 0.5)
    n_without_file = target - n_with_file

    with_file_texts = []
    for v, err, f in itertools.product(DEBUG_VERBS, ERRORS, FILES):
        with_file_texts.append((f"{v} {err} di file {f}.", err, f))
    RNG.shuffle(with_file_texts)
    with_file_texts = pad_to_target(with_file_texts, n_with_file)
    for text, err, f in with_file_texts:
        result.append(make_sample(text, "debug",
                                   sample_output("code_debugging", "debug", error=err, file=f)))

    without_file_texts = []
    for v, err in itertools.product(DEBUG_VERBS, ERRORS):
        without_file_texts.append((f"{v} {err} ini.", err))
    RNG.shuffle(without_file_texts)
    without_file_texts = pad_to_target(without_file_texts, n_without_file)
    for text, err in without_file_texts:
        result.append(make_sample(text, "debug",
                                   sample_output("code_debugging", "debug", error=err)))
    return result[:target]


# ===========================================================================
# 3. MODIFY (target 1500)
# ===========================================================================
def gen_modify(target: int = 1500) -> list[dict]:
    result = []
    texts = pad_to_target(combos(["", "Tolong", "Coba", "Bisa nggak"],
                                   [f"{req} di fungsi ini" for req in MODIFY_REQS], WRAPPERS), target)
    for text in texts:
        req = next(r for r in MODIFY_REQS if r.lower() in text.lower())
        result.append(make_sample(text.strip(), "modify",
                                   sample_output("code_modification", "modify", requirements=req)))
    return result[:target]


# ===========================================================================
# 4. REVIEW (target 1000)
# ===========================================================================
REVIEW_VERBS = ["Review", "Cek kualitas", "Tolong review", "Coba review", "Kasih feedback buat",
                "Analisa", "Kasih masukan buat", "Cek dong"]
REVIEW_OBJS = ["kode ini", "function ini", "script ini", "PR ini", "kode yang aku tulis",
               "implementasi ini", "class ini", "module ini", "commit ini", "branch ini"]
def gen_review(target: int = 1000) -> list[dict]:
    result = []
    texts = pad_to_target(combos(REVIEW_VERBS, REVIEW_OBJS, STATEMENT_WRAPPERS), target)
    for text in texts:
        result.append(make_sample(text, "review", sample_output("code_analysis", "review")))
    return result[:target]


# ===========================================================================
# 5. EXPLAIN (target 1000)
# ===========================================================================
def gen_explain(target: int = 1000) -> list[dict]:
    result = []
    n_lang = int(target * 0.6)
    n_generic = target - n_lang

    lang_texts = []
    for lang, task in itertools.product(LANGUAGES, TASKS):
        lang_texts.append((f"Jelasin cara kerja {task} pakai {lang}.", lang, task))
    RNG.shuffle(lang_texts)
    lang_texts = pad_to_target(lang_texts, n_lang)
    for text, lang, task in lang_texts:
        result.append(make_sample(text, "explain",
                                   sample_output("code_explanation", "explain", language=lang, requirements=task)))

    generic_texts = pad_to_target(combos(
        ["Jelasin", "Tolong jelasin", "Bisa jelasin"],
        ["kode ini", "function ini ngapain", "logika ini", "script ini",
         "kenapa ini bisa jalan", "gimana cara kerja ini"], STATEMENT_WRAPPERS), n_generic)
    for text in generic_texts:
        result.append(make_sample(text, "explain", sample_output("code_explanation", "explain")))
    return result[:target]


# ===========================================================================
# 6. REFACTOR (target 1000)
# ===========================================================================
REFACTOR_VERBS = ["Refactor", "Rapiin", "Bersihin", "Optimize", "Simplify", "Poles",
                   "Cleanup", "Restrukturisasi"]
REFACTOR_OBJS = ["function ini", "kode ini", "class ini", "module ini", "script ini",
                  "component ini", "file ini"]
REFACTOR_GOALS = ["biar lebih rapi", "biar lebih readable", "biar lebih efisien",
                    "biar performanya lebih bagus", "biar lebih maintainable", "",
                    "biar lebih pendek", "biar sesuai best practice", "biar lebih modular"]
def gen_refactor(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, o, g, suf in itertools.product(REFACTOR_VERBS, REFACTOR_OBJS, REFACTOR_GOALS,
                                            ["", " dong", " ya"]):
        text = f"{v} {o} {g}{suf}".strip()
        text = " ".join(text.split()) + "."
        texts.append((text, g))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, g in texts:
        params = {"requirements": g} if g else {}
        result.append(make_sample(text, "refactor", sample_output("refactoring", "refactor", **params)))
    return result[:target]


# ===========================================================================
# 7. ARCHITECTURE (target 1000)
# ===========================================================================
ARCH_VERBS = ["Gimana cara desain", "Bantuin desain", "Rancang", "Desain arsitektur buat",
              "Gimana approach yang bagus buat bikin", "Kasih saran desain buat",
              "Gimana strukturnya kalau mau bikin", "Tolong rancangin", "Coba desainin"]
def gen_architecture(target: int = 1000) -> list[dict]:
    result = []
    texts = []
    for v, req, suf in itertools.product(ARCH_VERBS, ARCH_REQS, ["?", " dong.", " ya.", " dong ya."]):
        texts.append((f"{v} {req}{suf}", req))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, req in texts:
        result.append(make_sample(text, "architecture",
                                   sample_output("architecture", "design", requirements=req)))
    return result[:target]


# ===========================================================================
# 8. TEST (target 500)
# ===========================================================================
TEST_VERBS = ["Bikinin unit test buat", "Tulisin test case buat", "Generate test buat",
              "Bikin testing buat", "Coba bikin test buat"]
TEST_OBJS = ["fungsi ini", "function login", "endpoint ini", "class ini", "module ini",
             "API ini", "komponen ini"]
def gen_test(target: int = 500) -> list[dict]:
    result = []
    texts = pad_to_target(combos(TEST_VERBS, TEST_OBJS, STATEMENT_WRAPPERS), target)
    for text in texts:
        result.append(make_sample(text, "test", sample_output("testing", "test")))
    return result[:target]


# ===========================================================================
# 9. AMBIGUOUS (target 500)
# ===========================================================================
AMBIG_CODING_PHRASES = [
    "Perbaiki kodenya.", "Benerin dong.", "Fix ini.", "Perbaiki fungsinya.",
    "Ini kenapa ya.", "Betulin deh.", "Perbaiki errornya.", "Fix bug-nya.",
    "Betulin kodenya dong.", "Perbaiki yang ini.", "Ini salah dimana ya.",
    "Coba dicek deh.", "Ini kok gini.", "Bantuin dong.",
    "Kok error terus ya.", "Ini gimana dong.", "Coba benerin.", "Aduh salah lagi.",
    "Ini bisa dibenerin nggak.", "Perbaiki dong ini.", "Ada yang salah nih.",
    "Ini gagal terus kenapa ya.", "Bantuin cek dong.", "Ini nggak jalan.",
    "Coba liat deh ini.", "Ini rusak kenapa ya.", "Perbaikin nih.",
    "Bisa dibetulin nggak.", "Ini eror mulu.", "Kok gini terus sih.",
    "Bantuin dong benerin.", "Tolong betulin ini.", "Ini masalahnya apa ya.",
]
AMBIG_SUFFIXES2 = ["", " dong", " ya", " deh", " sekarang", " aja", " nih", " beneran", " dong ya"]
def gen_ambiguous(target: int = 500) -> list[dict]:
    result = []
    combos_list = list(itertools.product(AMBIG_CODING_PHRASES, AMBIG_SUFFIXES2))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for base, suf in combos_list:
        text = base.rstrip(".") + suf + "."
        result.append(make_sample(text, "ambiguous",
                                   sample_output("code_modification", "modify"),
                                   label="ambiguous", note="file/fungsi mana yang dimaksud tidak jelas"))
    return result[:target]


# ===========================================================================
# 10. NEGATIVE (target 500)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Ingetin aku {task}.", "Jadwalin {task} dong.", "Cariin info soal {task}.",
    "Putar musik buat {task}.", "Kirim pesan soal {task} ke temen.",
    "Tolong catet {task}.", "Bikin reminder buat {task}.", "Update jadwal soal {task}.",
    "Cek jadwal {task} dong.", "Tambahin ke kalender soal {task}.",
    "Beritahu aku soal {task}.", "Cariin waktu buat {task}.",
]
NEGATIVE_TASKS = ["meeting jam 3", "deadline project", "beli galon", "rapat tim",
                   "jadwal dokter", "acara weekend", "reminder minum obat",
                   "laporan bulanan", "presentasi ke atasan", "agenda besok",
                   "checkup kesehatan", "renew paspor", "interview kerja",
                   "kumpul keluarga", "servis motor", "bayar tagihan",
                   "belanja bulanan", "olahraga pagi", "les bahasa Inggris",
                   "acara ulang tahun", "jemput anak sekolah", "arisan keluarga"]
def gen_negative(target: int = 500) -> list[dict]:
    result = []
    texts = []
    for t, task in itertools.product(NEGATIVE_TEMPLATES, NEGATIVE_TASKS):
        texts.append(t.format(task=task))
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        result.append(make_sample(text, "negative", None, label="negative",
                                   note="domain productivity/information/media, bukan coding"))
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
        "generate": (gen_generate, 2000),
        "debug": (gen_debug, 1500),
        "modify": (gen_modify, 1500),
        "review": (gen_review, 1000),
        "explain": (gen_explain, 1000),
        "refactor": (gen_refactor, 1000),
        "architecture": (gen_architecture, 1000),
        "test": (gen_test, 500),
        "ambiguous": (gen_ambiguous, 500),
        "negative": (gen_negative, 500),
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
    print(f"{'TOTAL':16s} {10000:8d} {total:8d}")


if __name__ == "__main__":
    main()
