"""
generate_persona_full.py
==========================
Generator dataset FULL untuk specialist "persona" (§33), target 12,000
sample di 9 task_category:

    character_call     : 1500
    character_switch   : 1500
    conversation       : 2500
    topic_extraction   : 2000
    roleplay_intent    : 1500
    dialogue_control   : 1000
    context_dependency : 1000
    ambiguous          :  500
    negative           :  500
    ------------------------
    TOTAL              : 12000
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260829)
OUT_DIR = Path(__file__).parent / "output" / "persona"

_counter = itertools.count(1)
def new_id() -> str:
    return f"persona_{next(_counter):06d}"


ACTION_TO_INTENT = {
    "call": "character_call", "switch": "character_switch",
    "talk": "conversation", "ask": "conversation",
    "roleplay": "roleplay",
    "continue": "dialogue_control", "resume_topic": "dialogue_control",
    "explain": "conversation",
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
    return {"domain": "persona", "intent": intent, "action": action,
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
PERSONAS = ["Mailin", "Aria", "Kaito", "Yuki", "Rei", "Nova", "Sena", "Kiro",
            "Zara", "Dara", "Airin", "Leo", "Cyra", "Bima"]
TOPICS = [
    "rencana liburan", "film terbaru", "masalah kerjaan", "resep masakan baru",
    "buku yang lagi dibaca", "berita politik", "rencana weekend", "hobi baru",
    "film horor", "series Netflix terbaru", "tren fashion", "startup idea",
    "investasi saham", "rencana nikah", "hubungan sama pacar", "keluarga",
    "kuliah", "skripsi", "kerjaan kantor", "kesehatan mental", "diet sehat",
    "olahraga rutin", "musik favorit", "game terbaru", "teknologi AI",
    "isu lingkungan", "traveling", "kucing peliharaan", "masa depan karir",
    "mimpi dan cita-cita", "persahabatan", "pertemanan lama", "resolusi tahun baru",
]
ROLES = [
    "asisten pribadi", "teman curhat", "guru bahasa Inggris", "motivator",
    "teman diskusi", "konsultan karir", "teman ngobrol santai",
    "pendamping belajar", "partner brainstorming", "teman cerita",
    "life coach", "teman debat", "editor tulisan", "teman roleplay cerita",
    "penasihat keuangan", "teman gaming", "guru matematika", "teman belajar bahasa Jepang",
    "asisten produktivitas", "teman sparring ide bisnis",
]


# ===========================================================================
# 1. CHARACTER_CALL (target 1500) & 2. CHARACTER_SWITCH (target 1500)
# ===========================================================================
CALL_VERBS = ["Panggil", "Panggilin", "Manggil", "Hubungin", "Call", "Panggilkan",
              "Sini panggil", "Coba panggil"]
SWITCH_VERBS = ["Ganti karakter ke", "Pindah ke", "Switch ke", "Ganti ke", "Beralih ke",
                "Ubah karakter jadi", "Ganti persona ke", "Tukar ke"]


def gen_character_call(target: int = 1500) -> list[dict]:
    result = []
    for text in pad_to_target(combos(CALL_VERBS, PERSONAS), target):
        p = next(x for x in PERSONAS if x.lower() in text.lower())
        result.append(make_sample(text, "character_call", sample_output("character_call", "call", "persona", p)))
    return result[:target]


def gen_character_switch(target: int = 1500) -> list[dict]:
    result = []
    for text in pad_to_target(combos(SWITCH_VERBS, PERSONAS), target):
        p = next(x for x in PERSONAS if x.lower() in text.lower())
        result.append(make_sample(text, "character_switch",
                                   sample_output("character_switch", "switch", "persona", p)))
    return result[:target]


# ===========================================================================
# 3. CONVERSATION (target 2500) & 4. TOPIC_EXTRACTION (target 2000)
# (Struktur output SAMA: action=talk/ask, target=persona, parameters={topic})
# ===========================================================================
TALK_VERBS = ["Suruh {p} bahas", "Ngobrol sama {p} soal", "Cerita ke {p} tentang",
              "Ajak {p} ngobrolin", "Diskusiin sama {p} soal"]
ASK_VERBS = ["Tanya {p} soal", "Tanyain {p} tentang", "Tanya ke {p} soal"]


def _gen_talk_like(target: int, category: str) -> list[dict]:
    result = []
    n_talk = target // 2
    n_ask = target - n_talk

    talk_texts = []
    for v, p, t in itertools.product(TALK_VERBS, PERSONAS, TOPICS):
        talk_texts.append((v.format(p=p) + f" {t}.", p, t))
    RNG.shuffle(talk_texts)
    talk_texts = pad_to_target(talk_texts, n_talk)
    for text, p, t in talk_texts:
        result.append(make_sample(text, category,
                                   sample_output("conversation", "talk", "persona", p, topic=t)))

    ask_texts = []
    for v, p, t in itertools.product(ASK_VERBS, PERSONAS, TOPICS):
        ask_texts.append((v.format(p=p) + f" {t}.", p, t))
    RNG.shuffle(ask_texts)
    ask_texts = pad_to_target(ask_texts, n_ask)
    for text, p, t in ask_texts:
        result.append(make_sample(text, category,
                                   sample_output("conversation", "ask", "persona", p, topic=t)))
    return result[:target]


def gen_conversation(target: int = 2500) -> list[dict]:
    return _gen_talk_like(target, "conversation")


def gen_topic_extraction(target: int = 2000) -> list[dict]:
    # Variasi kalimat lebih panjang/implicit (persona disebut lebih dulu + alasan)
    result = []
    n_each = target // 3
    base = _gen_talk_like(n_each * 2, "topic_extraction")
    result.extend(base)
    remaining = target - len(base)
    extra_texts = []
    for p, t in itertools.product(PERSONAS, TOPICS):
        extra_texts.append((f"Panggil {p} buat bahas {t}.", p, t))
        extra_texts.append((f"Minta {p} cerita soal {t} dong.", p, t))
    RNG.shuffle(extra_texts)
    extra_texts = pad_to_target(extra_texts, remaining)
    for text, p, t in extra_texts:
        result.append(make_sample(text, "topic_extraction",
                                   sample_output("conversation", "talk", "persona", p, topic=t)))
    return result[:target]


# ===========================================================================
# 5. ROLEPLAY_INTENT (target 1500)
# ===========================================================================
def gen_roleplay_intent(target: int = 1500) -> list[dict]:
    result = []
    texts = []
    for p, r in itertools.product(PERSONAS, ROLES):
        texts.append((f"{p}, jadi {r} ya.", p, r))
        texts.append((f"Jadiin {p} sebagai {r}.", p, r))
        texts.append((f"{p}, mulai sekarang kamu {r} aku ya.", p, r))
        texts.append((f"Coba deh {p} jadi {r}.", p, r))
        texts.append((f"{p}, peranin {r} dong.", p, r))
        texts.append((f"Aku mau {p} jadi {r} aku.", p, r))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, p, r in texts:
        result.append(make_sample(text, "roleplay_intent",
                                   sample_output("roleplay", "roleplay", "persona", p, role=r)))
    return result[:target]


# ===========================================================================
# 6. DIALOGUE_CONTROL (target 1000)
# ===========================================================================
CONTINUE_PHRASES = ["Lanjutin obrolan tadi.", "Terusin ngobrolnya.", "Lanjut ngobrol yang tadi.",
                     "Terusin obrolan kita.", "Lanjutin chat tadi.", "Lanjut ngobrol dong.",
                     "Terusin percakapan tadi.", "Lanjut ngomong yang tadi.", "Sambung obrolan kita.",
                     "Terusin diskusi tadi.", "Lanjutin cerita tadi.", "Terus ngobrolnya."]
RESUME_TOPIC_PHRASES = ["Balik ke topik sebelumnya.", "Balik ke pembahasan tadi.",
                          "Kembali ke topik yang tadi.", "Balik bahas yang tadi lagi.",
                          "Ke topik sebelumnya aja.", "Balik lagi ke yang tadi.",
                          "Kembaliin ke pembahasan awal.", "Balik ke obrolan sebelumnya.",
                          "Ke pembahasan yang tadi aja.", "Balik ke topik awal.",
                          "Kembali bahas yang sebelumnya.", "Balikin ke topik tadi."]
DIALOGUE_SUFFIXES = ["", " dong", " ya", " deh", " sekarang", " aja", " dulu", " ya deh"]


DIALOGUE_PREFIXES = ["", "Eh, ", "Btw, ", "Hmm, ", "Oke, "]
def gen_dialogue_control(target: int = 1000) -> list[dict]:
    result = []
    n_each = target // 2
    cont_texts = pad_to_target(list(itertools.product(DIALOGUE_PREFIXES, CONTINUE_PHRASES, DIALOGUE_SUFFIXES)), n_each)
    for prefix, base, suf in cont_texts:
        text = prefix + base.rstrip(".") + suf + "."
        result.append(make_sample(text, "dialogue_control",
                                   sample_output("dialogue_control", "continue", "conversation", "current")))
    remaining = target - n_each
    resume_texts = pad_to_target(list(itertools.product(DIALOGUE_PREFIXES, RESUME_TOPIC_PHRASES, DIALOGUE_SUFFIXES)), remaining)
    for prefix, base, suf in resume_texts:
        text = prefix + base.rstrip(".") + suf + "."
        result.append(make_sample(text, "dialogue_control",
                                   sample_output("dialogue_control", "resume_topic", "topic", "previous")))
    return result[:target]


# ===========================================================================
# 7. CONTEXT_DEPENDENCY (target 1000) & 8. AMBIGUOUS (target 500)
# (Struktur output SAMA: target=None, label=ambiguous)
# ===========================================================================
VAGUE_REFERENTS = ["dia", "dia lagi", "yang tadi", "orangnya", "karakternya", "dia deh", "yang itu"]
VAGUE_VERBS = ["Tanya", "Suruh", "Bilangin ke", "Cerita ke", "Ajak ngobrol", "Minta",
               "Bilang ke", "Ajak diskusi", "Sampein ke"]
VAGUE_ACTIONS_TEXT = ["soal itu", "jelasin", "soal itu lagi", "buat cerita", "ngobrol santai",
                        "soal masalah tadi", "buat jelasin lagi", "soal kejadian tadi",
                        "buat cerita lagi", "soal itu deh"]


def _gen_vague(target: int, category: str, note: str) -> list[dict]:
    result = []
    texts = []
    for v, ref, act in itertools.product(VAGUE_VERBS, VAGUE_REFERENTS, VAGUE_ACTIONS_TEXT):
        for suf in ["", " dong", " ya"]:
            texts.append(f"{v} {ref} {act}{suf}.")
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        action = "explain" if "jelasin" in text else "ask" if "tanya" in text.lower() else "talk"
        result.append(make_sample(text, category,
                                   sample_output("conversation", action, "persona", None),
                                   label="ambiguous", note=note))
    return result[:target]


def gen_context_dependency(target: int = 1000) -> list[dict]:
    return _gen_vague(target, "context_dependency", "butuh router_context: siapa 'dia', apa 'itu'")


def gen_ambiguous(target: int = 500) -> list[dict]:
    return _gen_vague(target, "ambiguous", "referent persona tidak jelas tanpa context")


# ===========================================================================
# 9. NEGATIVE (target 500)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Jelasin {topic} dengan bahasa Python.", "Apa itu {topic}?",
    "Gimana cara kerja {topic}?", "Definisi {topic} apa?",
    "Kapan {topic} pertama kali ditemukan?", "Siapa penemu {topic}?",
    "Coba jelasin {topic} dong.", "Kenapa {topic} itu penting?",
    "Contoh penerapan {topic} apa?", "Bedanya {topic} sama yang lain apa?",
]
NEGATIVE_TOPICS = ["rekursi", "machine learning", "fotosintesis", "gravitasi",
                    "algoritma sorting", "internet", "AI", "blockchain",
                    "quantum computing", "teori relativitas", "big bang", "DNA",
                    "enkripsi", "neural network", "cloud computing", "IoT",
                    "termodinamika", "evolusi", "genetika", "astrofisika"]


def gen_negative(target: int = 500) -> list[dict]:
    result = []
    texts = []
    for t, topic in itertools.product(NEGATIVE_TEMPLATES, NEGATIVE_TOPICS):
        texts.append(t.format(topic=topic))
        texts.append("Eh, " + t.format(topic=topic)[0].lower() + t.format(topic=topic)[1:])
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        result.append(make_sample(text, "negative", None, label="negative",
                                   note="technical/factual explanation -> domain coding/information, bukan persona"))
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
        "character_call": (gen_character_call, 1500),
        "character_switch": (gen_character_switch, 1500),
        "conversation": (gen_conversation, 2500),
        "topic_extraction": (gen_topic_extraction, 2000),
        "roleplay_intent": (gen_roleplay_intent, 1500),
        "dialogue_control": (gen_dialogue_control, 1000),
        "context_dependency": (gen_context_dependency, 1000),
        "ambiguous": (gen_ambiguous, 500),
        "negative": (gen_negative, 500),
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
    print(f"{'TOTAL':20s} {12000:8d} {total:8d}")


if __name__ == "__main__":
    main()
