"""
generate_media_full.py
========================
Generator dataset FULL untuk specialist "media" (§32), target 10,000
sample di 8 task_category:

    playback       : 3000
    search         : 1500
    queue          : 1500
    player_control : 1500
    streaming      : 1000
    metadata       :  500
    ambiguous      :  500
    negative       :  500
    ----------------------
    TOTAL          : 10000

Sama seperti generate_system_full.py: vocab generous dari awal (>200
kombinasi unik per action) supaya tidak perlu banyak iterasi.
"""
from __future__ import annotations
import itertools
import json
import random
from pathlib import Path

RNG = random.Random(20260828)
OUT_DIR = Path(__file__).parent / "output" / "media"

_counter = itertools.count(1)
def new_id() -> str:
    return f"media_{next(_counter):06d}"


ACTION_TO_INTENT = {
    "play": "playback", "resume": "playback",
    "search": "search_media",
    "queue_add": "queue_management", "queue_remove": "queue_management",
    "queue_clear": "queue_management", "queue_list": "queue_management",
    "pause": "media_control", "next": "media_control", "previous": "media_control",
    "repeat": "media_control", "repeat_off": "media_control",
    "shuffle": "media_control", "shuffle_off": "media_control", "stop": "media_control",
    "play_stream": "streaming", "resolve": "streaming",
    "query_metadata": "media_information",
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
    return {"domain": "media", "intent": intent, "action": action,
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
SONGS = [
    "Noah", "Sheila On 7", "Tulus", "Raisa", "Rich Brian", "Dewa 19",
    "Peterpan", "Coldplay", "Ed Sheeran", "Adele", "NIKI", "Isyana Sarasvati",
    "Hindia", "Fiersa Besari", "Payung Teduh", "Kunto Aji", "Yura Yunita",
    "Barasuara", "Efek Rumah Kaca", "Mocca", "Glenn Fredly", "Maliq & D'Essentials",
    "Kahitna", "Ari Lasso", "Afgan", "Rossa", "Andmesh", "Lyodra", "Mahalini",
    "Ariana Grande", "Taylor Swift", "The Weeknd", "Dua Lipa", "Billie Eilish",
]
ALBUMS = [
    "Konspirasi Alam Semesta", "Monokrom", "Diorama", "Album Biru", "Bumi",
    "Menari Dengan Bayangan", "Anti-Klimaks", "Vol. 1", "Kala", "Rasa Baru",
]
PLAYLISTS = [
    "playlist favorit", "playlist workout", "playlist galau", "playlist santai",
    "playlist road trip", "playlist study", "playlist chill", "playlist party",
    "playlist morning vibes", "playlist late night", "top hits 2026", "lagu kenangan",
]
GENRES = ["jazz", "EDM", "lo-fi", "klasik", "dangdut", "K-pop", "rock",
          "pop indonesia", "reggae", "akustik", "hip hop", "R&B"]
VIDEO_TOPICS = ["video tutorial Python", "video review HP", "vlog liburan",
                 "video game walkthrough", "video musik terbaru", "podcast Deddy Corbuzier",
                 "video motivasi", "video resep masakan", "highlight pertandingan bola",
                 "video komedi"]
STREAM_PLATFORMS = ["youtube", "twitch", "netflix", "disney+"]
METADATA_FIELDS = ["judul", "nama artis", "nama album", "lirik", "durasi", "genre", "tahun rilis"]

PLAYERS = ["spotify", "youtube music", "soundcloud", "joox", "resso", "apple music", "deezer"]


# ===========================================================================
# 1. PLAYBACK (target 3000)
# ===========================================================================
PLAY_VERBS = ["Putar", "Mainkan", "Play", "Puterin", "Setel"]
RESUME_VERBS = ["Lanjutin", "Resume", "Terusin", "Sambung"]


def gen_playback(target: int = 3000) -> list[dict]:
    result = []
    n_song = int(target * 0.45)
    n_album = int(target * 0.15)
    n_playlist = int(target * 0.15)
    n_genre = int(target * 0.1)
    n_resume = target - n_song - n_album - n_playlist - n_genre

    for text in pad_to_target(combos(PLAY_VERBS, SONGS), n_song):
        s = next(x for x in SONGS if x.lower() in text.lower())
        result.append(make_sample(text, "playback", sample_output("playback", "play", "artist", s)))
    for text in pad_to_target(combos(PLAY_VERBS, ALBUMS), n_album):
        a = next(x for x in ALBUMS if x.lower() in text.lower())
        result.append(make_sample(text, "playback", sample_output("playback", "play", "album", a)))
    for text in pad_to_target(combos(PLAY_VERBS, PLAYLISTS), n_playlist):
        p = next(x for x in PLAYLISTS if x.lower() in text.lower())
        result.append(make_sample(text, "playback", sample_output("playback", "play", "playlist", p)))
    for text in pad_to_target(combos(PLAY_VERBS, [f"musik {g}" for g in GENRES]), n_genre):
        g = next(x for x in GENRES if x.lower() in text.lower())
        result.append(make_sample(text, "playback", sample_output("playback", "play", "genre", g)))
    resume_texts = pad_to_target(combos(RESUME_VERBS, ["lagunya", "musiknya", "yang tadi", "playlist-nya"],
                                          STATEMENT_WRAPPERS), n_resume)
    for text in resume_texts:
        result.append(make_sample(text, "playback", sample_output("playback", "resume", "player", "current")))

    # tambahan: play + implicit app launch (§20)
    n_with_player = min(300, n_song // 3)
    withplayer_texts = []
    for v, s, p in itertools.product(PLAY_VERBS[:3], SONGS[:15], PLAYERS[:4]):
        withplayer_texts.append((f"{v} {s} di {p}.", s, p))
    RNG.shuffle(withplayer_texts)
    for text, s, p in withplayer_texts[:n_with_player]:
        samp = make_sample(text, "playback",
                            sample_output("playback", "play", "artist", s, player=p),
                            note="implicit app launch, §20 -- context.implicit_launch_allowed")
        result.append(samp)
    return result[:target]


# ===========================================================================
# 2. SEARCH (target 1500)
# ===========================================================================
SEARCH_VERBS = ["Cari", "Cariin", "Search", "Temuin"]
def gen_search(target: int = 1500) -> list[dict]:
    result = []
    n_song = int(target * 0.4)
    n_playlist = int(target * 0.25)
    n_artist = int(target * 0.2)
    n_genre = target - n_song - n_playlist - n_artist

    song_texts = []
    for v, s, p in itertools.product(SEARCH_VERBS, SONGS, PLAYERS):
        song_texts.append((f"{v} lagu {s} di {p}.", s))
    RNG.shuffle(song_texts)
    song_texts = pad_to_target(song_texts, n_song)
    for text, s in song_texts:
        result.append(make_sample(text, "search", sample_output("search_media", "search", "song", s)))

    for text in pad_to_target(combos(SEARCH_VERBS, PLAYLISTS), n_playlist):
        p = next(x for x in PLAYLISTS if x.lower() in text.lower())
        result.append(make_sample(text, "search", sample_output("search_media", "search", "playlist", p)))
    for text in pad_to_target(combos(SEARCH_VERBS, [f"lagu {s}" for s in SONGS]), n_artist):
        s = next(x for x in SONGS if x.lower() in text.lower())
        result.append(make_sample(text, "search", sample_output("search_media", "search", "artist", s)))
    for text in pad_to_target(combos(SEARCH_VERBS, [f"musik {g}" for g in GENRES]), n_genre):
        g = next(x for x in GENRES if x.lower() in text.lower())
        result.append(make_sample(text, "search", sample_output("search_media", "search", "genre", g)))
    return result[:target]


# ===========================================================================
# 3. QUEUE (target 1500)
# ===========================================================================
def gen_queue(target: int = 1500) -> list[dict]:
    result = []
    n_each = target // 4
    add_objs = ["lagu ini", "lagu Noah ke antrian", "playlist ini ke queue",
                "musik ini ke antrian", "lagu Tulus ke queue", "album ini ke antrian",
                "lagu berikutnya ke queue", "beberapa lagu ke antrian"]
    add_texts = pad_to_target(combos(["Tambahin", "Masukin", "Add", "Antriin", "Taruh"], add_objs), n_each)
    for text in add_texts:
        result.append(make_sample(text, "queue", sample_output("queue_management", "queue_add", "song", "current")))
    remove_objs = ["dari antrian", "lagu ini dari queue", "yang ini dari antrian",
                    "lagu itu dari queue", "yang barusan ditambahin", "lagu terakhir dari antrian"]
    remove_texts = pad_to_target(combos(["Hapus", "Buang", "Remove", "Keluarin", "Hilangin"],
                                          remove_objs, STATEMENT_WRAPPERS), n_each)
    for text in remove_texts:
        result.append(make_sample(text, "queue", sample_output("queue_management", "queue_remove", "song", "current")))
    clear_objs = ["antrian lagu", "queue-nya", "antrian musik", "semua antrian", "seluruh queue"]
    clear_texts = pad_to_target(combos(["Kosongin", "Bersihin", "Clear", "Hapus semua di", "Reset"],
                                         clear_objs, STATEMENT_WRAPPERS), n_each)
    for text in clear_texts:
        result.append(make_sample(text, "queue", sample_output("queue_management", "queue_clear", "queue", "current")))
    remaining = target - 3 * n_each
    list_objs = ["antrian lagu", "queue-nya", "antrian musik dong", "isi antrian", "daftar antrian"]
    list_texts = pad_to_target(combos(["Liat", "Tampilin", "Cek", "Kasih liat", "Buka"],
                                        list_objs, STATEMENT_WRAPPERS), remaining)
    for text in list_texts:
        result.append(make_sample(text, "queue", sample_output("queue_management", "queue_list", "queue", "current")))
    return result[:target]


# ===========================================================================
# 4. PLAYER_CONTROL (target 1500)
# ===========================================================================
CONTROL_OBJS = ["musiknya", "lagunya", "lagu ini", "musik", "playback-nya",
                 "lagu yang lagi diputer", "musiknya sekarang", "yang lagi jalan"]
def gen_player_control(target: int = 1500) -> list[dict]:
    result = []
    actions = {
        "pause": ["Pause", "Jeda", "Stop sebentar", "Berhentiin sebentar"],
        "next": ["Skip", "Next", "Lanjut ke lagu berikutnya", "Ganti lagu", "Skip ke lagu selanjutnya"],
        "previous": ["Balik ke lagu sebelumnya", "Previous", "Mundur satu lagu", "Balik ke lagu tadi"],
        "repeat": ["Ulangi", "Repeat", "Ulang terus", "Putar berulang"],
        "repeat_off": ["Matiin ulang", "Repeat off", "Berhenti ngulang", "Matiin repeat"],
        "shuffle": ["Acak", "Shuffle", "Random", "Acakin urutan"],
        "shuffle_off": ["Matiin acak", "Shuffle off", "Berhenti acak", "Matiin shuffle"],
        "stop": ["Matikan", "Stop", "Berhentiin", "Hentiin"],
    }
    n_each = target // len(actions)
    remainder = target - n_each * len(actions)
    for i, (action, verbs) in enumerate(actions.items()):
        n = n_each + (1 if i < remainder else 0)
        texts = pad_to_target(combos(verbs, CONTROL_OBJS, STATEMENT_WRAPPERS), n)
        note = "RULE 3 §16.2 -- stop lagu, BUKAN system.audio_control" if action == "stop" else None
        for text in texts:
            result.append(make_sample(text, "player_control",
                                       sample_output("media_control", action, "player", "current"), note=note))
    return result[:target]


# ===========================================================================
# 5. STREAMING (target 1000)
# ===========================================================================
def gen_streaming(target: int = 1000) -> list[dict]:
    result = []
    n_play = int(target * 0.7)
    n_resolve = target - n_play

    play_texts = []
    for v, topic, plat, suf in itertools.product(["Putar", "Mainkan", "Play", "Buka"], VIDEO_TOPICS,
                                                    STREAM_PLATFORMS, ["", " dong", " ya", " sekarang"]):
        play_texts.append((f"{v} {topic} di {plat}{suf}.", plat))
    RNG.shuffle(play_texts)
    play_texts = pad_to_target(play_texts, n_play)
    for text, plat in play_texts:
        result.append(make_sample(text, "streaming",
                                   sample_output("streaming", "play_stream", "video", "current", player=plat)))
    resolve_texts = pad_to_target(combos(["Resolve", "Buka", "Proses", "Load"],
                                          ["link video ini", "link ini", "url ini", "link yang aku kirim"],
                                          STATEMENT_WRAPPERS), n_resolve)
    for text in resolve_texts:
        result.append(make_sample(text, "streaming",
                                   sample_output("streaming", "resolve", "stream", "url_placeholder")))
    return result[:target]


# ===========================================================================
# 6. METADATA (target 500)
# ===========================================================================
def gen_metadata(target: int = 500) -> list[dict]:
    result = []
    objs = ["lagu ini", "musik ini", "lagu yang lagi diputer", "lagu ini deh",
            "musik yang lagi jalan", "lagu barusan", "yang lagi diputer sekarang"]
    prefixes = ["", "Eh, ", "Btw, ", "Hmm, "]
    texts = []
    for field, obj, prefix in itertools.product(METADATA_FIELDS, objs, prefixes):
        texts.append((f"{prefix}{field.capitalize()} {obj} apa?", field))
        texts.append((f"{prefix}Cek {field} {obj} dong.", field))
        texts.append((f"{prefix}Tau nggak {field} {obj}?", field))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text, field in texts:
        result.append(make_sample(text, "metadata",
                                   sample_output("media_information", "query_metadata", "song", "current", field=field)))
    return result[:target]


# ===========================================================================
# 7. AMBIGUOUS (target 500)
# ===========================================================================
AMBIG_MEDIA_PHRASES = [
    "Ganti yang lain.", "Ganti aja deh.", "Yang itu aja.", "Coba yang lain dulu.",
    "Ganti ke yang lain dong.", "Yang tadi aja deh.", "Bukan yang ini.",
    "Coba lagu lain.", "Ganti musiknya.", "Yang lain aja ah.",
    "Nggak yang ini deh.", "Ganti dulu deh.", "Coba yang beda.",
    "Bukan ini deh.", "Yang lain dulu ya.", "Ganti lagi.",
    "Coba yang lain aja.", "Ini bukan yang aku mau.", "Ganti ke lagu lain.",
    "Pengen yang lain nih.", "Coba deh yang lain.", "Ganti playlist-nya.",
    "Bukan yang itu deh.", "Yang lain dong, ini nggak enak.",
    "Ganti aja ke yang lain.", "Coba lagu yang lain deh.", "Bukan ini yang kumaksud.",
    "Ganti ke yang tadi aja.", "Yang barusan aja deh.", "Coba ganti deh.",
    "Ini kurang pas, ganti dong.", "Pindah ke yang lain.", "Ganti musik lain dong.",
]
AMBIG_SUFFIXES2 = ["", " dong", " ya", " deh", " sekarang", " aja", " nih", " deh sekarang", " dulu"]
def gen_ambiguous(target: int = 500) -> list[dict]:
    result = []
    combos_list = list(itertools.product(AMBIG_MEDIA_PHRASES, AMBIG_SUFFIXES2))
    RNG.shuffle(combos_list)
    combos_list = pad_to_target(combos_list, target)
    for base, suf in combos_list:
        text = base.rstrip(".") + suf + "."
        result.append(make_sample(text, "ambiguous",
                                   sample_output("playback", "play", "song", None),
                                   label="ambiguous",
                                   note="tidak jelas ganti lagu, playlist, atau player"))
    return result[:target]


# ===========================================================================
# 8. NEGATIVE (target 500)
# ===========================================================================
NEGATIVE_TEMPLATES = [
    "Berapa harga tiket konser {artist}?", "Kapan {artist} konser di Jakarta?",
    "Siapa member {artist}?", "Kapan album baru {artist} rilis?",
    "Berita terbaru soal {artist} apa?", "Umur {artist} berapa sih?",
    "Dari mana asal {artist}?", "Genre {artist} apa sih sebenarnya?",
    "Siapa yang nulis lagu {artist}?", "Kapan {artist} debut?",
    "Berapa banyak fans {artist}?", "Apa nama fandom {artist}?",
]
def gen_negative(target: int = 500) -> list[dict]:
    result = []
    texts = []
    for t, artist in itertools.product(NEGATIVE_TEMPLATES, SONGS):
        texts.append(t.format(artist=artist))
    texts = list(set(texts))
    RNG.shuffle(texts)
    texts = pad_to_target(texts, target)
    for text in texts:
        result.append(make_sample(text, "negative", None, label="negative",
                                   note="informational, bukan media control -> domain information"))
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
        "playback": (gen_playback, 3000),
        "search": (gen_search, 1500),
        "queue": (gen_queue, 1500),
        "player_control": (gen_player_control, 1500),
        "streaming": (gen_streaming, 1000),
        "metadata": (gen_metadata, 500),
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
