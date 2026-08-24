"""
templates.py
============
Kumpulan template kalimat SEDERHANA sebagai starting point dataset.

Ini BUKAN dataset final (§30-§37 menargetkan puluhan ribu sample per
specialist). Ini cuma seed kecil yang gampang kamu tambah/ubah nanti
(tinggal nambah entry di list/dict di bawah).

Struktur sengaja dipisah per domain supaya nanti gampang di-scale:
tinggal tambah domain baru / action baru tanpa ubah generator.py.
"""

# ---------------------------------------------------------------------------
# 1. SYSTEM SPECIALIST — contoh capability: application launch/close (§7, §31)
# ---------------------------------------------------------------------------

# capability "application" bukan LoRA terpisah (§2) — cuma target.value
APPLICATIONS = ["foot", "firefox", "spotify", "discord", "vscode", "terminal"]

SYSTEM_LAUNCH_TEMPLATES = [
    "Buka {app}",
    "Buka aplikasi {app}",
    "Buka {app} dong",
    "Jalankan {app}",
    "Tolong buka {app}",
    "Nyalain {app}",
]

SYSTEM_CLOSE_TEMPLATES = [
    "Tutup {app}",
    "Matiin {app}",
    "Keluar dari {app}",
    "Close {app}",
]

# Hard negative (§42): mirip lexical, beda intent — "cari file X" bukan launch
SYSTEM_HARD_NEGATIVE_TEMPLATES = [
    ("Cari file {app}", "system_hard_negative_search_not_launch"),
    ("Siapa yang install {app}?", "system_hard_negative_query_not_launch"),
]

# Negative murni (§41): tidak ada action system yang jelas / bukan domain system
SYSTEM_NEGATIVE_INPUTS = [
    "Halo, apa kabar?",
    "Ceritain dongeng singkat.",
    "Berapa hasil 12 x 8?",
]

# ---------------------------------------------------------------------------
# Task-category lain di System Specialist (§31): process, filesystem, shell,
# hardware, audio, display, network, system_query.
#
# BARU "application" yang diisi penuh di atas. Sisanya sengaja dikosongkan
# dulu (list kosong) supaya:
#   1. Strukturnya sudah sesuai §31 (task_composition.py udah nunggu ini)
#   2. Kamu tinggal isi list di bawah dengan pola yang sama kayak
#      SYSTEM_LAUNCH_TEMPLATES / SYSTEM_CLOSE_TEMPLATES
#
# Contoh isi nanti (process):
#   SYSTEM_PROCESS_TEMPLATES = ["Matiin proses {proc}", "Kill {proc}"]
#   PROCESSES = ["chrome", "spotify", "node"]
# ---------------------------------------------------------------------------

SYSTEM_PROCESS_TEMPLATES: list[str] = []       # TODO §31: process (target 2,000)
SYSTEM_FILESYSTEM_TEMPLATES: list[str] = []    # TODO §31: filesystem (target 3,000)
SYSTEM_SHELL_TEMPLATES: list[str] = []         # TODO §31: shell (target 2,000)
SYSTEM_HARDWARE_TEMPLATES: list[str] = []      # TODO §31: hardware (target 1,500)
SYSTEM_AUDIO_TEMPLATES: list[str] = []         # TODO §31: audio (target 1,000)
SYSTEM_DISPLAY_TEMPLATES: list[str] = []       # TODO §31: display (target 500)
SYSTEM_NETWORK_TEMPLATES: list[str] = []       # TODO §31: network (target 1,000)
SYSTEM_QUERY_TEMPLATES: list[str] = []         # TODO §31: system_query (target 1,000)


# ---------------------------------------------------------------------------
# 2. ROUTER — contoh lintas domain, single-intent & multi-intent (§29, §30)
# ---------------------------------------------------------------------------

# Kalimat single-domain sederhana per domain, dipakai Router untuk belajar
# segmentation + domain classification (bukan detail action)
ROUTER_SINGLE_DOMAIN_EXAMPLES: dict[str, list[str]] = {
    "system": [
        "Buka foot",
        "Matiin discord",
        "Restart komputer",
        "Cek penggunaan RAM",
    ],
    "media": [
        "Putar lagu Noah",
        "Pause musiknya",
        "Skip ke lagu berikutnya",
    ],
    "persona": [
        "Panggil Mailin buat ngobrol",
        "Ganti karakter ke Aria",
    ],
    "coding": [
        "Bikinin fungsi python buat sorting",
        "Cek bug di file main.py",
    ],
    "information": [
        "Cuaca hari ini gimana?",
        "Cari tahu ibu kota Prancis",
    ],
    "memory": [
        "Inget-inget kalau aku suka kopi hitam",
        "Kamu masih inget nama kucingku?",
    ],
    "productivity": [
        "Ingetin aku meeting jam 3 sore",
        "Tambahin todo beli galon",
    ],
}

# Negative / unknown untuk router (§30 — negative/unknown 1,000)
ROUTER_UNKNOWN_INPUTS = [
    "asdkjaslkdj qwerty",
    "...",
    "hmm",
]

# Contoh multi-intent (dua domain sekaligus, §5)
ROUTER_MULTI_INTENT_EXAMPLES = [
    {
        "input": "Buka foot lalu panggil Mailin buat bahas Paradeus.",
        "segments": [
            ("Buka foot", "system"),
            ("panggil Mailin buat bahas Paradeus", "persona"),
        ],
    },
    {
        "input": "Buka foot lalu putar lagu Noah.",
        "segments": [
            ("Buka foot", "system"),
            ("putar lagu Noah", "media"),
        ],
    },
    {
        "input": "Ingetin aku meeting jam 3 dan cariin cuaca besok.",
        "segments": [
            ("Ingetin aku meeting jam 3", "productivity"),
            ("cariin cuaca besok", "information"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Task-category lain di Router (§30): ambiguous, implicit_intent,
# domain_overlap_disambiguation, context_dependent, compound_command.
#
# Baru domain_overlap_disambiguation yang diisi 1 contoh (persis dari §5),
# sisanya stub kosong — pola pengisiannya sama kayak ROUTER_MULTI_INTENT_EXAMPLES
# di atas (list of dict {"input": ..., "segments": [(text, domain), ...]}).
# ---------------------------------------------------------------------------

# §5 — router hanya kasih HINT overlap, bukan resolve. Untuk versi simple ini
# kita simpan sebagai domain tunggal + note; field overlap_hint penuh
# (candidate_domains, resolution_rule) bisa ditambah ke metadata nanti.
ROUTER_DOMAIN_OVERLAP_EXAMPLES = [
    {
        "input": "Putar Noah di Spotify.",
        "segments": [("Putar Noah di Spotify", "media")],
        "note": "overlap candidate: media vs system, resolution_rule=media_with_app_context (§5, §16 RULE 2)",
    },
    {
        "input": "Matikan suara.",
        "segments": [("Matikan suara.", "system")],
        "note": "RULE 3 (§16.2): 'matikan suara'/mute -> system.audio_control, BUKAN media.stop",
    },
    {
        "input": "Matikan musik.",
        "segments": [("Matikan musik.", "media")],
        "note": "RULE 3 (§16.2): 'matikan musik'/stop lagu -> media.playback.stop, BUKAN system.audio_control",
    },
]

ROUTER_AMBIGUOUS_EXAMPLES: list[dict] = [
    # confidence borderline / makna ganda — Router ragu, bukan salah total
    {"input": "Matiin itu.", "segments": [("Matiin itu.", "unknown")],
     "note": "referent 'itu' tidak jelas: media, system, atau notification?"},
    {"input": "Yang tadi aja.", "segments": [("Yang tadi aja.", "unknown")],
     "note": "butuh context sebelumnya, tanpa itu domain tidak bisa dipastikan"},
]

ROUTER_IMPLICIT_INTENT_EXAMPLES: list[dict] = [
    # intent tidak dinyatakan eksplisit, tapi tersirat dari situasi
    {"input": "Berisik banget suaranya.", "segments": [("Berisik banget suaranya.", "system")],
     "note": "implicit: minta volume diturunkan/mute, bukan sekadar keluhan"},
    {"input": "Laptop udah lowbat nih.", "segments": [("Laptop udah lowbat nih.", "system")],
     "note": "implicit: minta cek status baterai (system_query), bukan cuma statement"},
]

ROUTER_CONTEXT_DEPENDENT_EXAMPLES: list[dict] = [
    # makna bergantung pada router_context (§6) dari turn sebelumnya
    {"input": "Ulang lagi dong.", "segments": [("Ulang lagi dong.", "media")],
     "note": "perlu last_domain=media & last_action=play dari router_context (§6)"},
    {"input": "Yang itu aja deh, jangan yang tadi.", "segments": [("Yang itu aja deh, jangan yang tadi.", "unknown")],
     "note": "butuh 2 turn context sebelumnya untuk resolve"},
]

ROUTER_COMPOUND_COMMAND_EXAMPLES: list[dict] = [
    # beberapa action berurutan TAPI dalam satu domain yang sama
    # (beda dari multi_intent yang lintas-domain)
    {"input": "Buka foot terus buka vscode juga.", "segments": [("Buka foot terus buka vscode juga.", "system")],
     "note": "compound dalam 1 domain (system): 2 action launch berurutan"},
    {"input": "Putar Noah terus lanjut ke Sheila On 7.", "segments": [("Putar Noah terus lanjut ke Sheila On 7.", "media")],
     "note": "compound dalam 1 domain (media): play lalu queue_add/next"},
]


# ---------------------------------------------------------------------------
# 3. MEDIA SPECIALIST (§8, §32)
# Format: dict[task_category] -> list of example dict
#   {"input", "intent", "action", "target": {"type","value"} atau None,
#    "parameters": {...} (optional), "label": "positive"/"hard_negative"/"negative",
#    "note": optional}
# ---------------------------------------------------------------------------

MEDIA_EXAMPLES: dict[str, list[dict]] = {
    "playback": [
        {"input": "Putar lagu Noah.", "intent": "playback", "action": "play",
         "target": {"type": "artist", "value": "Noah"}},
        {"input": "Putar Noah di Spotify.", "intent": "playback", "action": "play",
         "target": {"type": "artist", "value": "Noah"},
         "parameters": {"player": "spotify"},
         "note": "implicit app launch, §20 — context.implicit_launch_allowed"},
        {"input": "Lanjutin lagunya.", "intent": "playback", "action": "resume",
         "target": {"type": "player", "value": "current"}},
        {"input": "Putar album Konspirasi Alam Semesta.", "intent": "playback", "action": "play",
         "target": {"type": "album", "value": "Konspirasi Alam Semesta"}},
    ],
    "search": [
        {"input": "Cari lagu galau di Spotify.", "intent": "search_media", "action": "search",
         "target": {"type": "song", "value": "galau"}, "parameters": {"player": "spotify"}},
        {"input": "Cari playlist workout.", "intent": "search_media", "action": "search",
         "target": {"type": "playlist", "value": "workout"}},
    ],
    "queue": [
        {"input": "Tambahin lagu ini ke antrian.", "intent": "queue_management", "action": "queue_add",
         "target": {"type": "song", "value": "current"}},
        {"input": "Kosongin antrian lagu.", "intent": "queue_management", "action": "queue_clear",
         "target": {"type": "queue", "value": "current"}},
        {"input": "Liat antrian lagu dong.", "intent": "queue_management", "action": "queue_list",
         "target": {"type": "queue", "value": "current"}},
    ],
    "player_control": [
        {"input": "Pause musiknya.", "intent": "media_control", "action": "pause",
         "target": {"type": "player", "value": "current"}},
        {"input": "Skip ke lagu berikutnya.", "intent": "media_control", "action": "next",
         "target": {"type": "player", "value": "current"}},
        {"input": "Ulang lagu ini.", "intent": "media_control", "action": "repeat",
         "target": {"type": "player", "value": "current"}},
        {"input": "Matikan musik.", "intent": "media_control", "action": "stop",
         "target": {"type": "player", "value": "current"},
         "note": "RULE 3 §16.2 — stop lagu, BUKAN system.audio_control"},
    ],
    "streaming": [
        {"input": "Putar video ini di YouTube.", "intent": "streaming", "action": "play_stream",
         "target": {"type": "video", "value": "current"}, "parameters": {"player": "youtube"}},
        {"input": "Resolve link video ini.", "intent": "streaming", "action": "resolve",
         "target": {"type": "stream", "value": "url_placeholder"}},
    ],
    "metadata": [
        {"input": "Judul lagu ini apa?", "intent": "media_information", "action": "query_metadata",
         "target": {"type": "song", "value": "current"}},
    ],
    "ambiguous": [
        {"input": "Ganti yang lain.", "intent": "playback", "action": "play",
         "target": {"type": "song", "value": None}, "label": "ambiguous",
         "note": "tidak jelas ganti lagu, playlist, atau player"},
    ],
    "negative": [
        {"input": "Berapa harga tiket konser Noah?", "output": None, "label": "negative",
         "note": "informational, bukan media control -> domain information"},
    ],
}


# ---------------------------------------------------------------------------
# 4. PERSONA SPECIALIST (§9, §33)
# ---------------------------------------------------------------------------

PERSONA_EXAMPLES: dict[str, list[dict]] = {
    "character_call": [
        {"input": "Panggil Mailin.", "intent": "character_call", "action": "call",
         "target": {"type": "persona", "value": "mailin"}},
        {"input": "Panggilin Aria dong.", "intent": "character_call", "action": "call",
         "target": {"type": "persona", "value": "aria"}},
    ],
    "character_switch": [
        {"input": "Ganti karakter ke Aria.", "intent": "character_switch", "action": "switch",
         "target": {"type": "persona", "value": "aria"}},
        {"input": "Pindah ke Mailin aja.", "intent": "character_switch", "action": "switch",
         "target": {"type": "persona", "value": "mailin"}},
    ],
    "conversation": [
        {"input": "Suruh Mailin bahas Paradeus.", "intent": "conversation", "action": "talk",
         "target": {"type": "persona", "value": "mailin"}, "parameters": {"topic": "Paradeus"}},
        {"input": "Tanya Aria soal cuaca besok.", "intent": "conversation", "action": "ask",
         "target": {"type": "persona", "value": "aria"}, "parameters": {"topic": "cuaca besok"}},
    ],
    "topic_extraction": [
        {"input": "Panggil Mailin buat bahas rencana liburan.", "intent": "conversation", "action": "talk",
         "target": {"type": "persona", "value": "mailin"}, "parameters": {"topic": "rencana liburan"}},
        {"input": "Ngobrol sama Aria tentang film terbaru.", "intent": "conversation", "action": "talk",
         "target": {"type": "persona", "value": "aria"}, "parameters": {"topic": "film terbaru"}},
    ],
    "roleplay_intent": [
        {"input": "Aria, jadi jadi asisten pribadi aku ya.", "intent": "roleplay", "action": "roleplay",
         "target": {"type": "persona", "value": "aria"}, "parameters": {"role": "asisten pribadi"}},
    ],
    "dialogue_control": [
        {"input": "Lanjutin obrolan tadi.", "intent": "dialogue_control", "action": "continue",
         "target": {"type": "conversation", "value": "current"}},
        {"input": "Balik ke topik sebelumnya.", "intent": "dialogue_control", "action": "resume_topic",
         "target": {"type": "topic", "value": "previous"}},
    ],
    "context_dependency": [
        {"input": "Tanya dia lagi soal itu.", "intent": "conversation", "action": "ask",
         "target": {"type": "persona", "value": None}, "label": "ambiguous",
         "note": "butuh router_context: siapa 'dia', apa 'itu'"},
    ],
    "ambiguous": [
        {"input": "Suruh dia jelasin.", "intent": "conversation", "action": "explain",
         "target": {"type": "persona", "value": None}, "label": "ambiguous",
         "note": "referent persona tidak jelas tanpa context"},
    ],
    "negative": [
        {"input": "Jelasin rekursi dengan bahasa Python.", "output": None, "label": "negative",
         "note": "ini technical explanation -> domain coding, bukan persona"},
    ],
}


# ---------------------------------------------------------------------------
# 5. CODING SPECIALIST (§10, §34)
# Coding Specialist TIDAK menghasilkan kode — hanya structured request (§10, §53)
# ---------------------------------------------------------------------------

CODING_EXAMPLES: dict[str, list[dict]] = {
    "generate": [
        {"input": "Buat fungsi rekursif Python untuk mencari file.", "intent": "code_generation",
         "action": "generate", "parameters": {"language": "python", "requirements": "recursive file search"}},
        {"input": "Bikinin script bash buat backup folder.", "intent": "code_generation",
         "action": "generate", "parameters": {"language": "bash", "requirements": "backup folder"}},
    ],
    "debug": [
        {"input": "Kenapa function ini infinite loop ya?", "intent": "code_debugging", "action": "debug",
         "parameters": {"error": "infinite loop"}},
        {"input": "Debug error IndexError di file main.py.", "intent": "code_debugging", "action": "debug",
         "parameters": {"file": "main.py", "error": "IndexError"}},
    ],
    "modify": [
        {"input": "Tambahin error handling di fungsi ini.", "intent": "code_modification", "action": "modify",
         "parameters": {"requirements": "add error handling"}},
        {"input": "Ubah fungsi ini biar support async.", "intent": "code_modification", "action": "modify",
         "parameters": {"requirements": "convert to async"}},
    ],
    "review": [
        {"input": "Review kode ini dong.", "intent": "code_analysis", "action": "review",
         "parameters": {}},
    ],
    "explain": [
        {"input": "Jelasin rekursi dengan bahasa Python.", "intent": "code_explanation", "action": "explain",
         "parameters": {"language": "python", "requirements": "explain recursion"}},
        {"input": "Kode ini ngapain sih maksudnya?", "intent": "code_explanation", "action": "explain",
         "parameters": {}},
    ],
    "refactor": [
        {"input": "Refactor function ini biar lebih rapi.", "intent": "refactoring", "action": "refactor",
         "parameters": {}},
    ],
    "architecture": [
        {"input": "Gimana cara desain sistem antrian buat aplikasi chat?", "intent": "architecture",
         "action": "design", "parameters": {"requirements": "queue system design for chat app"}},
    ],
    "test": [
        {"input": "Bikinin unit test buat fungsi ini.", "intent": "testing", "action": "test",
         "parameters": {}},
    ],
    "ambiguous": [
        {"input": "Perbaiki kodenya.", "intent": "code_modification", "action": "modify",
         "parameters": {}, "label": "ambiguous", "note": "file/fungsi mana yang dimaksud tidak jelas"},
    ],
    "negative": [
        {"input": "Ingetin aku meeting jam 3.", "output": None, "label": "negative",
         "note": "domain productivity, bukan coding"},
    ],
}


# ---------------------------------------------------------------------------
# 6. INFORMATION SPECIALIST (§11, §35)
# ---------------------------------------------------------------------------

INFORMATION_EXAMPLES: dict[str, list[dict]] = {
    "search": [
        {"input": "Cari tahu ibu kota Prancis.", "intent": "search", "action": "search",
         "target": {"type": "web", "value": "ibu kota Prancis"}},
    ],
    "weather": [
        {"input": "Cek cuaca.", "intent": "weather", "action": "query",
         "target": {"type": "weather", "value": "today"}},
        {"input": "Cuaca besok gimana?", "intent": "weather", "action": "query",
         "target": {"type": "weather", "value": "tomorrow"}},
    ],
    "time": [
        {"input": "Jam berapa sekarang?", "intent": "time", "action": "query",
         "target": {"type": "time", "value": "now"}},
    ],
    "translation": [
        {"input": "Artiin 'good morning' ke bahasa Indonesia.", "intent": "translation", "action": "translate",
         "parameters": {"text": "good morning", "target_language": "id"}},
    ],
    "lookup": [
        {"input": "Siapa presiden Indonesia sekarang?", "intent": "lookup", "action": "lookup",
         "target": {"type": "entity", "value": "presiden Indonesia"}},
    ],
    "calculation": [
        {"input": "Berapa hasil 12 kali 8?", "intent": "calculation", "action": "calculate",
         "parameters": {"expression": "12 * 8"}},
    ],
    "comparison": [
        {"input": "Lebih bagus mana, iPhone atau Samsung?", "intent": "comparison", "action": "compare",
         "parameters": {"items": ["iPhone", "Samsung"]}},
    ],
    "knowledge": [
        {"input": "Apa itu fotosintesis?", "intent": "information_query", "action": "query",
         "target": {"type": "knowledge", "value": "fotosintesis"}},
    ],
    "ambiguous": [
        {"input": "Cari itu deh.", "intent": "search", "action": "search",
         "target": {"type": "web", "value": None}, "label": "ambiguous",
         "note": "objek pencarian tidak jelas tanpa context"},
    ],
    "negative": [
        {"input": "Ingetin aku soal itu besok.", "output": None, "label": "negative",
         "note": "domain productivity/memory, bukan information"},
    ],
}


# ---------------------------------------------------------------------------
# 7. MEMORY SPECIALIST (§12, §36)
# ---------------------------------------------------------------------------

MEMORY_EXAMPLES: dict[str, list[dict]] = {
    "remember": [
        {"input": "Ingat kalau aku suka musik Noah.", "intent": "memory_store", "action": "remember",
         "parameters": {"category": "preference", "content": "User likes Noah music"}},
        {"input": "Catet nama kucingku Milo.", "intent": "memory_store", "action": "remember",
         "parameters": {"category": "identity", "content": "User's cat name is Milo"}},
    ],
    "retrieve": [
        {"input": "Kamu masih inget nama kucingku?", "intent": "memory_retrieve", "action": "retrieve",
         "parameters": {"category": "identity", "query": "nama kucing"}},
    ],
    "update": [
        {"input": "Update, sekarang aku lebih suka kopi susu.", "intent": "memory_update", "action": "update",
         "parameters": {"category": "preference", "content": "User now prefers milk coffee"}},
    ],
    "forget": [
        {"input": "Lupain soal preferensi musik aku ya.", "intent": "memory_delete", "action": "forget",
         "parameters": {"category": "preference"}},
    ],
    "search": [
        {"input": "Cari memori soal rencana liburanku.", "intent": "memory_query", "action": "search",
         "parameters": {"query": "rencana liburan"}},
    ],
    "event_memory": [
        {"input": "Inget aku pernah cerita soal wawancara kerja minggu lalu.", "intent": "memory_store",
         "action": "remember", "parameters": {"category": "event", "content": "job interview last week"}},
    ],
    "preference_memory": [
        {"input": "Aku nggak suka pedas, tolong diinget.", "intent": "memory_store", "action": "remember",
         "parameters": {"category": "preference", "content": "User dislikes spicy food"}},
    ],
    "ambiguous": [
        {"input": "Inget yang tadi ya.", "intent": "memory_store", "action": "remember",
         "parameters": {"content": None}, "label": "ambiguous",
         "note": "konten yang mau diingat tidak jelas tanpa context"},
    ],
    "negative": [
        {"input": "Cari tahu ibu kota Jepang.", "output": None, "label": "negative",
         "note": "domain information, bukan memory"},
    ],
}


# ---------------------------------------------------------------------------
# 8. PRODUCTIVITY SPECIALIST (§13, §37)
# ---------------------------------------------------------------------------

PRODUCTIVITY_EXAMPLES: dict[str, list[dict]] = {
    "calendar": [
        {"input": "Bikin event meeting besok jam 10.", "intent": "calendar", "action": "create",
         "target": {"type": "event", "value": "meeting"}, "parameters": {"time": "besok 10:00"}},
    ],
    "reminder": [
        {"input": "Besok jam 8 ingatkan aku update project.", "intent": "reminder", "action": "create",
         "target": {"type": "reminder", "value": None},
         "parameters": {"time": "2026-08-23T08:00:00+07:00", "content": "update project"}},
    ],
    "todo": [
        {"input": "Tambahin todo beli galon.", "intent": "todo", "action": "create",
         "target": {"type": "todo", "value": "beli galon"}},
        {"input": "Tandain todo beli galon udah selesai.", "intent": "todo", "action": "complete",
         "target": {"type": "todo", "value": "beli galon"}},
    ],
    "schedule": [
        {"input": "Jadwalin olahraga tiap pagi jam 6.", "intent": "schedule", "action": "schedule",
         "target": {"type": "event", "value": "olahraga"}, "parameters": {"recurrence": "daily 06:00"}},
    ],
    "notification": [
        {"input": "Kirim notifikasi kalau meeting udah mulai.", "intent": "notification", "action": "notify",
         "target": {"type": "notification", "value": "meeting started"}},
    ],
    "communication": [
        {"input": "Balas pesan ini dengan 'oke siap'.", "intent": "communication", "action": "send",
         "target": {"type": "message", "value": "oke siap"},
         "note": "§3 — komunikasi sederhana masuk productivity"},
    ],
    "update_delete": [
        {"input": "Hapus reminder meeting besok.", "intent": "reminder", "action": "delete",
         "target": {"type": "reminder", "value": "meeting besok"}},
        {"input": "Ubah jadwal meeting jadi jam 2 siang.", "intent": "calendar", "action": "update",
         "target": {"type": "event", "value": "meeting"}, "parameters": {"time": "14:00"}},
    ],
    "ambiguous": [
        {"input": "Ubah itu jadi besok aja.", "intent": "calendar", "action": "update",
         "target": {"type": "event", "value": None}, "label": "ambiguous",
         "note": "event mana yang diubah tidak jelas tanpa context"},
    ],
    "negative": [
        {"input": "Ingat kalau aku suka kopi hitam.", "output": None, "label": "negative",
         "note": "domain memory, bukan productivity"},
    ],
}
