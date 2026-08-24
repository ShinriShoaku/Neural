"""
task_composition.py
====================
Target jumlah sample PER TASK-CATEGORY, diambil persis dari dokumen:

    §30  Router Dataset Composition
    §31  System Specialist Dataset
    §32  Media Specialist Dataset
    §33  Persona Specialist Dataset
    §34  Coding Specialist Dataset
    §35  Information Specialist Dataset
    §36  Memory Specialist Dataset
    §37  Productivity Specialist Dataset
    §38  Validator Dataset (tidak ada breakdown per-kategori eksplisit di
         dokumen — cuma total 8,000 di §39 — jadi dibiarkan tanpa target
         per task_category, lihat catatan di bawah)

Nama key task_category dinormalisasi dari judul di dokumen (spasi/slash
diganti underscore) supaya valid jadi nama file:
    "topic extraction"   -> "topic_extraction"
    "roleplay intent"    -> "roleplay_intent"
    "dialogue control"   -> "dialogue_control"
    "context dependency" -> "context_dependency"
    "event memory"       -> "event_memory"
    "preference memory"  -> "preference_memory"
    "update/delete"      -> "update_delete"
    "ambiguous/negative" -> "ambiguous_negative"

Dipakai untuk: (1) nama file output, (2) progress report di main.py.
"""

TASK_TARGETS: dict[str, dict[str, int]] = {
    "router": {
        # §30
        "single_intent": 3000,
        "multi_intent": 3000,
        "ambiguous": 1500,
        "implicit_intent": 500,
        "domain_overlap_disambiguation": 1000,
        "negative_unknown": 1000,
        "context_dependent": 1000,
        "compound_command": 1000,
    },
    "system": {
        # §31 (catatan: jumlah kolom ini total 16,000 di dokumen,
        # meski judul section bilang "Target awal: 15,000" — kemungkinan
        # typo di dokumen asli, dibiarkan apa adanya di sini)
        "application": 3000,
        "process": 2000,
        "filesystem": 3000,
        "shell": 2000,
        "hardware": 1500,
        "audio": 1000,
        "display": 500,
        "network": 1000,
        "system_query": 1000,
        "ambiguous_negative": 1000,
    },
    "media": {
        # §32
        "playback": 3000,
        "search": 1500,
        "queue": 1500,
        "player_control": 1500,
        "streaming": 1000,
        "metadata": 500,
        "ambiguous": 500,
        "negative": 500,
    },
    "persona": {
        # §33
        "character_call": 1500,
        "character_switch": 1500,
        "conversation": 2500,
        "topic_extraction": 2000,
        "roleplay_intent": 1500,
        "dialogue_control": 1000,
        "context_dependency": 1000,
        "ambiguous": 500,
        "negative": 500,
    },
    "coding": {
        # §34
        "generate": 2000,
        "debug": 1500,
        "modify": 1500,
        "review": 1000,
        "explain": 1000,
        "refactor": 1000,
        "architecture": 1000,
        "test": 500,
        "ambiguous": 500,
        "negative": 500,
    },
    "information": {
        # §35
        "search": 1500,
        "weather": 1000,
        "time": 500,
        "translation": 1000,
        "lookup": 1000,
        "calculation": 500,
        "comparison": 1000,
        "knowledge": 1000,
        "ambiguous": 250,
        "negative": 250,
    },
    "memory": {
        # §36
        "remember": 1500,
        "retrieve": 1000,
        "update": 1000,
        "forget": 750,
        "search": 750,
        "event_memory": 750,
        "preference_memory": 750,
        "ambiguous": 250,
        "negative": 250,
    },
    "productivity": {
        # §37
        "calendar": 1500,
        "reminder": 1500,
        "todo": 1000,
        "schedule": 1000,
        "notification": 750,
        "communication": 750,
        "update_delete": 750,
        "ambiguous": 375,
        "negative": 375,
    },
    "validator": {
        # §38 — dokumen TIDAK memberi breakdown angka per kategori negatif,
        # cuma total 8,000 (§39). Kategori di bawah diambil dari daftar
        # negative categories §38.1, tanpa target individual (None) —
        # main.py akan tampilkan count tanpa persentase untuk domain ini.
        "valid": None,
        "target_mismatch": None,
        "domain_mismatch": None,
        "missing_parameter": None,
        "hallucinated_parameter": None,
        "intent_mismatch": None,
        "unsupported_action": None,
        "parameter_mismatch": None,   # §38.1 ada, TIDAK ada di corruption pipeline §38.3 -> manual only
        "contradiction": None,        # idem
        "ambiguous": None,            # idem
    },
}

# Total target keseluruhan per §39 (buat referensi, bukan dipakai kode)
DATASET_TOTAL_TARGET = {
    "router": 12000,
    "system": 15000,  # lihat catatan di atas soal selisih breakdown
    "media": 10000,
    "persona": 12000,
    "coding": 10000,
    "information": 8000,
    "memory": 7000,
    "productivity": 8000,
    "validator": 8000,
}
