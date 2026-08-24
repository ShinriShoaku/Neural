"""
capabilities.py
================
Vocabulary registry per domain — dipakai untuk mengisi {intent_list},
{action_list}, {target_type_list} di system prompt specialist (§67.4).

Diambil persis dari §7-13. Idealnya ini di-generate dari capabilities.json
(§52) supaya prompt dan registry runtime tidak out-of-sync — tapi karena
capabilities.json (registry executor) belum dibuat di repo ini, untuk
sekarang ditulis langsung di sini. Kalau nanti capabilities.json dibuat,
tinggal ganti CAPABILITIES di bawah jadi hasil load dari file itu.
"""

CAPABILITIES: dict[str, dict[str, list[str]]] = {
    "system": {
        "intents": [
            "application_control", "process_control", "filesystem_operation",
            "shell_execution", "hardware_control", "audio_control",
            "display_control", "network_control", "system_query",
        ],
        "actions": [
            "launch", "close", "restart", "focus", "minimize", "maximize", "query_status",
            "list", "inspect", "start", "stop", "kill", "query_resource",
            "read", "write", "create", "copy", "move", "rename", "delete", "search",
            "execute", "query", "configure",
            "volume_get", "volume_set", "mute", "unmute", "device_list", "device_select",
            "brightness_get", "brightness_set", "display_list", "display_select",
            "status", "interface_list", "connection_query",
            "os_info", "kernel_info", "uptime", "resource_usage",
        ],
        "target_types": ["application", "process", "file", "directory", "device"],
    },
    "media": {
        "intents": ["playback", "search_media", "queue_management", "media_control",
                     "streaming", "media_information"],
        "actions": ["play", "pause", "resume", "stop", "next", "previous", "seek", "volume",
                     "shuffle", "repeat", "queue_add", "queue_remove", "queue_clear", "queue_list",
                     "search", "resolve", "play_stream", "query_metadata"],
        "target_types": ["song", "album", "artist", "playlist", "video", "stream", "player", "queue"],
    },
    "persona": {
        "intents": ["conversation", "character_call", "character_switch", "roleplay",
                     "character_query", "persona_context", "dialogue_control"],
        "actions": ["talk", "call", "switch", "ask", "answer", "continue", "introduce",
                     "explain", "roleplay", "change_topic", "resume_topic"],
        "target_types": ["persona", "character", "conversation", "topic"],
    },
    "coding": {
        "intents": ["code_generation", "code_modification", "code_debugging",
                     "code_analysis", "code_explanation", "architecture",
                     "refactoring", "testing"],
        "actions": ["generate", "modify", "fix", "debug", "analyze", "explain",
                     "refactor", "review", "test", "design"],
        "target_types": [],  # coding tidak pakai target, cuma parameters (§10)
    },
    "information": {
        "intents": ["information_query", "search", "weather", "time",
                     "translation", "calculation", "lookup", "comparison"],
        "actions": ["search", "query", "lookup", "translate", "calculate",
                     "compare", "retrieve", "summarize"],
        "target_types": ["web", "weather", "time", "knowledge", "document", "entity"],
    },
    "memory": {
        "intents": ["memory_store", "memory_retrieve", "memory_update",
                     "memory_delete", "memory_query", "memory_summarize"],
        "actions": ["remember", "retrieve", "update", "forget", "search", "summarize"],
        "target_types": [],  # memory pakai parameters.category (identity/preference/dst), bukan target
    },
    "productivity": {
        "intents": ["calendar", "reminder", "todo", "schedule",
                     "planning", "notification", "communication"],
        "actions": ["create", "update", "delete", "list", "complete",
                     "schedule", "remind", "notify", "send"],
        "target_types": ["event", "reminder", "todo", "notification", "message", "email"],
    },
}

ROUTER_DOMAINS = ["system", "media", "persona", "coding", "information",
                   "memory", "productivity", "unknown"]


def get_vocab_strings(domain: str) -> tuple[str, str, str]:
    """Return (intent_list, action_list, target_type_list) sebagai comma-separated string,
    persis format yang dipakai di §67.1 system prompt template."""
    cap = CAPABILITIES[domain]
    intent_list = ", ".join(cap["intents"])
    action_list = ", ".join(cap["actions"])
    target_type_list = ", ".join(cap["target_types"]) if cap["target_types"] else "(tidak ada — domain ini tidak pakai target)"
    return intent_list, action_list, target_type_list
