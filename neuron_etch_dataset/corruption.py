"""
corruption.py
=============
Implementasi Corruption Pipeline (§38.3) untuk membangkitkan Validator
dataset dari Task IR valid yang sudah ada di dataset specialist lain,
tanpa perlu menulis pasangan original/generated dari nol.

Setiap fungsi corrupt_* mengambil task IR dict yang VALID (hasil
SpecialistOutput.to_dict()) dan mengembalikan versi yang SUDAH DIRUSAK,
sesuai satu corruption_type di §38.3.

Mapping corruption_type -> reason (persis dari §38.3):
    swap_target        -> target_mismatch
    swap_domain         -> domain_mismatch
    drop_required_param -> missing_parameter
    inject_param        -> hallucinated_parameter
    swap_action         -> intent_mismatch
    swap_intent_far      -> unsupported_action

Catatan: `parameter_mismatch`, `contradiction`, `ambiguous` (§38.1) TIDAK
ada di daftar corruption otomatis §38.3 -> tetap harus manual/human-curated,
makanya tidak ada fungsi corrupt_* untuk itu di sini (lihat task_composition.py).
"""

from __future__ import annotations
import copy
import random
from typing import Optional

# Domain lain dipakai buat swap_domain (§38.3)
OTHER_DOMAINS = ["system", "media", "persona", "coding", "information", "memory", "productivity"]

# Value alternatif per target.type, buat swap_target (§38.3) — ambil value
# acak yang MASIH masuk akal di domain sama, supaya corruption realistis
ALT_TARGET_VALUES = {
    "application": ["firefox", "discord", "vscode", "spotify", "terminal", "foot"],
    "artist": ["Sheila On 7", "Noah", "Tulus", "Raisa"],
    "song": ["lagu lain", "lagu random"],
    "persona": ["mailin", "aria"],
    "reminder": ["reminder lain"],
    "event": ["event lain"],
    "todo": ["todo lain"],
}

# Action alternatif dalam domain yang sama (buat swap_action), diambil dari §7-13
ALT_ACTIONS_SAME_DOMAIN = {
    "system": ["launch", "close", "restart", "focus", "minimize", "kill", "query"],
    "media": ["play", "pause", "stop", "next", "previous", "queue_add"],
    "persona": ["call", "switch", "talk", "ask", "continue"],
    "coding": ["generate", "modify", "debug", "explain", "refactor", "test"],
    "information": ["search", "query", "translate", "calculate", "compare"],
    "memory": ["remember", "retrieve", "update", "forget", "search"],
    "productivity": ["create", "update", "delete", "complete", "notify"],
}

# Action dari domain LAIN (buat swap_intent_far, lebih ekstrem dari swap_action)
FAR_DOMAIN_ACTIONS = {
    "system": ("coding", "generate"),
    "media": ("productivity", "create"),
    "persona": ("information", "search"),
    "coding": ("system", "launch"),
    "information": ("media", "play"),
    "memory": ("persona", "call"),
    "productivity": ("memory", "remember"),
}

# Parameter wajib per (domain, intent) yang MASUK AKAL untuk di-drop (drop_required_param)
# — cuma contoh minimal, bukan daftar lengkap semua intent
REQUIRED_PARAM_BY_INTENT = {
    ("productivity", "reminder"): "time",
    ("coding", "code_generation"): "requirements",
    ("information", "translation"): "text",
    ("memory", "memory_store"): "content",
}


def corrupt_swap_target(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """swap_target -> reason: target_mismatch (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    target = corrupted.get("target")
    if not target or not target.get("type"):
        return None
    alt_values = ALT_TARGET_VALUES.get(target["type"])
    if not alt_values:
        return None
    candidates = [v for v in alt_values if v != target["value"]]
    if not candidates:
        return None
    target["value"] = rng.choice(candidates)
    return corrupted


def corrupt_swap_domain(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """swap_domain -> reason: domain_mismatch (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    current = corrupted.get("domain")
    candidates = [d for d in OTHER_DOMAINS if d != current]
    if not candidates:
        return None
    corrupted["domain"] = rng.choice(candidates)
    return corrupted


def corrupt_drop_required_param(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """drop_required_param -> reason: missing_parameter (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    key = (corrupted.get("domain"), corrupted.get("intent"))
    param_name = REQUIRED_PARAM_BY_INTENT.get(key)
    params = corrupted.get("parameters") or {}
    if not param_name or param_name not in params:
        return None
    del params[param_name]
    corrupted["parameters"] = params
    return corrupted


def corrupt_inject_param(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """inject_param -> reason: hallucinated_parameter (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    params = dict(corrupted.get("parameters") or {})
    # parameter yang TIDAK diminta di source.text — sengaja generik & mencolok
    params["priority"] = "urgent"
    corrupted["parameters"] = params
    return corrupted


def corrupt_swap_action(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """swap_action -> reason: intent_mismatch (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    domain = corrupted.get("domain")
    current_action = corrupted.get("action")
    candidates = [a for a in ALT_ACTIONS_SAME_DOMAIN.get(domain, []) if a != current_action]
    if not candidates:
        return None
    corrupted["action"] = rng.choice(candidates)
    return corrupted


def corrupt_swap_intent_far(task_ir: dict, rng: random.Random) -> Optional[dict]:
    """swap_intent_far -> reason: unsupported_action (§38.3)"""
    corrupted = copy.deepcopy(task_ir)
    domain = corrupted.get("domain")
    far = FAR_DOMAIN_ACTIONS.get(domain)
    if not far:
        return None
    far_domain, far_action = far
    corrupted["domain"] = far_domain
    corrupted["action"] = far_action
    return corrupted


# Registry corruption_type -> (fungsi, reason) — dipakai generator.py
CORRUPTORS = {
    "target_mismatch": corrupt_swap_target,
    "domain_mismatch": corrupt_swap_domain,
    "missing_parameter": corrupt_drop_required_param,
    "hallucinated_parameter": corrupt_inject_param,
    "intent_mismatch": corrupt_swap_action,
    "unsupported_action": corrupt_swap_intent_far,
}
