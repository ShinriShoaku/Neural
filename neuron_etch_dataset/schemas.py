"""
schemas.py
==========
Definisi struktur data dasar untuk dataset Neuron-Etch vNext.

Referensi ke dokumen arsitektur:
- §17  Canonical Task IR
- §29  Router Dataset (format)
- §40  Dataset Format (format umum semua specialist dataset)

Catatan: ini versi SIMPLE. Field yang ada di Task IR penuh (§17) seperti
`resolution`, `execution`, `confidence`, `risk`, `policy_decision`, `status`
belum semuanya diisi di sini — dataset training tidak butuh semua field
runtime itu, hanya bagian yang harus DIHASILKAN oleh model kecil (1B).

Field runtime (dihasilkan Python/Policy Engine saat eksekusi, bukan oleh
model) sengaja TIDAK dimasukkan ke dataset:
    execution, policy_decision, status, risk.effective
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Domain yang dikenali Router (§4)
DOMAINS = [
    "system",
    "media",
    "persona",
    "coding",
    "information",
    "memory",
    "productivity",
    "unknown",
]


@dataclass
class RouterSegment:
    """Satu segmen hasil segmentation Router (§5, §29)."""
    text: str
    domain: str  # salah satu dari DOMAINS


@dataclass
class RouterSample:
    """
    Satu sample dataset untuk router_core (§29).
    Router TIDAK perlu detail action — cukup segmentasi + domain.

    `task_category` = kategori komposisi dataset sesuai §30, mis.
    "single_intent", "multi_intent", "negative_unknown", dst.
    Field ini yang dipakai untuk MEMISAH dataset jadi file per-task,
    bukan cuma metadata biasa.
    """
    id: str
    input: str
    segments: list[RouterSegment]
    task_category: str  # single_intent | multi_intent | ambiguous | negative_unknown | ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "segments": [asdict(s) for s in self.segments],
            "task_category": self.task_category,
            "metadata": self.metadata,
        }


@dataclass
class SpecialistTarget:
    """Target dari sebuah action, mis. {"type": "application", "value": "foot"}"""
    type: str
    value: str


@dataclass
class SpecialistOutput:
    """Bagian `output` dari dataset specialist, sesuai §40."""
    domain: str
    intent: str
    action: str
    target: Optional[SpecialistTarget] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "domain": self.domain,
            "intent": self.intent,
            "action": self.action,
            "target": asdict(self.target) if self.target else None,
            "parameters": self.parameters,
        }
        return d


@dataclass
class SpecialistSample:
    """
    Satu sample dataset untuk specialist mana pun (§40).
    Dipakai untuk system/media/persona/coding/information/memory/productivity.

    Dua axis kategori yang BEDA, jangan ketuker:
    - task_category : kategori komposisi dataset sesuai §31-§37,
                       mis. "application", "process", "filesystem" (buat System).
                       Dipakai untuk MEMISAH dataset jadi file per-task.
    - label          : validitas sample itu sendiri — positive (task IR benar),
                        hard_negative (mirip lexical, beda intent, §42),
                        negative (bukan domain ini sama sekali, §41).
    """
    id: str
    input: str
    output: Optional[SpecialistOutput]  # None untuk negative/unknown sample
    task_category: str  # application | process | filesystem | ... (§31-37)
    context: dict[str, Any] = field(default_factory=dict)
    label: str = "positive"  # positive | negative | hard_negative | ambiguous
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "context": self.context,
            "output": self.output.to_dict() if self.output else None,
            "task_category": self.task_category,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class ValidatorSample:
    """
    Satu sample dataset untuk validator_core (§38).
    Format: pasangan original/generated/label/reason (§38.1), dihasilkan
    dari Task IR valid milik dataset specialist lain yang di-corrupt
    secara terprogram (§38.3) — atau dibiarkan valid apa adanya.

    task_category di sini = tipe corruption ("target_mismatch",
    "domain_mismatch", dst, lihat §38.3) atau "valid" untuk positive.
    """
    id: str
    original: str                 # teks asli, dari source.text task IR sumber
    generated: dict                # task IR (bisa valid, bisa hasil corrupt)
    label: str                    # "valid" | "invalid"
    task_category: str            # valid | target_mismatch | domain_mismatch | ...
    reason: Optional[str] = None  # None kalau valid, diisi kalau invalid (§38.1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original": self.original,
            "generated": self.generated,
            "label": self.label,
            "reason": self.reason,
            "task_category": self.task_category,
            "metadata": self.metadata,
        }
