"""
chatml_format.py
=================
Mengubah satu baris dataset (dari output/<domain>/*.jsonl, hasil generator
di root project) jadi list message ChatML [{"role", "content"}, ...] siap
ditokenisasi lewat tokenizer.apply_chat_template() (§65.2, §65.4).

Kenapa pakai messages list, bukan nulis manual string "<|im_start|>...":
tokenizer Qwen3.5 yang tahu persis token spesial & whitespace yang benar
lewat chat_template bawaan (Jinja). Nulis manual berisiko mismatch kalau
versi tokenizer beda. `enable_thinking=False` (§65.3) dan `tools=None`
(§65.4) dipassing ke apply_chat_template() nanti di build_training_data.py.

Tiga builder, satu per jenis adapter:
    build_router_messages()      -> §66.1
    build_specialist_messages()  -> §67.1 (generic pattern semua 7 specialist)
    build_validator_messages()   -> §68 (Tier 2 semantic validator)

CATATAN — bagian yang aku ASUMSIKAN karena tidak dirinci eksplisit di
dokumen (ditandai jelas di tiap tempat):
  1. confidence.score untuk specialist: dokumen bilang confidence HARUS
     dikalibrasi pakai validation set (§48), bukan diterima mentah dari
     model — tapi untuk *label training* kita tetap butuh angka target.
     Heuristik di bawah: positive/easy=0.97, positive/butuh context atau
     medium=0.85, hard_negative=0.90 (tetap yakin, cuma actionnya beda),
     ambiguous=0.45 (rendah, sesuai §48 "low confidence -> memang sering
     ambiguous"). Ganti manual di kolom `metadata.confidence_override`
     pada dataset kalau kamu mau nilai spesifik per-sample.
  2. Skema untuk sample "negative" (output=None, §41 wrong_domain dkk):
     dokumen tidak kasih skema JSON eksplisit untuk kasus ini di level
     specialist (§44 cuma kasih contoh untuk ambiguous). Aku pakai
     {"status": "rejected", "reason": "<task_category>"} — konsisten
     sama pola {"status": "ambiguous", "uncertainties": [...]} di §44.
     SESUAIKAN kalau kamu sudah punya skema resmi buat ini.
  3. router context (last_domain/last_action/session_turn_count, §66.1):
     dataset generator saat ini belum menyimpan context asli per-sample
     (khususnya buat kategori context_dependent yang PALING butuh ini).
     Default di bawah: "none"/"none"/1. Begitu kamu isi context asli di
     dataset (field baru di RouterSample.metadata), update read_context()
     di bawah supaya kepakai.
"""

from __future__ import annotations
import json
from typing import Any

from capabilities import get_vocab_strings


# ---------------------------------------------------------------------------
# ROUTER (§66.1)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """Kamu adalah Router untuk sistem asisten Liana. Tugasmu HANYA:
1. Segmentasi kalimat user menjadi satu atau lebih segment.
2. Klasifikasikan tiap segment ke salah satu domain berikut:
   system, media, persona, coding, information, memory, productivity, unknown
3. Deteksi apakah ada domain overlap (lihat daftar overlap di bawah).
4. Berikan confidence score (0.0-1.0) untuk tiap segment.

Kamu TIDAK menentukan action, target, atau parameter detail.
Kamu TIDAK menjawab pertanyaan user secara langsung.
Output HANYA JSON, tanpa penjelasan, tanpa markdown fence.

Overlap rules yang harus kamu ketahui (lihat §16.2 untuk detail penuh):
- Aplikasi media sebagai TARGET murni -> system
- Aplikasi media sebagai KONTEKS dari playback -> media
- "matikan suara/mute" -> system.audio; "matikan musik/stop lagu" -> media

Context sesi (Tier A, §6.2):
last_domain: {last_domain}
last_action: {last_action}
session_turn_count: {session_turn_count}

Schema output:
{{
  "segments": [
    {{
      "id": "seg_NNN",
      "text": "...",
      "domain": "...",
      "confidence": 0.0,
      "overlap_hint": {{
        "detected": false,
        "candidate_domains": [],
        "resolution_rule": null
      }}
    }}
  ]
}}"""


def _router_segment_confidence(task_category: str) -> float:
    # ASUMSI (lihat catatan di kepala file) — sesuaikan kalau kamu punya angka lain
    return {
        "single_intent": 0.96,
        "multi_intent": 0.93,
        "domain_overlap_disambiguation": 0.88,
        "compound_command": 0.90,
        "negative_unknown": 0.85,   # yakin bahwa ini "unknown", makanya confidence tetap tinggi
        "ambiguous": 0.45,
        "implicit_intent": 0.65,
        "context_dependent": 0.55,
    }.get(task_category, 0.7)


def build_router_messages(row: dict[str, Any],
                           last_domain: str = "none",
                           last_action: str = "none",
                           session_turn_count: int = 1) -> list[dict[str, str]]:
    system_prompt = ROUTER_SYSTEM_PROMPT.format(
        last_domain=last_domain, last_action=last_action, session_turn_count=session_turn_count,
    )

    task_category = row["task_category"]
    confidence = _router_segment_confidence(task_category)

    segments_out = []
    for i, seg in enumerate(row["segments"], start=1):
        overlap_hint = {"detected": False, "candidate_domains": [], "resolution_rule": None}
        if task_category == "domain_overlap_disambiguation":
            note = (row.get("metadata") or {}).get("note", "")
            # note formatnya "... resolution_rule=xxx ..." (lihat templates.py) -> parse ringan
            rule = None
            if "resolution_rule=" in note:
                rule = note.split("resolution_rule=")[1].split(",")[0].split(")")[0].strip()
            overlap_hint = {
                "detected": True,
                "candidate_domains": [seg["domain"], "system" if seg["domain"] != "system" else "media"],
                "resolution_rule": rule,
            }
        segments_out.append({
            "id": f"seg_{i:03d}",
            "text": seg["text"],
            "domain": seg["domain"],
            "confidence": confidence,
            "overlap_hint": overlap_hint,
        })

    assistant_json = json.dumps({"segments": segments_out}, ensure_ascii=False)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": assistant_json},
    ]


# ---------------------------------------------------------------------------
# SPECIALIST (§67.1, generic — dipakai semua 7 domain)
# ---------------------------------------------------------------------------

SPECIALIST_SYSTEM_PROMPT = """Kamu adalah Specialist domain "{domain}" untuk sistem Liana.
Tugasmu: ubah teks user menjadi Task IR JSON sesuai skema di bawah.
Kamu TIDAK menjawab user secara langsung.
Kamu TIDAK menghasilkan kode, dialog, atau konten panjang apa pun -
itu tugas model 8B.

Intent yang valid: {intent_list}
Action yang valid: {action_list}
Target type yang valid: {target_type_list}

Jika informasi tidak cukup untuk mengisi field wajib, JANGAN mengarang.
Set status menjadi "ambiguous" dan isi "uncertainties".

Context (Tier B, §6.2 - hanya diisi jika relevan):
{specialist_context}

Schema output:
{{
  "domain": "{domain}",
  "intent": "...",
  "action": "...",
  "target": {{ "type": "...", "value": "..." }},
  "parameters": {{}},
  "confidence": {{ "score": 0.0, "level": "...", "uncertainties": [] }}
}}

Output HANYA JSON, tanpa penjelasan, tanpa markdown fence."""


def _confidence_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _specialist_confidence(label: str, metadata: dict) -> float:
    override = (metadata or {}).get("confidence_override")
    if override is not None:
        return float(override)
    if label == "positive":
        return 0.85 if metadata.get("requires_context") or metadata.get("difficulty") == "medium" else 0.97
    if label == "hard_negative":
        return 0.90
    if label == "ambiguous":
        return 0.45
    return 0.5  # negative / fallback


def build_specialist_messages(domain: str, row: dict[str, Any]) -> list[dict[str, str]]:
    intent_list, action_list, target_type_list = get_vocab_strings(domain)
    context_str = json.dumps(row.get("context") or {}, ensure_ascii=False) if row.get("context") else "(tidak ada)"

    system_prompt = SPECIALIST_SYSTEM_PROMPT.format(
        domain=domain, intent_list=intent_list, action_list=action_list,
        target_type_list=target_type_list, specialist_context=context_str,
    )

    label = row.get("label", "positive")
    metadata = row.get("metadata") or {}
    output = row.get("output")

    if output is None:
        # negative sample (§41) — ASUMSI skema, lihat catatan di kepala file
        assistant_obj = {"status": "rejected", "reason": row.get("task_category", "wrong_domain")}
    elif label == "ambiguous":
        # §44 hallucination-guard pattern: field yang None -> masuk uncertainties
        uncertainties = []
        target = output.get("target")
        if target is not None and target.get("value") is None:
            uncertainties.append("target")
        for k, v in (output.get("parameters") or {}).items():
            if v is None:
                uncertainties.append(f"parameters.{k}")
        if not uncertainties:
            uncertainties = ["target"]
        assistant_obj = {
            "domain": domain,
            "intent": output["intent"],
            "action": output["action"],
            "status": "ambiguous",
            "uncertainties": uncertainties,
            "confidence": {"score": 0.45, "level": "low", "uncertainties": uncertainties},
        }
    else:
        score = _specialist_confidence(label, metadata)
        assistant_obj = {
            "domain": domain,
            "intent": output["intent"],
            "action": output["action"],
            "target": output.get("target"),
            "parameters": output.get("parameters") or {},
            "confidence": {"score": score, "level": _confidence_level(score), "uncertainties": []},
        }

    assistant_json = json.dumps(assistant_obj, ensure_ascii=False)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": assistant_json},
    ]


# ---------------------------------------------------------------------------
# VALIDATOR TIER 2 (§68)
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """Kamu adalah Validator semantik untuk sistem Liana.
Tugasmu: bandingkan teks asli user dengan Task IR yang dihasilkan
Specialist, lalu tentukan apakah interpretasinya masuk akal.

Kamu HANYA mengecek hal yang TIDAK bisa dicek string-matching sederhana:
- intent_mismatch (makna berbeda meski string mirip)
- contradiction (bertentangan dengan task lain dalam graph)
- ambiguous (confidence borderline, makna ganda)

Kamu TIDAK mengecek typo/string mismatch - itu sudah ditangani Tier 1.

Schema output:
{
  "valid": true/false,
  "reason": "..." atau null,
  "confidence": 0.0
}

Output HANYA JSON."""


def build_validator_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        f'Original: "{row["original"]}"\n'
        f'Generated: {json.dumps(row["generated"], ensure_ascii=False)}\n'
        f'Task lain dalam graph (jika ada): []'
    )

    is_valid = row["label"] == "valid"
    assistant_obj = {
        "valid": is_valid,
        "reason": None if is_valid else row.get("reason"),
        "confidence": 0.95 if is_valid else 0.90,
    }
    assistant_json = json.dumps(assistant_obj, ensure_ascii=False)

    return [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_json},
    ]
