"""
validator_chatml.py
=====================
Builder ChatML hierarchical (2 stage) untuk validator_core (§38).

BEDA dari 7 specialist: input validator bukan cuma kalimat user, tapi
PASANGAN (instruksi asli, Task IR yang mau dicek). Tugasnya klasifikasi,
bukan ekstraksi/generasi Task IR baru.

Stage 1 -- Valid/invalid classifier: cek apakah Task IR itu benar
    merepresentasikan instruksi asli. Output: {"label": "valid"|"invalid"}

Stage 2 -- (HANYA kalau stage1="invalid") klasifikasi 9-way alasan
    kenapa invalid. Output: {"reason": "..."}

"valid" TIDAK butuh stage2 sama sekali.
"""
from __future__ import annotations
import json

REASONS = ["target_mismatch", "domain_mismatch", "missing_parameter",
           "hallucinated_parameter", "intent_mismatch", "unsupported_action",
           "parameter_mismatch", "contradiction", "ambiguous"]

VALIDATOR_STAGE1_PROMPT = """Kamu adalah Validator. Kamu akan dikasih INSTRUKSI user dan TASK IR
(hasil interpretasi sistem atas instruksi itu, dalam format JSON).

Tugasmu HANYA menentukan apakah Task IR itu VALID (benar-benar
merepresentasikan instruksi) atau INVALID (ada kesalahan apapun --
domain salah, target salah, action salah, parameter kurang/nyasar/
nggak nyambung, atau kontradiktif).

Output HANYA JSON murni, tanpa markdown fence: {"label": "valid"|"invalid"}

Contoh:
Instruksi: "Buka spotify."
Task IR: {"domain": "system", "intent": "application_control", "action": "launch", "target": {"type": "application", "value": "spotify"}, "parameters": {}}
Output: {"label": "valid"}

Instruksi: "Buka spotify."
Task IR: {"domain": "media", "intent": "application_control", "action": "launch", "target": {"type": "application", "value": "spotify"}, "parameters": {}}
Output: {"label": "invalid"}"""


def build_stage1_messages(original: str, generated: dict, label: str) -> list[dict[str, str]]:
    user_content = f"Instruksi: \"{original}\"\nTask IR: {json.dumps(generated, ensure_ascii=False)}"
    return [
        {"role": "system", "content": VALIDATOR_STAGE1_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps({"label": label}, ensure_ascii=False)},
    ]


VALIDATOR_STAGE2_PROMPT = """Task IR ini SUDAH DIPASTIKAN invalid. Tentukan SALAH SATU alasan:

- target_mismatch (target.value salah, nggak sesuai yang disebut instruksi)
- domain_mismatch (domain salah total, harusnya domain lain)
- missing_parameter (ada parameter penting yang seharusnya ada tapi hilang)
- hallucinated_parameter (ada parameter yang TIDAK diminta instruksi, muncul dari mana aja)
- intent_mismatch (action salah, tapi domain/intent-nya masih benar)
- unsupported_action (domain DAN action-nya salah total, ke domain lain sama sekali)
- parameter_mismatch (isi parameter ada tapi nilainya nggak sesuai instruksi/nggak nyambung)
- contradiction (action-nya kebalikan dari yang diminta, mis. diminta nyalain tapi actionnya matiin)
- ambiguous (target/parameter isinya cuma placeholder generik kayak "itu"/"sesuatu", bukan nilai spesifik)

Output HANYA JSON murni, tanpa markdown fence: {"reason": "..."}

Contoh:
Instruksi: "Matiin bluetooth."
Task IR: {"domain": "system", "intent": "hardware_control", "action": "enable_device", "target": {"type": "device", "value": "bluetooth"}, "parameters": {}}
Output: {"reason": "contradiction"}"""


def build_stage2_messages(original: str, generated: dict, reason: str) -> list[dict[str, str]]:
    user_content = f"Instruksi: \"{original}\"\nTask IR: {json.dumps(generated, ensure_ascii=False)}"
    return [
        {"role": "system", "content": VALIDATOR_STAGE2_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps({"reason": reason}, ensure_ascii=False)},
    ]
