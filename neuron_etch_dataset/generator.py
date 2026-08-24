"""
generator.py
============
Mengubah contoh di templates.py jadi sample dataset sesuai schema.py,
DIPISAH PER TASK-CATEGORY sesuai §29-§38.

Setiap generate_*_dataset() mengembalikan dict:
    { "task_category_1": [Sample, ...], "task_category_2": [...], ... }

Ada 2 builder generik:
- build_specialist_dataset()  : dipakai semua specialist (§7-13) yang
                                 formatnya seragam (input/intent/action/
                                 target/parameters), lihat MEDIA_EXAMPLES
                                 dkk di templates.py.
- generate_validator_dataset(): khusus, karena formatnya beda total
                                 (original/generated/label/reason, §38.1)
                                 dan sumbernya dari corruption pipeline (§38.3)
                                 terhadap Task IR valid milik specialist lain.

System & Router tetap punya generator sendiri karena polanya beda
(System pakai template+permutasi APPLICATIONS, Router pakai segments).
"""

from __future__ import annotations
import itertools
import random

from schemas import (
    RouterSample,
    RouterSegment,
    SpecialistSample,
    SpecialistOutput,
    SpecialistTarget,
    ValidatorSample,
)
import templates as tpl
from corruption import CORRUPTORS

RNG = random.Random(42)  # seed tetap supaya hasil reproducible


# ---------------------------------------------------------------------------
# GENERIC BUILDER — dipakai semua specialist (§7-13) berformat seragam
# ---------------------------------------------------------------------------

def build_specialist_dataset(
    domain: str,
    examples_by_category: dict[str, list[dict]],
    id_prefix: str,
) -> dict[str, list[SpecialistSample]]:
    """
    examples_by_category: { task_category: [ {input, intent, action,
        target?, parameters?, label?, note?, output? (None utk negative)}, ... ] }
    """
    result: dict[str, list[SpecialistSample]] = {cat: [] for cat in examples_by_category}
    counter = itertools.count(1)

    for task_category, examples in examples_by_category.items():
        for ex in examples:
            sample_id = f"{id_prefix}_{next(counter):06d}"

            # sample negative eksplisit: output=None (bukan domain ini sama sekali)
            if "output" in ex and ex["output"] is None:
                result[task_category].append(SpecialistSample(
                    id=sample_id,
                    input=ex["input"],
                    output=None,
                    task_category=task_category,
                    label=ex.get("label", "negative"),
                    metadata={"note": ex.get("note")},
                ))
                continue

            target = None
            if ex.get("target") is not None:
                target = SpecialistTarget(type=ex["target"]["type"], value=ex["target"].get("value"))

            result[task_category].append(SpecialistSample(
                id=sample_id,
                input=ex["input"],
                output=SpecialistOutput(
                    domain=domain,
                    intent=ex["intent"],
                    action=ex["action"],
                    target=target,
                    parameters=ex.get("parameters", {}),
                ),
                task_category=task_category,
                label=ex.get("label", "positive"),
                metadata={"note": ex.get("note")} if ex.get("note") else {},
            ))

    return result


# ---------------------------------------------------------------------------
# SYSTEM SPECIALIST DATASET (§31) — generator khusus (template + permutasi app)
# ---------------------------------------------------------------------------

def generate_system_dataset() -> dict[str, list[SpecialistSample]]:
    result: dict[str, list[SpecialistSample]] = {
        "application": [], "process": [], "filesystem": [], "shell": [],
        "hardware": [], "audio": [], "display": [], "network": [],
        "system_query": [], "ambiguous_negative": [],
    }
    counter = itertools.count(1)

    def new_id() -> str:
        return f"system_{next(counter):06d}"

    for template in tpl.SYSTEM_LAUNCH_TEMPLATES:
        for app in tpl.APPLICATIONS:
            text = template.format(app=app)
            result["application"].append(SpecialistSample(
                id=new_id(), input=text,
                output=SpecialistOutput(domain="system", intent="application_control", action="launch",
                                         target=SpecialistTarget(type="application", value=app), parameters={}),
                task_category="application", label="positive",
                metadata={"difficulty": "easy", "ambiguity": False, "requires_context": False},
            ))

    for template in tpl.SYSTEM_CLOSE_TEMPLATES:
        for app in tpl.APPLICATIONS:
            text = template.format(app=app)
            result["application"].append(SpecialistSample(
                id=new_id(), input=text,
                output=SpecialistOutput(domain="system", intent="application_control", action="close",
                                         target=SpecialistTarget(type="application", value=app), parameters={}),
                task_category="application", label="positive",
                metadata={"difficulty": "easy", "ambiguity": False, "requires_context": False},
            ))

    for template, reason in tpl.SYSTEM_HARD_NEGATIVE_TEMPLATES:
        for app in tpl.APPLICATIONS:
            text = template.format(app=app)
            is_query = "query" in reason
            result["application"].append(SpecialistSample(
                id=new_id(), input=text,
                output=SpecialistOutput(
                    domain="system",
                    intent="information_query" if is_query else "filesystem_search",
                    action="query" if is_query else "search",
                    target=SpecialistTarget(type="application", value=app), parameters={}),
                task_category="application", label="hard_negative",
                metadata={"difficulty": "hard", "ambiguity": False, "requires_context": False, "note": reason},
            ))

    for text in tpl.SYSTEM_NEGATIVE_INPUTS:
        result["ambiguous_negative"].append(SpecialistSample(
            id=new_id(), input=text, output=None,
            task_category="ambiguous_negative", label="negative",
            metadata={"difficulty": "easy", "note": "bukan domain system"},
        ))

    # process/filesystem/shell/hardware/audio/display/network/system_query:
    # masih stub (§31 TODO di templates.py) -> tetap [] sampai diisi

    return result


# ---------------------------------------------------------------------------
# ROUTER DATASET (§30) — generator khusus (pakai segments, bukan output)
# ---------------------------------------------------------------------------

def generate_router_dataset() -> dict[str, list[RouterSample]]:
    result: dict[str, list[RouterSample]] = {
        "single_intent": [], "multi_intent": [], "ambiguous": [], "implicit_intent": [],
        "domain_overlap_disambiguation": [], "negative_unknown": [],
        "context_dependent": [], "compound_command": [],
    }
    counter = itertools.count(1)

    def new_id() -> str:
        return f"router_{next(counter):06d}"

    for domain, texts in tpl.ROUTER_SINGLE_DOMAIN_EXAMPLES.items():
        for text in texts:
            result["single_intent"].append(RouterSample(
                id=new_id(), input=text, segments=[RouterSegment(text=text, domain=domain)],
                task_category="single_intent", metadata={"difficulty": "easy"},
            ))

    for item in tpl.ROUTER_MULTI_INTENT_EXAMPLES:
        segments = [RouterSegment(text=t, domain=d) for t, d in item["segments"]]
        result["multi_intent"].append(RouterSample(
            id=new_id(), input=item["input"], segments=segments,
            task_category="multi_intent", metadata={"difficulty": "medium"},
        ))

    for item in tpl.ROUTER_DOMAIN_OVERLAP_EXAMPLES:
        segments = [RouterSegment(text=t, domain=d) for t, d in item["segments"]]
        result["domain_overlap_disambiguation"].append(RouterSample(
            id=new_id(), input=item["input"], segments=segments,
            task_category="domain_overlap_disambiguation",
            metadata={"difficulty": "medium", "note": item.get("note")},
        ))

    for text in tpl.ROUTER_UNKNOWN_INPUTS:
        result["negative_unknown"].append(RouterSample(
            id=new_id(), input=text, segments=[RouterSegment(text=text, domain="unknown")],
            task_category="negative_unknown", metadata={"difficulty": "easy"},
        ))

    for cat_name, examples in [
        ("ambiguous", tpl.ROUTER_AMBIGUOUS_EXAMPLES),
        ("implicit_intent", tpl.ROUTER_IMPLICIT_INTENT_EXAMPLES),
        ("context_dependent", tpl.ROUTER_CONTEXT_DEPENDENT_EXAMPLES),
        ("compound_command", tpl.ROUTER_COMPOUND_COMMAND_EXAMPLES),
    ]:
        for item in examples:
            segments = [RouterSegment(text=t, domain=d) for t, d in item["segments"]]
            result[cat_name].append(RouterSample(
                id=new_id(), input=item["input"], segments=segments,
                task_category=cat_name, metadata={"difficulty": "medium", "note": item.get("note")},
            ))

    return result


# ---------------------------------------------------------------------------
# MEDIA / PERSONA / CODING / INFORMATION / MEMORY / PRODUCTIVITY (§32-§37)
# semua pakai build_specialist_dataset() generik
# ---------------------------------------------------------------------------

def generate_media_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("media", tpl.MEDIA_EXAMPLES, "media")


def generate_persona_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("persona", tpl.PERSONA_EXAMPLES, "persona")


def generate_coding_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("coding", tpl.CODING_EXAMPLES, "coding")


def generate_information_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("information", tpl.INFORMATION_EXAMPLES, "information")


def generate_memory_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("memory", tpl.MEMORY_EXAMPLES, "memory")


def generate_productivity_dataset() -> dict[str, list[SpecialistSample]]:
    return build_specialist_dataset("productivity", tpl.PRODUCTIVITY_EXAMPLES, "productivity")


# ---------------------------------------------------------------------------
# VALIDATOR DATASET (§38) — via Automated Corruption Pipeline (§38.3)
# ---------------------------------------------------------------------------

def generate_validator_dataset(
    source_specialist_data: dict[str, dict[str, list[SpecialistSample]]],
) -> dict[str, list[ValidatorSample]]:
    """
    source_specialist_data: { domain_name: { task_category: [SpecialistSample, ...] } }
    Ambil sample POSITIVE dari semua specialist yang sudah di-generate (system,
    media, persona, dst), lalu:
      - simpan sebagian apa adanya sebagai task_category "valid"
      - corrupt sebagian dengan tiap corruption_type di corruption.py (§38.3)

    Ini persis alur §38.3: "ambil valid dari dataset specialist yang sudah
    ada + generate corrupt via script" — bukan menulis pasangan dari nol.
    """
    result: dict[str, list[ValidatorSample]] = {
        "valid": [], "target_mismatch": [], "domain_mismatch": [],
        "missing_parameter": [], "hallucinated_parameter": [],
        "intent_mismatch": [], "unsupported_action": [],
        "parameter_mismatch": [], "contradiction": [], "ambiguous": [],
        # parameter_mismatch/contradiction/ambiguous dibiarkan kosong (manual only, §38.3)
    }
    counter = itertools.count(1)

    def new_id() -> str:
        return f"validator_{next(counter):06d}"

    # kumpulkan semua sample positive (label == "positive", output != None)
    # dari seluruh specialist yang sudah digenerate, jadi satu pool sumber
    source_pool: list[SpecialistSample] = []
    for domain_data in source_specialist_data.values():
        for samples in domain_data.values():
            for s in samples:
                if s.label == "positive" and s.output is not None:
                    source_pool.append(s)

    for sample in source_pool:
        task_ir = sample.output.to_dict()
        task_ir["domain"] = sample.output.domain  # pastikan konsisten

        # 1) simpan versi VALID apa adanya
        result["valid"].append(ValidatorSample(
            id=new_id(), original=sample.input, generated=task_ir,
            label="valid", task_category="valid", reason=None,
        ))

        # 2) coba semua corruption_type otomatis (§38.3) — kalau corruptor
        #    return None (mis. tidak ada target buat di-swap), skip aja
        for corruption_type, corrupt_fn in CORRUPTORS.items():
            corrupted_ir = corrupt_fn(task_ir, RNG)
            if corrupted_ir is None:
                continue
            result[corruption_type].append(ValidatorSample(
                id=new_id(), original=sample.input, generated=corrupted_ir,
                label="invalid", task_category=corruption_type, reason=corruption_type,
            ))

    return result
