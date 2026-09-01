"""
run_full_pipeline.py
======================
Orkestrasi PENUH sistem Liana: router -> specialist -> validator, dalam
SATU proses, pakai SATU base model yang dimuat sekali, dengan 9 adapter
LoRA yang di-switch on-the-fly (bukan load ulang model 9x -- jauh lebih
hemat memori & lebih cepat).

Alur per input:
    1. Router (2 stage) -> pecah kalimat jadi 1+ segment {domain, text}
    2. Untuk tiap segment yang domain-nya dikenali (bukan "unknown"):
       panggil specialist adapter yang sesuai (2 stage) -> Task IR
    3. Kalau Task IR berhasil dihasilkan: panggil validator adapter
       (2 stage) -> {label: valid/invalid, reason kalau invalid}
    4. Cetak semua hasilnya rapi ke terminal

Cara pakai:
    # mode interaktif (loop terus, ketik 'exit'/'quit' buat keluar)
    python run_full_pipeline.py --base-model-dir ./models/Qwen3.5-0.8B

    # mode single-shot
    python run_full_pipeline.py --base-model-dir ./models/Qwen3.5-0.8B \\
        --input "Buka spotify terus putar lagu Noah."

Semua path adapter punya default (lihat --help), tapi bisa dioverride
kalau nama folder adapter kamu beda:
    python run_full_pipeline.py --base-model-dir ./models/Qwen3.5-0.8B \\
        --router-adapter ./adapters/router_core_v7 \\
        --system-adapter ./adapters/system_core_v1 \\
        ... dst
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

# ---------------------------------------------------------------------------
# Import semua modul chatml (harus dijalankan dari folder training/, atau
# folder yang sama dengan file2 ini)
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)  # neuron_etch_dataset/, tempat generate_*_full.py
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PARENT_DIR)

import chatml_format as routerc
import system_chatml as sysc
import media_chatml as medc
import persona_chatml as perc
import coding_chatml as codc
import information_chatml as infc
import memory_chatml as memc
import productivity_chatml as prodc
import validator_chatml as valc

# ACTION -> INTENT lookup buat domain yang stage2-nya memprediksi action
# langsung (system/media/persona) -- disimpan di generate_*_full.py, bukan
# di *_chatml.py, jadi diimpor terpisah.
from generate_system_full import ACTION_TO_INTENT as SYSTEM_ACTION_TO_INTENT
from generate_media_full import ACTION_TO_INTENT as MEDIA_ACTION_TO_INTENT
from generate_persona_full import ACTION_TO_INTENT as PERSONA_ACTION_TO_INTENT

SPECIALIST_DOMAINS = ["system", "media", "persona", "coding", "information", "memory", "productivity"]


class AdapterNotLoadedError(Exception):
    """Dilempar kalau adapter yang mau dipakai belum/tidak lagi dimuat ke
    model (misal habis di-unload lewat API server). Dipakai supaya server
    bisa kasih respons rapi ("adapter X belum dimuat") alih-alih crash."""
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        super().__init__(f"Adapter {adapter_name!r} belum dimuat ke model.")


# ===========================================================================
# GENERATE & PARSE HELPER (dipakai semua adapter)
# ===========================================================================
def get_stop_token_ids(tokenizer) -> list[int]:
    """<|im_end|> HARUS jadi stop token eksplisit, kalau tidak model bisa
    lanjut generate abis JSON selesai (nge-halusinasi giliran baru)."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)
    return list(ids)


def generate(model, tokenizer, adapter_name: str, system_prompt: str, user_text: str,
             max_new_tokens: int = 120) -> str:
    loaded = getattr(model, "peft_config", {})
    if adapter_name not in loaded:
        raise AdapterNotLoadedError(adapter_name)
    model.set_adapter(adapter_name)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=get_stop_token_ids(tokenizer),
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(raw_text: str):
    text = raw_text.strip()
    for marker in ("<|im_end|>", "\nuser\n", "\nassistant\n", "<|im_start|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


# ===========================================================================
# ROUTER
# ===========================================================================
def run_router(model, tokenizer, text: str, max_new_tokens: int) -> list[dict]:
    """Return list[{"domain": str, "text": str}]. Selalu return minimal 1
    segment (fallback domain='unknown' kalau ada yang gagal parse)."""
    raw1 = generate(model, tokenizer, "router", routerc.ROUTER_STAGE1_PROMPT, text, max_new_tokens=40)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category == "ambiguous":
        return [{"domain": "unknown", "text": text}]

    if category in ("single_intent", "implicit_intent"):
        raw2 = generate(model, tokenizer, "router", routerc.ROUTER_STAGE2_SINGLE_PROMPT, text, max_new_tokens=40)
        parsed2, _ = parse_json(raw2)
        domain = parsed2.get("domain") if isinstance(parsed2, dict) else None
        return [{"domain": domain or "unknown", "text": text}]

    if category == "multi_intent":
        raw2 = generate(model, tokenizer, "router", routerc.ROUTER_STAGE2_MULTI_PROMPT, text,
                         max_new_tokens=max_new_tokens)
        parsed2, _ = parse_json(raw2)
        segments = parsed2.get("segments") if isinstance(parsed2, dict) else (
            parsed2 if isinstance(parsed2, list) else None)
        if isinstance(segments, list) and segments:
            out = []
            for seg in segments:
                if isinstance(seg, dict) and seg.get("domain") and seg.get("text"):
                    out.append({"domain": seg["domain"], "text": seg["text"]})
            if out:
                return out
        return [{"domain": "unknown", "text": text}]

    # stage1 gagal / category tidak dikenal -> fallback aman
    return [{"domain": "unknown", "text": text}]


# ===========================================================================
# SPECIALIST: SYSTEM
# ===========================================================================
def run_system(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "system", sysc.SYSTEM_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in sysc.CATEGORIES or category == "ambiguous_negative":
        return None

    raw2 = generate(model, tokenizer, "system", sysc.STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict) or "action" not in parsed2:
        return None

    action = parsed2.get("action")
    return {
        "domain": "system",
        "intent": SYSTEM_ACTION_TO_INTENT.get(action, "unknown"),
        "action": action,
        "target": parsed2.get("target"),
        "parameters": parsed2.get("parameters", {}),
    }


# ===========================================================================
# SPECIALIST: MEDIA
# ===========================================================================
def run_media(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "media", medc.MEDIA_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in medc.CATEGORIES or category == "negative":
        return None

    raw2 = generate(model, tokenizer, "media", medc.STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict) or "action" not in parsed2:
        return None

    action = parsed2.get("action")
    return {
        "domain": "media",
        "intent": MEDIA_ACTION_TO_INTENT.get(action, "unknown"),
        "action": action,
        "target": parsed2.get("target"),
        "parameters": parsed2.get("parameters", {}),
    }


# ===========================================================================
# SPECIALIST: PERSONA
# ===========================================================================
def run_persona(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "persona", perc.PERSONA_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in perc.CATEGORIES:
        return None
    group = perc.STAGE1_TO_STAGE2_GROUP.get(category)
    if group is None:  # negative
        return None

    raw2 = generate(model, tokenizer, "persona", perc.STAGE2_PROMPTS[group], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict) or "action" not in parsed2:
        return None

    action = parsed2.get("action")
    return {
        "domain": "persona",
        "intent": PERSONA_ACTION_TO_INTENT.get(action, "unknown"),
        "action": action,
        "target": parsed2.get("target"),
        "parameters": parsed2.get("parameters", {}),
    }


# ===========================================================================
# SPECIALIST: CODING
# ===========================================================================
def run_coding(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "coding", codc.CODING_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in codc.CATEGORIES or category == "negative":
        return None

    if category in codc.NO_STAGE2_CATEGORIES:  # review, test
        return {"domain": "coding", "intent": codc.CATEGORY_TO_INTENT[category],
                "action": codc.CATEGORY_TO_ACTION[category], "target": None, "parameters": {}}

    if category == "ambiguous":
        raw2 = generate(model, tokenizer, "coding", codc.STAGE2_PROMPTS["ambiguous"], text,
                         max_new_tokens=max_new_tokens)
        parsed2, _ = parse_json(raw2)
        action = parsed2.get("action") if isinstance(parsed2, dict) else None
        return {"domain": "coding", "intent": codc.CATEGORY_TO_INTENT["ambiguous"],
                "action": action, "target": None, "parameters": {}}

    raw2 = generate(model, tokenizer, "coding", codc.STAGE2_PROMPTS[category], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    parameters = parsed2.get("parameters", {}) if isinstance(parsed2, dict) else {}
    return {"domain": "coding", "intent": codc.CATEGORY_TO_INTENT[category],
            "action": codc.CATEGORY_TO_ACTION[category], "target": None, "parameters": parameters}


# ===========================================================================
# SPECIALIST: INFORMATION
# ===========================================================================
def run_information(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "information", infc.INFORMATION_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in infc.CATEGORIES or category == "negative":
        return None

    if category in infc.NO_STAGE2_CATEGORIES:  # ambiguous
        return {"domain": "information", "intent": infc.CATEGORY_TO_INTENT["ambiguous"],
                "action": "search", "target": infc.AMBIGUOUS_CONSTANT_OUTPUT["target"],
                "parameters": infc.AMBIGUOUS_CONSTANT_OUTPUT["parameters"]}

    raw2 = generate(model, tokenizer, "information", infc.STAGE2_PROMPTS[category], text,
                     max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict):
        return None
    return {"domain": "information", "intent": infc.CATEGORY_TO_INTENT[category],
            "action": infc.CATEGORY_TO_ACTION[category],
            "target": parsed2.get("target"), "parameters": parsed2.get("parameters", {})}


# ===========================================================================
# SPECIALIST: MEMORY
# ===========================================================================
def run_memory(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "memory", memc.MEMORY_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in memc.CATEGORIES or category == "negative":
        return None

    group = memc.STAGE1_TO_STAGE2_GROUP.get(category)
    if group is None:  # ambiguous
        return {"domain": "memory", "intent": memc.CATEGORY_TO_INTENT["ambiguous"],
                "action": "remember", "target": None,
                "parameters": memc.AMBIGUOUS_CONSTANT_OUTPUT["parameters"]}

    raw2 = generate(model, tokenizer, "memory", memc.STAGE2_PROMPTS[group], text, max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    parameters = parsed2.get("parameters", {}) if isinstance(parsed2, dict) else {}
    return {"domain": "memory", "intent": memc.CATEGORY_TO_INTENT[category],
            "action": memc.CATEGORY_TO_ACTION[category], "target": None, "parameters": parameters}


# ===========================================================================
# SPECIALIST: PRODUCTIVITY
# ===========================================================================
def _productivity_build_full_output(category: str, payload: dict) -> dict:
    if category == "todo":
        return {"intent": "todo", "action": payload.get("action"), "target": payload.get("target"),
                "parameters": {}}
    if category == "update_delete":
        return {"intent": payload.get("intent"), "action": payload.get("action"),
                "target": payload.get("target"), "parameters": payload.get("parameters", {})}
    return {"intent": prodc.CATEGORY_TO_INTENT[category], "action": prodc.CATEGORY_TO_ACTION[category],
            "target": payload.get("target"), "parameters": payload.get("parameters", {})}


def run_productivity(model, tokenizer, text: str, max_new_tokens: int) -> dict | None:
    raw1 = generate(model, tokenizer, "productivity", prodc.PRODUCTIVITY_STAGE1_PROMPT, text, max_new_tokens=30)
    parsed1, _ = parse_json(raw1)
    category = parsed1.get("category") if isinstance(parsed1, dict) else None
    if category not in prodc.CATEGORIES or category == "negative":
        return None

    if category in prodc.NO_STAGE2_CATEGORIES:  # ambiguous
        out = dict(prodc.AMBIGUOUS_CONSTANT_OUTPUT)
        out["domain"] = "productivity"
        return out

    raw2 = generate(model, tokenizer, "productivity", prodc.STAGE2_PROMPTS[category], text,
                     max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    if not isinstance(parsed2, dict):
        return None
    full = _productivity_build_full_output(category, parsed2)
    full["domain"] = "productivity"
    return full


SPECIALIST_RUNNERS = {
    "system": run_system, "media": run_media, "persona": run_persona,
    "coding": run_coding, "information": run_information, "memory": run_memory,
    "productivity": run_productivity,
}


# ===========================================================================
# VALIDATOR
# ===========================================================================
def run_validator(model, tokenizer, original: str, task_ir: dict, max_new_tokens: int) -> dict:
    user_content = f"Instruksi: \"{original}\"\nTask IR: {json.dumps(task_ir, ensure_ascii=False)}"

    raw1 = generate(model, tokenizer, "validator", valc.VALIDATOR_STAGE1_PROMPT, user_content, max_new_tokens=20)
    parsed1, _ = parse_json(raw1)
    label = parsed1.get("label") if isinstance(parsed1, dict) else None
    if label not in ("valid", "invalid"):
        return {"label": "unknown", "reason": None}

    if label == "valid":
        return {"label": "valid", "reason": None}

    raw2 = generate(model, tokenizer, "validator", valc.VALIDATOR_STAGE2_PROMPT, user_content,
                     max_new_tokens=max_new_tokens)
    parsed2, _ = parse_json(raw2)
    reason = parsed2.get("reason") if isinstance(parsed2, dict) else None
    return {"label": "invalid", "reason": reason}


# ===========================================================================
# ORKESTRASI PENUH
# ===========================================================================
def run_full_pipeline(model, tokenizer, text: str, max_new_tokens: int = 120) -> dict:
    t0 = time.time()

    if "router" not in getattr(model, "peft_config", {}):
        return {"input": text, "segments": [], "elapsed_sec": 0.0,
                "error": "Adapter 'router' belum dimuat -- pipeline tidak bisa jalan tanpa router."}

    try:
        segments = run_router(model, tokenizer, text, max_new_tokens)
    except AdapterNotLoadedError as e:
        return {"input": text, "segments": [], "elapsed_sec": round(time.time() - t0, 2),
                "error": f"Router gagal jalan: {e}"}

    results = []
    for seg in segments:
        domain, seg_text = seg["domain"], seg["text"]
        entry = {"domain": domain, "text": seg_text, "task_ir": None, "validation": None, "note": None}

        if domain == "unknown":
            entry["note"] = "router tidak bisa nentuin domain (ambiguous/gagal parse)"
        elif domain not in SPECIALIST_RUNNERS:
            entry["note"] = f"domain {domain!r} tidak dikenal sistem"
        else:
            try:
                task_ir = SPECIALIST_RUNNERS[domain](model, tokenizer, seg_text, max_new_tokens)
            except AdapterNotLoadedError as e:
                entry["note"] = f"adapter specialist {domain!r} belum dimuat -- segment ini dilewati"
                task_ir = None
                results.append(entry)
                continue

            if task_ir is None:
                entry["note"] = "specialist bilang ini bukan command actionable (negative/ambiguous)"
            else:
                entry["task_ir"] = task_ir
                try:
                    entry["validation"] = run_validator(model, tokenizer, seg_text, task_ir, max_new_tokens)
                except AdapterNotLoadedError:
                    entry["validation"] = None
                    entry["note"] = "Task IR berhasil dibuat, tapi adapter validator belum dimuat (belum divalidasi)"

        results.append(entry)

    return {"input": text, "segments": results, "elapsed_sec": round(time.time() - t0, 2), "error": None}


# ===========================================================================
# CETAK HASIL
# ===========================================================================
def print_result(result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"INPUT: {result['input']}")
    print(f"{'=' * 70}")
    if result.get("error"):
        print(f"\033[91mERROR: {result['error']}\033[0m")
        return
    for i, seg in enumerate(result["segments"], 1):
        print(f"\n[Segment {i}] domain={seg['domain']!r}")
        print(f"  teks: {seg['text']!r}")
        if seg["note"]:
            print(f"  -> {seg['note']}")
            continue
        print(f"  Task IR: {json.dumps(seg['task_ir'], ensure_ascii=False, indent=2)}")
        v = seg["validation"]
        if v["label"] == "valid":
            print(f"  Validator: \033[92mVALID\033[0m")
        elif v["label"] == "invalid":
            print(f"  Validator: \033[91mINVALID\033[0m (alasan: {v['reason']})")
        else:
            print(f"  Validator: tidak bisa menentukan (parsing gagal)")
    print(f"\n(selesai dalam {result['elapsed_sec']}s)")


# ===========================================================================
# LOAD MODEL + SEMUA ADAPTER
# ===========================================================================
ADAPTER_NAMES = ["router", "system", "media", "persona", "coding",
                  "information", "memory", "productivity", "validator"]


def load_pipeline(args):
    print(f"Loading base model dari {args.base_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    adapter_paths = {
        "router": args.router_adapter, "system": args.system_adapter,
        "media": args.media_adapter, "persona": args.persona_adapter,
        "coding": args.coding_adapter, "information": args.information_adapter,
        "memory": args.memory_adapter, "productivity": args.productivity_adapter,
        "validator": args.validator_adapter,
    }

    first_name = ADAPTER_NAMES[0]
    print(f"Loading adapter {first_name!r} dari {adapter_paths[first_name]} ...")
    model = PeftModel.from_pretrained(base_model, adapter_paths[first_name], adapter_name=first_name)

    for name in ADAPTER_NAMES[1:]:
        print(f"Loading adapter {name!r} dari {adapter_paths[name]} ...")
        model.load_adapter(adapter_paths[name], adapter_name=name)

    model.eval()
    print("Semua adapter dimuat. Siap dipakai.\n")
    return model, tokenizer


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--router-adapter", default="./adapters/router_core_v7")
    parser.add_argument("--system-adapter", default="./adapters/system_core_v1")
    parser.add_argument("--media-adapter", default="./adapters/media_core_v1")
    parser.add_argument("--persona-adapter", default="./adapters/persona_core_v1")
    parser.add_argument("--coding-adapter", default="./adapters/coding_core_v1")
    parser.add_argument("--information-adapter", default="./adapters/information_core_v1")
    parser.add_argument("--memory-adapter", default="./adapters/memory_core_v1")
    parser.add_argument("--productivity-adapter", default="./adapters/productivity_core_v1")
    parser.add_argument("--validator-adapter", default="./adapters/validator_core_v1")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--input", default=None, help="Mode single-shot: satu kalimat, langsung keluar setelah selesai.")
    parser.add_argument("--json", action="store_true", help="Cetak hasil sebagai JSON mentah (buat dipakai skrip lain).")
    args = parser.parse_args()

    for name, path in [
        ("router", args.router_adapter), ("system", args.system_adapter),
        ("media", args.media_adapter), ("persona", args.persona_adapter),
        ("coding", args.coding_adapter), ("information", args.information_adapter),
        ("memory", args.memory_adapter), ("productivity", args.productivity_adapter),
        ("validator", args.validator_adapter),
    ]:
        if not os.path.isdir(path):
            print(f"PERINGATAN: folder adapter {name!r} tidak ketemu di {path!r}. "
                  f"Cek lagi --{name}-adapter atau nama folder adapter kamu.")

    model, tokenizer = load_pipeline(args)

    if args.input is not None:
        result = run_full_pipeline(model, tokenizer, args.input, args.max_new_tokens)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
        return

    print("Mode interaktif. Ketik kalimat, tekan Enter. Ketik 'exit' atau 'quit' buat keluar.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa.")
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            print("Sampai jumpa.")
            break
        result = run_full_pipeline(model, tokenizer, text, args.max_new_tokens)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)


if __name__ == "__main__":
    main()
