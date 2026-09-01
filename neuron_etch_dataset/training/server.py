"""
Liana Pipeline FastAPI Server
=============================

Web UI untuk:
- Load 1 base model sekali.
- Load adapter LoRA secara modular.
- Enable/disable adapter per domain.
- Menjalankan router -> specialist -> validator.
- Menampilkan log pipeline secara realtime via polling.
- Reload model/adapters tanpa restart server.

Contoh:
    python server.py --host 0.0.0.0 --port 8000

UI:
    http://127.0.0.1:8000

Struktur adapter default mengikuti run_full_pipeline.py:
    router
    system
    media
    persona
    coding
    information
    memory
    productivity
    validator
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# PATH / IMPORT
# ============================================================================

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PARENT_DIR))

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

import chatml_format as routerc
import system_chatml as sysc
import media_chatml as medc
import persona_chatml as perc
import coding_chatml as codc
import information_chatml as infc
import memory_chatml as memc
import productivity_chatml as prodc
import validator_chatml as valc

from generate_system_full import ACTION_TO_INTENT as SYSTEM_ACTION_TO_INTENT
from generate_media_full import ACTION_TO_INTENT as MEDIA_ACTION_TO_INTENT
from generate_persona_full import ACTION_TO_INTENT as PERSONA_ACTION_TO_INTENT


# ============================================================================
# CONFIG
# ============================================================================

ADAPTER_NAMES = [
    "router",
    "system",
    "media",
    "persona",
    "coding",
    "information",
    "memory",
    "productivity",
    "validator",
]

SPECIALIST_DOMAINS = [
    "system",
    "media",
    "persona",
    "coding",
    "information",
    "memory",
    "productivity",
]

DEFAULT_ADAPTERS = {
    "router": "./adapters/router_core_v7",
    "system": "./adapters/system_core_v1",
    "media": "./adapters/media_core_v1",
    "persona": "./adapters/persona_core_v1",
    "coding": "./adapters/coding_core_v1",
    "information": "./adapters/information_core_v1",
    "memory": "./adapters/memory_core_v1",
    "productivity": "./adapters/productivity_core_v1",
    "validator": "./adapters/validator_core_v1",
}

# Adapter yang aktif secara default.
DEFAULT_ENABLED = {
    name: True for name in ADAPTER_NAMES
}


# ============================================================================
# LOGGING UNTUK UI
# ============================================================================

class UILogger:
    def __init__(self, max_lines: int = 1000):
        self._lock = threading.Lock()
        self._lines: list[dict[str, Any]] = []
        self._sequence = 0
        self.max_lines = max_lines

    def add(self, message: str, level: str = "INFO") -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            item = {
                "id": self._sequence,
                "time": time.strftime("%H:%M:%S"),
                "level": level.upper(),
                "message": str(message),
            }
            self._lines.append(item)

            if len(self._lines) > self.max_lines:
                self._lines = self._lines[-self.max_lines:]

            return item

    def get(self, after: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [x for x in self._lines if x["id"] > after]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


log = UILogger()


# Redirect stdout/stderr dari proses inference ke UI.
class StreamToLogger:
    def __init__(self, level: str = "INFO"):
        self.level = level

    def write(self, text: str):
        text = text.rstrip()
        if text:
            for line in text.splitlines():
                if line.strip():
                    log.add(line, self.level)

    def flush(self):
        pass


# ============================================================================
# STATE
# ============================================================================

class PipelineState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.base_model_dir: str | None = None
        self.adapter_paths: dict[str, str] = dict(DEFAULT_ADAPTERS)
        self.enabled: dict[str, bool] = dict(DEFAULT_ENABLED)

        self.loaded_adapters: list[str] = []
        self.loading = False
        self.ready = False
        self.error: str | None = None

        # Generation tidak boleh berjalan bersamaan pada model yang sama.
        self.inference_lock = threading.Lock()

        self.load_lock = threading.Lock()


state = PipelineState()


# ============================================================================
# GENERATE / PARSE
# ============================================================================

def get_stop_token_ids(tokenizer) -> list[int]:
    ids = set()

    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        ids.add(im_end_id)

    return list(ids)


def generate(
    model,
    tokenizer,
    adapter_name: str,
    system_prompt: str,
    user_text: str,
    max_new_tokens: int = 120,
) -> str:

    if adapter_name not in state.loaded_adapters:
        raise RuntimeError(
            f"Adapter '{adapter_name}' tidak aktif/terload."
        )

    model.set_adapter(adapter_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=get_stop_token_ids(tokenizer),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def parse_json(raw_text: str):
    text = raw_text.strip()

    for marker in (
        "<|im_end|>",
        "\nuser\n",
        "\nassistant\n",
        "<|im_start|>",
    ):
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

    except json.JSONDecodeError as exc:
        return None, str(exc)


# ============================================================================
# ROUTER
# ============================================================================

def run_router(model, tokenizer, text: str, max_new_tokens: int):
    raw1 = generate(
        model,
        tokenizer,
        "router",
        routerc.ROUTER_STAGE1_PROMPT,
        text,
        max_new_tokens=40,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[ROUTER/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[ROUTER/STAGE1] JSON parse error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    log.add(f"[ROUTER] category={category!r}")

    if category == "ambiguous":
        return [{"domain": "unknown", "text": text}]

    if category in ("single_intent", "implicit_intent"):
        raw2 = generate(
            model,
            tokenizer,
            "router",
            routerc.ROUTER_STAGE2_SINGLE_PROMPT,
            text,
            max_new_tokens=40,
        )

        parsed2, error2 = parse_json(raw2)

        log.add(f"[ROUTER/STAGE2] raw={raw2}")

        if error2:
            log.add(
                f"[ROUTER/STAGE2] JSON parse error: {error2}",
                "WARNING",
            )

        domain = (
            parsed2.get("domain")
            if isinstance(parsed2, dict)
            else None
        )

        return [
            {
                "domain": domain or "unknown",
                "text": text,
            }
        ]

    if category == "multi_intent":
        raw2 = generate(
            model,
            tokenizer,
            "router",
            routerc.ROUTER_STAGE2_MULTI_PROMPT,
            text,
            max_new_tokens=max_new_tokens,
        )

        parsed2, error2 = parse_json(raw2)

        log.add(f"[ROUTER/STAGE2-MULTI] raw={raw2}")

        if error2:
            log.add(
                f"[ROUTER/STAGE2-MULTI] JSON parse error: {error2}",
                "WARNING",
            )

        segments = (
            parsed2.get("segments")
            if isinstance(parsed2, dict)
            else (
                parsed2
                if isinstance(parsed2, list)
                else None
            )
        )

        if isinstance(segments, list) and segments:
            out = []

            for seg in segments:
                if (
                    isinstance(seg, dict)
                    and seg.get("domain")
                    and seg.get("text")
                ):
                    out.append(
                        {
                            "domain": seg["domain"],
                            "text": seg["text"],
                        }
                    )

            if out:
                return out

        return [{"domain": "unknown", "text": text}]

    return [{"domain": "unknown", "text": text}]


# ============================================================================
# SPECIALIST
# ============================================================================

def run_system(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "system",
        sysc.SYSTEM_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[SYSTEM/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[SYSTEM] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in sysc.CATEGORIES or category == "ambiguous_negative":
        return None

    raw2 = generate(
        model,
        tokenizer,
        "system",
        sysc.STAGE2_PROMPTS[category],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[SYSTEM/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[SYSTEM] JSON error: {error2}", "WARNING")

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


def run_media(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "media",
        medc.MEDIA_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[MEDIA/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[MEDIA] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in medc.CATEGORIES or category == "negative":
        return None

    raw2 = generate(
        model,
        tokenizer,
        "media",
        medc.STAGE2_PROMPTS[category],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[MEDIA/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[MEDIA] JSON error: {error2}", "WARNING")

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


def run_persona(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "persona",
        perc.PERSONA_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[PERSONA/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[PERSONA] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in perc.CATEGORIES:
        return None

    group = perc.STAGE1_TO_STAGE2_GROUP.get(category)

    if group is None:
        return None

    raw2 = generate(
        model,
        tokenizer,
        "persona",
        perc.STAGE2_PROMPTS[group],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[PERSONA/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[PERSONA] JSON error: {error2}", "WARNING")

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


def run_coding(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "coding",
        codc.CODING_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[CODING/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[CODING] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in codc.CATEGORIES or category == "negative":
        return None

    if category in codc.NO_STAGE2_CATEGORIES:
        return {
            "domain": "coding",
            "intent": codc.CATEGORY_TO_INTENT[category],
            "action": codc.CATEGORY_TO_ACTION[category],
            "target": None,
            "parameters": {},
        }

    if category == "ambiguous":
        raw2 = generate(
            model,
            tokenizer,
            "coding",
            codc.STAGE2_PROMPTS["ambiguous"],
            text,
            max_new_tokens=max_new_tokens,
        )

        parsed2, error2 = parse_json(raw2)

        log.add(f"[CODING/STAGE2] raw={raw2}")

        if error2:
            log.add(f"[CODING] JSON error: {error2}", "WARNING")

        action = parsed2.get("action") if isinstance(parsed2, dict) else None

        return {
            "domain": "coding",
            "intent": codc.CATEGORY_TO_INTENT["ambiguous"],
            "action": action,
            "target": None,
            "parameters": {},
        }

    raw2 = generate(
        model,
        tokenizer,
        "coding",
        codc.STAGE2_PROMPTS[category],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[CODING/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[CODING] JSON error: {error2}", "WARNING")

    parameters = (
        parsed2.get("parameters", {})
        if isinstance(parsed2, dict)
        else {}
    )

    return {
        "domain": "coding",
        "intent": codc.CATEGORY_TO_INTENT[category],
        "action": codc.CATEGORY_TO_ACTION[category],
        "target": None,
        "parameters": parameters,
    }


def run_information(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "information",
        infc.INFORMATION_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[INFORMATION/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[INFORMATION] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in infc.CATEGORIES or category == "negative":
        return None

    if category in infc.NO_STAGE2_CATEGORIES:
        return {
            "domain": "information",
            "intent": infc.CATEGORY_TO_INTENT["ambiguous"],
            "action": "search",
            "target": infc.AMBIGUOUS_CONSTANT_OUTPUT["target"],
            "parameters": infc.AMBIGUOUS_CONSTANT_OUTPUT["parameters"],
        }

    raw2 = generate(
        model,
        tokenizer,
        "information",
        infc.STAGE2_PROMPTS[category],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[INFORMATION/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[INFORMATION] JSON error: {error2}", "WARNING")

    if not isinstance(parsed2, dict):
        return None

    return {
        "domain": "information",
        "intent": infc.CATEGORY_TO_INTENT[category],
        "action": infc.CATEGORY_TO_ACTION[category],
        "target": parsed2.get("target"),
        "parameters": parsed2.get("parameters", {}),
    }


def run_memory(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "memory",
        memc.MEMORY_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[MEMORY/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[MEMORY] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in memc.CATEGORIES or category == "negative":
        return None

    group = memc.STAGE1_TO_STAGE2_GROUP.get(category)

    if group is None:
        return {
            "domain": "memory",
            "intent": memc.CATEGORY_TO_INTENT["ambiguous"],
            "action": "remember",
            "target": None,
            "parameters": memc.AMBIGUOUS_CONSTANT_OUTPUT["parameters"],
        }

    raw2 = generate(
        model,
        tokenizer,
        "memory",
        memc.STAGE2_PROMPTS[group],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[MEMORY/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[MEMORY] JSON error: {error2}", "WARNING")

    parameters = (
        parsed2.get("parameters", {})
        if isinstance(parsed2, dict)
        else {}
    )

    return {
        "domain": "memory",
        "intent": memc.CATEGORY_TO_INTENT[category],
        "action": memc.CATEGORY_TO_ACTION[category],
        "target": None,
        "parameters": parameters,
    }


def _productivity_build_full_output(category, payload):
    if category == "todo":
        return {
            "intent": "todo",
            "action": payload.get("action"),
            "target": payload.get("target"),
            "parameters": {},
        }

    if category == "update_delete":
        return {
            "intent": payload.get("intent"),
            "action": payload.get("action"),
            "target": payload.get("target"),
            "parameters": payload.get("parameters", {}),
        }

    return {
        "intent": prodc.CATEGORY_TO_INTENT[category],
        "action": prodc.CATEGORY_TO_ACTION[category],
        "target": payload.get("target"),
        "parameters": payload.get("parameters", {}),
    }


def run_productivity(model, tokenizer, text, max_new_tokens):
    raw1 = generate(
        model,
        tokenizer,
        "productivity",
        prodc.PRODUCTIVITY_STAGE1_PROMPT,
        text,
        max_new_tokens=30,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[PRODUCTIVITY/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[PRODUCTIVITY] JSON error: {error1}", "WARNING")

    category = parsed1.get("category") if isinstance(parsed1, dict) else None

    if category not in prodc.CATEGORIES or category == "negative":
        return None

    if category in prodc.NO_STAGE2_CATEGORIES:
        out = dict(prodc.AMBIGUOUS_CONSTANT_OUTPUT)
        out["domain"] = "productivity"
        return out

    raw2 = generate(
        model,
        tokenizer,
        "productivity",
        prodc.STAGE2_PROMPTS[category],
        text,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[PRODUCTIVITY/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[PRODUCTIVITY] JSON error: {error2}", "WARNING")

    if not isinstance(parsed2, dict):
        return None

    full = _productivity_build_full_output(category, parsed2)
    full["domain"] = "productivity"

    return full


SPECIALIST_RUNNERS = {
    "system": run_system,
    "media": run_media,
    "persona": run_persona,
    "coding": run_coding,
    "information": run_information,
    "memory": run_memory,
    "productivity": run_productivity,
}


# ============================================================================
# VALIDATOR
# ============================================================================

def run_validator(model, tokenizer, original, task_ir, max_new_tokens):
    user_content = (
        f'Instruksi: "{original}"\n'
        f"Task IR: {json.dumps(task_ir, ensure_ascii=False)}"
    )

    raw1 = generate(
        model,
        tokenizer,
        "validator",
        valc.VALIDATOR_STAGE1_PROMPT,
        user_content,
        max_new_tokens=20,
    )

    parsed1, error1 = parse_json(raw1)

    log.add(f"[VALIDATOR/STAGE1] raw={raw1}")

    if error1:
        log.add(f"[VALIDATOR] JSON error: {error1}", "WARNING")

    label = parsed1.get("label") if isinstance(parsed1, dict) else None

    if label not in ("valid", "invalid"):
        return {
            "label": "unknown",
            "reason": None,
        }

    if label == "valid":
        return {
            "label": "valid",
            "reason": None,
        }

    raw2 = generate(
        model,
        tokenizer,
        "validator",
        valc.VALIDATOR_STAGE2_PROMPT,
        user_content,
        max_new_tokens=max_new_tokens,
    )

    parsed2, error2 = parse_json(raw2)

    log.add(f"[VALIDATOR/STAGE2] raw={raw2}")

    if error2:
        log.add(f"[VALIDATOR] JSON error: {error2}", "WARNING")

    reason = (
        parsed2.get("reason")
        if isinstance(parsed2, dict)
        else None
    )

    return {
        "label": "invalid",
        "reason": reason,
    }


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_pipeline(text: str, max_new_tokens: int = 120):
    if not state.ready or state.model is None:
        raise RuntimeError("Model belum siap. Load model terlebih dahulu.")

    if not state.enabled.get("router", False):
        raise RuntimeError(
            "Router adapter disabled. Router wajib aktif untuk pipeline."
        )

    t0 = time.time()

    model = state.model
    tokenizer = state.tokenizer

    log.add("=" * 60)
    log.add(f"[INPUT] {text}")

    segments = run_router(
        model,
        tokenizer,
        text,
        max_new_tokens,
    )

    log.add(f"[ROUTER] segments={json.dumps(segments, ensure_ascii=False)}")

    results = []

    for index, seg in enumerate(segments, 1):
        domain = seg["domain"]
        seg_text = seg["text"]

        log.add(
            f"[SEGMENT {index}] domain={domain!r} text={seg_text!r}"
        )

        entry = {
            "domain": domain,
            "text": seg_text,
            "task_ir": None,
            "validation": None,
            "note": None,
        }

        if domain == "unknown":
            entry["note"] = (
                "Router tidak bisa menentukan domain "
                "(ambiguous/gagal parse)."
            )

        elif domain not in SPECIALIST_RUNNERS:
            entry["note"] = f"Domain {domain!r} tidak dikenal sistem."

        elif not state.enabled.get(domain, False):
            entry["note"] = (
                f"Adapter specialist '{domain}' disabled."
            )

            log.add(
                f"[SPECIALIST] {domain} disabled -> skip",
                "WARNING",
            )

        else:
            log.add(f"[SPECIALIST] running adapter={domain}")

            task_ir = SPECIALIST_RUNNERS[domain](
                model,
                tokenizer,
                seg_text,
                max_new_tokens,
            )

            if task_ir is None:
                entry["note"] = (
                    "Specialist menganggap input bukan "
                    "command actionable."
                )

                log.add(
                    f"[SPECIALIST] {domain} -> no actionable Task IR"
                )

            else:
                entry["task_ir"] = task_ir

                log.add(
                    "[TASK IR] "
                    + json.dumps(
                        task_ir,
                        ensure_ascii=False,
                    )
                )

                if state.enabled.get("validator", False):
                    entry["validation"] = run_validator(
                        model,
                        tokenizer,
                        seg_text,
                        task_ir,
                        max_new_tokens,
                    )

                    log.add(
                        "[VALIDATOR] "
                        + json.dumps(
                            entry["validation"],
                            ensure_ascii=False,
                        )
                    )
                else:
                    entry["note"] = (
                        "Validator adapter disabled; "
                        "Task IR tidak divalidasi."
                    )

                    log.add(
                        "[VALIDATOR] disabled -> skip",
                        "WARNING",
                    )

        results.append(entry)

    elapsed = round(time.time() - t0, 3)

    result = {
        "input": text,
        "segments": results,
        "elapsed_sec": elapsed,
    }

    log.add(f"[DONE] elapsed={elapsed}s")
    log.add("=" * 60)

    return result


# ============================================================================
# MODEL LOADING
# ============================================================================

def unload_model():
    if state.model is not None:
        try:
            del state.model
        except Exception:
            pass

    if state.tokenizer is not None:
        try:
            del state.tokenizer
        except Exception:
            pass

    state.model = None
    state.tokenizer = None
    state.loaded_adapters = []
    state.ready = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline(
    base_model_dir: str,
    adapter_paths: dict[str, str],
    enabled: dict[str, bool],
):
    if not enabled.get("router", False):
        raise ValueError(
            "Router wajib enabled. Pipeline membutuhkan router."
        )

    enabled_names = [
        name
        for name in ADAPTER_NAMES
        if enabled.get(name, False)
    ]

    missing = []

    for name in enabled_names:
        path = adapter_paths.get(name, "")

        if not path:
            missing.append(f"{name}: path kosong")

        elif not os.path.isdir(path):
            missing.append(
                f"{name}: folder tidak ditemukan -> {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Adapter yang enabled tidak valid:\n"
            + "\n".join(missing)
        )

    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(
            f"Base model tidak ditemukan: {base_model_dir}"
        )

    log.add(f"[MODEL] Loading tokenizer: {base_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.add("[MODEL] Loading base model...")

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_dir,
        dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        ),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    log.add(
        "[MODEL] Base model loaded "
        f"(CUDA={torch.cuda.is_available()})"
    )

    first_name = enabled_names[0]

    log.add(
        f"[ADAPTER] Loading '{first_name}' -> "
        f"{adapter_paths[first_name]}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[first_name],
        adapter_name=first_name,
    )

    loaded = [first_name]

    for name in enabled_names[1:]:
        log.add(
            f"[ADAPTER] Loading '{name}' -> "
            f"{adapter_paths[name]}"
        )

        model.load_adapter(
            adapter_paths[name],
            adapter_name=name,
        )

        loaded.append(name)

    model.eval()

    log.add(
        "[READY] Adapter aktif: "
        + ", ".join(loaded)
    )

    return model, tokenizer, loaded


def load_pipeline_background(
    base_model_dir: str,
    adapter_paths: dict[str, str],
    enabled: dict[str, bool],
):
    with state.load_lock:
        if state.loading:
            return

        state.loading = True
        state.ready = False
        state.error = None

    try:
        log.add("[LOAD] Starting model reload...")

        old_model = state.model

        # Jangan mengganti state model sampai model baru berhasil loaded.
        model, tokenizer, loaded = load_pipeline(
            base_model_dir,
            adapter_paths,
            enabled,
        )

        state.model = model
        state.tokenizer = tokenizer
        state.base_model_dir = base_model_dir
        state.adapter_paths = dict(adapter_paths)
        state.enabled = dict(enabled)
        state.loaded_adapters = loaded
        state.ready = True
        state.error = None

        if old_model is not None and old_model is not model:
            try:
                del old_model
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.add("[LOAD] Reload selesai. Pipeline READY.")

    except Exception as exc:
        state.ready = False
        state.error = str(exc)

        log.add(
            f"[LOAD] FAILED: {type(exc).__name__}: {exc}",
            "ERROR",
        )

    finally:
        state.loading = False


# ============================================================================
# API MODELS
# ============================================================================

class LoadRequest(BaseModel):
    base_model_dir: str

    adapters: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_ADAPTERS)
    )

    enabled: dict[str, bool] = Field(
        default_factory=lambda: dict(DEFAULT_ENABLED)
    )


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1)
    max_new_tokens: int = Field(
        default=120,
        ge=1,
        le=2048,
    )


# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(
    title="Liana Pipeline Server",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/api/status")
async def api_status():
    return {
        "ready": state.ready,
        "loading": state.loading,
        "error": state.error,
        "base_model_dir": state.base_model_dir,
        "loaded_adapters": state.loaded_adapters,
        "enabled": state.enabled,
        "adapter_paths": state.adapter_paths,
        "cuda": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


@app.get("/api/logs")
async def api_logs(after: int = 0):
    return {
        "logs": log.get(after),
    }


@app.delete("/api/logs")
async def api_clear_logs():
    log.clear()
    return {"ok": True}


@app.post("/api/load")
async def api_load(req: LoadRequest):
    if state.loading:
        raise HTTPException(
            status_code=409,
            detail="Model sedang loading.",
        )

    enabled = {
        name: bool(req.enabled.get(name, False))
        for name in ADAPTER_NAMES
    }

    adapters = {
        name: req.adapters.get(
            name,
            DEFAULT_ADAPTERS[name],
        )
        for name in ADAPTER_NAMES
    }

    # Validasi router.
    if not enabled["router"]:
        raise HTTPException(
            status_code=400,
            detail="Router wajib enabled.",
        )

    thread = threading.Thread(
        target=load_pipeline_background,
        args=(
            req.base_model_dir,
            adapters,
            enabled,
        ),
        daemon=True,
    )

    thread.start()

    return {
        "ok": True,
        "message": "Model loading dimulai.",
    }


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline belum ready."
                + (
                    f" Error: {state.error}"
                    if state.error
                    else ""
                )
            ),
        )

    # Karena inference adalah operasi blocking, jalankan di thread.
    def worker():
        with state.inference_lock:
            return run_full_pipeline(
                req.text,
                req.max_new_tokens,
            )

    try:
        result = await asyncio.to_thread(worker)

        return {
            "ok": True,
            "result": result,
        }

    except Exception as exc:
        log.add(
            f"[REQUEST ERROR] {type(exc).__name__}: {exc}",
            "ERROR",
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================================
# SIMPLE WEB UI
# ============================================================================

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Liana Pipeline</title>

<style>
:root {
    color-scheme: dark;
    --bg: #0b0d10;
    --panel: #12161b;
    --panel2: #181d23;
    --border: #272d35;
    --text: #e8edf2;
    --muted: #8e99a6;
    --accent: #8ab4f8;
    --good: #7bd88f;
    --warn: #f2c66d;
    --bad: #f17c7c;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

button,
input,
textarea {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    width: min(1200px, calc(100% - 32px));
    margin: 24px auto;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
}

.title h1 {
    margin: 0;
    font-size: 22px;
}

.title p {
    margin: 5px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.status {
    border: 1px solid var(--border);
    background: var(--panel);
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 12px;
}

.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--bad);
    margin-right: 6px;
}

.dot.ready {
    background: var(--good);
}

.grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 14px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

.panel-head {
    padding: 13px 15px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    font-weight: 700;
}

.panel-body {
    padding: 14px;
}

label {
    display: block;
    margin-bottom: 6px;
    color: var(--muted);
    font-size: 12px;
}

input[type="text"],
input[type="number"],
textarea {
    width: 100%;
    background: var(--panel2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 9px;
    outline: none;
    padding: 9px 10px;
}

input:focus,
textarea:focus {
    border-color: var(--accent);
}

textarea {
    min-height: 115px;
    resize: vertical;
}

.field {
    margin-bottom: 13px;
}

.adapters {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.adapter {
    display: grid;
    grid-template-columns: 22px 92px 1fr;
    gap: 8px;
    align-items: center;
    padding: 8px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--panel2);
}

.adapter input[type="text"] {
    padding: 6px 7px;
    font-size: 11px;
}

.adapter-name {
    font-size: 12px;
    font-weight: 600;
}

.actions {
    display: flex;
    gap: 8px;
    margin-top: 13px;
}

button {
    border: 1px solid var(--border);
    background: var(--panel2);
    color: var(--text);
    border-radius: 9px;
    padding: 9px 12px;
}

button:hover {
    border-color: #414a56;
}

button.primary {
    background: #202a38;
    border-color: #40536c;
}

button.danger {
    color: var(--bad);
}

.result {
    min-height: 260px;
}

.result pre,
.log pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace;
    font-size: 12px;
    line-height: 1.55;
}

.result-content {
    padding: 14px;
    max-height: 520px;
    overflow: auto;
}

.log {
    margin-top: 14px;
}

.log-content {
    height: 390px;
    overflow: auto;
    padding: 10px 14px;
    background: #090b0e;
}

.log-line {
    font-family:
        "JetBrains Mono",
        "Cascadia Code",
        monospace;
    font-size: 11px;
    line-height: 1.5;
    padding: 2px 0;
    color: #b9c2cc;
}

.log-time {
    color: #697581;
}

.log-level {
    display: inline-block;
    width: 58px;
}

.log-level.INFO {
    color: var(--accent);
}

.log-level.WARNING {
    color: var(--warn);
}

.log-level.ERROR {
    color: var(--bad);
}

.empty {
    color: var(--muted);
    font-size: 13px;
}

.meta {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 8px;
}

.badge {
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 8px;
    font-size: 11px;
    color: var(--muted);
}

@media (max-width: 850px) {
    .grid {
        grid-template-columns: 1fr;
    }
}
</style>
</head>

<body>
<div class="app">

<header>
    <div class="title">
        <h1>Liana Pipeline</h1>
        <p>Router → Specialist → Validator</p>
    </div>

    <div class="status">
        <span id="statusDot" class="dot"></span>
        <span id="statusText">Offline</span>
    </div>
</header>

<div class="grid">

<section>

<div class="panel">
    <div class="panel-head">Model</div>

    <div class="panel-body">

        <div class="field">
            <label>Base Model Directory</label>
            <input
                id="baseModel"
                type="text"
                placeholder="./models/Qwen3.5-0.8B"
            >
        </div>

        <div class="panel-head"
             style="margin: 0 -14px 12px; border-top: 1px solid var(--border);">
            Adapters
        </div>

        <div id="adapterList" class="adapters"></div>

        <div class="actions">
            <button
                class="primary"
                onclick="loadModel()"
                id="loadBtn"
            >
                Load / Reload
            </button>
        </div>

    </div>
</div>

</section>


<section>

<div class="panel">
    <div class="panel-head">Pipeline Input</div>

    <div class="panel-body">

        <div class="field">
            <label>Instruction</label>
            <textarea
                id="inputText"
                placeholder="Contoh: Buka Spotify terus putar lagu Noah."
            ></textarea>
        </div>

        <div class="field">
            <label>Max New Tokens</label>
            <input
                id="maxTokens"
                type="number"
                value="120"
                min="1"
                max="2048"
            >
        </div>

        <div class="actions">
            <button
                class="primary"
                onclick="runPipeline()"
                id="runBtn"
            >
                Run Pipeline
            </button>

            <button onclick="clearResult()">
                Clear
            </button>
        </div>

    </div>
</div>


<div class="panel result" style="margin-top:14px;">
    <div class="panel-head">Result</div>

    <div id="result" class="result-content">
        <div class="empty">
            Belum ada hasil.
        </div>
    </div>
</div>

</section>

</div>


<div class="panel log">
    <div class="panel-head"
         style="display:flex; justify-content:space-between; align-items:center;">
        <span>Live Logs</span>
        <button
            class="danger"
            onclick="clearLogs()"
            style="padding:5px 8px; font-size:11px;"
        >
            Clear
        </button>
    </div>

    <div id="logs" class="log-content"></div>
</div>

</div>


<script>

const adapterDefaults = {
    router: "./adapters/router_core_v7",
    system: "./adapters/system_core_v1",
    media: "./adapters/media_core_v1",
    persona: "./adapters/persona_core_v1",
    coding: "./adapters/coding_core_v1",
    information: "./adapters/information_core_v1",
    memory: "./adapters/memory_core_v1",
    productivity: "./adapters/productivity_core_v1",
    validator: "./adapters/validator_core_v1"
};

const adapterNames = [
    "router",
    "system",
    "media",
    "persona",
    "coding",
    "information",
    "memory",
    "productivity",
    "validator"
];

let lastLogId = 0;
let loading = false;


function createAdapters() {
    const root = document.getElementById("adapterList");

    root.innerHTML = "";

    for (const name of adapterNames) {
        const row = document.createElement("div");
        row.className = "adapter";

        row.innerHTML = `
            <input
                type="checkbox"
                id="enable-${name}"
                checked
                ${name === "router" ? "disabled" : ""}
            >

            <div class="adapter-name">${name}</div>

            <input
                type="text"
                id="path-${name}"
                value="${adapterDefaults[name]}"
            >
        `;

        root.appendChild(row);
    }
}


function collectConfig() {
    const adapters = {};
    const enabled = {};

    for (const name of adapterNames) {
        adapters[name] =
            document.getElementById(`path-${name}`).value.trim();

        enabled[name] =
            document.getElementById(`enable-${name}`).checked;
    }

    // Router selalu aktif.
    enabled.router = true;

    return {
        base_model_dir:
            document.getElementById("baseModel").value.trim(),

        adapters,
        enabled
    };
}


async function loadModel() {
    if (loading) return;

    const config = collectConfig();

    if (!config.base_model_dir) {
        alert("Base model directory belum diisi.");
        return;
    }

    loading = true;

    document.getElementById("loadBtn").disabled = true;

    try {
        const response = await fetch("/api/load", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (!response.ok) {
            alert(data.detail || "Gagal load model.");
        }

    } catch (error) {
        alert(error.toString());

    } finally {
        document.getElementById("loadBtn").disabled = false;
        loading = false;
    }
}


async function runPipeline() {
    const text =
        document.getElementById("inputText").value.trim();

    if (!text) {
        return;
    }

    const maxTokens =
        Number(document.getElementById("maxTokens").value);

    const btn = document.getElementById("runBtn");

    btn.disabled = true;
    btn.textContent = "Running...";

    try {
        const response = await fetch("/api/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text,
                max_new_tokens: maxTokens
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Pipeline error");
        }

        document.getElementById("result").innerHTML =
            `<pre>${escapeHtml(
                JSON.stringify(data.result, null, 2)
            )}</pre>`;

    } catch (error) {
        document.getElementById("result").innerHTML =
            `<pre style="color:var(--bad)">${escapeHtml(
                error.toString()
            )}</pre>`;

    } finally {
        btn.disabled = false;
        btn.textContent = "Run Pipeline";
    }
}


async function pollStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        const dot =
            document.getElementById("statusDot");

        const text =
            document.getElementById("statusText");

        if (data.ready) {
            dot.classList.add("ready");
            text.textContent =
                `Ready · ${data.loaded_adapters.length} adapters`;
        } else if (data.loading) {
            dot.classList.remove("ready");
            text.textContent = "Loading...";
        } else {
            dot.classList.remove("ready");
            text.textContent =
                data.error ? "Error" : "Not loaded";
        }

    } catch (_) {
        const dot =
            document.getElementById("statusDot");

        dot.classList.remove("ready");

        document.getElementById("statusText")
            .textContent = "Offline";
    }
}


async function pollLogs() {
    try {
        const response =
            await fetch(`/api/logs?after=${lastLogId}`);

        const data = await response.json();

        const container =
            document.getElementById("logs");

        for (const item of data.logs) {
            const line =
                document.createElement("div");

            line.className = "log-line";

            line.innerHTML =
                `<span class="log-time">[${escapeHtml(item.time)}]</span> ` +
                `<span class="log-level ${escapeHtml(item.level)}">` +
                `[${escapeHtml(item.level)}]</span> ` +
                `${escapeHtml(item.message)}`;

            container.appendChild(line);

            lastLogId = Math.max(
                lastLogId,
                Number(item.id)
            );
        }

        // Auto-scroll kalau user sedang dekat bawah.
        if (
            container.scrollHeight -
            container.scrollTop -
            container.clientHeight < 80
        ) {
            container.scrollTop =
                container.scrollHeight;
        }

    } catch (_) {
        // Ignore polling error.
    }
}


async function clearLogs() {
    await fetch("/api/logs", {
        method: "DELETE"
    });

    document.getElementById("logs").innerHTML = "";
    lastLogId = 0;
}


function clearResult() {
    document.getElementById("result").innerHTML =
        `<div class="empty">Belum ada hasil.</div>`;
}


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


createAdapters();

setInterval(pollStatus, 1000);
setInterval(pollLogs, 400);

pollStatus();
pollLogs();

</script>

</body>
</html>
"""


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Liana FastAPI Pipeline Server"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--base-model-dir",
        default=None,
        help="Optional: langsung load model saat server start.",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Uvicorn reload untuk development.",
    )

    args = parser.parse_args()

    if args.base_model_dir:
        enabled = dict(DEFAULT_ENABLED)
        adapters = dict(DEFAULT_ADAPTERS)

        threading.Thread(
            target=load_pipeline_background,
            args=(
                args.base_model_dir,
                adapters,
                enabled,
            ),
            daemon=True,
        ).start()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
