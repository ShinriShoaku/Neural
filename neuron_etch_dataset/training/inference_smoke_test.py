"""
inference_smoke_test.py
========================
Load base model + 1 adapter, kirim 1 kalimat, lihat apakah outputnya JSON
valid sesuai skema. Ini BUKAN eval harness lengkap (§70) — cuma sanity
check cepat setelah training selesai.

Menerapkan §65.4 (enable_thinking=False, add_generation_prompt=True) dan
guardrail parsing §69 (strip markdown fence, retry sekali kalau JSON invalid).

Cara pakai:
    # Router
    python inference_smoke_test.py --adapter router \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/router_core \\
        --text "Buka foot lalu putar lagu Noah"

    # Specialist
    python inference_smoke_test.py --adapter system \\
        --base-model-dir ./models/Qwen3.5-0.8B \\
        --adapter-dir ./adapters/system_specialist \\
        --text "Buka foot"
"""

from __future__ import annotations
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

from chatml_format import build_router_messages, build_specialist_messages

SPECIALIST_DOMAINS = ["system", "media", "persona", "coding", "information", "memory", "productivity"]


def build_prompt_messages(adapter: str, text: str) -> list[dict]:
    # Fake row: cuma perlu system+user, assistant-nya dibuang (index [:2])
    fake_row = {
        "input": text, "output": None, "label": "negative",
        "task_category": "smoke_test", "metadata": {},
        "segments": [{"text": text, "domain": "unknown"}],
    }
    if adapter == "router":
        return build_router_messages(fake_row)[:2]
    elif adapter in SPECIALIST_DOMAINS:
        return build_specialist_messages(adapter, fake_row)[:2]
    else:
        raise ValueError("Smoke test validator butuh --original & --generated-json, "
                          "belum didukung mode --text sederhana ini.")


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 200) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,  # §65.3, §65.4
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_with_guardrail(raw_text: str) -> tuple[dict | None, str | None]:
    """§69 — strip markdown fence, json.loads, return (parsed, error)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True,
                         choices=["router"] + SPECIALIST_DOMAINS)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading base model dari {args.base_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Loading adapter dari {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = build_prompt_messages(args.adapter, args.text)

    print(f"\nInput: {args.text}")
    raw = generate(model, tokenizer, messages, args.max_new_tokens)
    print(f"Raw output:\n{raw}")

    parsed, err = parse_with_guardrail(raw)
    if parsed is not None:
        print(f"\nJSON valid:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")
        return

    # retry sekali sesuai §69
    print(f"\nJSON parse gagal ({err}), retry sekali dengan instruksi tambahan...")
    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": "Output JSON valid saja."},
    ]
    raw2 = generate(model, tokenizer, retry_messages, args.max_new_tokens)
    print(f"Raw output (retry):\n{raw2}")
    parsed2, err2 = parse_with_guardrail(raw2)
    if parsed2 is not None:
        print(f"\nJSON valid (setelah retry):\n{json.dumps(parsed2, indent=2, ensure_ascii=False)}")
    else:
        print(f"\nTetap gagal parse: {err2} -> status=failed, reason=invalid_json_output (§69)")


if __name__ == "__main__":
    main()
