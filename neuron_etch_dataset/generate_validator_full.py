"""
generate_validator_full.py
=============================
Generator dataset FULL untuk validator_core (§38), target 8,000 sample.

Beda dari 7 specialist lain: validator TIDAK punya vocab sendiri --
datanya diambil dari Task IR VALID yang sudah dihasilkan 7 specialist
(system/media/persona/coding/information/memory/productivity), lalu
sebagian dibiarkan valid apa adanya, sebagian di-CORRUPT terprogram.

Semua mapping korupsi (nilai target alternatif, action per intent,
parameter yang wajib ada, dst) di-DERIVE OTOMATIS dari data specialist
yang sudah di-generate -- bukan hardcode manual seperti corruption.py
stub asli (yang cakupannya sempit, cuma sebagian kecil vocab).

10 kategori (9 invalid + 1 valid), ~800 masing-masing:
    valid                   -- Task IR asli, tidak diubah
    target_mismatch         -- target.value ditukar ke value lain yang masuk akal
    domain_mismatch         -- domain ditukar ke domain lain
    missing_parameter       -- parameter yang biasanya ada di-drop
    hallucinated_parameter  -- parameter asing (priority=urgent) disisipkan
    intent_mismatch         -- action ditukar ke action lain DALAM intent yang sama
    unsupported_action      -- domain+action ditukar total ke domain lain
    parameter_mismatch      -- (BARU) nilai 1 parameter ditukar ke nilai dari
                               sample lain dengan key sama (isinya jadi nggak
                               nyambung sama input text)
    contradiction           -- (BARU) action ditukar ke LAWAN semantiknya
                               (mute<->unmute, enable<->disable, dst)
    ambiguous               -- (BARU) target/parameter value diganti jadi
                               placeholder generik ("itu"/"sesuatu")

3 kategori terakhir awalnya "manual only" di desain asli (§38.1 vs §38.3),
di sini diimplementasi otomatis juga supaya validator_core punya cakupan
penuh 10 kategori.
"""
from __future__ import annotations
import copy
import glob
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

RNG = random.Random(20260903)
NEURAL_ROOT = Path(__file__).parent
OUT_DIR = NEURAL_ROOT / "output" / "validator"
SPECIALIST_DOMAINS = ["system", "media", "persona", "coding", "information", "memory", "productivity"]

_counter = itertools.count(1)
def new_id() -> str:
    return f"validator_{next(_counter):06d}"


# ===========================================================================
# 1. MUAT SEMUA TASK IR VALID DARI 7 SPECIALIST
# ===========================================================================
def load_positive_rows() -> list[dict]:
    rows = []
    for domain in SPECIALIST_DOMAINS:
        for fp in glob.glob(str(NEURAL_ROOT / "output" / domain / "*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("output") is not None and d.get("label", "positive") == "positive":
                        rows.append(d)
    return rows


# ===========================================================================
# 2. DERIVE VOCAB KORUPSI OTOMATIS DARI DATA
# ===========================================================================
def build_corruption_vocab(rows: list[dict]):
    alt_target_values = defaultdict(set)          # target.type -> {values}
    actions_by_domain_intent = defaultdict(set)    # (domain,intent) -> {actions}
    actions_by_domain = defaultdict(set)           # domain -> {actions} (semua intent)
    domain_action_pairs = []                       # [(domain, action), ...] buat far-swap
    param_key_values = defaultdict(list)           # param_key -> [values...] (buat parameter_mismatch)
    param_key_freq = defaultdict(lambda: defaultdict(int))  # (domain,intent) -> {param_key: count}
    domain_intent_count = defaultdict(int)         # (domain,intent) -> total sample

    for row in rows:
        out = row["output"]
        domain, intent, action = out["domain"], out["intent"], out["action"]
        target = out.get("target")
        params = out.get("parameters") or {}

        if target and target.get("type") and target.get("value") is not None:
            alt_target_values[target["type"]].add(target["value"] if isinstance(target["value"], str) else str(target["value"]))

        actions_by_domain_intent[(domain, intent)].add(action)
        actions_by_domain[domain].add(action)
        domain_action_pairs.append((domain, action))

        domain_intent_count[(domain, intent)] += 1
        for k, v in params.items():
            param_key_freq[(domain, intent)][k] += 1
            if isinstance(v, str):
                param_key_values[k].append(v)

    # required param = muncul di >=70% sample utk (domain,intent) itu
    required_param_by_intent = {}
    for key, count in domain_intent_count.items():
        req = [k for k, c in param_key_freq[key].items() if c / count >= 0.7]
        if req:
            required_param_by_intent[key] = req

    return {
        "alt_target_values": {k: list(v) for k, v in alt_target_values.items()},
        "actions_by_domain_intent": {k: list(v) for k, v in actions_by_domain_intent.items()},
        "actions_by_domain": {k: list(v) for k, v in actions_by_domain.items()},
        "domain_action_pairs": domain_action_pairs,
        "param_key_values": dict(param_key_values),
        "required_param_by_intent": required_param_by_intent,
    }


OPPOSITE_ACTIONS = {
    "mute": "unmute", "unmute": "mute",
    "enable_device": "disable_device", "disable_device": "enable_device",
    "enable_wifi": "disable_wifi", "disable_wifi": "enable_wifi",
    "connect_wifi": "disconnect_wifi", "disconnect_wifi": "connect_wifi",
    "volume_up": "volume_down", "volume_down": "volume_up",
    "brightness_up": "brightness_down", "brightness_down": "brightness_up",
    "repeat": "repeat_off", "repeat_off": "repeat",
    "shuffle": "shuffle_off", "shuffle_off": "shuffle",
    "queue_add": "queue_remove", "queue_remove": "queue_add",
    "create": "delete", "delete": "create",
    "create_file": "delete_file", "delete_file": "create_file",
    "create_folder": "delete_file",
    "launch": "close", "close": "launch",
    "pause": "resume", "resume": "pause",
    "next": "previous", "previous": "next",
}
AMBIGUOUS_PLACEHOLDERS = ["itu", "sesuatu", "yang tadi", "entah apa", "hal itu"]


# ===========================================================================
# 3. FUNGSI CORRUPT_*
# ===========================================================================
def corrupt_valid(task_ir: dict, vocab: dict, rng: random.Random):
    return copy.deepcopy(task_ir)


def corrupt_target_mismatch(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    target = corrupted.get("target")
    if not target or not target.get("type"):
        return None
    alts = vocab["alt_target_values"].get(target["type"], [])
    candidates = [v for v in alts if v != target["value"]]
    if not candidates:
        return None
    target["value"] = rng.choice(candidates)
    return corrupted


def corrupt_domain_mismatch(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    current = corrupted.get("domain")
    candidates = [d for d in SPECIALIST_DOMAINS if d != current]
    corrupted["domain"] = rng.choice(candidates)
    return corrupted


def corrupt_missing_parameter(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    key = (corrupted["domain"], corrupted["intent"])
    required = vocab["required_param_by_intent"].get(key, [])
    params = corrupted.get("parameters") or {}
    candidates = [p for p in required if p in params]
    if not candidates:
        return None
    del params[rng.choice(candidates)]
    corrupted["parameters"] = params
    return corrupted


def corrupt_hallucinated_parameter(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    params = dict(corrupted.get("parameters") or {})
    params["priority"] = "urgent"
    corrupted["parameters"] = params
    return corrupted


def corrupt_intent_mismatch(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    key = (corrupted["domain"], corrupted["intent"])
    current_action = corrupted["action"]
    candidates = [a for a in vocab["actions_by_domain_intent"].get(key, []) if a != current_action]
    if not candidates:
        # fallback: action lain di domain yang sama
        candidates = [a for a in vocab["actions_by_domain"].get(corrupted["domain"], []) if a != current_action]
    if not candidates:
        return None
    corrupted["action"] = rng.choice(candidates)
    return corrupted


def corrupt_unsupported_action(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    current_domain = corrupted["domain"]
    candidates = [(d, a) for d, a in vocab["domain_action_pairs"] if d != current_domain]
    if not candidates:
        return None
    far_domain, far_action = rng.choice(candidates)
    corrupted["domain"] = far_domain
    corrupted["action"] = far_action
    return corrupted


def corrupt_parameter_mismatch(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    params = dict(corrupted.get("parameters") or {})
    str_keys = [k for k, v in params.items() if isinstance(v, str)]
    if not str_keys:
        return None
    key = rng.choice(str_keys)
    pool = [v for v in vocab["param_key_values"].get(key, []) if v != params[key]]
    if not pool:
        return None
    params[key] = rng.choice(pool)
    corrupted["parameters"] = params
    return corrupted


def corrupt_contradiction(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    current_action = corrupted["action"]
    opposite = OPPOSITE_ACTIONS.get(current_action)
    if not opposite:
        return None
    key = (corrupted["domain"], corrupted["intent"])
    if opposite not in vocab["actions_by_domain_intent"].get(key, []):
        return None  # opposite harus valid action utk intent yg sama (biar tetap masuk akal)
    corrupted["action"] = opposite
    return corrupted


def corrupt_ambiguous(task_ir: dict, vocab: dict, rng: random.Random):
    corrupted = copy.deepcopy(task_ir)
    target = corrupted.get("target")
    params = dict(corrupted.get("parameters") or {})
    str_param_keys = [k for k, v in params.items() if isinstance(v, str)]
    choices = []
    if target and target.get("value") is not None:
        choices.append("target")
    if str_param_keys:
        choices.append("param")
    if not choices:
        return None
    pick = rng.choice(choices)
    if pick == "target":
        target["value"] = rng.choice(AMBIGUOUS_PLACEHOLDERS)
    else:
        params[rng.choice(str_param_keys)] = rng.choice(AMBIGUOUS_PLACEHOLDERS)
        corrupted["parameters"] = params
    return corrupted


CORRUPTORS = {
    "valid": corrupt_valid,
    "target_mismatch": corrupt_target_mismatch,
    "domain_mismatch": corrupt_domain_mismatch,
    "missing_parameter": corrupt_missing_parameter,
    "hallucinated_parameter": corrupt_hallucinated_parameter,
    "intent_mismatch": corrupt_intent_mismatch,
    "unsupported_action": corrupt_unsupported_action,
    "parameter_mismatch": corrupt_parameter_mismatch,
    "contradiction": corrupt_contradiction,
    "ambiguous": corrupt_ambiguous,
}
REASON_BY_CATEGORY = {
    "target_mismatch": "target_mismatch", "domain_mismatch": "domain_mismatch",
    "missing_parameter": "missing_parameter", "hallucinated_parameter": "hallucinated_parameter",
    "intent_mismatch": "intent_mismatch", "unsupported_action": "unsupported_action",
    "parameter_mismatch": "parameter_mismatch", "contradiction": "contradiction",
    "ambiguous": "ambiguous",
}


# ===========================================================================
# 4. GENERATE PER KATEGORI (retry sampling sumber lain kalau corruptor gagal)
# ===========================================================================
def generate_category(category: str, target: int, rows: list[dict], vocab: dict) -> list[dict]:
    corrupt_fn = CORRUPTORS[category]
    result = []
    seen_keys = set()
    shuffled = list(rows)
    RNG.shuffle(shuffled)
    idx = 0
    attempts = 0
    max_attempts = target * 20 + 2000  # jaga2 supaya tidak infinite loop kalau corruptor sering None

    while len(result) < target and attempts < max_attempts:
        row = shuffled[idx % len(shuffled)]
        idx += 1
        attempts += 1

        corrupted = corrupt_fn(row["output"], vocab, RNG)
        if corrupted is None:
            continue

        dedup_key = (row["input"], json.dumps(corrupted, sort_keys=True))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        label = "valid" if category == "valid" else "invalid"
        reason = None if category == "valid" else REASON_BY_CATEGORY[category]
        result.append({
            "id": new_id(),
            "original": row["input"],
            "generated": corrupted,
            "label": label,
            "reason": reason,
            "task_category": category,
            "metadata": {},
        })

    return result


# ===========================================================================
# MAIN
# ===========================================================================
def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    print("Memuat semua Task IR valid dari 7 specialist...")
    rows = load_positive_rows()
    print(f"Total sample sumber: {len(rows)}")

    print("Derive vocab korupsi otomatis dari data...")
    vocab = build_corruption_vocab(rows)
    print(f"  target types: {len(vocab['alt_target_values'])}")
    print(f"  (domain,intent) pairs: {len(vocab['actions_by_domain_intent'])}")
    print(f"  required-param intents: {len(vocab['required_param_by_intent'])}")

    categories = list(CORRUPTORS.keys())
    total_target = 8000
    n_valid = total_target // 2  # 4000
    n_invalid_each = (total_target - n_valid) // (len(categories) - 1)  # ~444 each x 9

    targets = {"valid": n_valid}
    for cat in categories:
        if cat != "valid":
            targets[cat] = n_invalid_each

    print(f"\n{'task_category':24s} {'target':>8s} {'actual':>8s}")
    print("-" * 44)
    total = 0
    for cat in categories:
        samples = generate_category(cat, targets[cat], rows, vocab)
        save_jsonl(samples, OUT_DIR / f"{cat}.jsonl")
        print(f"{cat:24s} {targets[cat]:8d} {len(samples):8d}")
        total += len(samples)
    print("-" * 44)
    print(f"{'TOTAL':24s} {total_target:8d} {total:8d}")


if __name__ == "__main__":
    main()
