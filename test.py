import json
from safetensors import safe_open
from transformers import AutoModelForCausalLM

BASE = "./models/Qwen3.5-0.8B"
ADAPTER = "./adapters/router_core_v7"

# 1. Lihat target_modules yang tercatat di adapter_config
cfg = json.load(open(f"{ADAPTER}/adapter_config.json"))
print("target_modules:", cfg.get("target_modules"))

# 2. Lihat nama-nama module MLP yang beneran ada di base model sekarang
base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True)
mlp_names = set()
for name, _ in base.named_modules():
    if "mlp" in name and ("proj" in name):
        mlp_names.add(name)
print("\ncontoh nama modul mlp di base model sekarang:")
for n in sorted(mlp_names)[:20]:
    print(" ", n)

# 3. Lihat nama key yang ada di checkpoint adapter
with safe_open(f"{ADAPTER}/adapter_model.safetensors", framework="pt", device="cpu") as f:
    ckpt_keys = list(f.keys())
print("\ncontoh key mlp di checkpoint adapter:")
for k in sorted(k for k in ckpt_keys if "mlp" in k)[:20]:
    print(" ", k)
