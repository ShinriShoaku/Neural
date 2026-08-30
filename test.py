from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    "./models/Qwen3.5-0.8B", trust_remote_code=True,
)
mlp_names = sorted(n for n, _ in base.named_modules() if "mlp" in n and "proj" in n)
print(mlp_names[:10])