"""
api_server.py
===============
FastAPI server buat sistem Liana (router -> specialist -> validator).
Alur kerjanya SAMA PERSIS dengan run_full_pipeline.py versi CLI -- file
ini cuma bungkus HTTP-nya, semua logika inti (generate/parse/run_router/
run_system/dst) diimpor langsung dari run_full_pipeline.py, tidak
diduplikasi.

Base model dimuat SEKALI saat startup. 9 adapter LoRA bisa di-load/unload
kapan saja lewat endpoint API (atau lewat UI di "/") tanpa perlu restart
server -- jadi bisa dites adapter mana aja yang lagi aktif.

Jalanin:
    cd training/
    export BASE_MODEL_DIR=./models/Qwen3.5-0.8B
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Config lain (opsional, override path adapter default) lewat environment
variable, lihat bagian ADAPTER_PATHS di bawah.

Contoh curl:
    # jalanin pipeline penuh
    curl -X POST http://localhost:8000/run \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Buka spotify terus putar lagu Noah."}'

    # lihat status semua adapter
    curl http://localhost:8000/adapters

    # matiin adapter media (buat tes fallback / hemat VRAM)
    curl -X POST http://localhost:8000/adapters/media/unload

    # nyalain lagi
    curl -X POST http://localhost:8000/adapters/media/load

    # load semua adapter sekaligus
    curl -X POST http://localhost:8000/adapters/load_all

    # buka UI di browser
    open http://localhost:8000/
"""
from __future__ import annotations
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

import run_full_pipeline as pipeline

# ===========================================================================
# KONFIGURASI (lewat environment variable, ada default yang sama kayak CLI)
# ===========================================================================
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "./models/Qwen3.5-0.8B")

ADAPTER_PATHS = {
    "router": os.environ.get("ROUTER_ADAPTER", "./adapters/router_core_v7"),
    "system": os.environ.get("SYSTEM_ADAPTER", "./adapters/system_core_v1"),
    "media": os.environ.get("MEDIA_ADAPTER", "./adapters/media_core_v1"),
    "persona": os.environ.get("PERSONA_ADAPTER", "./adapters/persona_core_v1"),
    "coding": os.environ.get("CODING_ADAPTER", "./adapters/coding_core_v1"),
    "information": os.environ.get("INFORMATION_ADAPTER", "./adapters/information_core_v1"),
    "memory": os.environ.get("MEMORY_ADAPTER", "./adapters/memory_core_v1"),
    "productivity": os.environ.get("PRODUCTIVITY_ADAPTER", "./adapters/productivity_core_v1"),
    "validator": os.environ.get("VALIDATOR_ADAPTER", "./adapters/validator_core_v1"),
}
# Adapter mana aja yang di-load OTOMATIS saat server nyala. Default: semua.
# Bisa dibatasi lewat env var LOAD_ADAPTERS_ON_START="router,system,media"
_load_on_start_raw = os.environ.get("LOAD_ADAPTERS_ON_START", "all")
LOAD_ADAPTERS_ON_START = (
    list(ADAPTER_PATHS.keys()) if _load_on_start_raw.strip().lower() == "all"
    else [x.strip() for x in _load_on_start_raw.split(",") if x.strip()]
)


# ===========================================================================
# STATE GLOBAL (1 model, 1 tokenizer, dipakai bareng semua request)
# ===========================================================================
class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.lock = threading.Lock()  # serialize akses ke model (set_adapter itu shared mutable state)

    @property
    def loaded_adapters(self) -> set[str]:
        if self.model is None:
            return set()
        return set(getattr(self.model, "peft_config", {}).keys())


STATE = ModelState()


def _load_base_model_and_first_adapters():
    print(f"Loading base model dari {BASE_MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    to_load = [n for n in LOAD_ADAPTERS_ON_START if n in ADAPTER_PATHS]
    if not to_load:
        raise RuntimeError("LOAD_ADAPTERS_ON_START kosong / tidak valid -- minimal harus ada 1 adapter.")

    first = to_load[0]
    print(f"Loading adapter {first!r} dari {ADAPTER_PATHS[first]} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATHS[first], adapter_name=first)

    for name in to_load[1:]:
        print(f"Loading adapter {name!r} dari {ADAPTER_PATHS[name]} ...")
        model.load_adapter(ADAPTER_PATHS[name], adapter_name=name)

    model.eval()
    print(f"Server siap. Adapter aktif: {sorted(to_load)}")
    STATE.model = model
    STATE.tokenizer = tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in ADAPTER_PATHS.items():
        if not os.path.isdir(path):
            print(f"PERINGATAN: folder adapter {name!r} tidak ketemu di {path!r}.")
    await run_in_threadpool(_load_base_model_and_first_adapters)
    yield
    STATE.model = None
    STATE.tokenizer = None


app = FastAPI(title="Liana Pipeline API", lifespan=lifespan)


# ===========================================================================
# SCHEMA REQUEST/RESPONSE
# ===========================================================================
class RunRequest(BaseModel):
    text: str
    max_new_tokens: int = 120


class AdapterActionResponse(BaseModel):
    name: str
    loaded: bool
    message: str


# ===========================================================================
# ENDPOINT: health
# ===========================================================================
@app.get("/health")
def health():
    return {
        "status": "ok" if STATE.model is not None else "model belum siap",
        "model_loaded": STATE.model is not None,
        "loaded_adapters": sorted(STATE.loaded_adapters),
    }


# ===========================================================================
# ENDPOINT: jalankan pipeline
# ===========================================================================
@app.post("/run")
async def run(req: RunRequest):
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi sebentar.")

    def _do_run():
        with STATE.lock:
            return pipeline.run_full_pipeline(STATE.model, STATE.tokenizer, req.text, req.max_new_tokens)

    result = await run_in_threadpool(_do_run)
    return result


# ===========================================================================
# ENDPOINT: kelola adapter
# ===========================================================================
@app.get("/adapters")
def list_adapters():
    loaded = STATE.loaded_adapters
    return [
        {"name": name, "path": path, "loaded": name in loaded}
        for name, path in ADAPTER_PATHS.items()
    ]


@app.post("/adapters/{name}/load", response_model=AdapterActionResponse)
async def load_adapter(name: str):
    if name not in ADAPTER_PATHS:
        raise HTTPException(status_code=404, detail=f"Adapter {name!r} tidak dikenal.")
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    if name in STATE.loaded_adapters:
        return AdapterActionResponse(name=name, loaded=True, message="sudah dimuat sebelumnya")

    path = ADAPTER_PATHS[name]
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Folder adapter tidak ketemu: {path}")

    def _do_load():
        with STATE.lock:
            STATE.model.load_adapter(path, adapter_name=name)

    await run_in_threadpool(_do_load)
    return AdapterActionResponse(name=name, loaded=True, message=f"berhasil dimuat dari {path}")


@app.post("/adapters/{name}/unload", response_model=AdapterActionResponse)
async def unload_adapter(name: str):
    if name not in ADAPTER_PATHS:
        raise HTTPException(status_code=404, detail=f"Adapter {name!r} tidak dikenal.")
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    if name not in STATE.loaded_adapters:
        return AdapterActionResponse(name=name, loaded=False, message="memang belum dimuat")

    def _do_unload():
        with STATE.lock:
            STATE.model.delete_adapter(name)

    await run_in_threadpool(_do_unload)
    return AdapterActionResponse(name=name, loaded=False, message="berhasil di-unload, VRAM dibebaskan")


@app.post("/adapters/load_all")
async def load_all_adapters():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    results = []

    def _do_load_all():
        with STATE.lock:
            for name, path in ADAPTER_PATHS.items():
                if name in STATE.loaded_adapters:
                    results.append({"name": name, "loaded": True, "message": "sudah dimuat"})
                    continue
                if not os.path.isdir(path):
                    results.append({"name": name, "loaded": False, "message": f"folder tidak ketemu: {path}"})
                    continue
                STATE.model.load_adapter(path, adapter_name=name)
                results.append({"name": name, "loaded": True, "message": "berhasil dimuat"})

    await run_in_threadpool(_do_load_all)
    return results


@app.post("/adapters/unload_all")
async def unload_all_adapters():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    results = []

    def _do_unload_all():
        with STATE.lock:
            for name in list(STATE.loaded_adapters):
                STATE.model.delete_adapter(name)
                results.append({"name": name, "loaded": False})

    await run_in_threadpool(_do_unload_all)
    return results


# ===========================================================================
# UI SEDERHANA (1 halaman, vanilla JS, tanpa build step)
# ===========================================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>Liana Pipeline</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 16px; background: #0f1115; color: #e6e6e6; }
  h1 { font-size: 20px; }
  h2 { font-size: 15px; color: #9aa4b2; margin-top: 28px; }
  textarea { width: 100%; height: 70px; font-size: 15px; padding: 10px; border-radius: 8px; border: 1px solid #333; background: #1a1d24; color: #e6e6e6; box-sizing: border-box; }
  button { background: #4f8cff; color: white; border: none; padding: 9px 18px; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 8px; }
  button:hover { background: #3d78e6; }
  button.small { padding: 4px 10px; font-size: 12px; }
  button.danger { background: #e05555; }
  button.danger:hover { background: #c94444; }
  .adapters { display: flex; flex-wrap: wrap; gap: 8px; }
  .adapter-chip { display: flex; align-items: center; gap: 8px; background: #1a1d24; border: 1px solid #333; border-radius: 20px; padding: 6px 6px 6px 14px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; }
  .dot.on { background: #4caf50; }
  .dot.off { background: #555; }
  .result { margin-top: 20px; }
  .segment { background: #1a1d24; border: 1px solid #333; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge.valid { background: #16351f; color: #4caf50; }
  .badge.invalid { background: #3a1a1a; color: #e05555; }
  .badge.note { background: #332b12; color: #d4a83a; }
  pre { background: #0f1115; padding: 10px; border-radius: 6px; overflow-x: auto; font-size: 12.5px; }
  #loadall, #unloadall { margin-left: 6px; }
</style>
</head>
<body>
  <h1>🧠 Liana Pipeline</h1>
  <p style="color:#9aa4b2; font-size:13px;">router &rarr; specialist &rarr; validator, 1 base model + 9 adapter LoRA</p>

  <h2>Adapter aktif</h2>
  <div class="adapters" id="adapters">memuat...</div>
  <div style="margin-top:10px;">
    <button class="small" id="loadall" onclick="loadAll()">Load semua</button>
    <button class="small danger" id="unloadall" onclick="unloadAll()">Unload semua</button>
  </div>

  <h2>Coba kalimat</h2>
  <textarea id="text" placeholder="Contoh: Buka spotify terus putar lagu Noah.">Buka spotify terus putar lagu Noah.</textarea><br>
  <button onclick="runPipeline()">Jalankan</button>
  <span id="status" style="margin-left:10px; color:#9aa4b2; font-size:13px;"></span>

  <div class="result" id="result"></div>

<script>
async function fetchAdapters() {
  const res = await fetch('/adapters');
  const data = await res.json();
  const el = document.getElementById('adapters');
  el.innerHTML = '';
  data.forEach(a => {
    const chip = document.createElement('div');
    chip.className = 'adapter-chip';
    chip.innerHTML = `
      <span class="dot ${a.loaded ? 'on' : 'off'}"></span>
      <span>${a.name}</span>
      <button class="small" onclick="toggleAdapter('${a.name}', ${a.loaded})">${a.loaded ? 'Unload' : 'Load'}</button>
    `;
    el.appendChild(chip);
  });
}

async function toggleAdapter(name, currentlyLoaded) {
  const action = currentlyLoaded ? 'unload' : 'load';
  await fetch(`/adapters/${name}/${action}`, { method: 'POST' });
  fetchAdapters();
}

async function loadAll() {
  await fetch('/adapters/load_all', { method: 'POST' });
  fetchAdapters();
}

async function unloadAll() {
  await fetch('/adapters/unload_all', { method: 'POST' });
  fetchAdapters();
}

async function runPipeline() {
  const text = document.getElementById('text').value;
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  statusEl.textContent = 'menjalankan...';
  resultEl.innerHTML = '';
  try {
    const res = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    statusEl.textContent = data.elapsed_sec ? `selesai dalam ${data.elapsed_sec}s` : '';
    if (data.error) {
      resultEl.innerHTML = `<div class="segment"><span class="badge invalid">ERROR</span><p>${data.error}</p></div>`;
      return;
    }
    data.segments.forEach(seg => {
      let html = `<div class="segment"><b>domain:</b> ${seg.domain} &nbsp; <b>teks:</b> "${seg.text}"<br>`;
      if (seg.note) {
        html += `<span class="badge note">CATATAN</span> ${seg.note}`;
      } else {
        html += `<pre>${JSON.stringify(seg.task_ir, null, 2)}</pre>`;
        if (seg.validation) {
          if (seg.validation.label === 'valid') {
            html += `<span class="badge valid">VALID</span>`;
          } else if (seg.validation.label === 'invalid') {
            html += `<span class="badge invalid">INVALID</span> alasan: ${seg.validation.reason}`;
          } else {
            html += `<span class="badge note">tidak diketahui</span>`;
          }
        }
      }
      html += `</div>`;
      resultEl.innerHTML += html;
    });
  } catch (e) {
    statusEl.textContent = '';
    resultEl.innerHTML = `<div class="segment"><span class="badge invalid">ERROR</span> ${e}</div>`;
  }
}

fetchAdapters();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
