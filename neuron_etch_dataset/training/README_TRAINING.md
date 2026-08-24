# Training pipeline — Neuron-Etch vNext

Pipeline lengkap dari dataset mentah (`../output/`) sampai adapter LoRA
siap dipakai, mengikuti §65-72 dokumen arsitektur.

## 0. Sebelum mulai — batasan yang perlu kamu tahu

1. **Repo ID Qwen3.5 belum bisa dipastikan.** Qwen3.5 kemungkinan rilis
   setelah training-data cutoff-ku, jadi `download_model.py` pakai
   tebakan `Qwen/Qwen3.5-0.8B`. **Cek dulu** ke https://huggingface.co/Qwen
   sebelum download, ganti dengan `--repo-id` kalau namanya beda.
2. **Dataset masih kecil.** Ini dibuat buat kamu lanjutkan sendiri (lihat
   percakapan sebelumnya) — beberapa kategori task masih 0 sample. Training
   akan tetap JALAN dengan dataset kecil, tapi hasilnya tidak akan bagus.
   `build_training_data.py` akan warning kategori mana yang < 10 sample.
3. **Beberapa bagian skema training aku asumsikan** karena tidak dirinci
   eksplisit di dokumen (nilai `confidence` numerik, skema JSON untuk
   sample "negative"/rejected). Detail lengkap ada di komentar kepala
   file `chatml_format.py` — baca itu dan sesuaikan kalau kamu sudah
   punya angka/skema resmi.

## 1. Setup environment

```bash
cd training
pip install -r requirements.txt
```

Butuh GPU dengan VRAM cukup untuk model 0.8B + LoRA (biasanya muat di
GPU 8GB+ dengan bf16). Bisa jalan di CPU juga tapi jauh lebih lambat.

## 2. Download base model — ke FOLDER, bukan cache

```bash
python download_model.py --target ./models/Qwen3.5-0.8B
```

File model akan ada FISIK di `./models/Qwen3.5-0.8B/` (bukan symlink ke
`~/.cache/huggingface`). Kalau model utama kegedean untuk hardware kamu,
pakai fallback (§65.1):

```bash
python download_model.py --target ./models/Qwen3.5-0.6B --fallback
```

**Cek model apa saja yang sudah kedownload:**

```bash
python download_model.py --list --models-root ./models
```

**Uninstall (hapus dari disk kalau mau bebasin ruang / ganti model):**

```bash
python download_model.py --uninstall --target ./models/Qwen3.5-0.8B
# atau tanpa konfirmasi interaktif (buat script/CI):
python download_model.py --uninstall --target ./models/Qwen3.5-0.8B --yes
```

Uninstall cuma hapus folder lokal itu — tidak menyentuh apa pun di
`~/.cache/huggingface`, karena cache itu memang tidak pernah dipakai
sama sekali oleh `download_model.py` (lihat catatan di kepala file).

## 3. Lengkapi dataset (bagian kamu)

Edit file di `../templates.py` untuk isi kategori yang masih kosong
(lihat pesan `STUB (belum diisi)` waktu jalanin `../main.py`). Setelah
nambah contoh, generate ulang dataset mentah:

```bash
cd ..
python main.py
cd training
```

## 4. Build training data (convert ke ChatML)

```bash
python build_training_data.py
```

Ini baca semua `../output/<domain>/*.jsonl`, ubah jadi
`training_data/<adapter>.train.jsonl` + `.val.jsonl` format:

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Buka foot"},
  {"role": "assistant", "content": "{\"domain\":\"system\",...}"}
]}
```

## 5. Training — urutan yang direkomendasikan (§59)

Jangan latih 9 adapter sekaligus. Urutan: Router → System → Persona →
Media (jalankan benchmark §58 dulu) → Information → Coding → Memory →
Productivity → Validator. Atau kalau mau MVP dulu (§60): Router, System,
Persona, Media, Validator saja (5 adapter).

Ada 2 pilihan script training — hasil akhirnya (format adapter) **identik**,
tinggal pilih salah satu:

| | `train_lora.py` | `train_lora_unsloth.py` |
|---|---|---|
| Dependency | transformers + peft biasa | + library `unsloth` |
| Kecepatan training | baseline | ~2-4x lebih cepat (kernel dipatch) |
| VRAM | baseline | jauh lebih hemat, ada opsi 4-bit (QLoRA) |
| Hardware | CPU/GPU apa saja | GPU NVIDIA (CUDA) saja |
| Adapter hasil | format PEFT standar | format PEFT standar (sama persis) |

Kalau kamu punya GPU NVIDIA, `train_lora_unsloth.py` biasanya pilihan
lebih enak — model 0.8B jadi muat & cepat dilatih bahkan di GPU consumer
kecil. Install dulu (di luar `requirements.txt` karena command-nya beda
tergantung versi CUDA kamu):

```bash
pip install unsloth
# kalau gagal / butuh versi CUDA spesifik, ikuti:
# https://github.com/unslothai/unsloth#installation
```

```bash
python train_lora.py \
    --adapter router \
    --base-model-dir ./models/Qwen3.5-0.8B \
    --train-file training_data/router.train.jsonl \
    --val-file training_data/router.val.jsonl \
    --output-dir ./adapters/router_core \
    --epochs 3
```

atau versi Unsloth (argumen SAMA PERSIS, cukup ganti nama script; ada
tambahan `--load-in-4bit` kalau VRAM sangat terbatas):

```bash
python train_lora_unsloth.py \
    --adapter router \
    --base-model-dir ./models/Qwen3.5-0.8B \
    --train-file training_data/router.train.jsonl \
    --val-file training_data/router.val.jsonl \
    --output-dir ./adapters/router_core \
    --epochs 3
```

Ulangi untuk tiap adapter (ganti `--adapter`, `--train-file`, `--val-file`,
`--output-dir`). Hyperparameter default (`--lora-r 16 --lora-alpha 32
--lr 2e-4 --batch-size 4 --grad-accum 4`) adalah titik awal yang wajar
untuk model 0.8B — sesuaikan lagi kalau overfitting (dataset kecil) atau
underfitting.

Hasil training kesimpan 2 tempat (sama untuk kedua script):
- `./adapters/<adapter_name>/` — raw output PEFT (buat lanjut training / debug)
- `./adapters_versioned/<name>.v1.0.safetensors` — nama resmi sesuai §72.1
  (`router_core.v1.0.safetensors`, `system_specialist.v1.0.safetensors`, dst)

Karena adapter yang dihasilkan Unsloth tetap format PEFT standar,
`inference_smoke_test.py` bisa load adapter dari kedua script tanpa
perlu Unsloth terinstall di mesin yang dipakai buat serving/inference.

## 6. Smoke test

```bash
python inference_smoke_test.py --adapter system \
    --base-model-dir ./models/Qwen3.5-0.8B \
    --adapter-dir ./adapters/system_specialist \
    --text "Buka foot"
```

Ini bukan eval harness resmi (§70) — cuma cek cepat outputnya JSON valid
dan masuk akal. Untuk eval resmi per §70, siapkan golden test set format
§70.2 secara terpisah (jangan dari data training) dan bikin script eval
sendiri yang hitung metrik §70.3 (domain_accuracy dkk) + confusion matrix
§70.4.

## 7. Sebelum promosi ke "production" (§72.2)

Adapter baru TIDAK BOLEH gantikan versi aktif tanpa lolos regression gate:
1. Full run ke golden test set versi baru
2. Semua metrik §70.3 tidak boleh turun > toleransi yang kamu sepakati
3. Hard negative set (§42) khususnya tidak boleh regresi

File ini semua di luar scope pipeline training — perlu dibangun terpisah
kalau kamu sudah siap ke tahap itu.
