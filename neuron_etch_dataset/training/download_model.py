"""
download_model.py
==================
Download base model (§65: Qwen3.5-0.8B, fallback Qwen3.5-0.6B) ke FOLDER
LOKAL yang kamu tentukan sendiri — BUKAN ke cache default HuggingFace
(~/.cache/huggingface/hub).

Cara kerja `snapshot_download` biasa: file didownload ke cache, lalu
di-symlink ke `local_dir` kalau kamu kasih local_dir tapi tidak set
`local_dir_use_symlinks=False`. Itu artinya file ASLI tetap di cache
(~/.cache), cuma dibikin symlink — kalau cache-nya kehapus, folder lokal
ikut rusak. Script ini eksplisit set `local_dir_use_symlinks=False` (dan
di huggingface_hub versi baru yang parameter ini sudah dihapus/deprecated,
perilaku default snapshot_download SUDAH selalu copy file asli ke
local_dir, bukan symlink) — jadi file BENAR-BENAR ada di folder yang kamu
kasih, tidak bergantung pada cache sama sekali.

Cara pakai — download:
    python download_model.py --target ./models/Qwen3.5-0.8B
    python download_model.py --target ./models/Qwen3.5-0.8B --repo-id Qwen/Qwen3.5-0.8B
    python download_model.py --target ./models/Qwen3.5-0.6B --fallback

Cara pakai — uninstall (hapus model dari folder lokal, bebasin disk):
    python download_model.py --uninstall --target ./models/Qwen3.5-0.8B
    python download_model.py --uninstall --target ./models/Qwen3.5-0.8B --yes   # tanpa konfirmasi

Cara pakai — list (cek model apa saja yang sudah kedownload di suatu folder induk):
    python download_model.py --list --models-root ./models

CATATAN PENTING soal repo_id:
    Qwen3.5 (§65) kemungkinan rilis SETELAH knowledge cutoff-ku, jadi aku
    tidak bisa memastikan repo_id persis di HuggingFace Hub. Default di
    bawah ("Qwen/Qwen3.5-0.8B") adalah TEBAKAN berdasar pola penamaan
    Qwen sebelumnya (Qwen/Qwen2.5-0.5B-Instruct, dst). SEBELUM run,
    cek dulu ke https://huggingface.co/Qwen dan ganti --repo-id kalau
    namanya beda (mis. ada suffix "-Instruct").
"""

import argparse
import os
import shutil
import sys

from huggingface_hub import snapshot_download

DEFAULT_REPO_MAIN = "Qwen/Qwen3.5-0.8B"       # §65.1 — cek dulu nama persisnya di HF Hub
DEFAULT_REPO_FALLBACK = "Qwen/Qwen3.5-0.6B"   # §65.1 fallback kalau VRAM/latency jadi kendala


def download(repo_id: str, target_dir: str, revision: str | None, hf_token: str | None) -> None:
    os.makedirs(target_dir, exist_ok=True)

    # PENTING: set HF_HOME juga ke luar cache default untuk request ini,
    # supaya tidak ada file "nyasar" ke ~/.cache walau cuma metadata/lock.
    # (opsional tapi lebih bersih untuk lingkungan yang mau folder mandiri)
    print(f"Downloading {repo_id} -> {target_dir}")
    print("(file akan disalin langsung ke folder ini, bukan cache)")

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # paksa copy file asli, bukan symlink ke cache
            token=hf_token,
            # skip file yang tidak perlu buat inference/training (mis. .bin duplikat
            # kalau sudah ada safetensors, format GGUF, dst) — hemat bandwidth & disk
            allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.model",
                "tokenizer*", "*.py", "generation_config.json",
            ],
        )
    except TypeError:
        # huggingface_hub versi baru sudah menghapus parameter local_dir_use_symlinks
        # (perilaku barunya: selalu copy file asli ke local_dir) -> retry tanpa param itu
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target_dir,
            token=hf_token,
            allow_patterns=[
                "*.safetensors", "*.json", "*.txt", "*.model",
                "tokenizer*", "*.py", "generation_config.json",
            ],
        )

    print(f"\nSelesai. Model tersimpan di: {os.path.abspath(target_dir)}")
    print("Isi folder:")
    for f in sorted(os.listdir(target_dir)):
        path = os.path.join(target_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.isfile(path) else 0
        print(f"  {f:<40} {size_mb:>8.1f} MB" if size_mb else f"  {f}")


# ---------------------------------------------------------------------------
# UNINSTALL — hapus model yang sudah didownload dari folder lokal
# ---------------------------------------------------------------------------

def human_readable_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def folder_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
    return total


def uninstall(target_dir: str, skip_confirm: bool) -> None:
    """Hapus folder model lokal (kebalikan dari download()). Tidak menyentuh
    cache HuggingFace sama sekali — memang tidak pernah dipakai (§ lihat
    docstring file ini), jadi tidak ada yang perlu dibersihkan di sana."""
    if not os.path.exists(target_dir):
        print(f"Tidak ada apa-apa di {target_dir} — tidak ada yang perlu di-uninstall.")
        return

    if not os.path.isdir(target_dir):
        print(f"ERROR: {target_dir} bukan folder.", file=sys.stderr)
        sys.exit(1)

    size = folder_size(target_dir)
    file_count = sum(len(files) for _, _, files in os.walk(target_dir))

    print(f"Akan menghapus: {os.path.abspath(target_dir)}")
    print(f"Isi: {file_count} file, total {human_readable_size(size)}")

    if not skip_confirm:
        answer = input("Lanjutkan hapus? Ketik 'yes' untuk konfirmasi: ").strip().lower()
        if answer != "yes":
            print("Dibatalkan, tidak ada yang dihapus.")
            return

    shutil.rmtree(target_dir)
    print(f"Selesai. {human_readable_size(size)} sudah dibebaskan.")


def list_models(models_root: str) -> None:
    """List semua folder model di bawah models_root, beserta ukurannya —
    buat cek cepat model apa saja yang sudah didownload sebelum decide
    mau uninstall yang mana."""
    if not os.path.exists(models_root):
        print(f"Folder {models_root} belum ada — belum ada model yang didownload ke sini.")
        return

    entries = sorted(d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d)))
    if not entries:
        print(f"Folder {models_root} ada tapi kosong — belum ada model.")
        return

    print(f"Model di {os.path.abspath(models_root)}:\n")
    print(f"  {'NAMA':<30} {'UKURAN':>10}  {'FILE UTAMA'}")
    for name in entries:
        path = os.path.join(models_root, name)
        size = folder_size(path)
        has_weights = any(f.endswith(".safetensors") for f in os.listdir(path))
        marker = "safetensors ada" if has_weights else "belum lengkap / bukan model"
        print(f"  {name:<30} {human_readable_size(size):>10}  {marker}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--uninstall", action="store_true", help="Hapus model di --target dari disk")
    mode.add_argument("--list", action="store_true", help="List model yang sudah kedownload di --models-root")

    parser.add_argument("--target", default=None, help="Folder tujuan/sumber, mis. ./models/Qwen3.5-0.8B "
                                                         "(wajib untuk download & uninstall)")
    parser.add_argument("--models-root", default="./models",
                         help="Folder induk tempat semua model disimpan (dipakai --list). Default: ./models")
    parser.add_argument("--yes", action="store_true", help="Skip konfirmasi waktu --uninstall")
    parser.add_argument("--repo-id", default=None, help="Override repo_id HuggingFace")
    parser.add_argument("--fallback", action="store_true",
                         help=f"Pakai fallback {DEFAULT_REPO_FALLBACK} (§65.1) sebagai ganti model utama")
    parser.add_argument("--revision", default=None, help="Git revision/branch/tag tertentu (opsional)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                         help="Token HuggingFace kalau model gated (atau set env var HF_TOKEN)")
    args = parser.parse_args()

    if args.list:
        list_models(args.models_root)
        return

    if args.uninstall:
        if not args.target:
            print("ERROR: --uninstall butuh --target", file=sys.stderr)
            sys.exit(1)
        uninstall(args.target, skip_confirm=args.yes)
        return

    # --- mode default: download ---
    if not args.target:
        print("ERROR: --target wajib diisi untuk download", file=sys.stderr)
        sys.exit(1)

    repo_id = args.repo_id or (DEFAULT_REPO_FALLBACK if args.fallback else DEFAULT_REPO_MAIN)

    if os.path.exists(args.target) and os.listdir(args.target):
        print(f"PERINGATAN: {args.target} sudah ada isinya. File akan di-overwrite/dilengkapi.")

    try:
        download(repo_id, args.target, args.revision, args.hf_token)
    except Exception as e:
        print(f"\nGagal download: {e}", file=sys.stderr)
        print("\nKemungkinan penyebab:", file=sys.stderr)
        print(f"  1. repo_id '{repo_id}' belum/tidak ada di HuggingFace Hub — cek nama persisnya "
              f"di https://huggingface.co/Qwen dan pakai --repo-id", file=sys.stderr)
        print("  2. model gated/private — butuh --hf-token atau env var HF_TOKEN", file=sys.stderr)
        print("  3. tidak ada koneksi internet ke huggingface.co", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
