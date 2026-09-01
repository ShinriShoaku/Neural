#!/usr/bin/env python3
"""
Liana Pipeline Tester & Benchmark
=================================

Script untuk:
1. Cek status server pipeline.
2. Kirim beberapa input ke /run.
3. Tampilkan hasil parsing & validasi per segment.
4. Benchmark waktu eksekusi per input.

Usage:
    python test_pipeline.py --url http://127.0.0.1:8000 --max-tokens 120

Dependencies:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:
    print("Error: 'requests' belum terinstall.")
    print("Install dengan: pip install requests")
    sys.exit(1)


# ============================================================================
# KONFIGURASI DEFAULT
# ============================================================================

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MAX_TOKENS = 120

# Contoh input untuk menguji berbagai domain specialist.
TEST_INPUTS = [
    # System / Media
    "Buka aplikasi Spotify dan putar lagu terbaru.",
    # Media
    "Cari video tutorial Python di YouTube.",
    # Persona
    "Panggil saya dengan nama 'miku-sama'.",
    # Coding
    "Buatkan fungsi Python untuk menghitung faktorial.",
    # Information
    "Cari tahu cuaca hari ini di Jakarta.",
    # Memory
    "Ingatkan saya bahwa meeting besok jam 9 pagi.",
    # Productivity
    "Buatkan todo list untuk hari ini: belajar, olahraga, tidur.",
    # Multi-intent / Ambiguous
    "Buka YouTube terus cari video tentang machine learning.",
    # Negative / non-actionable
    "Halo, apa kabar?",
]


# ============================================================================
# UTILS
# ============================================================================

def print_banner():
    print("=" * 70)
    print("  LIANA PIPELINE TESTER & BENCHMARK")
    print("=" * 70)
    print()


def print_section(title: str):
    print()
    print(f"┌{'─' * 68}┐")
    print(f"│ {title:<66} │")
    print(f"└{'─' * 68}┘")


def print_box(label: str, content: str, width: int = 68):
    lines = content.splitlines()
    print(f"┌─ {label} {'─' * (width - len(label) - 4)}┐")
    for line in lines:
        safe = line[:width - 2]
        print(f"│ {safe:<{width - 2}}│")
    print(f"└{'─' * width}┘")


def print_json_box(label: str, data: Any, width: int = 68):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print_box(label, text, width)


def print_separator():
    print("─" * 70)


# ============================================================================
# API CLIENT  (kompatibel dengan api_server.py)
# ============================================================================

class PipelineClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_health(self) -> dict[str, Any]:
        """GET /health -> {status, model_loaded, loaded_adapters}"""
        resp = self.session.get(self._url("/health"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_adapters(self) -> list[dict]:
        """GET /adapters -> list[{name, path, loaded}]"""
        resp = self.session.get(self._url("/adapters"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def post_run(self, text: str, max_new_tokens: int = 120) -> dict[str, Any]:
        """POST /run -> {elapsed_sec, segments, error?}"""
        resp = self.session.post(
            self._url("/run"),
            json={"text": text, "max_new_tokens": max_new_tokens},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def post_load_adapter(self, name: str) -> dict[str, Any]:
        resp = self.session.post(self._url(f"/adapters/{name}/load"), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def post_unload_adapter(self, name: str) -> dict[str, Any]:
        resp = self.session.post(self._url(f"/adapters/{name}/unload"), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def post_load_all(self) -> list[dict]:
        resp = self.session.post(self._url("/adapters/load_all"), timeout=120)
        resp.raise_for_status()
        return resp.json()

    def post_unload_all(self) -> list[dict]:
        resp = self.session.post(self._url("/adapters/unload_all"), timeout=120)
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# BENCHMARK
# ============================================================================

@dataclass
class BenchmarkResult:
    input_text: str
    ok: bool
    elapsed_sec: float
    result: dict[str, Any] | None
    error: str | None


def run_benchmark(
    client: PipelineClient,
    inputs: list[str],
    max_new_tokens: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for idx, text in enumerate(inputs, 1):
        print_separator()
        print(f"  TEST #{idx}/{len(inputs)}")
        print(f"  Input: {text[:60]}{'...' if len(text) > 60 else ''}")
        print()

        t0 = time.time()
        try:
            data = client.post_run(text, max_new_tokens)
            elapsed = time.time() - t0

            # api_server.py mengembalikan {error?, elapsed_sec, segments}
            if data.get("error"):
                print(f"  ❌ Pipeline Error: {data['error']}")
                results.append(
                    BenchmarkResult(
                        input_text=text,
                        ok=False,
                        elapsed_sec=elapsed,
                        result=None,
                        error=data["error"],
                    )
                )
                continue

            segments = data.get("segments", [])
            pipeline_time = data.get("elapsed_sec", 0.0)
            print(f"  ✅ Pipeline OK  (server_time={pipeline_time:.3f}s | total={elapsed:.3f}s)")
            print()

            for seg in segments:
                domain = seg.get("domain", "?")
                seg_text = seg.get("text", "")
                task_ir = seg.get("task_ir")
                validation = seg.get("validation")
                note = seg.get("note")

                print(f"    ┌─ Domain: {domain}")
                print(f"    │  Teks  : {seg_text[:55]}{'...' if len(seg_text) > 55 else ''}")
                if task_ir:
                    print(f"    │  Intent : {task_ir.get('intent', 'N/A')}")
                    print(f"    │  Action : {task_ir.get('action', 'N/A')}")
                    print(f"    │  Target : {task_ir.get('target', 'N/A')}")
                    params = task_ir.get("parameters", {})
                    if params:
                        print(f"    │  Params : {json.dumps(params, ensure_ascii=False)}")
                if validation:
                    label = validation.get("label", "?")
                    reason = validation.get("reason")
                    print(f"    │  Valid  : {label}")
                    if reason:
                        print(f"    │  Reason : {reason}")
                if note:
                    print(f"    │  Note   : {note}")
                print(f"    └")
                print()

            results.append(
                BenchmarkResult(
                    input_text=text,
                    ok=True,
                    elapsed_sec=elapsed,
                    result=data,
                    error=None,
                )
            )

        except requests.HTTPError as exc:
            elapsed = time.time() - t0
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            print(f"  ❌ HTTP Error: {detail}")
            results.append(
                BenchmarkResult(
                    input_text=text,
                    ok=False,
                    elapsed_sec=elapsed,
                    result=None,
                    error=detail,
                )
            )

        except requests.RequestException as exc:
            elapsed = time.time() - t0
            print(f"  ❌ Network Error: {exc}")
            results.append(
                BenchmarkResult(
                    input_text=text,
                    ok=False,
                    elapsed_sec=elapsed,
                    result=None,
                    error=str(exc),
                )
            )

    return results


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list[BenchmarkResult]):
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total = len(results)
    success = sum(1 for r in results if r.ok)
    failed = total - success
    avg_time = sum(r.elapsed_sec for r in results) / total if total else 0
    max_time = max((r.elapsed_sec for r in results), default=0)
    min_time = min((r.elapsed_sec for r in results), default=0)

    print(f"  Total tests : {total}")
    print(f"  Success     : {success}")
    print(f"  Failed      : {failed}")
    print(f"  Avg time    : {avg_time:.3f}s")
    print(f"  Min time    : {min_time:.3f}s")
    print(f"  Max time    : {max_time:.3f}s")
    print()

    if failed:
        print("  Failed inputs:")
        for r in results:
            if not r.ok:
                print(f"    - {r.input_text[:50]}... | Error: {r.error}")
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test & Benchmark Liana Pipeline Server"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max new tokens per generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--custom-input",
        default=None,
        help="Gunakan single custom input, abaikan test set default.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="File berisi list input (satu per baris).",
    )
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Load semua adapter sebelum mulai benchmark.",
    )

    args = parser.parse_args()

    print_banner()

    client = PipelineClient(args.url)

    # ── Cek health server ──────────────────────────────────────────────
    print_section("SERVER HEALTH")
    try:
        health = client.get_health()
        model_loaded = health.get("model_loaded", False)
        adapters = health.get("loaded_adapters", [])
        print(f"  Status       : {health.get('status', 'unknown')}")
        print(f"  Model loaded : {'✅ YES' if model_loaded else '❌ NO'}")
        print(f"  Adapters     : {', '.join(adapters) if adapters else '(none)'}")
    except requests.RequestException as exc:
        print(f"  ❌ Gagal terhubung ke server: {exc}")
        sys.exit(1)

    if not model_loaded:
        print()
        print("  ⚠️  Model belum siap. Load model terlebih dahulu.")
        print(f"     Buka {args.url} di browser atau load adapter via API.")
        sys.exit(1)

    # ── Load semua adapter (opsional) ──────────────────────────────────
    if args.load_all:
        print_section("LOAD ALL ADAPTERS")
        load_results = client.post_load_all()
        for item in load_results:
            name = item.get("name", "?")
            loaded = item.get("loaded", False)
            msg = item.get("message", "")
            icon = "✅" if loaded else "❌"
            print(f"  {icon} {name:<15} {msg}")
        print()

    # ── Tentukan input list ────────────────────────────────────────────
    if args.custom_input:
        inputs = [args.custom_input]
    elif args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                inputs = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ File tidak ditemukan: {args.input_file}")
            sys.exit(1)
    else:
        inputs = TEST_INPUTS

    print(f"  Total input yang akan diuji: {len(inputs)}")

    # ── Jalankan benchmark ─────────────────────────────────────────────
    print_section("BENCHMARK START")
    overall_t0 = time.time()

    results = run_benchmark(
        client,
        inputs,
        args.max_tokens,
    )

    overall_elapsed = time.time() - overall_t0

    print()
    print(f"  Overall elapsed: {overall_elapsed:.3f}s")

    # ── Summary ────────────────────────────────────────────────────────
    print_summary(results)

    print("Done.")


if __name__ == "__main__":
    main()
