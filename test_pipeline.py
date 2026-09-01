#!/usr/bin/env python3
"""
Liana Pipeline Tester & Benchmark
=================================

Script untuk:
1. Cek status server pipeline.
2. Kirim beberapa input ke /api/generate.
3. Tampilkan log realtime selama proses.
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
    # System
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
        # truncate + pad
        safe = line[:width - 2]
        print(f"│ {safe:<{width - 2}}│")
    print(f"└{'─' * width}┘")


def print_json_box(label: str, data: Any, width: int = 68):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    print_box(label, text, width)


def print_separator():
    print("─" * 70)


# ============================================================================
# API CLIENT
# ============================================================================

class PipelineClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_status(self) -> dict[str, Any]:
        resp = self.session.get(self._url("/api/status"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_logs(self, after: int = 0) -> list[dict]:
        resp = self.session.get(
            self._url("/api/logs"),
            params={"after": after},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("logs", [])

    def post_generate(self, text: str, max_new_tokens: int = 120) -> dict[str, Any]:
        resp = self.session.post(
            self._url("/api/generate"),
            json={"text": text, "max_new_tokens": max_new_tokens},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()


# ============================================================================
# LOG POLLER
# ============================================================================

class LogPoller:
    def __init__(self, client: PipelineClient):
        self.client = client
        self.last_log_id = 0
        self._seen_ids: set[int] = set()

    def poll_new_logs(self) -> list[dict]:
        """Ambil log baru sejak terakhir dipanggil."""
        try:
            logs = self.client.get_logs(after=self.last_log_id)
        except requests.RequestException:
            return []

        new_logs = []
        for item in logs:
            log_id = item.get("id", 0)
            if log_id not in self._seen_ids:
                self._seen_ids.add(log_id)
                new_logs.append(item)
                if log_id > self.last_log_id:
                    self.last_log_id = log_id

        return new_logs

    def print_new_logs(self) -> int:
        """Cetak log baru, kembalikan jumlah log yang dicetak."""
        logs = self.poll_new_logs()
        for item in logs:
            t = item.get("time", "??:??:??")
            level = item.get("level", "INFO")
            msg = item.get("message", "")
            level_color = {
                "INFO": "[36m",      # cyan
                "WARNING": "[33m",   # yellow
                "ERROR": "[31m",     # red
            }.get(level, "[0m")
            reset = "[0m"
            print(
                f"  [{t}] {level_color}[{level:>7}]{reset} {msg}"
            )
        return len(logs)


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
    poll_logs: bool = True,
) -> list[BenchmarkResult]:
    poller = LogPoller(client) if poll_logs else None
    results: list[BenchmarkResult] = []

    for idx, text in enumerate(inputs, 1):
        print_separator()
        print(f"  TEST #{idx}/{len(inputs)}")
        print(f"  Input: {text[:60]}{'...' if len(text) > 60 else ''}")
        print()

        t0 = time.time()
        try:
            data = client.post_generate(text, max_new_tokens)
            elapsed = time.time() - t0
            ok = data.get("ok", False)
            result = data.get("result")

            # Tampilkan hasil
            if ok and result:
                segments = result.get("segments", [])
                pipeline_time = result.get("elapsed_sec", 0.0)
                print(f"  ✅ Pipeline OK  (server_time={pipeline_time:.3f}s | total={elapsed:.3f}s)")
                print()
                for seg in segments:
                    domain = seg.get("domain", "?")
                    task_ir = seg.get("task_ir")
                    validation = seg.get("validation")
                    note = seg.get("note")

                    print(f"    ┌─ Domain: {domain}")
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
            else:
                print(f"  ⚠️  Response tidak OK: {data}")

            results.append(
                BenchmarkResult(
                    input_text=text,
                    ok=ok,
                    elapsed_sec=elapsed,
                    result=result,
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

        # Poll log setelah setiap test
        if poller:
            time.sleep(0.3)  # beri waktu server menulis log
            count = poller.print_new_logs()
            if count:
                print()

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
        "--no-logs",
        action="store_true",
        help="Nonaktifkan polling log.",
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

    args = parser.parse_args()

    print_banner()

    client = PipelineClient(args.url)

    # ── Cek status server ──────────────────────────────────────────────
    print_section("SERVER STATUS")
    try:
        status = client.get_status()
        print(f"  Ready        : {'✅ YES' if status.get('ready') else '❌ NO'}")
        print(f"  Loading      : {'⏳ YES' if status.get('loading') else '   NO'}")
        print(f"  CUDA         : {'✅' if status.get('cuda') else '❌'} {status.get('gpu') or 'CPU'}")
        print(f"  Loaded adapters: {', '.join(status.get('loaded_adapters', []))}")
        if status.get("error"):
            print(f"  Error        : {status['error']}")
    except requests.RequestException as exc:
        print(f"  ❌ Gagal terhubung ke server: {exc}")
        sys.exit(1)

    if not status.get("ready"):
        print()
        print("  ⚠️  Server belum ready. Load model terlebih dahulu via Web UI.")
        print(f"     Buka {args.url} di browser untuk load model.")
        sys.exit(1)

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

    print()
    print(f"  Total input yang akan diuji: {len(inputs)}")

    # ── Jalankan benchmark ─────────────────────────────────────────────
    print_section("BENCHMARK START")
    overall_t0 = time.time()

    results = run_benchmark(
        client,
        inputs,
        args.max_tokens,
        poll_logs=not args.no_logs,
    )

    overall_elapsed = time.time() - overall_t0

    print()
    print(f"  Overall elapsed: {overall_elapsed:.3f}s")

    # ── Summary ────────────────────────────────────────────────────────
    print_summary(results)

    # ── Final log dump (kalau belum semua keambil) ─────────────────────
    if not args.no_logs:
        print_section("REMAINING LOGS")
        poller = LogPoller(client)
        poller.last_log_id = 0  # reset supaya ambil semua log terakhir
        poller._seen_ids.clear()
        count = poller.print_new_logs()
        if not count:
            print("  (tidak ada log tambahan)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
