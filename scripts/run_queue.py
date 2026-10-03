#!/usr/bin/env python3
"""Run the benchmark strictly one language at a time.

    python3 scripts/run_queue.py sfw sig sil snw tpm vag xon
    python3 scripts/run_queue.py --remaining          # every language without a finished result

Why serial: the GPU is shared, and running several languages (each with its own
synthesis and judge containers) at once made every one of them slow and memory
hungry. One language at a time, and within it one stage and one model at a time
(synthesis, OmniVoice, the ASR judge and SpeechBERTScore each load a single model
and release it), keeps the footprint small and predictable.

Each language is its own run_benchmark.py process, so the log has one
"BENCHMARKING: <iso>" ... "All benchmarks finished" pair per language, which is
what scripts/sync_results.py reads to decide a language is done. A language that
ends incomplete is re-run, up to --repairs more times, before the queue moves on.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import yamlio  # noqa: E402


def complete(iso):
    path = ROOT / "benchmarks" / f"{iso}.yaml"
    try:
        doc = yamlio.load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    rows = doc.get("benchmarks") or []
    if not rows or not all("composite" in b and not b.get("partial") for b in rows):
        return False
    total = doc.get("num_samples") or 200
    shares = sorted((b.get("failed") or 0) / total for b in rows)
    return shares[len(shares) // 2] < 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("isos", nargs="*", help="languages, in the order to run them")
    ap.add_argument("--remaining", action="store_true", help="every language not yet complete")
    ap.add_argument("--repairs", type=int, default=2, help="extra attempts for a language that ends incomplete")
    args = ap.parse_args()

    from benchmark.config import all_isos

    isos = args.isos or ([i for i in all_isos() if not complete(i)] if args.remaining else [])
    if not isos:
        ap.error("give languages, or --remaining")
    print(f"{time.strftime('%F %T')} queue: {' '.join(isos)}", flush=True)

    for iso in isos:
        for attempt in range(1 + args.repairs):
            print(f"{time.strftime('%F %T')} === {iso} (attempt {attempt + 1})", flush=True)
            subprocess.run([sys.executable, "-u", "scripts/run_benchmark.py", "--iso", iso], cwd=ROOT)
            if complete(iso):
                print(f"{time.strftime('%F %T')} === {iso} complete", flush=True)
                break
            print(f"{time.strftime('%F %T')} === {iso} incomplete; "
                  f"{'retrying' if attempt < args.repairs else 'giving up, moving on'}", flush=True)
    print(f"{time.strftime('%F %T')} queue finished", flush=True)


if __name__ == "__main__":
    main()
