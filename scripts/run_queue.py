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

--parallel K runs up to K languages at once. Use it when the work is light (for example
adding one small model to finished languages): each language still goes through its stages
in order, and the GPU-memory and judge-slot gates in run_benchmark.py / h200_run.sh keep K
languages from overrunning the card. Each language then gets its own log file
(<log-prefix>_<iso>.log), because the publisher pairs a language's "BENCHMARKING" line with
its "All benchmarks finished" line by position in a single file.
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import yamlio  # noqa: E402


def complete(iso, only_model=None):
    path = ROOT / "benchmarks" / f"{iso}.yaml"
    try:
        doc = yamlio.load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    rows = doc.get("benchmarks") or []
    if not rows or not all("composite" in b and not b.get("partial") for b in rows):
        return False
    if only_model and not any(b.get("model", "").startswith(only_model) for b in rows):
        return False                     # the model's row is not in the results yet
    total = doc.get("num_samples") or 200
    shares = sorted((b.get("failed") or 0) / total for b in rows)
    return shares[len(shares) // 2] < 0.5


def running_isos():
    """Languages that already have a run_benchmark process (never start a duplicate)."""
    out = subprocess.run(["pgrep", "-af", "scripts/run_benchmark.py --iso"], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        if "pgrep" in line or "bash -c" in line or "--iso" not in line:
            continue
        found.add(line.split("--iso")[1].split()[0])
    return found


def registered(iso, model):
    from benchmark.evaluate import load_tts_models

    return any(m["name"] == model or m.get("model_id") == model for m in load_tts_models(iso))


def run_language(iso, args, log_path=None):
    """All attempts for one language. Returns True once its results are complete."""
    cmd = [sys.executable, "-u", "scripts/run_benchmark.py", "--iso", iso] + (
        ["--only-model", args.only_model] if args.only_model else [])
    for attempt in range(1 + args.repairs):
        if iso in running_isos():
            print(f"{time.strftime('%F %T')} === {iso} is already running; leaving it", flush=True)
            return None
        print(f"{time.strftime('%F %T')} === {iso} (attempt {attempt + 1})", flush=True)
        if log_path:
            with open(log_path, "a") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        else:
            subprocess.run(cmd, cwd=ROOT)
        if complete(iso, args.only_model):
            print(f"{time.strftime('%F %T')} === {iso} complete", flush=True)
            return True
        print(f"{time.strftime('%F %T')} === {iso} incomplete; "
              f"{'retrying' if attempt < args.repairs else 'giving up, moving on'}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("isos", nargs="*", help="languages, in the order to run them")
    ap.add_argument("--remaining", action="store_true", help="every language not yet complete")
    ap.add_argument("--repairs", type=int, default=2, help="extra attempts for a language that ends incomplete")
    ap.add_argument("--only-model", default=None, help="add just this registered model to each language")
    ap.add_argument("--parallel", type=int, default=1, help="languages to run at once (default 1: strictly serial)")
    ap.add_argument("--log-dir", default=None, help="with --parallel: write one log per language here")
    ap.add_argument("--log-prefix", default="queue", help="log file prefix, <prefix>_<iso>.log")
    args = ap.parse_args()

    from benchmark.config import all_isos

    isos = args.isos
    if not isos and args.remaining:
        isos = [i for i in all_isos()
                if (not args.only_model or registered(i, args.only_model)) and not complete(i, args.only_model)]
    if not isos:
        ap.error("give languages, or --remaining")
    print(f"{time.strftime('%F %T')} queue: {' '.join(isos)}  (parallel={args.parallel})", flush=True)

    todo, lock = list(isos), threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                iso = todo.pop(0)
            log = Path(args.log_dir) / f"{args.log_prefix}_{iso}.log" if args.log_dir and args.parallel > 1 else None
            run_language(iso, args, log)
            time.sleep(2)

    threads = []
    for k in range(max(1, args.parallel)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(20)      # stagger the starts so model loads do not all hit the GPU at once
    for t in threads:
        t.join()
    print(f"{time.strftime('%F %T')} queue finished", flush=True)


if __name__ == "__main__":
    main()
