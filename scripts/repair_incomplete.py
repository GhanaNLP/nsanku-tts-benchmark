#!/usr/bin/env python3
"""Re-run languages that ended incomplete.

A language can reach its end with a stage missing: a CUDA OOM on the shared GPU
while the judge loads, a stage that died under an older worker script. Its YAML
then has no composite (or every model failing), and the sync refuses to publish
it. Every stage resumes from its cache, so re-running the language only does the
missing work. This finds such languages and re-runs them, a few times at most.

    python3 scripts/repair_incomplete.py --once --dry-run   # list candidates
    python3 scripts/repair_incomplete.py --watch            # check every 15 min

A language is a candidate when it is claimed, not running now, and its YAML
fails the same completeness test the sync applies.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import yamlio  # noqa: E402
PROJECTS = ROOT.parent  # claims/ lives next to the checkout on the GPU box
MAX_REPAIRS = 3


def complete(path):
    """Same test as scripts/sync_results.py: composites everywhere, sane failure rate."""
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


def running():
    out = subprocess.run(["pgrep", "-af", "run_benchmark.py --iso"],
                         capture_output=True, text=True).stdout
    return {line.split("--iso")[1].split()[0] for line in out.splitlines()
            if "--iso" in line and "pgrep" not in line and "bash -c" not in line}


def candidates(repairs):
    claimed = sorted(p.name for p in (PROJECTS / "claims").glob("*") if p.is_dir())
    busy = running()
    out = []
    for iso in claimed:
        yml = ROOT / "benchmarks" / f"{iso}.yaml"
        if iso in busy or not yml.exists() or complete(yml):
            continue
        if repairs.get(iso, 0) >= MAX_REPAIRS:
            continue
        out.append(iso)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    args = ap.parse_args()

    state = PROJECTS / "repairs.json"
    repairs = json.loads(state.read_text()) if state.exists() else {}
    while True:
        todo = candidates(repairs)
        print(f"{time.strftime('%H:%M:%S')} incomplete and idle: {todo or 'none'}", flush=True)
        if not args.dry_run:
            for iso in todo[:1]:  # one at a time: they share the GPU
                repairs[iso] = repairs.get(iso, 0) + 1
                state.write_text(json.dumps(repairs))
                print(f"  re-running {iso} (repair {repairs[iso]} of {MAX_REPAIRS})", flush=True)
                with open(PROJECTS / f"repair_{iso}.log", "a") as log:
                    subprocess.run([sys.executable, "-u", "scripts/run_benchmark.py", "--iso", iso],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if args.once or not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
