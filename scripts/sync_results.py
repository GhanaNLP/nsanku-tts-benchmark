#!/usr/bin/env python3
"""Pull finished languages from the H200, rebuild the summary, commit and push.

Only languages whose run has *finished* are published. A language's YAML is
rewritten at the end of each of its runs, so a YAML found on disk mid-run (or
left over from an earlier smoke test) is not a result. The H200 logs say which
runs completed: each language prints "BENCHMARKING: <iso>" when it starts and
"All benchmarks finished" when it ends.

Usage:
    python3 scripts/sync_results.py             # pull, rebuild, commit, push
    python3 scripts/sync_results.py --no-push   # pull and rebuild only
    python3 scripts/sync_results.py --host h200 --remote /mnt/.../projects
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.config import all_isos  # noqa: E402

REMOTE = "/mnt/volume_d2wey28/projects"
LOGS = ["full_A.log", "full_B.log", "claim_*.log"]


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, text=True, check=True, **kw)


def finished_languages(host, remote):
    """ISOs whose run printed 'All benchmarks finished', in log order per file."""
    done = set()
    for pattern in LOGS:
        # One file at a time: the order of lines within a log is what pairs a
        # language with its completion line.
        listing = subprocess.run(
            ["ssh", host, f"ls {remote}/{pattern} 2>/dev/null"],
            capture_output=True, text=True).stdout.split()
        for log in listing:
            out = subprocess.run(
                ["ssh", host, f"grep -aE 'BENCHMARKING:|All benchmarks finished' {log}"],
                capture_output=True, text=True).stdout
            current = None
            for line in out.splitlines():
                m = re.search(r"BENCHMARKING:\s+(\S+)", line)
                if m:
                    current = m.group(1)
                elif "All benchmarks finished" in line and current:
                    done.add(current)
                    current = None
    return done & set(all_isos())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="h200")
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    done = sorted(finished_languages(args.host, args.remote))
    print(f"{len(done)} finished language(s): {' '.join(done)}")
    if not done:
        return

    bench = ROOT / "benchmarks"
    bench.mkdir(exist_ok=True)
    remote_dir = f"{args.remote}/nsanku-tts-benchmark/benchmarks"
    pulled = []
    for iso in done:
        r = subprocess.run(["scp", "-q", f"{args.host}:{remote_dir}/{iso}.yaml",
                            str(bench / f"{iso}.yaml")])
        if r.returncode == 0:
            pulled.append(iso)
        else:
            print(f"  could not fetch {iso}.yaml")

    # A YAML that is not from a finished run is not a result: drop stale ones.
    for path in bench.glob("*.yaml"):
        if path.stem not in pulled:
            path.unlink()

    sh([sys.executable, "scripts/build_summary.py", *pulled])
    sh(["git", "add", "-A", "benchmarks", "space/bundled_data.json"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("nothing new to publish")
        return
    sh(["git", "commit", "-q", "-m",
        f"results: {len(pulled)} language(s) complete\n\n"
        "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"])
    if args.no_push:
        print("committed (not pushed)")
        return
    for remote in ("origin", "michseth"):
        sh(["git", "push", "-q", remote, "main"])
    print("pushed")


if __name__ == "__main__":
    main()
