#!/usr/bin/env python3
"""Pull finished languages from the H200, publish their Listen clips, rebuild the
summary, commit and push.

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

import yaml

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


def complete(path):
    """A language is publishable only if every model has a composite (both metrics ran)."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    rows = doc.get("benchmarks") or []
    return bool(rows) and all("composite" in b and not b.get("partial") for b in rows)


def publish_audio(host, remote, isos):
    """Upload the Listen clips on the GPU box, before the feed that points at them.

    The two helper files are copied over rather than pulled: that checkout's
    tracked YAMLs are rewritten by the runs, so a git pull there can conflict.
    """
    repo = f"{remote}/nsanku-tts-benchmark"
    for rel in ("benchmark/clips.py", "scripts/publish_audio.py"):
        subprocess.run(["scp", "-q", str(ROOT / rel), f"{host}:{repo}/{rel}"], check=True)
    r = subprocess.run(
        ["ssh", host, f"cd {repo} && bash scripts/h200_run.sh score python "
                      f"scripts/publish_audio.py --iso {' '.join(isos)}"],
        capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip() and "%|" not in l]
    print("\n".join(lines[-len(isos) - 1:]))
    if r.returncode != 0:
        sys.exit(f"publishing the Listen clips failed:\n{r.stderr[-400:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="h200")
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip uploading the Listen clips")
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
        if r.returncode != 0:
            print(f"  could not fetch {iso}.yaml")
        elif not complete(bench / f"{iso}.yaml"):
            # A stage can die (e.g. a CUDA OOM on the shared GPU) and the language
            # still reach its end; publishing that would show half a result.
            print(f"  NOT publishing {iso}: results are incomplete (a metric is missing)")
        else:
            pulled.append(iso)

    # A YAML that is not from a finished run is not a result: drop stale ones.
    for path in bench.glob("*.yaml"):
        if path.stem not in pulled:
            path.unlink()

    if not args.no_audio:
        publish_audio(args.host, args.remote, pulled)

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
