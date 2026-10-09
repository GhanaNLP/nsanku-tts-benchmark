#!/usr/bin/env python3
"""Publish finished languages to GitHub from the GPU box itself.

Runs on the H200, next to the benchmark, so results reach GitHub with nothing else
(a laptop, a Claude Code session) needing to be online. It does what
scripts/sync_results.py does from outside, minus the ssh hops:

  1. which languages finished           (the benchmark logs, read locally)
  2. which of those are complete        (every model has a composite; sane failure rate)
  3. upload their Listen clips          (Hugging Face; only new or changed ones)
  4. rebuild benchmarks/summary.json    (the Space's live feed)
  5. commit and push                    (a deploy key scoped to this one repo)

It works in its own clean clone (--publish-dir), never in the directory the benchmark
runs in, so it cannot disturb a run, and the benchmark's own working tree can stay as
messy as it likes. A language is only ever published once complete, so a half-finished
or failed one stays off the board.

    python3 scripts/publish_from_gpu.py --once --no-push        # try it
    python3 scripts/publish_from_gpu.py --watch --interval 1200 # every 20 minutes
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Re-use the checks the laptop-side sync applies, so both agree on what "complete" means.
from scripts.sync_results import _previous_samples, _sample_signature, complete  # noqa: E402

PROJECTS = Path("/mnt/volume_d2wey28/projects")
LOGS = ["full_A.log", "full_B.log", "claim_*.log", "repair_[a-z]*.log", "queue*.log"]
GIT_USER = ("nsanku benchmark bot", "noreply@ghananlp.org")


def run(cmd, cwd=None, check=True, **kw):
    return subprocess.run(cmd, cwd=cwd, text=True, check=check, **kw)


def finished_languages(projects):
    """ISOs whose run printed 'All benchmarks finished', read from the local logs."""
    from benchmark.config import all_isos

    done = set()
    for pattern in LOGS:
        for log in sorted(projects.glob(pattern)):
            out = subprocess.run(
                ["grep", "-aE", "BENCHMARKING:|All benchmarks finished", str(log)],
                capture_output=True, text=True).stdout
            current = None
            for line in out.replace("\r", "\n").splitlines():
                m = re.search(r"BENCHMARKING:\s+(\S+)", line)
                if m:
                    current = m.group(1)
                elif "All benchmarks finished" in line and current:
                    done.add(current)
                    current = None
    return done & set(all_isos())


def ensure_clone(publish_dir, repo_url):
    if not (publish_dir / ".git").exists():
        run(["git", "clone", "-q", repo_url, str(publish_dir)])
    run(["git", "config", "user.name", GIT_USER[0]], cwd=publish_dir)
    run(["git", "config", "user.email", GIT_USER[1]], cwd=publish_dir)
    key = Path.home() / ".ssh" / "nsanku_deploy"
    run(["git", "config", "core.sshCommand", f"ssh -i {key} -o IdentitiesOnly=yes"], cwd=publish_dir)


def cycle(args):
    pub = Path(args.publish_dir)
    bench = Path(args.bench_root)
    ensure_clone(pub, args.repo)
    # Start from the newest main so a push is a fast-forward. Local edits here are only
    # ever generated files, so a hard reset to origin is safe.
    run(["git", "fetch", "-q", "origin", "main"], cwd=pub)
    run(["git", "reset", "-q", "--hard", "origin/main"], cwd=pub)

    done = sorted(finished_languages(PROJECTS))
    print(f"{time.strftime('%F %T')} finished: {len(done)} language(s)", flush=True)
    bdir = pub / "benchmarks"
    previous = _previous_samples(bdir / "summary.json")

    pulled = []
    for iso in done:
        src = bench / "benchmarks" / f"{iso}.yaml"
        if not src.exists():
            continue
        dst = bdir / f"{iso}.yaml"
        dst.write_bytes(src.read_bytes())
        if complete(dst):
            pulled.append(iso)
        else:
            print(f"  NOT publishing {iso}: results are incomplete", flush=True)
    for path in bdir.glob("*.yaml"):
        if path.stem not in pulled:
            path.unlink()

    todo = [iso for iso in pulled if _sample_signature(bdir / f"{iso}.yaml") != previous.get(iso)]
    if todo and not args.no_audio:
        print(f"  uploading Listen clips: {' '.join(todo)}", flush=True)
        # Inside the image that has huggingface_hub; the clips live under the benchmark dir.
        # publish_audio reads the yamls from the benchmark's own benchmarks/ directory.
        run(["bash", "scripts/h200_run.sh", "score", "python", "scripts/publish_audio.py", "--iso", *todo],
            cwd=bench)

    run([sys.executable, "scripts/build_summary.py", *pulled], cwd=pub)
    run(["git", "add", "-A", "benchmarks", "space/bundled_data.json"], cwd=pub)
    if run(["git", "diff", "--cached", "--quiet"], cwd=pub, check=False).returncode == 0:
        print("  nothing new to publish", flush=True)
        return
    run(["git", "commit", "-q", "-m",
         f"results: {len(pulled)} language(s) complete (published from the GPU box)\n\n"
         "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"], cwd=pub)
    if args.no_push:
        print("  committed (not pushed)", flush=True)
        return
    for attempt in range(3):
        if run(["git", "push", "-q", "origin", "main"], cwd=pub, check=False).returncode == 0:
            print(f"  pushed {len(pulled)} language(s)", flush=True)
            return
        run(["git", "pull", "-q", "--rebase", "origin", "main"], cwd=pub, check=False)
    print("  push failed after 3 attempts; will retry next cycle", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", default=str(PROJECTS / "nsanku-tts-benchmark"))
    ap.add_argument("--publish-dir", default=str(PROJECTS / "nsanku-tts-benchmark-publish"))
    ap.add_argument("--repo", default="git@github.com:GhanaNLP/nsanku-tts-benchmark.git")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    while True:
        try:
            cycle(args)
        except Exception as e:  # keep the watcher alive; the next cycle retries
            print(f"{time.strftime('%F %T')} cycle failed: {e}", flush=True)
        if args.once or not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
