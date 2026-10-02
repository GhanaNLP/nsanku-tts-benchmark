#!/usr/bin/env python3
"""Master benchmark runner for nsanku-tts-benchmark.

Orchestrates multi-model synthesis and scoring across languages using the
isolated container environments:
  1. Synthesises standard models in `tts` container (torch 2.5)
  2. Synthesises OmniVoice models in `omni` container (torch 2.8 + transformers 5)
  3. Scores CER with the language's ASR judge in the `asr` container
  4. Scores SpeechBERTScore in the `score` container (torch 2.8 + WavLM L6 precision)
  5. Assembles benchmarks/{iso}.yaml (composite of both) in the `score` container

Supports running a single language or all 12 Ghanaian languages sequentially.
Incremental and resumable: clips already synthesised or scored on disk are skipped.

Usage:
    python scripts/run_benchmark.py --iso twi_asante
    python scripts/run_benchmark.py --all-languages
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.config import ISO_TO_NAME, SPEECH_EVAL_SOURCES


# GPU memory (GB) a stage needs free before it starts. The GPU is shared with other
# jobs, and a stage that loads a model onto a full card dies with CUDA OOM.
# Measured: a VoxCPM/MMS synthesis process holds ~5 GB, WavLM scoring ~3 GB, the 7B ASR
# judge ~20 GB. A stage with nothing left to do still waits, so keep the small ones small.
NEED_FREE_GB = {"synth": 8, "omni": 8, "asr": 22, "score": 6}
ATTEMPTS = 3
RETRY_WAIT_S = 300


def gpu_free_gb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.split()
        return float(out[0]) / 1024.0
    except Exception:
        return None


def wait_for_gpu(stage, patience_s=21600):
    """Wait until the card has room for this stage. Returns False if it never did:
    starting anyway would just run the shared GPU out of memory."""
    need = NEED_FREE_GB.get(stage)
    if not need:
        return True
    waited = 0
    while waited < patience_s:
        free = gpu_free_gb()
        if free is None or free >= need:
            return True
        if waited % 600 == 0:
            print(f"    waiting for GPU memory: {free:.0f} GB free, {stage} needs {need} GB", flush=True)
        time.sleep(60)
        waited += 60
    return False


def run_cmd(cmd, desc="", stage=None):
    """Run one stage, retrying on failure. Every stage is resumable, so a retry
    only does the work that is still missing. Returns True on success."""
    for attempt in range(1, ATTEMPTS + 1):
        if not wait_for_gpu(stage):
            print(f"    no GPU room for {stage} after waiting; counting as a failed attempt", flush=True)
            continue
        print(f"\n>>> [{desc}] {' '.join(cmd)}" + (f"  (attempt {attempt})" if attempt > 1 else ""), flush=True)
        t0 = time.time()
        res = subprocess.run(cmd, cwd=REPO)
        dt = time.time() - t0
        if res.returncode == 0:
            print(f"DONE ({dt:.1f}s)", flush=True)
            return True
        print(f"FAILED ({dt:.1f}s): {' '.join(cmd)} (exit {res.returncode})", file=sys.stderr, flush=True)
        if attempt < ATTEMPTS:
            time.sleep(RETRY_WAIT_S)
    return False


def benchmark_language(iso, limit=None):
    """Run every stage for one language. Returns the stages that failed for good."""
    lang_name = ISO_TO_NAME.get(iso, iso)
    print(f"\n{'#' * 70}")
    print(f"  BENCHMARKING: {iso} ({lang_name})")
    print(f"{'#' * 70}")

    limit_args = ["--limit", str(limit)] if limit else []
    py = ["python", "-m", "benchmark.evaluate"]
    run = ["bash", "scripts/h200_run.sh"]
    stages = [
        # (stage image, command, description)
        ("synth", py + ["synthesize", "--iso", iso, "--stack", "tts"] + limit_args, f"TTS Synthesis for {iso}"),
        ("omni", py + ["synthesize", "--iso", iso, "--stack", "omni"] + limit_args, f"OmniVoice Synthesis for {iso}"),
        ("asr", py + ["score-cer", "--iso", iso] + limit_args, f"CER for {iso}"),
        ("score", py + ["score-sbs", "--iso", iso] + limit_args, f"SpeechBERTScore for {iso}"),
        ("score", py + ["assemble", "--iso", iso], f"Assemble {iso}"),
    ]
    failed = []
    for image, cmd, desc in stages:
        if not run_cmd(run + [image] + cmd, desc, stage=image):
            failed.append(desc)
    return failed


def main():
    parser = argparse.ArgumentParser(description="Run nsanku-tts-benchmark across languages.")
    parser.add_argument("--iso", default=None, help="Single language ISO to run (e.g. twi_asante)")
    parser.add_argument("--all-languages", action="store_true", help="Run all 12 benchmark languages")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit per model (default: all 200)")
    args = parser.parse_args()

    if not args.iso and not args.all_languages:
        parser.error("Specify either --iso <language> or --all-languages")

    languages = sorted(SPEECH_EVAL_SOURCES.keys()) if args.all_languages else [args.iso]

    t_start = time.time()
    incomplete = {}
    for i, iso in enumerate(languages, 1):
        print(f"\n[{i}/{len(languages)}] Starting {iso}...")
        failed = benchmark_language(iso, limit=args.limit)
        if failed:
            incomplete[iso] = failed

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    if incomplete:
        # The sync publishes a language only after "All benchmarks finished", so an
        # incomplete one is never published; re-run it to fill the gaps.
        for iso, failed in incomplete.items():
            print(f"INCOMPLETE {iso}: stage(s) failed after {ATTEMPTS} attempts: {'; '.join(failed)}")
        print(f"Ended after {total_time/60:.1f} minutes with incomplete languages.")
    else:
        print(f"All benchmarks finished in {total_time/60:.1f} minutes.")
    print(f"{'=' * 70}")

    # Show cross-language leaderboard
    run_cmd(["bash", "scripts/h200_run.sh", "score", "python", "scripts/leaderboard.py"], "Leaderboard")


if __name__ == "__main__":
    main()
