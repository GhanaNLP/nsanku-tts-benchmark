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


def run_cmd(cmd, desc=""):
    print(f"\n>>> [{desc}] {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED ({dt:.1f}s): {' '.join(cmd)} (exit {res.returncode})", file=sys.stderr)
    else:
        print(f"DONE ({dt:.1f}s)")
    return res.returncode == 0


def benchmark_language(iso, limit=None):
    lang_name = ISO_TO_NAME.get(iso, iso)
    print(f"\n{'#' * 70}")
    print(f"  BENCHMARKING: {iso} ({lang_name})")
    print(f"{'#' * 70}")

    limit_args = ["--limit", str(limit)] if limit else []

    # Stage 1a: Synthesise standard models in tts image
    cmd_tts = [
        "bash", "scripts/h200_run.sh", "synth",
        "python", "-m", "benchmark.evaluate", "synthesize",
        "--iso", iso, "--stack", "tts"
    ] + limit_args
    run_cmd(cmd_tts, f"TTS Synthesis for {iso}")

    # Stage 1b: Synthesise OmniVoice in omni image
    cmd_omni = [
        "bash", "scripts/h200_run.sh", "omni",
        "python", "-m", "benchmark.evaluate", "synthesize",
        "--iso", iso, "--stack", "omni"
    ] + limit_args
    run_cmd(cmd_omni, f"OmniVoice Synthesis for {iso}")

    # Stage 2a: CER with the language's ASR judge (asr image)
    run_cmd(["bash", "scripts/h200_run.sh", "asr",
             "python", "-m", "benchmark.evaluate", "score-cer", "--iso", iso] + limit_args,
            f"CER for {iso}")

    # Stage 2b: SpeechBERTScore (score image)
    run_cmd(["bash", "scripts/h200_run.sh", "score",
             "python", "-m", "benchmark.evaluate", "score-sbs", "--iso", iso] + limit_args,
            f"SpeechBERTScore for {iso}")

    # Stage 3: join both metrics into benchmarks/{iso}.yaml
    run_cmd(["bash", "scripts/h200_run.sh", "score",
             "python", "-m", "benchmark.evaluate", "assemble", "--iso", iso],
            f"Assemble {iso}")


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
    for i, iso in enumerate(languages, 1):
        print(f"\n[{i}/{len(languages)}] Starting {iso}...")
        benchmark_language(iso, limit=args.limit)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All benchmarks finished in {total_time/60:.1f} minutes.")
    print(f"{'=' * 70}")

    # Show cross-language leaderboard
    run_cmd(["bash", "scripts/h200_run.sh", "score", "python", "scripts/leaderboard.py"], "Leaderboard")


if __name__ == "__main__":
    main()
