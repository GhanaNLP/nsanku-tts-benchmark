#!/usr/bin/env python3
"""Assemble benchmarks/*.yaml from the per-language results HF Jobs produced.

Jobs write a results/{iso}.json into the audio dataset repo rather than
committing to git, so a job needs only an HF token. This pulls those results
together into the YAMLs the leaderboard reads, which are then committed here.

Run:  python3 scripts/assemble_benchmarks.py
      python3 scripts/assemble_benchmarks.py --repo ghananlpcommunity/nsanku-tts-benchmark-audio
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get(
        "NSANKU_TTS_AUDIO_REPO", "ghananlpcommunity/nsanku-tts-benchmark-audio"))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    from benchmark.asr import judge_for
    from benchmark.config import ISO_TO_NAME, NUM_SAMPLES, language_categories
    from benchmark.evaluate import save_benchmark

    local = Path(snapshot_download(
        repo_id=args.repo, repo_type="dataset",
        allow_patterns=["results/*.json"],
        token=os.environ.get("HF_TOKEN") or None,
    ))
    files = sorted((local / "results").glob("*.json"))
    if not files:
        print("no results/*.json in the dataset repo yet")
        return

    for path in files:
        iso = path.stem
        results = json.loads(path.read_text(encoding="utf-8"))
        if not results:
            continue
        spec = judge_for(iso)
        samples = max(
            (c.get("samples", 0) for r in results for c in r.get("per_category", {}).values()),
            default=NUM_SAMPLES,
        )
        save_benchmark(
            iso,
            ISO_TO_NAME.get(iso, iso),
            results,
            categories=[c["category"] for c in language_categories(iso)],
            num_samples_per_category=samples,
            judge={"model": spec["model"], "cer_on_real_speech": spec["judge_cer"]} if spec else None,
        )
    print(f"\nassembled {len(files)} language file(s) into benchmarks/")


if __name__ == "__main__":
    main()
