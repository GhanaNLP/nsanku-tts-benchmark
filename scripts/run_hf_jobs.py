#!/usr/bin/env python3
"""Submit the benchmark to HuggingFace Jobs.

One job per (language, model) for synthesis, one per language for scoring.
Jobs are independent containers, so a failure costs one unit rather than the
run, and the whole thing survives this machine being closed.

Each job clones the repo at a pinned ref and installs the pinned requirements
for its stage — synthesis and scoring cannot share an environment (fairseq2
pins torch 2.8, voxcpm/f5-tts need torch 2.5).

Run:  python3 scripts/run_hf_jobs.py --stage synth
      python3 scripts/run_hf_jobs.py --stage score --langs twi-asa,ewe
      python3 scripts/run_hf_jobs.py --stage all --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO_URL = "https://github.com/GhanaNLP/nsanku-tts-benchmark"
NAMESPACE = os.environ.get("NSANKU_TTS_HF_NAMESPACE", "ghananlpcommunity")

# Base images carry the CUDA runtime; the rest is pip-installed per stage.
IMAGES = {
    "synth": "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
    "score": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
}
APT = {
    # ffmpeg: F5-TTS loads its reference clip through pydub.
    # espeak-ng-data: nano-twi ships a slimmed copy missing en_dict.
    "synth": "ffmpeg espeak-ng-data git",
    "score": "ffmpeg git",
}
# Hardware follows the work: a hosted model is a network call whichever stage
# it is in, omniASR-LLM-7B needs real headroom, everything else fits an A10G.
FLAVORS = {"synth": "a10g-small", "score": "l40sx1", "api": "cpu-upgrade"}


def is_api_model(meta):
    return "api" in (meta.get("architecture", "") + meta.get("license", "")).lower()


def build_command(stage, subset, model, ref, samples, force, flavor):
    unit = (f"--stage {stage} --subset {subset}"
            + (f" --model {model}" if model else "")
            + (f" --samples {samples}" if samples else "")
            + (" --force" if force else ""))
    # A CPU flavor has no CUDA; the judge decides which one it needs.
    device = "cpu" if flavor.startswith("cpu") else "cuda"
    script = (
        "set -eux; "
        f"export NSANKU_DEVICE={device}; "
        f"apt-get update && apt-get install -y --no-install-recommends {APT[stage]}; "
        f"git clone --depth 1 --branch {ref} {REPO_URL} /app; "
        "cd /app; "
        f"pip install --no-cache-dir -r requirements/{'tts' if stage == 'synth' else 'asr'}.txt; "
        f"python -m benchmark.job {unit} --device $NSANKU_DEVICE"
    )
    return ["bash", "-lc", script]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synth", "score", "all"], default="all")
    ap.add_argument("--langs", default="", help="comma/space separated subsets")
    ap.add_argument("--model", default="", help="only models matching this substring")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref", default="main", help="git ref the jobs check out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import run_job

    from benchmark.asr import judge_for
    from benchmark.dataset import available_subsets, subset_to_iso
    from benchmark.evaluate import load_tts_models

    subsets = [s for s in args.langs.replace(",", " ").split() if s] or available_subsets()
    token = os.environ.get("HF_TOKEN")
    secrets = {"HF_TOKEN": token, "KHAYA_API_KEY": os.environ.get("KHAYA_API_KEY", "")}

    planned = []
    if args.stage in ("all", "synth"):
        for subset in subsets:
            for info in load_tts_models(subset):
                if args.model and args.model.lower() not in info["name"].lower():
                    continue
                flavor = FLAVORS["api"] if is_api_model(info) else FLAVORS["synth"]
                planned.append(("synth", subset, info["name"], flavor))
    if args.stage in ("all", "score"):
        for subset in subsets:
            spec = judge_for(subset_to_iso(subset))
            if spec is None:
                continue
            # A hosted judge is a network call, not a GPU workload.
            flavor = FLAVORS["api"] if spec["kind"] == "khaya" else FLAVORS["score"]
            planned.append(("score", subset, None, flavor))

    print(f"{len(planned)} job(s) to submit as {NAMESPACE}")
    for stage, subset, model, flavor in planned:
        label = f"{stage} {subset}" + (f" / {model}" if model else "")
        if args.dry_run:
            print(f"  [dry-run] {label}  ({flavor})")
            continue
        job = run_job(
            image=IMAGES[stage],
            command=build_command(stage, subset, model, args.ref,
                                  args.samples, args.force, flavor),
            flavor=flavor,
            secrets=secrets,
            namespace=NAMESPACE,
            timeout="12h",
            token=token,
        )
        print(f"  submitted {label}  ({flavor})  -> {job.id}")


if __name__ == "__main__":
    main()
