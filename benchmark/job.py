"""Run one unit of benchmark work inside a HuggingFace Job.

A job is a fresh container with no shared filesystem, so the unit has to pull
what it needs and push what it produced:

    pull   existing clips / transcriptions for this (language, model) from the
           audio dataset repo, so a re-run resumes instead of starting over
    run    synthesis or scoring, exactly as the local and Modal paths do
    push   clips, transcriptions and a per-model result JSON back to that repo

The YAMLs the leaderboard reads are assembled from those result JSONs by
scripts/assemble_benchmarks.py and committed to GitHub, so a job needs only an
HF token — no git credentials.

Run (inside a job):
    python -m benchmark.job --stage synth --subset twi-asa --model org/name
    python -m benchmark.job --stage score --subset twi-asa
"""

import argparse
import json
import os
import sys
from pathlib import Path

WORK = Path(os.environ.get("NSANKU_TTS_WORK_DIR", "/data"))
AUDIO_REPO = os.environ.get("NSANKU_TTS_AUDIO_REPO", "ghananlpcommunity/nsanku-tts-audio")
RESULTS_PREFIX = "results"


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def _prepare_dirs():
    """Point the benchmark at this container's scratch space."""
    audio, trans = WORK / "audio", WORK / "transcriptions"
    audio.mkdir(parents=True, exist_ok=True)
    trans.mkdir(parents=True, exist_ok=True)
    os.environ["NSANKU_TTS_AUDIO_DIR"] = str(audio)
    os.environ["NSANKU_TTS_TRANSCRIPTIONS_DIR"] = str(trans)
    os.environ.setdefault("HF_HOME", str(WORK / "hf-cache"))
    return audio, trans


def _pull(api, patterns, dest):
    """Fetch just the paths this unit needs, not the whole 1.7 GB repo."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=AUDIO_REPO,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as e:
        print(f"  nothing to resume ({type(e).__name__}: {str(e)[:120]})")


def _push(api, folder, path_in_repo, message):
    if not any(Path(folder).rglob("*")):
        return
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=AUDIO_REPO,
        repo_type="dataset",
        commit_message=message,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synth", "score"], required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--model", help="synth only: the model to run")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.samples:
        os.environ["NSANKU_TTS_NUM_SAMPLES"] = str(args.samples)
    audio_dir, trans_dir = _prepare_dirs()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmark.dataset import subset_to_iso
    from benchmark.evaluate import score_language, synthesize_language

    api = _api()
    iso = subset_to_iso(args.subset)

    if args.stage == "synth":
        if not args.model:
            raise SystemExit("--model is required for the synth stage")
        slug = args.model.replace("/", "__")
        _pull(api, [f"audio/{iso}/*/{slug}/*"], WORK)
        synthesize_language(args.subset, model_filter=args.model,
                            device=args.device, force=args.force)
        _push(api, audio_dir, "audio", f"Synthesis: {args.model} on {iso}")
        return

    # Scoring needs every model's clips for this language, and any rows
    # already scored so a re-run only fills the gaps.
    _pull(api, [f"audio/{iso}/*", f"transcriptions/{iso}_*"], WORK)
    results = score_language(args.subset, device=args.device, force=args.force)
    _push(api, trans_dir, "transcriptions", f"Transcriptions: {iso}")

    out = WORK / "results"
    out.mkdir(exist_ok=True)
    (out / f"{iso}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _push(api, out, RESULTS_PREFIX, f"Scores: {iso}")
    print(f"\nScored {len(results)} model(s) for {iso}")


if __name__ == "__main__":
    main()
