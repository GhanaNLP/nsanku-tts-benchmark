#!/usr/bin/env python3
"""Upload the Listen clips for finished languages to the audio dataset.

For each language YAML this picks the sample sentence (benchmark/clips.py) and
uploads, for every model that read it, that one synthesised clip, plus the real
recording of the same sentence. Already-published files are skipped, so it is
cheap to run on every sync.

    audio/{iso}/read_aloud/{model with / as __}/{key}.wav   model output
    refs/{iso}/{key}.wav                                    human recording

Runs wherever the clips live (the GPU box), inside the score image:
    scripts/h200_run.sh score python scripts/publish_audio.py --iso ada dag
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import config  # noqa: E402
from benchmark.clips import pick_sample  # noqa: E402

REPO = "ghananlpcommunity/nsanku-tts-benchmark-audio"


def slug(model):
    return model.replace("/", "__")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iso", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    existing = set(api.list_repo_files(REPO, repo_type="dataset"))
    total = 0
    for iso in args.iso:
        path = config.BENCHMARK_DIR / f"{iso}.yaml"
        if not path.exists():
            print(f"{iso}: no results yet")
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        sample = pick_sample(doc.get("benchmarks", []))
        if not sample:
            print(f"{iso}: nothing scored")
            continue
        key = sample["key"]
        wanted = []
        for model in sample["models"]:
            src = config.AUDIO_DIR / iso / config.DEFAULT_CATEGORY / slug(model) / f"{key}.wav"
            wanted.append((f"audio/{iso}/{config.DEFAULT_CATEGORY}/{slug(model)}/{key}.wav", src))
        wanted.append((f"refs/{iso}/{key}.wav", config.REFERENCE_DIR / iso / f"{key}.wav"))

        ops, missing = [], []
        for remote, src in wanted:
            if remote in existing:
                continue
            if not src.exists():
                missing.append(str(src))
                continue
            ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(src)))
        print(f"{iso}: sentence {key} - {len(ops)} to upload, "
              f"{len(wanted) - len(ops) - len(missing)} already there, {len(missing)} missing")
        for m in missing:
            print(f"   missing {m}")
        if ops and not args.dry_run:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"Listen clips: {iso} ({key})")
            total += len(ops)
    print(f"uploaded {total} file(s)")


if __name__ == "__main__":
    main()
