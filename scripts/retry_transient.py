#!/usr/bin/env python3
"""Re-synthesise clips that failed for reasons that are not the model's.

A clip whose recorded error is transport trouble (timeout, dropped connection,
rate limit) or a bug since fixed is re-attempted, and only those clips. Clips
the model itself failed on (empty output, no symbol, no audio in the response)
are left alone: every model gets one attempt per sentence.

    scripts/h200_run.sh synth python scripts/retry_transient.py [--iso ada ...]

Afterwards re-run score-cer, score-sbs and assemble for the languages listed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark import config  # noqa: E402
from benchmark.dataset import load_samples  # noqa: E402
from benchmark.evaluate import clip_dir, load_tts_models, synthesize_language  # noqa: E402

TRANSIENT = ("'time' is not defined", "Read timed out", "ConnectionPool",
             "Connection aborted", "Timeout", "timed out", "Max retries exceeded")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iso", nargs="*", help="limit to these languages")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    touched = []
    for iso in args.iso or config.all_isos():
        todo = {}
        for info in load_tts_models(iso):
            path = clip_dir(iso, config.DEFAULT_CATEGORY, info["name"]) / "errors.json"
            if not path.exists():
                continue
            errors = json.loads(path.read_text(encoding="utf-8"))
            keys = [k for k, v in errors.items() if any(t in str(v) for t in TRANSIENT)]
            if keys:
                todo[info["name"]] = keys
        if not todo:
            continue
        samples = load_samples(iso, extract_references=True)
        by_key = {str(s.index): s for s in samples}
        for name, keys in todo.items():
            subset = [by_key[k] for k in keys if k in by_key]
            print(f"{iso} {name}: retrying {len(subset)} clip(s) {[s.key for s in subset]}")
            if subset and not args.dry_run:
                synthesize_language(iso, model_filter=name, samples=subset, stack="tts")
                touched.append(iso)
    print("languages to re-score:", " ".join(sorted(set(touched))) or "none")


if __name__ == "__main__":
    main()
