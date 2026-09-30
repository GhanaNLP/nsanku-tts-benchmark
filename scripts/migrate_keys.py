#!/usr/bin/env python3
"""Rename pre-multi-source results to the ``<source>_<row>`` key format.

Before a language could draw from several text sources, samples were keyed by
bare row index (``00012.wav``, ``"12"``) in that language's single config. The
rows in that config are still valid samples under the new scheme, so instead of
re-synthesising and re-scoring them this renames them in place:

    audio/{iso}/read_aloud/{model}/00012.wav  ->  bible_00012.wav
    references/{iso}/00012.wav                ->  bible_00012.wav
    errors.json and benchmarks/{iso}.scores.json keys likewise

Idempotent: keys that already carry a source are left alone.

Usage:  python3 scripts/migrate_keys.py [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark import config  # noqa: E402

BARE_WAV = re.compile(r"^(\d{5})\.wav$")


def rekey(mapping, source):
    out, changed = {}, 0
    for k, v in mapping.items():
        if str(k).isdigit():
            out[f"{source}_{int(k):05d}"] = v
            changed += 1
        else:
            out[k] = v
    return out, changed


def rename_wavs(folder, source, dry):
    n = 0
    for wav in sorted(folder.glob("*.wav")):
        m = BARE_WAV.match(wav.name)
        if not m:
            continue
        target = wav.with_name(f"{source}_{m.group(1)}.wav")
        if not target.exists():
            if not dry:
                wav.rename(target)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for iso, source in config.LEGACY_SOURCE.items():
        clips = refs = 0
        audio_root = config.AUDIO_DIR / iso / config.DEFAULT_CATEGORY
        for model_dir in sorted(p for p in audio_root.glob("*") if p.is_dir()):
            clips += rename_wavs(model_dir, source, args.dry_run)
            errors = model_dir / "errors.json"
            if errors.exists():
                data, changed = rekey(json.loads(errors.read_text(encoding="utf-8")), source)
                if changed and not args.dry_run:
                    errors.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        ref_dir = config.REFERENCE_DIR / iso
        if ref_dir.is_dir():
            refs = rename_wavs(ref_dir, source, args.dry_run)

        scores = config.BENCHMARK_DIR / f"{iso}.scores.json"
        rescored = 0
        if scores.exists():
            doc = json.loads(scores.read_text(encoding="utf-8"))
            for model, rows in doc.get("per_sample", {}).items():
                doc["per_sample"][model], c = rekey(rows, source)
                rescored += c
            if rescored and not args.dry_run:
                scores.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"{iso}: {clips} clips, {refs} references, {rescored} cached scores "
              f"-> '{source}_*'{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
