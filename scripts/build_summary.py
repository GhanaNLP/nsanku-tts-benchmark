#!/usr/bin/env python3
"""Build the compact leaderboard feed from benchmarks/{iso}.yaml.

The per-language YAMLs carry every scored clip, which is what makes them
auditable but also megabytes each. The Space only needs the per-model numbers,
so it reads this summary instead: one small JSON for all languages.

    benchmarks/summary.json   served from GitHub raw (live as results are pushed)
    space/bundled_data.json   same content, shipped inside the Space as a fallback

Usage:  python3 scripts/build_summary.py [iso ...]   (default: every language that has a YAML)
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.config import BENCHMARK_DIR, ISO_TO_NAME, all_isos  # noqa: E402

# Everything on a model row except the per-clip entries.
DROP = {"entries"}


def build(only=None):
    out = {}
    for iso in all_isos():
        if only is not None and iso not in only:
            continue
        path = BENCHMARK_DIR / f"{iso}.yaml"
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = [{k: v for k, v in b.items() if k not in DROP}
                for b in doc.get("benchmarks", [])]
        if not rows:
            continue
        out[iso] = {
            "iso_639_3": iso,
            "language": ISO_TO_NAME.get(iso, doc.get("language", iso)),
            "num_samples": doc.get("num_samples"),
            "updated": doc.get("updated"),
            "judge": doc.get("judge"),
            "encoder": doc.get("encoder"),
            "benchmarks": rows,
        }
    return out


def main():
    data = build(set(sys.argv[1:]) or None)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for target in (BENCHMARK_DIR / "summary.json", ROOT / "space" / "bundled_data.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    models = sum(len(d["benchmarks"]) for d in data.values())
    print(f"{len(data)} languages, {models} model rows, {len(text) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
