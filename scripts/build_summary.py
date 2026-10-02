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

from benchmark.config import BENCHMARK_DIR, ISO_TO_NAME, SPEECH_EVAL_SOURCES, all_isos  # noqa: E402
from benchmark.clips import pick_sample  # noqa: E402
from benchmark import yamlio  # noqa: E402
from benchmark.evaluate import load_tts_models  # noqa: E402

# Everything on a model row except the per-clip entries.
DROP = {"entries"}

# Languages missing from languages/ghana_languages.yaml.
FAMILY_FALLBACK = {"aha": "Kwa", "kpo": "Kwa"}


def families():
    path = ROOT / "languages" / "ghana_languages.yaml"
    out = dict(FAMILY_FALLBACK)
    if path.exists():
        for lang in yamlio.load(path.read_text(encoding="utf-8")).get("languages", []):
            if lang.get("family"):
                out[lang["iso_639_3"]] = lang["family"]
    return out


def build(only=None):
    out = {}
    family = families()
    for iso in all_isos():
        if only is not None and iso not in only:
            continue
        path = BENCHMARK_DIR / f"{iso}.yaml"
        if not path.exists():
            continue
        doc = yamlio.load(path.read_text(encoding="utf-8")) or {}
        registry = {m["name"]: m for m in load_tts_models(iso)}
        rows = []
        for b in doc.get("benchmarks", []):
            row = {k: v for k, v in b.items() if k not in DROP}
            row["license"] = registry.get(b["model"], {}).get("license")
            rows.append(row)
        if not rows:
            continue
        scored = {r["model"] for r in rows}
        # The sentence behind the Listen buttons (see benchmark/clips.py).
        sample = pick_sample(doc.get("benchmarks", []))
        # Text sources that actually contributed samples (a configured source
        # can end up empty), so the domain count on the board is honest.
        domains = sorted({src for r in rows for src in (r.get("per_source") or {})},
                         key=list(SPEECH_EVAL_SOURCES[iso]).index)
        out[iso] = {
            "iso_639_3": iso,
            "language": ISO_TO_NAME.get(iso, doc.get("language", iso)),
            "family": family.get(iso),
            "domains": domains,
            "sample": sample,
            # Registered for this language but no valid output: shown as failed.
            "missing": sorted(n for n in registry if n not in scored),
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
