"""Aggregate per-language benchmark results into a cross-language leaderboard.

benchmarks/{iso}.yaml holds one file per language. This joins them into a
single table so models can be compared across languages, and reports coverage
honestly: a model that only speaks two languages should not look like the
overall winner by default.

Usage:
    PYTHONPATH=. python scripts/leaderboard.py
    PYTHONPATH=. python scripts/leaderboard.py --results-dir /path/to/benchmarks
"""

import argparse
from pathlib import Path

import yaml

from benchmark.config import BENCHMARK_DIR, ENCODER_LAYER, ENCODER_MODEL, ISO_TO_NAME


def load_results(results_dir=None):
    """Read every per-language result file into {iso: document}."""
    d = Path(results_dir) if results_dir else BENCHMARK_DIR
    out = {}
    for path in sorted(d.glob("*.yaml")):
        if path.name.startswith("."):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"  WARN: {path.name} is not valid YAML ({e}); skipping")
            continue
        if not doc or "benchmarks" not in doc:
            continue
        out[doc.get("iso_639_3", path.stem)] = doc
    return out


def collect(results):
    """Join per-language scores into {model: {scores_by_lang}}.

    Collects composite score ((1 - CER) + SBS)/2, character accuracy,
    and SpeechBERTScore.
    """
    models = {}
    for iso, doc in results.items():
        for entry in doc["benchmarks"]:
            name = entry["model"]
            row = models.setdefault(name, {
                "composite": {},
                "accuracy": {},
                "sbs": {},
                "ref": False,
            })
            comp = entry.get("composite", entry.get("sbs"))
            acc = entry.get("accuracy")
            sbs = entry.get("sbs")
            n = entry.get("num_scored", 0)

            if comp is not None:
                row["composite"][iso] = (comp, n)
            if acc is not None:
                row["accuracy"][iso] = (acc, n)
            if sbs is not None:
                row["sbs"][iso] = (sbs, n)
            row["ref"] = row["ref"] or bool(entry.get("uses_reference"))
    return models


def main():
    parser = argparse.ArgumentParser(description="cross-language leaderboard")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {Path(args.results_dir) if args.results_dir else BENCHMARK_DIR}")
        return

    isos = sorted(results)
    models = collect(results)

    print(f"\nLanguages: {len(isos)}  models: {len(models)}")
    print(f"Scoring:   Composite = (Character Accuracy [1-CER] + SpeechBERTScore) / 2")
    print(f"Encoder:   {ENCODER_MODEL} layer {ENCODER_LAYER}")
    for iso in isos:
        print(f"  {iso:<14} {ISO_TO_NAME.get(iso, iso)}")

    full, partial = [], []
    for name, row in models.items():
        comp_scores = {iso: v[0] for iso, v in row["composite"].items() if v[0] is not None}
        if not comp_scores:
            continue
        mean_comp = sum(comp_scores.values()) / len(comp_scores)

        acc_scores = [v[0] for v in row["accuracy"].values() if v[0] is not None]
        mean_acc = sum(acc_scores) / len(acc_scores) if acc_scores else None

        sbs_scores = [v[0] for v in row["sbs"].values() if v[0] is not None]
        mean_sbs = sum(sbs_scores) / len(sbs_scores) if sbs_scores else None

        record = (mean_comp, mean_acc, mean_sbs, len(comp_scores), name, comp_scores, row["ref"])
        (full if len(comp_scores) == len(isos) else partial).append(record)

    full.sort(key=lambda r: -r[0])
    partial.sort(key=lambda r: -r[0])

    w = max((len(r[4]) for r in full + partial), default=20)

    def show(rows, note=""):
        if not rows:
            return
        print(f"\n{note}" if note else "")
        print(f"  {'model':<{w}}  {'langs':>6}  {'Composite':>9}  {'Acc(1-CER)':>10}  {'SBS':>7}   per-language Composite")
        print("  " + "-" * (w + 62))
        for mean_comp, mean_acc, mean_sbs, n_langs, name, comp_scores, uses_ref in rows:
            cells = " ".join(f"{comp_scores[iso]:.3f}" if iso in comp_scores else "  -  "
                             for iso in isos)
            acc_str = f"{mean_acc:>10.4f}" if mean_acc is not None else "         -"
            sbs_str = f"{mean_sbs:>7.4f}" if mean_sbs is not None else "      -"
            tag = " (ref)" if uses_ref else ""
            print(f"  {name:<{w}}  {n_langs:>3}/{len(isos):<2}  {mean_comp:>9.4f}  {acc_str}  {sbs_str}   {cells}{tag}")

    show(full, f"Full coverage — all {len(isos)} languages (ranked by Composite)")
    show(partial,
         f"Partial coverage — NOT comparable to the table above "
         f"(averaged over the languages each model happens to speak)")

    if not full:
        print("\nNo model covers every language yet, so there is no "
              "like-for-like ranking to report.")
    elif len(full) > 1:
        best, worst = full[0], full[-1]
        n = len(full) - 1
        print(f"\n{n} model{'' if n == 1 else 's'} separated by "
              f"{best[0] - worst[0]:.4f} Composite Score "
              f"({worst[0]:.4f} to {best[0]:.4f}).")
        if best[0] - worst[0] < 0.05:
            print("  NOTE: that spread is small. SpeechBERTScore is sensitive to "
                  "sample size, and differences this narrow should not be read "
                  "as a ranking on their own.")
    if partial:
        print(f"\n{len(partial)} further model{'s' if len(partial) != 1 else ''} excluded from the ranking "
              "because they do not cover every language.")


if __name__ == "__main__":
    main()
