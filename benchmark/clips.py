"""Choose the clip a language's Listen buttons play.

One sentence per language, read by every model, so the buttons compare like with
like, plus the real recording of that sentence. It is picked to be *typical*
rather than flattering: for each model the clip's composite is compared with that
model's own median, and the sentence whose clips sit closest to their medians
across all models wins. A best-case clip would make every model sound better
than it is; a worst-case one would be unfair to the models that stumble on it.

Used by scripts/build_summary.py (which records the choice in the feed) and
scripts/publish_audio.py (which uploads exactly those clips), so both always
agree.
"""

import statistics

MIN_WORDS, MAX_WORDS = 6, 20
LENGTH_PENALTY = 0.05


def clip_composite(row):
    if row.get("sbs") is None or row.get("cer") is None:
        return None
    return (row["sbs"] + (1.0 - row["cer"])) / 2.0


def pick_sample(benchmarks):
    """Return {key, text, source, models} for a language, or None.

    *benchmarks* is the ``benchmarks`` list of a language YAML; ``models`` lists
    the models that have that clip scored (and therefore on disk).
    """
    per_model = {}
    for b in benchmarks:
        comps = {k: c for k, r in (b.get("entries") or {}).items()
                 if (c := clip_composite(r)) is not None}
        if comps:
            per_model[b["model"]] = comps
    if not per_model:
        return None

    common = set.intersection(*(set(c) for c in per_model.values()))
    if not common:
        # No sentence was read by everyone: fall back to the one most models read.
        counts = {}
        for comps in per_model.values():
            for k in comps:
                counts[k] = counts.get(k, 0) + 1
        top = max(counts.values())
        common = {k for k, n in counts.items() if n == top}

    medians = {m: statistics.median(c.values()) for m, c in per_model.items()}
    text_of = {}
    for b in benchmarks:
        for k, r in (b.get("entries") or {}).items():
            text_of.setdefault(k, (r.get("text") or "", r.get("source")))

    def cost(key):
        gaps = [abs(c[key] - medians[m]) for m, c in per_model.items() if key in c]
        words = len(text_of.get(key, ("",))[0].split())
        penalty = 0.0 if MIN_WORDS <= words <= MAX_WORDS else LENGTH_PENALTY
        return (sum(gaps) / len(gaps) + penalty, key)

    key = min(common, key=cost)
    text, source = text_of.get(key, ("", None))
    return {
        "key": key,
        "text": text,
        "source": source or key.rsplit("_", 1)[0],
        "models": sorted(m for m, c in per_model.items() if key in c),
    }
