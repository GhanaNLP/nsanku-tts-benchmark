"""Orchestrate TTS evaluation for a single language.

Each model is scored on every text sample by CTC forced-alignment.
The ranking metric is mean alignment score (closer to 0 = better).

Incremental: results are keyed by sample *index* (the row index in the
ghana-sentences subset). Bumping NUM_SAMPLES only scores the new rows.
Pre-computed samples are reused; only the delta is re-evaluated.

Results format (benchmarks/{iso}.yaml):

    iso_639_3: twi_asante
    num_samples: 200
    benchmarks:
      - model: org/model
        score: -0.42
        entries:
          "0": {text: ..., score: -0.50, num_frames: 123}
          "1": {text: ..., score: null, error: ...}
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    BENCHMARK_DIR,
    NUM_SAMPLES,
    ISO_TO_NAME,
    ALIGNMENT_LANG_MAP,
    TTS_LANG_MAP,
)
from .dataset import load_text_samples, subset_to_iso

logger = logging.getLogger(__name__)


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_tts_models(subset):
    """Load models eligible for *subset* from data/tts_models.json."""
    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    if not path.exists():
        return []
    with open(path) as f:
        all_models = json.load(f)
    iso = subset_to_iso(subset)
    ret = []
    for m in all_models:
        langs = m.get("languages", [])
        if iso in langs or "all" in langs:
            ret.append(m)
    return ret


def _scored_indices(benchmark_path, model_id):
    """Return the set of sample indices already scored for *model_id*."""
    if not benchmark_path.exists():
        return set()
    with open(benchmark_path) as f:
        data = yaml.safe_load(f) or {}
    for b in data.get("benchmarks", []):
        if b.get("model") != model_id:
            continue
        entries = b.get("entries") or {}
        scored = set()
        for sample_idx in entries:
            try:
                scored.add(int(sample_idx))
            except (TypeError, ValueError):
                continue
        return scored
    return set()


def save_benchmark(iso_code, language, results, num_samples=None):
    """Merge results into benchmarks/{iso}.yaml, ranked by alignment score."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{iso_code}.yaml"

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = yaml.safe_load(f) or {}

    merged = {r["model"]: r for r in existing.get("benchmarks", [])}
    for r in results:
        merged[r["model"]] = r

    ranked = sorted(
        merged.values(),
        key=lambda x: (x.get("score") is None, -(x.get("score") if x.get("score") is not None else -1e9)),
    )
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples": num_samples if num_samples is not None else NUM_SAMPLES,
        "scoring": "alignment_score",
        "updated": _today(),
        "benchmarks": ranked,
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved: {path}")


def _score_samples(samples, audio_fn, lang, device="cuda"):
    """Score each sample: generate audio via audio_fn, then align.

    Args:
        samples: list of {"text": str, "index": int}
        audio_fn: callable(text) -> audio_bytes (WAV)
        lang: alignment language code (see ALIGNMENT_LANG_MAP)

    Returns:
        (entries: dict[key=str(index), {text, score, num_frames}],
         valid_count: int)
    """
    from .alignment import compute_alignment_score_from_bytes

    entries = {}
    valid = 0

    for sample in samples:
        text = sample["text"]
        idx = sample["index"]
        try:
            audio_bytes = audio_fn(text)
            result = compute_alignment_score_from_bytes(
                audio_bytes, text, lang=lang, device=device,
            )
            score = result["score"]
            if score is None:
                entries[str(idx)] = {"text": text, "score": None, "error": "no_frames"}
            else:
                entries[str(idx)] = {
                    "text": text,
                    "score": round(score, 4),
                    "num_frames": result["num_frames"],
                }
                valid += 1
        except Exception as e:
            entries[str(idx)] = {
                "text": text,
                "score": None,
                "error": str(e)[:200],
            }

    return entries, valid


def evaluate_language(subset, model_filter=None, device="cuda", force=False, samples=None):
    """Benchmark all TTS models for a language subset.

    Args:
        subset: ghana-sentences subset name (twi-asa, dag, ...)
        model_filter: substring to filter model IDs by
        device: torch device ("cuda")
        force: re-score all samples even if already done
        samples: optional pre-fetched sample list (from load_text_samples)

    Samples with indices already present in the per-model ``entries`` map are
    skipped (incremental).  ``force=True`` disables that and re-scores all.
    """
    iso = subset_to_iso(subset)
    language = ISO_TO_NAME.get(iso, subset)
    al_lang = ALIGNMENT_LANG_MAP.get(iso, iso)
    tts_lang = TTS_LANG_MAP.get(iso, iso)

    print(f"\n{'=' * 60}")
    print(f"  Evaluating {subset} ({language}, {iso})")
    print(f"{'=' * 60}")

    if samples is None:
        samples = load_text_samples(subset, offset=0, limit=NUM_SAMPLES)
    if not samples:
        print(f"  No text samples for {subset} — skipping")
        return []
    print(f"  Text samples in range: {len(samples)} (first idx={samples[0]['index']}, last idx={samples[-1]['index']})")

    models = load_tts_models(subset)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No TTS models to evaluate for {subset}")
        return []

    benchmark_path = BENCHMARK_DIR / f"{iso}.yaml"
    results = []

    for model_info in models:
        model_id = model_info["name"]
        print(f"\n  {'-' * 50}")
        print(f"  [{model_id}]")
        print(f"  {'-' * 50}")

        # Determine which samples need scoring (incremental)
        fresh = samples
        if not force:
            scored = _scored_indices(benchmark_path, model_id)
            fresh = [s for s in samples if s["index"] not in scored]
            if not fresh:
                print(f"    All {len(samples)} samples already scored — skipping")
                continue
            print(f"    New samples to score: {len(fresh)} ({len(scored)} already done, skipping)")

        try:
            from .models import load_tts_model
            tts_model = load_tts_model(model_id, device=device, subset=subset, meta=model_info)

            def audio_fn(text, _model=tts_model, _lang=tts_lang):
                return _model.synthesize(text, lang=_lang)

            print(f"    Synthesising + aligning {len(fresh)} samples...")
            t0 = time.time()
            entries, valid = _score_samples(fresh, audio_fn, lang=al_lang, device=device)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"    Failed: {str(e)[:200]}")
            save_benchmark(iso, language, [{
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "owner": model_id.split("/")[0],
                "architecture": model_info.get("architecture", "unknown"),
                "score": None,
                "error": str(e)[:200],
                "source": "evaluated",
            }])
            continue

        # Merge new entries into existing benchmark file for this model
        existing_entries = {}
        if benchmark_path.exists():
            with open(benchmark_path) as f:
                data = yaml.safe_load(f) or {}
            for b in data.get("benchmarks", []):
                if b.get("model") == model_id:
                    existing_entries = b.get("entries") or {}
                    break
        existing_entries.update(entries)

        valid_scores = [v["score"] for v in existing_entries.values() if v.get("score") is not None]
        total_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        result = {
            "model": model_id,
            "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
            "owner": model_id.split("/")[0],
            "architecture": model_info.get("architecture", "unknown"),
            "score": round(total_score, 4) if total_score is not None else None,
            "samples": len(existing_entries),
            "valid": len(valid_scores),
            "avg_seconds": round(elapsed / max(len(fresh), 1), 2),
            "entries": existing_entries,
            "source": "evaluated",
        }
        results.append(result)
        if total_score is not None:
            print(f"    ALIGNMENT SCORE: {total_score:.4f}  ({len(valid_scores)} valid of {len(existing_entries)} sampled, {elapsed:.0f}s)")
        else:
            print(f"    No valid output")
        save_benchmark(iso, language, [result], num_samples=max(len(existing_entries), NUM_SAMPLES))

        try:
            tts_model.cleanup()
        except Exception:
            pass

    return results