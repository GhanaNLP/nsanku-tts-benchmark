"""Orchestrate TTS evaluation for a single language.

Each model is scored on every text sample by CTC forced-alignment.
The ranking metric is mean alignment score (closer to 0 = better).
"""

import csv
import json
import time
from pathlib import Path

import yaml

from .config import (
    BENCHMARK_DIR, AUDIO_DIR, NUM_SAMPLES, SUBSET_TO_ISO, ISO_TO_NAME,
)
from .dataset import load_text_samples, subset_to_iso
from .alignment import compute_alignment_score_from_bytes, load_audio_from_bytes


def _today():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%d")


def load_tts_models(subset):
    """Load models eligible for *subset* from data/tts_models.json."""
    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    if not path.exists():
        return []
    with open(path) as f:
        all_models = json.load(f)
    iso = subset_to_iso(subset)
    return [
        m for m in all_models
        if iso in m.get("languages", []) or "all" in m.get("languages", [])
    ]


def save_benchmark(iso_code, language, results):
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
        key=lambda x: (x.get("score") is None, -(x.get("score") or -1e9)),
    )
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples": NUM_SAMPLES,
        "scoring": "alignment_score",
        "benchmarks": ranked,
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved: {path}")


def _done_models(iso_code):
    path = BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {
        b["model"]
        for b in data.get("benchmarks", [])
        if b.get("score") is not None
    }


def _score_samples(samples, audio_fn, lang, device="cuda"):
    """Score each sample: generate audio via audio_fn, then align.

    audio_fn(text) -> audio_bytes

    Returns:
        dict with score (mean alignment), per_sample list, valid count.
    """
    total_score = 0.0
    valid = 0
    per_sample = []

    for sample in samples:
        text = sample["text"]
        idx = sample["index"]
        try:
            audio_bytes = audio_fn(text)
            result = compute_alignment_score_from_bytes(
                audio_bytes, text, lang=lang, device=device,
            )
            score = result["score"]
            total_score += score
            valid += 1
            per_sample.append({
                "index": idx,
                "text": text,
                "score": round(score, 4),
                "num_frames": result["num_frames"],
            })
        except Exception as e:
            per_sample.append({
                "index": idx,
                "text": text,
                "score": None,
                "error": str(e)[:200],
            })

    if valid == 0:
        return None, per_sample, 0
    return round(total_score / valid, 4), per_sample, valid


def evaluate_language(subset, model_filter=None, device="cuda", force=False):
    """Benchmark all TTS models for a language subset.

    Each model synthesises every text sample; alignment score is the metric.
    """
    iso = subset_to_iso(subset)
    language = ISO_TO_NAME.get(iso, subset)

    print(f"\n{'=' * 60}")
    print(f"  Evaluating {subset} ({language}, {iso})")
    print(f"{'=' * 60}")

    samples = load_text_samples(subset)
    if not samples:
        print(f"  No text samples for {subset} — skipping")
        return
    print(f"  Text samples: {len(samples)}")

    models = load_tts_models(subset)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No TTS models to evaluate for {subset}")
        return

    done = set() if force else _done_models(iso)
    pending = [m for m in models if m["name"] not in done]
    if not pending:
        print(f"  All {len(models)} models already benchmarked for {subset}")
        return
    print(f"  Models: {len(pending)} pending (of {len(models)})")

    results = []
    for model_info in pending:
        model_id = model_info["name"]
        print(f"\n  {'-' * 50}")
        print(f"  [{model_id}]")
        print(f"  {'-' * 50}")

        try:
            from .models import load_tts_model
            tts_model = load_tts_model(model_id, device=device)
        except Exception as load_err:
            err = str(load_err)
            reason = "unsupported_model" if "requires IPA" in err else "load_failed"
            print(f"    Failed to load: {err[:160]}")
            results.append({
                "model": model_id,
                "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
                "owner": model_id.split("/")[0],
                "architecture": model_info.get("architecture", "unknown"),
                "score": None,
                "error": reason,
                "source": "evaluated",
            })
            save_benchmark(iso, language, results)
            continue

        def audio_fn(text, _model=tts_model):
            return _model.synthesize(text, lang=iso)

        print(f"    Synthesising + aligning {len(samples)} samples...")
        t0 = time.time()
        score, per_sample, valid = _score_samples(
            samples, audio_fn, lang=iso, device=device,
        )
        elapsed = time.time() - t0

        result = {
            "model": model_id,
            "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
            "owner": model_id.split("/")[0],
            "architecture": model_info.get("architecture", "unknown"),
            "score": score,
            "samples": len(samples),
            "valid": valid,
            "avg_seconds": round(elapsed / max(len(samples), 1), 2),
            "per_sample": per_sample,
            "source": "evaluated",
        }
        if score is not None:
            print(f"    ALIGNMENT SCORE: {score:.4f}  ({valid}/{len(samples)} valid, {elapsed:.1f}s)")
        else:
            result["error"] = "no_valid_output"
            print(f"    No valid output")

        results.append(result)
        tts_model.cleanup()
        save_benchmark(iso, language, results)

    _print_leaderboard(subset)
    return results


def _print_leaderboard(subset):
    iso = subset_to_iso(subset)
    path = BENCHMARK_DIR / f"{iso}.yaml"
    if not path.exists():
        print(f"\n  No results saved for {subset}")
        return
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    ranked = sorted(
        [b for b in data.get("benchmarks", []) if b.get("score") is not None],
        key=lambda x: -(x["score"]),
    )
    print(f"\n  Leaderboard for {subset} (ranked by alignment score):")
    for rank, b in enumerate(ranked[:10], 1):
        print(f"    {rank}. {b['model'][:42]:42s}  score: {b['score']:.4f}")
