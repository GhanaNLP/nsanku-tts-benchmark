"""Orchestrate TTS evaluation for a single language.

Scoring is intelligibility, not acoustics: every clip is transcribed by the
best ASR model for that language (data/asr_judges.json) and compared to the
text the model was asked to read.  The ranking metric is mean CER — lower is
better, and it is directly comparable to the ASR leaderboard's CER because
both use the same normalisation (benchmark/metrics.py).

The run is two-stage, because the judges are too large to sit next to a TTS
model on one GPU, and because re-scoring should never mean re-synthesising:

    stage 1  synthesize_language()  ->  audio/{iso}/{model}/{index}.wav
    stage 2  score_language()       ->  benchmarks/{iso}.yaml

Incremental: results are keyed by sample *index* (the row index in the
ghana-sentences subset). Bumping NUM_SAMPLES only scores the new rows.

Results format (benchmarks/{iso}.yaml):

    iso_639_3: twi_asante
    num_samples: 200
    scoring: cer
    judge: {model: Qlerqly/griot-nano-1, cer_on_real_speech: 0.2212}
    benchmarks:
      - model: org/model
        cer: 0.31
        wer: 0.55
        score: 0.31
        entries:
          "3": {text: ..., hyp: ..., cer: 0.28, wer: 0.51}
          "17": {text: ..., error: ...}
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import (
    AUDIO_DIR,
    BENCHMARK_DIR,
    NUM_SAMPLES,
    ISO_TO_NAME,
    TTS_LANG_MAP,
)
from .dataset import load_text_samples, subset_to_iso

logger = logging.getLogger(__name__)

# Written by stage 1 next to a model's clips, read by stage 2.
ERRORS_FILE = "synth_errors.json"


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slug(model_id):
    return model_id.replace("/", "__")


def load_tts_models(subset):
    """Load models eligible for *subset* from data/tts_models.json."""
    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
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
    with open(benchmark_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for b in data.get("benchmarks", []):
        if b.get("model") != model_id:
            continue
        entries = b.get("entries") or {}
        scored = set()
        for sample_idx, entry in entries.items():
            # Errored samples stay unscored so a later run retries them,
            # rather than being skipped forever.
            if not isinstance(entry, dict) or entry.get("cer") is None:
                continue
            try:
                scored.add(int(sample_idx))
            except (TypeError, ValueError):
                continue
        return scored
    return set()


def save_benchmark(iso_code, language, results, num_samples=None, judge=None):
    """Merge results into benchmarks/{iso}.yaml, ranked by CER (lower first)."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{iso_code}.yaml"

    existing = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    merged = {r["model"]: r for r in existing.get("benchmarks", [])}
    for r in results:
        merged[r["model"]] = r

    ranked = sorted(
        merged.values(),
        key=lambda x: (x.get("score") is None, x.get("score") if x.get("score") is not None else 0),
    )
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples": num_samples if num_samples is not None else NUM_SAMPLES,
        "scoring": "cer",
        "judge": judge or existing.get("judge"),
        "updated": _today(),
        "benchmarks": ranked,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Saved: {path}")


# ── Stage 1: synthesis ───────────────────────────────────────────────────────


def synthesize_language(subset, model_filter=None, device="cuda", force=False, samples=None):
    """Synthesise every sample for every fitted model into the audio dir.

    Clips already on disk are left alone unless *force*, so a re-run only
    fills gaps.  Returns a list of (model_id, written, failed).
    """
    iso = subset_to_iso(subset)
    language = ISO_TO_NAME.get(iso, subset)
    tts_lang = TTS_LANG_MAP.get(iso, iso)

    print(f"\n{'=' * 60}")
    print(f"  Synthesising {subset} ({language}, {iso})")
    print(f"{'=' * 60}")

    if samples is None:
        samples = load_text_samples(subset, offset=0, limit=NUM_SAMPLES)
    if not samples:
        print(f"  No text samples for {subset} — skipping")
        return []
    print(f"  Samples: {len(samples)} (first idx={samples[0]['index']}, "
          f"last idx={samples[-1]['index']})")

    models = load_tts_models(subset)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No TTS models for {subset}")
        return []

    _write_text_index(iso, language, samples)

    results = []
    for model_info in models:
        model_id = model_info["name"]
        out_dir = AUDIO_DIR / iso / _slug(model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        todo = [
            s for s in samples
            if force or not (out_dir / f"{s['index']:05d}.wav").exists()
        ]
        print(f"\n  [{model_id}] {len(todo)} to synthesise "
              f"({len(samples) - len(todo)} already on disk)")
        if not todo:
            results.append((model_id, 0, 0))
            continue

        errors = _load_errors(out_dir)
        written = failed = 0
        t0 = time.time()
        try:
            from .models import load_tts_model

            tts_model = load_tts_model(model_id, device=device, subset=subset, meta=model_info)
        except Exception as e:
            print(f"    Failed to load: {str(e)[:200]}")
            for s in todo:
                errors[str(s["index"])] = str(e)[:200]
            _save_errors(out_dir, errors)
            results.append((model_id, 0, len(todo)))
            continue

        for sample in todo:
            key = str(sample["index"])
            try:
                audio = tts_model.synthesize(sample["text"], lang=tts_lang)
                (out_dir / f"{sample['index']:05d}.wav").write_bytes(audio)
                errors.pop(key, None)
                written += 1
            except Exception as e:
                errors[key] = str(e)[:200]
                failed += 1

        _save_errors(out_dir, errors)
        elapsed = time.time() - t0
        print(f"    {written} written, {failed} failed "
              f"({elapsed:.0f}s, {elapsed / max(len(todo), 1):.1f}s/sample)")
        results.append((model_id, written, failed))

        try:
            tts_model.cleanup()
        except Exception:
            pass

    return results


def _write_text_index(iso, language, samples):
    """Write the clip-index -> sentence map next to a language's clips."""
    lang_dir = AUDIO_DIR / iso
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Clips of samples the filter no longer selects would otherwise sit
    # around looking current.  Sibling containers may be doing the same.
    pool_names = {f"{s['index']:05d}.wav" for s in samples}
    for stale in lang_dir.glob("*/*.wav"):
        if stale.name not in pool_names:
            stale.unlink(missing_ok=True)

    tmp = lang_dir / f"TEXTS.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{language} ({iso}) — clip index -> sentence\n\n")
        for sample in samples:
            f.write(f"{sample['index']:05d}  {sample['text']}\n")
    tmp.replace(lang_dir / "TEXTS.txt")


def _load_errors(model_dir):
    path = model_dir / ERRORS_FILE
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_errors(model_dir, errors):
    with open(model_dir / ERRORS_FILE, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


# ── Stage 2: ASR scoring ─────────────────────────────────────────────────────


def score_language(subset, device="cuda", force=False, samples=None):
    """Transcribe every synthesised clip and score it against its text."""
    from .asr import judge_for, load_judge
    from .metrics import compute_metrics

    iso = subset_to_iso(subset)
    language = ISO_TO_NAME.get(iso, subset)

    print(f"\n{'=' * 60}")
    print(f"  Scoring {subset} ({language}, {iso})")
    print(f"{'=' * 60}")

    spec = judge_for(iso)
    if spec is None:
        print(f"  No ASR judge for {iso} — skipping")
        return []
    print(f"  Judge: {spec['model']} (its own CER on real speech: {spec['judge_cer']})")

    if samples is None:
        samples = load_text_samples(subset, offset=0, limit=NUM_SAMPLES)
    if not samples:
        print(f"  No text samples for {subset} — skipping")
        return []
    text_by_index = {s["index"]: s["text"] for s in samples}

    models = load_tts_models(subset)
    if not models:
        return []

    benchmark_path = BENCHMARK_DIR / f"{iso}.yaml"
    judge_meta = {"model": spec["model"], "cer_on_real_speech": spec["judge_cer"]}
    judge = load_judge(iso, device=device)
    results = []

    try:
        for model_info in models:
            model_id = model_info["name"]
            audio_dir = AUDIO_DIR / iso / _slug(model_id)
            print(f"\n  [{model_id}]")
            if not audio_dir.is_dir():
                print("    No clips — synthesise first")
                continue

            already = set() if force else _scored_indices(benchmark_path, model_id)
            synth_errors = _load_errors(audio_dir)

            entries = {}
            pending = []
            for index in sorted(text_by_index):
                key = str(index)
                if index in already:
                    continue
                wav_path = audio_dir / f"{index:05d}.wav"
                if not wav_path.exists():
                    entries[key] = {
                        "text": text_by_index[index],
                        "cer": None,
                        "error": synth_errors.get(key, "no audio synthesised"),
                    }
                    continue
                pending.append((index, wav_path))

            t0 = time.time()
            hyps = judge.transcribe_many([p.read_bytes() for _, p in pending])
            for (index, _), hyp in zip(pending, hyps):
                reference = text_by_index[index]
                scores = compute_metrics(reference, hyp)
                entry = {
                    "text": reference,
                    "hyp": hyp,
                    "cer": scores["cer"],
                    "wer": scores["wer"],
                }
                if not hyp:
                    # The judge heard nothing. That is a real intelligibility
                    # signal, but it can also be a flaky endpoint, so flag it
                    # rather than quietly scoring it as a total miss.
                    entry["empty_hyp"] = True
                entries[str(index)] = entry
            transcribed = len(pending)
            elapsed = time.time() - t0

            result = _merge_model_result(
                benchmark_path, model_info, entries, text_by_index, elapsed,
                transcribed, judge_meta,
            )
            results.append(result)
            if result["cer"] is not None:
                print(f"    CER {result['cer']:.4f}  WER {result['wer']:.4f}  "
                      f"({result['valid']} valid of {result['samples']}, {elapsed:.0f}s)")
            else:
                print("    No valid output")
            save_benchmark(
                iso, language, [result],
                num_samples=max(len(text_by_index), NUM_SAMPLES),
                judge=judge_meta,
            )
    finally:
        try:
            judge.cleanup()
        except Exception:
            pass

    return results


def _merge_model_result(benchmark_path, model_info, entries, text_by_index,
                        elapsed, transcribed, judge_meta):
    """Fold fresh entries into whatever this model already had scored."""
    model_id = model_info["name"]

    existing_entries = {}
    if benchmark_path.exists():
        with open(benchmark_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for b in data.get("benchmarks", []):
            if b.get("model") == model_id:
                existing_entries = b.get("entries") or {}
                break

    # Drop entries for samples no longer in the pool — otherwise a change to
    # the sample filter leaves orphans skewing the average.
    pool = {str(i) for i in text_by_index}
    existing_entries = {k: v for k, v in existing_entries.items() if k in pool}
    existing_entries.update(entries)

    cers = [v["cer"] for v in existing_entries.values() if v.get("cer") is not None]
    wers = [v["wer"] for v in existing_entries.values() if v.get("wer") is not None]
    mean_cer = sum(cers) / len(cers) if cers else None
    mean_wer = sum(wers) / len(wers) if wers else None

    return {
        "model": model_id,
        "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
        "owner": model_id.split("/")[0],
        "architecture": model_info.get("architecture", "unknown"),
        "cer": round(mean_cer, 4) if mean_cer is not None else None,
        "wer": round(mean_wer, 4) if mean_wer is not None else None,
        "score": round(mean_cer, 4) if mean_cer is not None else None,
        "samples": len(existing_entries),
        "valid": len(cers),
        "avg_seconds": round(elapsed / max(transcribed, 1), 2),
        "judge": judge_meta,
        "entries": existing_entries,
        "source": "evaluated",
    }
