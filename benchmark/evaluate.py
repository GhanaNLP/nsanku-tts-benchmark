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
    DEFAULT_CATEGORY,
    NUM_SAMPLES,
    ISO_TO_NAME,
    ROOT,
    TTS_LANG_MAP,
    language_categories,
)
from .dataset import load_text_samples, subset_to_iso

logger = logging.getLogger(__name__)

# Written by stage 1 next to a model's clips, read by stage 2.
ERRORS_FILE = "synth_errors.json"

# Per-sample ASR output, mirroring the ASR benchmark's transcriptions/ dir:
# transcriptions/{iso}_{category}_{model}.csv
TRANSCRIPTIONS_DIR = Path(
    os.environ.get("NSANKU_TTS_TRANSCRIPTIONS_DIR", str(ROOT / "transcriptions"))
)


def clip_dir(iso, category, model_id):
    """Where one model's clips for one language and domain live."""
    return AUDIO_DIR / iso / category / _slug(model_id)


def save_transcriptions(iso, category, model_id, entries):
    """Write the judge's per-sample output so any score can be checked."""
    import csv

    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTIONS_DIR / f"{iso}_{category}_{_slug(model_id)}.csv"
    rows = sorted(entries.items(), key=lambda kv: int(kv[0]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "reference", "hypothesis", "wer", "cer"])
        for key, entry in rows:
            writer.writerow([
                key,
                entry.get("text", ""),
                entry.get("hyp", "") if entry.get("error") is None else f"ERROR: {entry['error']}",
                entry.get("wer", ""),
                entry.get("cer", ""),
            ])
    return path


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


def load_transcriptions(iso, category, model_id):
    """Read back a model's per-sample rows so a re-run only fills the gaps."""
    import csv

    path = TRANSCRIPTIONS_DIR / f"{iso}_{category}_{_slug(model_id)}.csv"
    if not path.exists():
        return {}
    entries = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            hyp = row.get("hypothesis") or ""
            entry = {"text": row.get("reference", ""), "hyp": hyp}
            # An errored sample stays unscored so a later run retries it.
            if hyp.startswith("ERROR: "):
                entry.update({"cer": None, "error": hyp[len("ERROR: "):]})
            else:
                try:
                    entry["cer"] = float(row["cer"])
                    entry["wer"] = float(row["wer"])
                except (TypeError, ValueError, KeyError):
                    entry["cer"] = None
            entries[row["sample_id"]] = entry
    return entries


def save_benchmark(iso_code, language, results, categories=None,
                   num_samples_per_category=None, judge=None):
    """Merge results into benchmarks/{iso}.yaml, ranked by CER (lower first).

    The file is a summary — per-sample references and hypotheses live in
    transcriptions/{iso}_{category}_{model}.csv, as in the ASR benchmark.
    """
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
        "num_samples_per_category": num_samples_per_category
        or existing.get("num_samples_per_category", NUM_SAMPLES),
        "categories": categories or existing.get("categories", [DEFAULT_CATEGORY]),
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
    """Synthesise every sample of every domain for every fitted model.

    Clips already on disk are left alone unless *force*, so a re-run only
    fills gaps.  Returns a list of (model_id, category, written, failed).
    """
    iso = subset_to_iso(subset)
    language = ISO_TO_NAME.get(iso, subset)
    tts_lang = TTS_LANG_MAP.get(iso, iso)

    print(f"\n{'=' * 60}")
    print(f"  Synthesising {subset} ({language}, {iso})")
    print(f"{'=' * 60}")

    models = load_tts_models(subset)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if not models:
        print(f"  No TTS models for {subset}")
        return []

    results = []
    for cat in language_categories(iso):
        category = cat["category"]
        cat_samples = samples or load_text_samples(cat["subset"], offset=0, limit=NUM_SAMPLES)
        if not cat_samples:
            print(f"  [{category}] no text samples — skipping")
            continue
        print(f"\n  domain: {category} — {len(cat_samples)} samples "
              f"(first idx={cat_samples[0]['index']}, last idx={cat_samples[-1]['index']})")
        _write_text_index(iso, language, category, cat_samples)

        for model_info in models:
            model_id = model_info["name"]
            out_dir = clip_dir(iso, category, model_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            todo = [
                s for s in cat_samples
                if force or not (out_dir / f"{s['index']:05d}.wav").exists()
            ]
            print(f"\n  [{model_id}] {len(todo)} to synthesise "
                  f"({len(cat_samples) - len(todo)} already on disk)")
            if not todo:
                results.append((model_id, category, 0, 0))
                continue

            errors = _load_errors(out_dir)
            written = failed = 0
            t0 = time.time()
            try:
                from .models import load_tts_model

                tts_model = load_tts_model(
                    model_id, device=device, subset=subset, iso=iso, meta=model_info
                )
            except Exception as e:
                print(f"    Failed to load: {str(e)[:200]}")
                for s in todo:
                    errors[str(s["index"])] = str(e)[:200]
                _save_errors(out_dir, errors)
                results.append((model_id, category, 0, len(todo)))
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
            results.append((model_id, category, written, failed))

            try:
                tts_model.cleanup()
            except Exception:
                pass

    return results


def _write_text_index(iso, language, category, samples):
    """Write the clip-index -> sentence map next to a domain's clips."""
    cat_dir = AUDIO_DIR / iso / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Clips of samples the filter no longer selects would otherwise sit
    # around looking current.  Sibling containers may be doing the same.
    pool_names = {f"{s['index']:05d}.wav" for s in samples}
    for stale in cat_dir.glob("*/*.wav"):
        if stale.name not in pool_names:
            stale.unlink(missing_ok=True)

    tmp = cat_dir / f"TEXTS.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{language} ({iso}) — {category} — clip index -> sentence\n\n")
        for sample in samples:
            f.write(f"{sample['index']:05d}  {sample['text']}\n")
    tmp.replace(cat_dir / "TEXTS.txt")


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

    models = load_tts_models(subset)
    if not models:
        return []

    categories = language_categories(iso)
    judge_meta = {"model": spec["model"], "cer_on_real_speech": spec["judge_cer"]}
    judge = load_judge(iso, device=device)

    # model -> category -> stats
    per_model = {m["name"]: {} for m in models}
    sample_clips = {}
    sample_counts = []

    try:
        for cat in categories:
            category = cat["category"]
            cat_samples = samples or load_text_samples(cat["subset"], offset=0, limit=NUM_SAMPLES)
            if not cat_samples:
                continue
            sample_counts.append(len(cat_samples))
            text_by_index = {s["index"]: s["text"] for s in cat_samples}
            print(f"\n  domain: {category} ({len(cat_samples)} samples)")

            for model_info in models:
                model_id = model_info["name"]
                audio_dir = clip_dir(iso, category, model_id)
                print(f"    [{model_id}]")
                if not audio_dir.is_dir():
                    print("      no clips — synthesise first")
                    continue

                entries = {} if force else load_transcriptions(iso, category, model_id)
                entries = {k: v for k, v in entries.items() if int(k) in text_by_index}
                synth_errors = _load_errors(audio_dir)

                pending = []
                for index in sorted(text_by_index):
                    key = str(index)
                    if entries.get(key, {}).get("cer") is not None:
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
                        # The judge heard nothing. That is a real
                        # intelligibility signal, but it can also be a flaky
                        # endpoint, so flag it rather than quietly scoring it
                        # as a total miss.
                        entry["empty_hyp"] = True
                    entries[str(index)] = entry
                elapsed = time.time() - t0

                save_transcriptions(iso, category, model_id, entries)
                scored_ids = sorted(
                    (int(k) for k, e in entries.items() if e.get("cer") is not None)
                )
                if scored_ids and model_id not in sample_clips:
                    # A clip the judge could actually read, so the demo on the
                    # dashboard is representative rather than a failure case.
                    sample_clips[model_id] = (category, scored_ids[0])
                cers = [e["cer"] for e in entries.values() if e.get("cer") is not None]
                wers = [e["wer"] for e in entries.values() if e.get("wer") is not None]
                per_model[model_id][category] = {
                    "wer": round(sum(wers) / len(wers), 4) if wers else None,
                    "cer": round(sum(cers) / len(cers), 4) if cers else None,
                    "samples": len(entries),
                    "valid": len(cers),
                    "avg_seconds_per_sample": round(elapsed / max(len(pending), 1), 2),
                }
                if cers:
                    print(f"      CER {sum(cers) / len(cers):.4f}  "
                          f"({len(cers)} valid of {len(entries)}, {elapsed:.0f}s)")
                else:
                    print("      no valid output")
    finally:
        try:
            judge.cleanup()
        except Exception:
            pass

    results = []
    for model_info in models:
        model_id = model_info["name"]
        by_cat = {c: v for c, v in per_model[model_id].items() if v}
        if not by_cat:
            continue
        results.append(_model_result(model_info, by_cat, judge_meta,
                                     sample_clips.get(model_id)))

    save_benchmark(
        iso, language, results,
        categories=[c["category"] for c in categories],
        num_samples_per_category=max(sample_counts) if sample_counts else NUM_SAMPLES,
        judge=judge_meta,
    )
    return results


def _model_result(model_info, by_category, judge_meta, sample_clip=None):
    """Average a model's per-domain scores into one row.

    Domains are weighted equally: a model is not credited for a register it
    happens to have more sentences in.
    """
    model_id = model_info["name"]
    cers = [v["cer"] for v in by_category.values() if v["cer"] is not None]
    wers = [v["wer"] for v in by_category.values() if v["wer"] is not None]
    mean_cer = round(sum(cers) / len(cers), 4) if cers else None
    mean_wer = round(sum(wers) / len(wers), 4) if wers else None
    result = {
        "model": model_id,
        "model_url": model_info.get("url", f"https://huggingface.co/{model_id}"),
        "owner": model_id.split("/")[0],
        "architecture": model_info.get("architecture", "unknown"),
        "cer": mean_cer,
        "wer": mean_wer,
        "score": mean_cer,
        "samples": sum(v["samples"] for v in by_category.values()),
        "valid": sum(v["valid"] for v in by_category.values()),
        "judge": judge_meta,
        "per_category": by_category,
        "source": "evaluated",
    }
    if sample_clip:
        category, index = sample_clip
        result["sample_category"] = category
        result["sample_clip"] = f"{index:05d}"
    return result
