"""Orchestrate TTS evaluation for a language: SpeechBERTScore *and* CER.

Every sample is a ghana-speech-eval row: a sentence plus a real recording of it.
Two metrics grade the same synthesised clip:

  * CER: an ASR judge transcribes the clip and the transcript is compared with
    the sentence (intelligibility).
  * SpeechBERTScore: the clip is compared with the real recording in a frozen
    SSL encoder's feature space (acoustic similarity, speaker, prosody).

The published score is the composite (SBS + (1 - CER)) / 2.

    SpeechBERTScore = (1/N_gen) * sum_i max_j cos(z_gen_i, z_ref_j)

Both sides go through the same frozen SSL encoder, so the score reflects how
close the generated waveform lands to the human recording in the encoder's
feature space -- intelligibility, speaker similarity, and prosody together,
weighted by whatever the encoder happens to represent at the chosen layer.

The run is two-stage, because scoring should never mean re-synthesising and
because a TTS model and the encoder rarely both fit on one GPU:

    stage 1  synthesize_language()  ->  audio/{iso}/{model}/{index}.wav
    stage 2  score_sbs_language() + score_cer_language()
    stage 3  assemble_language()    ->  benchmarks/{iso}.yaml

Incremental: results are keyed by sample key (``<source>_<row>``, the text source
and row in ghana-speech-eval). Bumping NUM_SAMPLES only scores the new rows.

Results format (benchmarks/{iso}.yaml):

    iso_639_3: twi_asante
    num_samples: 200
    scoring: speechbertscore
    encoder: {model: microsoft/wavlm-large, layer: 6, variant: precision}
    benchmarks:
      - model: org/model-ref
        sbs: 0.742
        score: 0.742
        entries:
          "3": {text: ..., sbs: 0.761, gen_sec: 4.1, ref_sec: 3.9}
          "17": {text: ..., error: ...}
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import config
from .config import (
    DEFAULT_CATEGORY,
    ENCODER_LAYER,
    ENCODER_MODEL,
    HF_TOKEN,
    MAX_JUDGE_SECONDS,
    ISO_TO_NAME,
    NUM_SAMPLES,
    SBS_VARIANT,
    TTS_LANG_MAP,
    all_isos,
)
from .dataset import load_samples, source_of, write_manifest

logger = logging.getLogger(__name__)

ERRORS_FILE = "errors.json"


# ── Model registry ───────────────────────────────────────────────────────────


def load_tts_models(iso):
    """Models eligible for *iso*, expanded into their inference modes.

    A model that can read both with and without a reference clip is two
    entries — "<model>-ref" and "<model>-noref" — scored separately, because
    the two are different systems from a user's point of view and the
    difference is worth measuring rather than deciding.
    """
    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        all_models = json.load(f)
    ret = []
    for m in all_models:
        langs = m.get("languages", [])
        if iso not in langs and "all" not in langs:
            continue
        modes = m.get("modes") or (["ref"] if m.get("uses_reference") else ["noref"])
        mode_langs = m.get("mode_languages") or {}
        for mode in modes:
            mode_iso = mode_langs.get(mode)
            if mode_iso is not None and iso not in mode_iso and "all" not in mode_iso:
                continue
            entry = {**m, "model_id": m["name"], "mode": mode}
            # Any run that used a reference clip says so in its name, so the
            # board needs no separate marker. A model that only runs without one
            # keeps its plain name.
            if mode == "ref" or len(modes) > 1:
                entry["name"] = f"{m['name']}-{mode}"
            entry["uses_reference"] = mode == "ref"
            ret.append(entry)
    return ret


def clip_dir(iso, category, model_id):
    """Where one model's synthesised clips live for a language.

    Resolved per call rather than at import, so overriding the output location
    (as the tests do) actually redirects the write instead of quietly filling
    the repository's real audio/ directory.
    """
    return config.AUDIO_DIR / iso / category / model_id.replace("/", "__")


def _slug(model_id):
    return model_id.replace("/", "__")


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


# ── Stage 1: synthesis ───────────────────────────────────────────────────────


def synthesize_language(iso, model_filter=None, device="cuda", force=False, samples=None,
                        stack=None):
    """Synthesise every sample of every model for one language.

    Clips already on disk are left alone unless *force*, so a re-run only fills
    gaps. Returns a list of (model_name, written, failed).
    """
    from .models import prompt_scope

    language = ISO_TO_NAME.get(iso, iso)
    tts_lang = TTS_LANG_MAP.get(iso, iso)
    category = DEFAULT_CATEGORY

    print(f"\n{'=' * 60}")
    print(f"  Synthesising {iso} ({language})")
    print(f"{'=' * 60}")

    models = load_tts_models(iso)
    if model_filter:
        models = [m for m in models if model_filter.lower() in m["name"].lower()]
    if stack == "omni":
        models = [m for m in models if m.get("stack") == "omni"]
    elif stack == "tts":
        # Everything that runs in the standard image; OmniVoice needs its own.
        models = [m for m in models if m.get("stack") != "omni"]
    if not models:
        print(f"  No TTS models for {iso}")
        return []

    samples = samples or load_samples(iso, limit=NUM_SAMPLES)
    if not samples:
        print(f"  no usable samples for {iso} — skipping")
        return []
    write_manifest(iso, samples)
    print(f"  {len(samples)} samples, first idx={samples[0].key}, "
          f"last idx={samples[-1].key}")

    results = []
    for model_info in models:
        model_id = model_info["model_id"]
        name = model_info["name"]
        out_dir = clip_dir(iso, category, name)
        out_dir.mkdir(parents=True, exist_ok=True)

        todo = [s for s in samples
                if force or not (out_dir / f"{s.key}.wav").exists()]
        print(f"\n  [{name}] {len(todo)} to synthesise "
              f"({len(samples) - len(todo)} already on disk)")
        if not todo:
            results.append((name, 0, 0))
            continue

        errors = _load_errors(out_dir)
        written = failed = 0
        t0 = time.time()
        try:
            from .models import load_tts_model

            tts_model = load_tts_model(
                model_id, device=device, iso=iso, meta=model_info
            )
        except Exception as e:
            print(f"    Failed to load: {str(e)[:200]}")
            for s in todo:
                errors[str(s.index)] = str(e)[:200]
            _save_errors(out_dir, errors)
            results.append((name, 0, len(todo)))
            continue

        seen = set()
        for sample in todo:
            key = str(sample.index)
            try:
                # prompt_scope publishes this row so a ref-mode model's
                # reference_clip() returns a *different* row, never the
                # recording SpeechBERTScore will score against.
                with prompt_scope(iso, sample.index):
                    spoken = tts_model.prepare_text(sample.text, iso)
                    audio = tts_model.synthesize(spoken, lang=tts_lang)
                (out_dir / f"{sample.key}.wav").write_bytes(audio)
                errors.pop(key, None)
                written += 1
            except Exception as e:
                msg = str(e)[:300]
                errors[key] = msg
                failed += 1
                if msg not in seen:
                    seen.add(msg)
                    print(f"      [first error] {msg}")

        _save_errors(out_dir, errors)
        rate = written / max(1e-9, time.time() - t0)
        print(f"    {written} written, {failed} failed ({rate:.2f}/s)")
        try:
            del tts_model
        except Exception:
            pass
        results.append((name, written, failed))

    return results


# ── Stage 2a: SpeechBERTScore ────────────────────────────────────────────────


def score_sbs_language(iso, device="cuda", force=False, samples=None):
    """Score every synthesised clip in *iso* against its real reference.

    Writes only the per-sample cache; ``assemble_language`` builds the YAML.
    Returns {model: mean_sbs}.
    """
    from .speechbertscore import SpeechBERTScorer, self_check

    print(f"\n{'=' * 60}")
    print(f"  SpeechBERTScore: {iso} ({ISO_TO_NAME.get(iso, iso)})")
    print(f"{'=' * 60}")

    samples = samples or load_samples(iso, limit=NUM_SAMPLES, extract_references=True)
    if not samples:
        print(f"  no samples for {iso}")
        return {}

    encoder_tag = {"model": ENCODER_MODEL, "layer": ENCODER_LAYER, "variant": SBS_VARIANT}
    cache = load_score_cache(iso)
    if cache and cache.get("encoder") == encoder_tag:
        per_sample = cache.get("per_sample", {})
    else:
        if cache:
            print(f"  cache was built with {cache.get('encoder')}, not "
                  f"{encoder_tag} — discarding and re-scoring")
        per_sample = _seed_from_yaml(iso, encoder_tag)

    models = load_tts_models(iso)
    scorer = None
    means = {}
    for model_info in models:
        name = model_info["name"]
        out_dir = clip_dir(iso, DEFAULT_CATEGORY, name)
        clips = [s for s in samples if (out_dir / f"{s.key}.wav").exists()]
        if not clips:
            print(f"  [{name}] no clips — skipping")
            continue

        import soundfile as sf

        scores = []
        for sample in clips:
            key = str(sample.index)
            cached = per_sample.get(name, {}).get(key)
            if not force and cached and "sbs" in cached:
                scores.append(cached["sbs"])
                continue
            if scorer is None:
                scorer = SpeechBERTScorer(device=device)
                if not self_check(scorer):
                    raise RuntimeError("scorer failed its determinism self-check")
                print(f"  encoder: {ENCODER_MODEL} layer {ENCODER_LAYER} "
                      f"({SBS_VARIANT} variant)")
            try:
                gen, gen_sr = sf.read(out_dir / f"{sample.key}.wav",
                                      dtype="float32", always_2d=False)
                ref, ref_sr = sf.read(sample.reference_path,
                                      dtype="float32", always_2d=False)
                value = scorer.score(gen, gen_sr, ref, ref_sr)
                if value is None:
                    raise ValueError("empty audio")
                per_sample.setdefault(name, {})[key] = {
                    "text": sample.text,
                    "sbs": round(float(value), 4),
                    "gen_sec": round(len(gen) / gen_sr, 2),
                    "ref_sec": round(len(ref) / ref_sr, 2),
                }
                scores.append(float(value))
            except Exception as e:
                per_sample.setdefault(name, {})[key] = {
                    "text": sample.text, "error": str(e)[:200]}

        if scores:
            means[name] = sum(scores) / len(scores)
            print(f"  [{name}] SBS {means[name]:.4f} over {len(scores)} samples")

    if scorer is not None:
        scorer.cleanup()
    save_score_cache(iso, per_sample, encoder_tag)
    return means


def _seed_from_yaml(iso, encoder_tag):
    """Recover per-sample SBS from a published results file when no cache exists.

    The YAML carries every scored clip, so a fresh checkout does not have to
    re-score results that are already in git. Only trusted when the file was
    produced by the same encoder configuration.
    """
    doc = _load_existing(iso)
    if doc.get("encoder") != encoder_tag:
        return {}
    seeded = {}
    for entry in doc.get("benchmarks", []):
        rows = {}
        for key, row in (entry.get("entries") or {}).items():
            if "sbs" in row:
                rows[key] = {k: row[k] for k in ("text", "sbs", "gen_sec", "ref_sec") if k in row}
            elif "error" in row:
                rows[key] = {"text": row.get("text"), "error": row["error"]}
        if rows:
            seeded[entry["model"]] = rows
    return seeded


# ── Stage 2b: CER via the language's ASR judge ───────────────────────────────


def score_cer_language(iso, device="cuda", force=False, samples=None):
    """Transcribe every synthesised clip with the language's ASR judge and
    compare it to the sentence that was synthesised.

    The samples are the same ghana-speech-eval rows SpeechBERTScore uses, so
    both metrics grade the same clips. Returns {model: mean_cer}.
    """
    from .asr import judge_for, load_judge
    from .metrics import compute_metrics

    print(f"\n{'=' * 60}")
    print(f"  CER: {iso} ({ISO_TO_NAME.get(iso, iso)})")
    print(f"{'=' * 60}")

    spec = judge_for(iso)
    if spec is None:
        print(f"  No ASR judge for {iso} — skipping")
        return {}
    print(f"  Judge: {spec['model']} (its own CER on real speech: {spec['judge_cer']})")

    samples = samples or load_samples(iso, limit=NUM_SAMPLES, extract_references=False)
    if not samples:
        return {}
    text_by_key = {str(s.index): s.text for s in samples}

    cache = load_cer_cache(iso)
    if cache.get("judge") != spec["model"]:
        cache = {"judge": spec["model"], "per_sample": {}}
    per_sample = cache["per_sample"]

    judge = None
    means = {}
    try:
        for model_info in load_tts_models(iso):
            name = model_info["name"]
            out_dir = clip_dir(iso, DEFAULT_CATEGORY, name)
            rows = per_sample.setdefault(name, {})
            pending = []
            for s in samples:
                key = str(s.index)
                if not force and (rows.get(key, {}).get("cer") is not None
                                  or "error" in rows.get(key, {})):
                    continue
                wav = out_dir / f"{s.key}.wav"
                if not wav.exists():
                    continue
                if _clip_seconds(wav) > MAX_JUDGE_SECONDS:
                    rows[key] = {"error": f"clip longer than the judge limit "
                                          f"({MAX_JUDGE_SECONDS:.0f}s)"}
                    continue
                pending.append((key, wav))
            if pending:
                if judge is None:
                    judge = load_judge(iso, device=device)
                hyps = judge.transcribe_many([p.read_bytes() for _, p in pending])
                for (key, _), hyp in zip(pending, hyps):
                    m = compute_metrics(text_by_key[key], hyp)
                    rows[key] = {"hyp": hyp, "cer": m["cer"], "wer": m["wer"]}
                    if not hyp:
                        # Silence is a real intelligibility signal but can also
                        # be a flaky endpoint, so flag it.
                        rows[key]["empty_hyp"] = True
            cers = [rows[k]["cer"] for k in text_by_key
                    if rows.get(k, {}).get("cer") is not None]
            if cers:
                means[name] = sum(cers) / len(cers)
                print(f"  [{name}] CER {means[name]:.4f} over {len(cers)} clips")
    finally:
        if judge is not None:
            try:
                judge.cleanup()
            except Exception:
                pass

    save_cer_cache(iso, cache)
    return means


def _clip_seconds(path):
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


# ── Stage 3: assemble the published YAML ─────────────────────────────────────


def assemble_language(iso, samples=None):
    """Join the SBS and CER caches into benchmarks/{iso}.yaml.

    Only the current sample set is reported. The caches can hold rows from
    earlier runs (other filters, a larger NUM_SAMPLES); averaging those in
    would mix sample sets and let the two metrics grade different clips.

    Composite is (SBS + (1 - CER)) / 2 and exists only when both metrics were
    scored for the model; a model with one metric is listed with ``score`` set
    to that metric and ``partial: true`` so the board can say so.
    """
    from .asr import judge_for

    encoder_tag = {"model": ENCODER_MODEL, "layer": ENCODER_LAYER, "variant": SBS_VARIANT}
    sbs_cache = load_score_cache(iso)
    sbs_rows = sbs_cache.get("per_sample", {}) if sbs_cache.get("encoder") == encoder_tag \
        else _seed_from_yaml(iso, encoder_tag)
    cer_rows = load_cer_cache(iso).get("per_sample", {})

    if samples is None:
        samples = load_samples(iso, limit=NUM_SAMPLES, extract_references=False)
    allowed = {str(s.index) for s in samples}

    benchmarks = []
    for info in load_tts_models(iso):
        name = info["name"]
        s_rows = {k: v for k, v in sbs_rows.get(name, {}).items()
                  if k in allowed and "sbs" in v}
        c_rows = {k: v for k, v in cer_rows.get(name, {}).items()
                  if k in allowed and v.get("cer") is not None}
        if not s_rows and not c_rows:
            continue

        entries = {}
        for key in sorted(set(s_rows) | set(c_rows)):
            row = {"source": source_of(key)}
            row.update(s_rows.get(key) or {})
            c = c_rows.get(key)
            if c:
                row.setdefault("text", sbs_rows.get(name, {}).get(key, {}).get("text"))
                row["cer"], row["wer"] = c["cer"], c["wer"]
            entries[key] = row

        entry = {
            "model": name,
            "model_url": info.get("url", f"https://huggingface.co/{info.get('model_id', name)}"),
            "mode": info.get("mode", "noref"),
            "owner": name.split("/")[0],
            "architecture": info.get("architecture", "unknown"),
            "model_class": info.get("model_class", "non-llm"),
            "params": info.get("params", "?"),
            "uses_reference": info.get("uses_reference", False),
        }
        sbs = round(sum(v["sbs"] for v in s_rows.values()) / len(s_rows), 4) if s_rows else None
        cer = round(sum(v["cer"] for v in c_rows.values()) / len(c_rows), 4) if c_rows else None
        wer = round(sum(v["wer"] for v in c_rows.values()) / len(c_rows), 4) if c_rows else None
        if sbs is not None:
            entry["sbs"], entry["num_scored"] = sbs, len(s_rows)
        if cer is not None:
            entry.update(cer=cer, wer=wer, accuracy=round(max(0.0, 1.0 - cer), 4),
                         cer_num_scored=len(c_rows))
        if sbs is not None and cer is not None:
            entry["composite"] = round((entry["accuracy"] + sbs) / 2.0, 4)
            entry["score"] = entry["composite"]
        else:
            entry["score"] = sbs if sbs is not None else cer
            entry["partial"] = True
        entry["per_source"] = _per_source(entries)
        entry["entries"] = entries
        benchmarks.append(entry)

    if not benchmarks:
        return None

    spec = judge_for(iso)
    payload = {
        "iso_639_3": iso,
        "language": ISO_TO_NAME.get(iso, iso),
        "category": DEFAULT_CATEGORY,
        "num_samples": len(samples),
        "scoring": "composite (sbs + (1 - cer)) / 2",
        "encoder": encoder_tag,
        "judge": {"model": spec["model"], "cer_on_real_speech": spec["judge_cer"]} if spec else None,
        "benchmarks": sorted(benchmarks, key=lambda b: (-b["score"], b["model"])),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    class _NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    config.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = config.BENCHMARK_DIR / f"{iso}.yaml"
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, allow_unicode=True, sort_keys=False,
                  Dumper=_NoAliasDumper, width=100)
    tmp.replace(path)
    return path


def _per_source(entries):
    """Mean SBS / CER per text source, so a model's result can be read by register."""
    acc = {}
    for row in entries.values():
        a = acc.setdefault(row["source"], {"sbs": [], "cer": []})
        if "sbs" in row:
            a["sbs"].append(row["sbs"])
        if "cer" in row:
            a["cer"].append(row["cer"])
    out = {}
    for source, a in sorted(acc.items()):
        d = {}
        if a["sbs"]:
            d["sbs"] = round(sum(a["sbs"]) / len(a["sbs"]), 4)
        if a["cer"]:
            d["cer"] = round(sum(a["cer"]) / len(a["cer"]), 4)
        d["n"] = max(len(a["sbs"]), len(a["cer"]))
        out[source] = d
    return out


# ── Result caches ────────────────────────────────────────────────────────────


def _load_existing(iso):
    path = config.BENCHMARK_DIR / f"{iso}.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}


def _cache_path(iso):
    """Per-sample SBS cache. Kept out of the YAML so the published file stays
    readable, but it must outlive the process or re-runs would re-score every
    clip."""
    return config.BENCHMARK_DIR / f"{iso}.scores.json"


def _read_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    tmp.replace(path)
    return path


def load_score_cache(iso):
    return _read_json(_cache_path(iso))


def save_score_cache(iso, per_sample, encoder):
    """Tagged with the encoder: scores from a different encoder or layer are
    not comparable, so a changed encoder discards the cache."""
    return _write_json(_cache_path(iso), {"encoder": encoder, "per_sample": per_sample})


def _cer_cache_path(iso):
    return config.BENCHMARK_DIR / f"{iso}.cer.json"


def load_cer_cache(iso):
    return _read_json(_cer_cache_path(iso))


def save_cer_cache(iso, cache):
    """Tagged with the judge: CER from a different ASR model is not comparable."""
    return _write_json(_cer_cache_path(iso), cache)


def synthesize_all(model_filter=None, device="cuda", force=False, limit=None):
    results = {}
    for iso in all_isos():
        if limit:
            samples = load_samples(iso, limit=limit)
            results[iso] = synthesize_language(
                iso, model_filter=model_filter, device=device,
                force=force, samples=samples)
        else:
            results[iso] = synthesize_language(
                iso, model_filter=model_filter, device=device, force=force)
    return results


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="nsanku-TTS benchmark")
    parser.add_argument(
        "command",
        choices=["synthesize", "score-sbs", "score-cer", "assemble", "score", "all"],
        help="score = score-sbs + score-cer + assemble (needs both stacks in one "
             "environment); the H200 runner uses separate images per stage.")
    parser.add_argument("--iso", action="append", help="limit to these languages")
    parser.add_argument("--model", help="substring filter on model name")
    parser.add_argument("--stack", choices=["tts", "omni"],
                        help="synthesize only the models that run in this image")
    parser.add_argument("--limit", type=int, help="samples per language")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for iso in args.iso or all_isos():
        cmd = args.command
        # Reference audio is only needed by SBS; CER just needs the text.
        need_refs = cmd in ("score-sbs", "score", "all")
        # assemble needs the sample set too, to report only those rows.
        samples = load_samples(iso, limit=args.limit or NUM_SAMPLES,
                               extract_references=need_refs)
        if cmd in ("synthesize", "all"):
            synthesize_language(iso, model_filter=args.model, device=args.device,
                                force=args.force, samples=samples, stack=args.stack)
        if cmd in ("score-sbs", "score", "all"):
            score_sbs_language(iso, device=args.device, force=args.force, samples=samples)
        if cmd in ("score-cer", "score", "all"):
            score_cer_language(iso, device=args.device, force=args.force, samples=samples)
        if cmd in ("assemble", "score", "all"):
            print(f"  wrote {assemble_language(iso, samples)}")


if __name__ == "__main__":
    main()
