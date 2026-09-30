"""End-to-end pipeline check with a fake dataset, no network and no GPU.

Proves the wiring in evaluate.py -- model expansion, synthesis, prompt scoping,
scoring, aggregation and the YAML shape -- using a stub TTS model and a stub
ghana-speech-eval. Everything except the real encoder and real speech is faked;
the metric itself is covered by scripts/test_metric.py.

Run:  PYTHONPATH=. python scripts/test_pipeline.py
"""

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    extra = f"  {detail}" if detail else ""
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{extra}")


def repo_snapshot(config):
    """Filenames in the real repo's output dirs, to prove the test wrote nothing."""
    out = set()
    for d in (config.ROOT / "benchmarks", config.ROOT / "data"):
        if d.exists():
            out |= {f"{d.name}/{p.name}" for p in d.iterdir()}
    return out


def main():
    tmp = Path(tempfile.mkdtemp())

    from benchmark import config, dataset, evaluate, models, speechbertscore

    # Point every output path at the temp dir before anything writes.
    config.AUDIO_DIR = tmp / "audio"
    config.BENCHMARK_DIR = tmp / "benchmarks"
    config.REFERENCE_DIR = tmp / "references"
    config.DATA_DIR = tmp / "data"
    repo_before = repo_snapshot(config)

    iso = "twi_asante"
    n_rows = 12
    sr = 16000
    rng = np.random.default_rng(0)

    # ── a fake ghana-speech-eval ───────────────────────────────────────────
    # Every row is usable, so the +500 offset row is usable too. If the offset
    # row were rejected, load_prompt_sample returns None and reference_clip
    # silently falls back to the published clip -- which would make the
    # disjointness check below pass for entirely the wrong reason.
    rows = [{"text": f"Row number {i} has some words",
             "length": 5.0,
             "audio": {"bytes": b""}} for i in range(n_rows)]
    dataset.config_rows = lambda i: rows

    print("\n== sample filtering ==")
    for text, dur, want in [
        ("Row number 0 has some words", 5.0, True),
        ("Row number 0 has some words", 1.0, False),      # too short to be a stable utterance
        ("Row number 0 has some words", 40.0, False),     # too long
        ("one two three four", 5.0, False),               # too few words
        (" ".join(["word"] * 40), 5.0, False),            # too many words
        ("", 5.0, False),                                 # no text
    ]:
        check(f"is_usable({dur}s, {len(text.split())} words) == {want}",
              dataset.is_usable(text, dur) is want)

    def fake_write_reference(sample, wav_bytes):
        out = sample.reference_path
        out.parent.mkdir(parents=True, exist_ok=True)
        tone = rng.normal(0, 0.05, sr * 4).astype(np.float32)
        sf.write(out, tone, sr, subtype="PCM_16")
        return out

    dataset._write_reference = fake_write_reference

    # The scoped prompt path must not touch the network. If it ever falls back
    # to the published per-language clip, this raises instead of silently
    # making the disjointness check pass for the wrong reason.
    import huggingface_hub

    def no_network(*a, **k):
        raise AssertionError(
            "the row-scoped prompt path reached the network; it fell back to "
            "the published clip, so prompt disjointness was not exercised")

    real_dl = huggingface_hub.hf_hub_download
    huggingface_hub.hf_hub_download = no_network

    # ── a stub TTS model that records what prompt it was handed ────────────
    prompts_seen = []

    class StubTTS:
        def __init__(self):
            self.model_id = "stub/model"

        def prepare_text(self, text, iso):
            return text

        def synthesize(self, text, lang="twi"):
            clip, _text, _src = models.reference_clip(
                iso, {"recipe": None}, token=None)
            prompts_seen.append(clip)
            # A tone that is not the reference, so the score is below 1.0.
            t = np.arange(int(sr * 4)) / sr
            out = np.sin(2 * np.pi * 300 * t).astype(np.float32) * 0.1
            buf = tmp / "gen.wav"
            sf.write(buf, out, sr, subtype="PCM_16")
            return buf.read_bytes()

    models.load_tts_model = lambda *a, **k: StubTTS()

    # ── model expansion ────────────────────────────────────────────────────
    print("\n== model expansion ==")
    entry = {"name": "org/Model", "languages": [iso],
             "modes": ["ref", "noref"],
             "mode_languages": {"ref": [iso], "noref": [iso]}}
    real_registry = config.ROOT / "data" / "tts_models.json"
    saved_registry = real_registry.read_bytes() if real_registry.exists() else None
    real_registry.write_text(json.dumps([entry]))
    try:
        return run_pipeline(config, dataset, evaluate, models, speechbertscore,
                            rows, fake_write_reference, tmp, iso,
                            samples_expected=3, repo_before=repo_before, n_rows=n_rows)
    finally:
        if saved_registry is not None:
            real_registry.write_bytes(saved_registry)
        huggingface_hub.hf_hub_download = real_dl


def run_pipeline(config, dataset, evaluate, models, speechbertscore,
                 rows, fake_write_reference, tmp, iso, samples_expected,
                 repo_before, n_rows):
    import shutil

    expanded = evaluate.load_tts_models(iso)
    names = [m["name"] for m in expanded]
    check("both modes are reported separately",
          names == ["org/Model-ref", "org/Model-noref"], str(names))
    check("the ref entry is flagged as using a reference",
          expanded[0]["uses_reference"] is True)
    check("the noref entry is not", expanded[1]["uses_reference"] is False)
    check("a language outside the model's list is excluded",
          evaluate.load_tts_models("ewe") == [], str(evaluate.load_tts_models("ewe")))

    # A mode the model does not offer for this language must be dropped.
    real_registry = config.ROOT / "data" / "tts_models.json"
    saved = real_registry.read_bytes()
    narrow = {"name": "org/Model", "languages": [iso], "modes": ["ref", "noref"],
              "mode_languages": {"ref": [iso], "noref": []}}
    real_registry.write_text(json.dumps([narrow]))
    check("a mode unavailable for this language is dropped",
          [m["name"] for m in evaluate.load_tts_models(iso)] == ["org/Model-ref"],
          str([m["name"] for m in evaluate.load_tts_models(iso)]))
    real_registry.write_bytes(saved)

    # ── synthesis + scoring ────────────────────────────────────────────────
    print("\n== synthesis ==")
    samples = dataset.load_samples(iso, limit=samples_expected)
    check("the sample loader honours its limit",
          len(samples) == samples_expected,
          f"got {len(samples)} requested {samples_expected}")
    check("reference audio was written for each sample",
          all(s.reference_path.exists() for s in samples))

    class StubTTS:
        def __init__(self):
            self.model_id = "stub/model"

        def prepare_text(self, text, iso):
            return text

        def synthesize(self, text, lang="twi"):
            clip, _text, _src = models.reference_clip(iso, {}, token=None)
            prompts_seen.append(clip)
            t = np.arange(int(16000 * 4)) / 16000
            out = (np.sin(2 * np.pi * 300 * t) * 0.1).astype(np.float32)
            buf = tmp / "gen.wav"
            sf.write(buf, out, 16000, subtype="PCM_16")
            return buf.read_bytes()

    prompts_seen = []
    models.load_tts_model = lambda *a, **k: StubTTS()

    evaluate.synthesize_language(iso, device="cpu", samples=samples)
    out_dir = evaluate.clip_dir(iso, config.DEFAULT_CATEGORY, "org/Model-ref")
    clips = sorted(out_dir.glob("*.wav"))
    check("clips were synthesised", len(clips) == samples_expected, f"{len(clips)} clips")
    check("a manifest was written",
          (config.DATA_DIR / f"manifest_{iso}.json").exists())

    scored_refs = {s.reference_path.name for s in samples}
    check("the stub really did prompt (guard is not vacuous)",
          len(prompts_seen) == samples_expected * 2,
          f"{len(prompts_seen)} prompts for "
          f"{samples_expected} samples x 2 modes")
    check("every prompt came from the scoped path, not the fallback",
          all("/references/" in p for p in prompts_seen),
          f"prompts={sorted({p for p in prompts_seen})[:2]}")
    check("no ref-mode prompt was the clip being scored",
          not (scored_refs & {Path(p).name for p in prompts_seen}),
          f"prompts={sorted({Path(p).name for p in prompts_seen})}")
    check("every prompt is exactly the configured offset away (mod config length)",
          {Path(p).name for p in prompts_seen} ==
          {f"{(s.index + config.PROMPT_ROW_OFFSET) % n_rows:05d}.wav"
           for s in samples},
          f"prompts={sorted({Path(p).name for p in prompts_seen})}")

    print("\n== scoring ==")

    class StubScorer:
        """Deterministic stand-in for the real encoder-backed scorer."""

        def __init__(self, *a, **k):
            self.model_name = config.ENCODER_MODEL
            self.layer = config.ENCODER_LAYER

        def score(self, gen, gen_sr, ref, ref_sr):
            return round(len(gen) / max(1, gen_sr) / 100.0, 4)

        def cleanup(self):
            pass

    speechbertscore.SpeechBERTScorer = StubScorer
    speechbertscore.self_check = lambda s: True

    means = evaluate.score_sbs_language(iso, device="cpu", samples=samples)
    check("SBS produced a mean per model", len(means) == 2, str(means))

    print("\n== CER ==")
    from benchmark import asr

    class StubJudge:
        """Hears the sentence back with the last word dropped for the second model."""
        def transcribe_many(self, wavs):
            return ["one two three four" for _ in wavs]
        def cleanup(self):
            pass

    asr.load_judge = lambda iso, device="cuda": StubJudge()
    asr.judge_for = lambda iso: {"model": "stub/judge", "judge_cer": 0.1}
    cer_means = evaluate.score_cer_language(iso, device="cpu", samples=samples)
    check("CER produced a mean per model", len(cer_means) == 2, str(cer_means))
    check("CER is in [0, 1+]", all(v >= 0 for v in cer_means.values()), str(cer_means))
    check("the cer cache is tagged with its judge",
          evaluate.load_cer_cache(iso).get("judge") == "stub/judge")

    print("\n== yaml output ==")
    path = evaluate.assemble_language(iso)
    doc = yaml.safe_load(path.read_text())
    check("yaml is written", path.exists(), str(path))
    check("scoring is the composite", doc["scoring"].startswith("composite"))
    check("the encoder config is recorded",
          doc["encoder"] == {"model": config.ENCODER_MODEL,
                             "layer": config.ENCODER_LAYER, "variant": "precision"},
          str(doc["encoder"]))
    check("the judge is recorded", doc["judge"]["model"] == "stub/judge")
    check("the bulky per-sample cache is not dumped to yaml",
          "per_sample" not in doc, str(list(doc)))
    top = doc["benchmarks"][0]
    check("every clip was scored", top["num_scored"] == samples_expected)
    check("composite = (sbs + (1 - cer)) / 2",
          abs(top["composite"] - (top["sbs"] + top["accuracy"]) / 2) < 1e-3
          and abs(top["accuracy"] - (1 - top["cer"])) < 1e-3, str(top["composite"]))
    check("score is the composite and none are marked partial",
          all(b["score"] == b["composite"] and not b.get("partial")
              for b in doc["benchmarks"]))
    check("entries carry sbs, cer and the text",
          all({"text", "sbs", "cer", "gen_sec", "ref_sec"} <= set(e)
              for e in top["entries"].values()))
    check("the models are ranked by score descending",
          [b["score"] for b in doc["benchmarks"]] ==
          sorted([b["score"] for b in doc["benchmarks"]], reverse=True))
    check("ref/noref variants are kept as separate rows",
          {b["model"] for b in doc["benchmarks"]} ==
          {"org/Model-ref", "org/Model-noref"})

    print("\n== incremental re-score ==")
    calls = {"n": 0}

    class CountingScorer(StubScorer):
        def score(self, *a, **k):
            calls["n"] += 1
            return super().score(*a, **k)

    speechbertscore.SpeechBERTScorer = CountingScorer
    evaluate.score_sbs_language(iso, device="cpu", samples=samples)
    check("re-scoring reuses the cached scores",
          calls["n"] == 0, f"{calls['n']} clips re-encoded")

    # The cache must survive a process restart, or "only scores the new rows"
    # is a claim the code does not honour.
    check("a score cache file was written",
          evaluate.load_score_cache(iso) != {}, "empty cache")
    saved = evaluate.load_score_cache(iso)
    check("the cache is tagged with its encoder",
          saved.get("encoder", {}).get("model") == config.ENCODER_MODEL,
          str(saved.get("encoder")))

    # A different encoder must invalidate it rather than mix configurations.
    evaluate.save_score_cache(iso, saved.get("per_sample", {}),
                              {"model": "some/other-encoder", "layer": 0,
                               "variant": "precision"})
    # The published yaml also seeds a missing cache (same-encoder only), so
    # drop it to isolate the cache-invalidation behaviour.
    (config.BENCHMARK_DIR / f"{iso}.yaml").unlink()
    fresh = {"n": 0}

    class FreshScorer(StubScorer):
        def score(self, *a, **k):
            fresh["n"] += 1
            return super().score(*a, **k)

    speechbertscore.SpeechBERTScorer = FreshScorer
    evaluate.score_sbs_language(iso, device="cpu", samples=samples)
    check("a changed encoder forces a re-score",
          fresh["n"] == samples_expected * 2,
          f"{fresh['n']} clips re-encoded (expected {samples_expected * 2})")

    shutil.rmtree(tmp, ignore_errors=True)
    repo_now = repo_snapshot(config)
    leaked = sorted(set(repo_before) ^ set(repo_now))
    check("nothing was written outside the temp dir", not leaked, str(leaked))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
