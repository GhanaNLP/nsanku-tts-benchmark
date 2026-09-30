"""Verify benchmark 2's scorer reproduces the validated pilot numbers.

The layer sweep picked wavlm-large layer 6 with content sensitivity ~0.317
and speed penalty ~0.014. If this module cannot reproduce that, something
between the pilot and the package has changed.
"""

import numpy as np

from benchmark.config import SPEECH_EVAL_CONFIGS
from benchmark.dataset import load_samples
from benchmark.speechbertscore import SpeechBERTScorer, self_check, speechbertscore


def resample_roundtrip(a, factor):
    n_out = max(8, int(len(a) / factor))
    x_old = np.linspace(0, 1, len(a), dtype=np.float32)
    x_new = np.linspace(0, 1, n_out, dtype=np.float32)
    stretched = np.interp(x_new, x_old, a).astype(np.float32)
    x_b = np.linspace(0, 1, len(stretched), dtype=np.float32)
    x_a = np.linspace(0, 1, len(a), dtype=np.float32)
    return np.interp(x_a, x_b, stretched).astype(np.float32)


def main():
    scorer = SpeechBERTScorer(device="cuda")
    print(f"  {scorer.model_name} layer {scorer.layer}/{scorer.n_layers}")

    print("\n== determinism guard ==")
    ok = self_check(scorer)
    print(f"  self_check: {ok}")

    import soundfile as sf

    # The same six languages the encoder sweep used. ISO codes, not subset
    # names: dagaare/gonja/kasem are ghana-sentences subsets, and the codes here
    # are dga/gjn/xsm. Silently dropping unknown codes would quietly shrink
    # coverage, so an unexpected code is an error, not a filter.
    langs = ["twi_asante", "ewe", "dga", "gjn", "fat", "xsm"]
    unknown = [l for l in langs if l not in SPEECH_EVAL_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown language codes: {unknown}")

    content, speed = [], []
    for iso in langs:
        samples = load_samples(iso, limit=8)
        wavs = []
        for s in samples:
            a, sr = sf.read(s.reference_path, dtype="float32", always_2d=False)
            wavs.append((a, sr))
        for i, (w, sr) in enumerate(wavs):
            z_ref = scorer.features(w, sr)
            ow, osr = wavs[(i + 1) % len(wavs)]
            content.append(speechbertscore(scorer.features(ow, osr), z_ref))
            speed.append(speechbertscore(
                scorer.features(resample_roundtrip(w, 0.85), sr), z_ref))
        print(f"  {iso}: {len(samples)} samples, refs extracted")

    cs = 1 - float(np.mean(content))
    sp = 1 - float(np.mean(speed))
    print("\n== reproduces pilot? ==")
    print(f"  content sensitivity: {cs:.4f}   (pilot wavlm-large L6: 0.3171)")
    print(f"  speed penalty:       {sp:.4f}   (pilot wavlm-large L6: 0.0138)")

    # Also verify a mismatched pair cannot score near 1.0
    print(f"  self-similarity:     {speechbertscore(scorer.features(wavs[0][0], wavs[0][1]), scorer.features(wavs[0][0], wavs[0][1])):.6f}")

    delta = abs(cs - 0.3171)
    print(f"\n  delta from pilot: {delta:.4f}")
    print("  MATCH" if delta < 0.05 else "  MISMATCH -- investigate before building on this")


if __name__ == "__main__":
    main()
