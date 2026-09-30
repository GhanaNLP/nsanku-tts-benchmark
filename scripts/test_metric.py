"""Offline tests for the SpeechBERTScore module and prompt disjointness.

These need no GPU and no network: the encoder is a randomly-initialised tiny
WavLM, so what is under test is our code (feature extraction, layer indexing,
the Eq. 6 computation, resampling) rather than the pretrained weights.
"""

import sys

import numpy as np
import torch

# The local test box has torch 2.4, while the installed torchao (imported by
# transformers' quantizer registry) needs torch >= 2.6 and cannot even be
# imported. We neither quantise nor load pretrained weights here, so we hide
# torchao: assigning None in sys.modules is the documented way to block a
# module, and it makes transformers' availability check report False. The H200
# runs torch 2.8 and needs none of this.
sys.modules.setdefault("torchao", None)

from benchmark.speechbertscore import (
    SpeechBERTScorer,
    speechbertscore,
    to_mono_16k,
    validate_layer,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f'  {detail}' if detail else ''}")


def tiny_scorer(layer=2, device="cpu"):
    """A WavLM with random weights and 4 layers, built without any download."""
    from transformers import WavLMConfig, WavLMModel

    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    model = WavLMModel(cfg).to(device).eval()
    return _wrap(model, layer, device)


def _wrap(model, layer, device):
    """Build a scorer around an already-constructed model, skipping __init__."""
    scorer = SpeechBERTScorer.__new__(SpeechBERTScorer)
    scorer.model_name = "tiny-wavlm-test"
    scorer.layer = layer
    scorer.device = torch.device(device)
    scorer.dtype = torch.float32
    scorer.model = model
    scorer.n_layers = model.config.num_hidden_layers
    return scorer


# ── the metric itself ────────────────────────────────────────────────────────


def test_metric_math():
    print("\n== Eq. 6 on known features ==")
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    check("identical features score 1.0", abs(speechbertscore(a, b) - 1.0) < 1e-6,
          f"got {speechbertscore(a, b):.6f}")

    c = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    # each generated frame matches the reference frame it equals -> 1.0
    check("permutation-invariant matching", abs(speechbertscore(c, b) - 1.0) < 1e-6,
          f"got {speechbertscore(c, b):.6f}")

    d = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    e = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    check("orthogonal features score 0.0", abs(speechbertscore(d, e)) < 1e-6,
          f"got {speechbertscore(d, e):.6f}")

    # precision: one good frame among many bad ones must not be rescued
    f = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    g = torch.tensor([[1.0, 0.0]])
    check("precision penalises a lone matching frame",
          abs(speechbertscore(f, g) - 0.25) < 1e-6,
          f"got {speechbertscore(f, g):.6f} (expected 0.25)")


def test_chunking():
    print("\n== chunked == single-shot ==")
    torch.manual_seed(1)
    a = torch.nn.functional.normalize(torch.randn(1000, 64), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(733, 64), dim=-1)
    one = speechbertscore(a, b, chunk=4096)
    many = speechbertscore(a, b, chunk=7)
    # Chunked accumulation happens in float32, so a different chunk size sums
    # in a different order and may differ in the last bits. Only the order
    # changes, not the value, so the tolerance is float32 epsilon, not exact
    # equality.
    check("chunk size does not change the score", abs(one - many) < 1e-6,
          f"4096:{one:.10f} 7:{many:.10f} (delta {abs(one - many):.2e})")


def test_empty_inputs():
    print("\n== degenerate inputs ==")
    a = torch.randn(4, 8)
    check("empty generated returns None",
          speechbertscore(torch.zeros(0, 8), a) is None)
    check("empty reference returns None", speechbertscore(a, torch.zeros(0, 8)) is None)


# ── resampling ───────────────────────────────────────────────────────────────


def test_resample():
    print("\n== resampling to 16 kHz ==")
    tone48 = np.sin(2 * np.pi * 300 * np.arange(48000) / 48000).astype(np.float32)
    out = to_mono_16k(tone48, 48000)
    check("length scales 48k -> 16k", abs(len(out) - 16000) <= 1, f"got {len(out)}")
    check("dtype is float32", out.dtype == np.float32, str(out.dtype))

    already = np.zeros(8000, dtype=np.float32)
    check("16 kHz input passes through unchanged",
          to_mono_16k(already, 16000).shape == already.shape)

    stereo = np.stack([np.ones(1000), np.zeros(1000)], axis=-1)
    check("stereo is mixed to mono", to_mono_16k(stereo, 16000).shape == (1000,))

    check("sub-1-sample input yields empty, not a crash",
          len(to_mono_16k(np.zeros(1, dtype=np.float32), 48000)) == 0)


# ── encoder plumbing ─────────────────────────────────────────────────────────


def test_encoder_layer_indexing():
    print("\n== encoder plumbing (random-weight tiny WavLM) ==")
    check("rejects a layer past the last block",
          _raises(lambda: validate_layer(24, 24, "wavlm"), ValueError))
    check("rejects a negative layer",
          _raises(lambda: validate_layer(24, -1, "wavlm"), ValueError))
    check("accepts the last valid layer", validate_layer(24, 23) == 23)

    scorer = tiny_scorer(layer=2)
    wav = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000).astype(np.float32)
    z = scorer.features(wav, 16000)
    check("features are 2-D frames x dim", z is not None and z.dim() == 2, str(tuple(z.shape)))
    check("features are L2-normalised",
          abs(z.norm(dim=-1).mean().item() - 1.0) < 1e-4,
          f"mean norm {z.norm(dim=-1).mean().item():.6f}")

    again = scorer.features(wav, 16000)
    check("repeated encode is bit-identical", torch.equal(z, again))

    self_sim = speechbertscore(z, z)
    check("self-similarity is exactly 1.0", abs(self_sim - 1.0) < 1e-4, f"got {self_sim:.6f}")

    higher = tiny_scorer(layer=3).features(wav, 16000)
    check("a different layer gives different features", not torch.equal(z, higher))

    check("empty audio yields None",
          scorer.features(np.zeros(0, dtype=np.float32), 16000) is None)


def test_score_wrapper():
    print("\n== scorer.score() ==")
    scorer = tiny_scorer()
    a = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000).astype(np.float32)
    b = np.sin(2 * np.pi * 210 * np.arange(16000) / 16000).astype(np.float32)
    v = scorer.score(a, 16000, a, 16000)
    check("identical clip scores ~1.0", v is not None and abs(v - 1.0) < 1e-4, f"got {v}")
    v2 = scorer.score(a, 16000, b, 16000)
    check("a different clip scores below 1.0", v2 is not None and v2 < 1.0, f"got {v2}")
    # scoring must not depend on which sample rate a model happened to emit
    a24 = np.sin(2 * np.pi * 200 * np.arange(24000) / 24000).astype(np.float32)
    v3 = scorer.score(a24, 24000, a, 16000)
    check("48 kHz output scores the same as 16 kHz", abs(v3 - v) < 0.05,
          f"16k:{v:.6f} 24k:{v3:.6f}")


# ── prompt disjointness ──────────────────────────────────────────────────────


def test_prompt_disjointness():
    print("\n== ref-mode prompt is never the scoring reference ==")
    from benchmark import dataset, models

    asked = []

    def k(n):
        return f"bible_{n:05d}"

    class FakePrompt:
        def __init__(self, index):
            self.index = index
            self.text = "a different sentence entirely"
            self.reference_path = f"/tmp/fake_{index}.wav"

    real_loader = dataset.load_prompt_sample

    def fake_loader(iso, key):
        asked.append((iso, key))
        return FakePrompt(k(dataset.split_key(key)[1] + 500))

    dataset.load_prompt_sample = fake_loader
    try:
        # A recipe override wins over the row-derived prompt, so a model that
        # needs one specific voice can still be prompted that way.
        from types import SimpleNamespace

        recipe = SimpleNamespace(REFERENCE_CLIP="/x.wav", REFERENCE_TEXT="hi")
        with models.prompt_scope("twi_asante", k(5)):
            clip, text, source = models.reference_clip(
                "twi_asante", {"recipe": recipe}, token=None)
        check("a recipe override wins", clip == "/x.wav" and text == "hi",
              f"{clip} / {text}")
        check("the override did not consult the dataset", asked == [], str(asked))

        # ...unless it pins the very row being scored, which would grade
        # copying rather than synthesis.
        collide = SimpleNamespace(
            REFERENCE_CLIP=str(dataset.reference_path_for("twi_asante", k(5))),
            REFERENCE_TEXT="hi",
        )
        with models.prompt_scope("twi_asante", k(5)):
            clip, text, source = models.reference_clip(
                "twi_asante", {"recipe": collide}, token=None)
        check("an override colliding with the scored row is refused",
              clip != str(dataset.reference_path_for("twi_asante", k(5))),
              f"clip={clip}")
        check("the collision fell back to an offset row",
              asked[-1] == ("twi_asante", k(5)), f"asked for {asked[-1]}")

        # The shipped recipes all set both knobs to None: that must mean
        # "no override", not "override with None".
        none_recipe = SimpleNamespace(REFERENCE_CLIP=None, REFERENCE_TEXT=None)
        with models.prompt_scope("twi_asante", k(7)):
            clip, text, source = models.reference_clip(
                "twi_asante", {"recipe": none_recipe}, token=None)
        check("a None override uses the row-derived prompt",
              clip == "/tmp/fake_bible_00507.wav", f"clip={clip}")

        for row in (0, 17, 431):
            with models.prompt_scope("twi_asante", k(row)):
                clip, text, source = models.reference_clip(
                    "twi_asante", {}, token=None)
            check(f"row {row}: the prompt row is not the scored row",
                  asked[-1] == ("twi_asante", k(row)), f"asked for {asked[-1]}")
            # Compare whole paths: substring-matching row 0 against
            # "fake_00500.wav" would spuriously pass.
            check(f"row {row}: the prompt is exactly {row + 500} rows away",
                  clip == f"/tmp/fake_{k(row + 500)}.wav", f"clip={clip}")
            check(f"row {row}: the prompt is not the scoring reference",
                  clip != str(dataset.reference_path_for("twi_asante", k(row))),
                  f"clip={clip}")
            check(f"row {row}: source records the row it used",
                  k(row) in source, f"source={source}")

        check("the scope is torn down afterwards",
              models._PROMPT_SCOPE is None, str(models._PROMPT_SCOPE))

        with models.prompt_scope("ewe", 3):
            with models.prompt_scope("dag", 9):
                check("a nested scope overrides", models._PROMPT_SCOPE == ("dag", 9),
                      str(models._PROMPT_SCOPE))
            check("nesting restores the outer scope",
                  models._PROMPT_SCOPE == ("ewe", 3), str(models._PROMPT_SCOPE))
    finally:
        dataset.load_prompt_sample = real_loader


def test_unscoped_fallback():
    print("\n== unscoped reference_clip falls back to the published clip ==")
    import json
    import tempfile
    from pathlib import Path

    from benchmark import dataset, models
    import huggingface_hub

    def boom(iso, key):
        raise AssertionError("an unscoped call must not consult the dataset")

    tmp = Path(tempfile.mkdtemp())
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({
        "twi_asante": {"audio": "twi_asante/ref.wav", "text": "published",
                       "source": "test"},
    }))
    clip_file = tmp / "ref.wav"
    clip_file.write_bytes(b"RIFF")

    def fake_dl(repo, filename, **kw):
        return str(manifest) if filename.endswith("manifest.json") else str(clip_file)

    real_loader = dataset.load_prompt_sample
    real_dl = huggingface_hub.hf_hub_download
    dataset.load_prompt_sample = boom
    huggingface_hub.hf_hub_download = fake_dl
    try:
        clip, text, source = models.reference_clip("twi_asante", {}, token=None)
        check("without a scope it returns the published clip",
              clip == str(clip_file), f"clip={clip}")
        check("the published transcript comes back too", text == "published", text)
    except Exception as e:
        check("without a scope it returns the published clip", False,
              f"raised {type(e).__name__}: {e}")
    finally:
        dataset.load_prompt_sample = real_loader
        huggingface_hub.hf_hub_download = real_dl


def _raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False


if __name__ == "__main__":
    test_metric_math()
    test_chunking()
    test_empty_inputs()
    test_resample()
    test_encoder_layer_indexing()
    test_score_wrapper()
    test_prompt_disjointness()
    test_unscoped_fallback()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
        raise SystemExit(1)
