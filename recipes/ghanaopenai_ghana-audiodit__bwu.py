"""Evaluation recipe for ghanaopenai/ghana-audiodit — Buli (bwu).

Architecture: LongCat-AudioDiT 1.4B (flow-matching DiT)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

AudioDiT flow-matching text-to-speech for Ghanaian languages.
STEPS is diffusion steps (16 default); CFG_STRENGTH is guidance (4.0 default);
SEED selects the voice identity.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Buli is
synthesised with this model on the next benchmark run.
"""


STEPS = 16
CFG_STRENGTH = 4.0
SEED = 42
SPEED = 1.0


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
