"""Evaluation recipe for ghananlpcommunity/tekyerema-tts-ewe — Ewe (ewe).

Architecture: Tekyerema Ewe VITS (0.03B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

SPEAKER picks a voice from the checkpoint's speaker file; left as None
the first speaker is used, which is what a multi-speaker VITS needs to
synthesise at all.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Ewe is
synthesised with this model on the next benchmark run.
"""


SPEAKER = None


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
