"""Evaluation recipe for ghananlpcommunity/F5-TTS-OpenBible-Ewe-ref — Ewe (ewe).

Architecture: F5-TTS (DiT + Vocos)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

F5-TTS reads in the voice of a reference clip, which is its recommended
inference setting. The clip is real recorded speech in this language from
ghana-speech-eval; REFERENCE_TEXT and REFERENCE_CLIP override it if a
better one exists. SPEED and NFE_STEP are the usual F5 knobs.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Ewe is
synthesised with this model on the next benchmark run.
"""


REFERENCE_TEXT = None
REFERENCE_CLIP = None
SPEED = 1.0
NFE_STEP = 32


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
