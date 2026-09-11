"""Evaluation recipe for KhayaAI/khaya-tts-v2 — Gonja (gjn).

Architecture: Khaya AI TTS v2 API
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

LANGUAGE_CODE is the Khaya API's language parameter. SPEAKER_ID
picks the voice (male_low, male_high, female); None uses the default.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Gonja is
synthesised with this model on the next benchmark run.
"""


LANGUAGE_CODE = 'gjn'
SPEAKER_ID = None


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
