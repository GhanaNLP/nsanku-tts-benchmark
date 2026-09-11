"""Evaluation recipe for ghananlpcommunity/nano-twi — Asante Twi (twi_asante).

Architecture: Matcha-TTS + Vocos ONNX
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

NOISE_SCALE and LENGTH_SCALE are the Matcha-TTS sampling knobs:
LENGTH_SCALE above 1.0 slows the speech down. SPEAKER_ID picks the voice.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Asante Twi is
synthesised with this model on the next benchmark run.
"""


NOISE_SCALE = 1.0
LENGTH_SCALE = 1.0
SPEAKER_ID = 0


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
