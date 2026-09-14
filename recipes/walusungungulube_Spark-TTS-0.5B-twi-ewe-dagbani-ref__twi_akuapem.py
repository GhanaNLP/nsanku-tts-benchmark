"""Evaluation recipe for walusungungulube/Spark-TTS-0.5B-twi-ewe-dagbani-ref — Akuapem Twi (twi_akuapem).

Architecture: Spark-TTS (0.5B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

Spark-TTS voice cloning mode using a reference clip for voice prompting.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Akuapem Twi is
synthesised with this model on the next benchmark run.
"""


REFERENCE_TEXT = None
REFERENCE_CLIP = None
GENDER = 'female'
PITCH = 'moderate'
SPEED = 'moderate'


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
