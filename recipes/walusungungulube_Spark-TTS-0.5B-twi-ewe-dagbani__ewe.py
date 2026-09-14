"""Evaluation recipe for walusungungulube/Spark-TTS-0.5B-twi-ewe-dagbani — Ewe (ewe).

Architecture: Spark-TTS (0.5B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

GENDER, PITCH, and SPEED control Spark-TTS's controllable voice attributes.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Ewe is
synthesised with this model on the next benchmark run.
"""


GENDER = 'female'
PITCH = 'moderate'
SPEED = 'moderate'


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
