"""Evaluation recipe for ghananlpcommunity/kasanoma-twi — Akuapem Twi (twi_akuapem).

Architecture: Kasanoma Piper VITS (0.06B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

VOICE picks one of the 12 exposed Piper voices. The model card's own
measurements rank them differently for pure Twi (twi-6 best) and for
code-switched text (twi-1 best, but 21st of 30 on pure Twi), so the
right voice depends on what this language's sentences look like.
LENGTH_SCALE above 1.0 slows the speech down.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Akuapem Twi is
synthesised with this model on the next benchmark run.
"""


VOICE = 'twi-6'
SYNTH_LANGUAGE = 'twi'
LENGTH_SCALE = 1.0
NOISE_SCALE = 0.667
NOISE_W = 0.8


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
