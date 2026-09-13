"""Evaluation recipe for k2-fsa/OmniVoice-design — Fante (fat).

Architecture: OmniVoice (voice cloning + voice design)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

OmniVoice reads this language in a voice described by the
VOICE_DESIGN instruct (gender, age, pitch, style, accent, comma
separated). Language is auto-detected; leave VOICE_DESIGN = None
and OmniVoice picks an automatic voice for everything.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Fante is
synthesised with this model on the next benchmark run.
"""


VOICE_DESIGN = None


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
