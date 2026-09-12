"""Evaluation recipe for k2-fsa/OmniVoice-ref — Gurene (gur).

Architecture: OmniVoice (voice cloning)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

OmniVoice clones the voice of a reference clip: real recorded speech
in this language from ghana-speech-eval. REFERENCE_TEXT and
REFERENCE_CLIP override it if a better one exists.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Gurene is
synthesised with this model on the next benchmark run.
"""


REFERENCE_TEXT = None
REFERENCE_CLIP = None


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
