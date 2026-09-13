"""Evaluation recipe for k2-fsa/OmniVoice-noref — Dagbani (dag).

Architecture: OmniVoice (voice cloning + voice design)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

OmniVoice auto mode: no reference clip, no voice design — the
model picks both the voice and the language itself.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Dagbani is
synthesised with this model on the next benchmark run.
"""


# no knobs — pure auto (model picks voice and language)


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
