"""Evaluation recipe for ghananlpcommunity/F5-TTS-OpenBible-Ewe — Ewe (ewe).

Architecture: F5-TTS (DiT + Vocos)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

F5-TTS is zero-shot: it copies the voice of a reference clip. REFERENCE_TEXT
is that clip's transcript and MUST be in this language — the clip is
synthesised once from this text and cached. SPEED scales the speaking rate.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Ewe is
synthesised with this model on the next benchmark run.
"""


REFERENCE_TEXT = 'Mawu lɔ̃ ame sia ame eye wòkpɔa wo dzi ɣesiaɣi.'
SPEED = 1.0
NFE_STEP = 32


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
