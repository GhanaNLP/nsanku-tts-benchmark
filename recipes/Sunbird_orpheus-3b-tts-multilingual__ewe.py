"""Evaluation recipe for Sunbird/orpheus-3b-tts-multilingual — Ewe (ewe).

Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

Orpheus-3B has no language-conditioning input: the voice it reads in — and
therefore the language — is chosen by prepending a ``speaker_id`` tag to the
prompt.  The Ewe-trained voice is ``slr129_ewe_0001`` from the Sunbird/tts
corpus (SLR129, 1 speaker).

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Ewe is
synthesised with this model on the next benchmark run.
"""


SPEAKER_ID = "slr129_ewe_0001"


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)