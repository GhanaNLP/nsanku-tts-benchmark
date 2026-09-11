"""Evaluation recipe for ghananlpcommunity/ghana-tts-36k — Kasem (xsm).

Architecture: VoxCPM v1 (0.7B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

CFG_VALUE is the guidance scale and INFERENCE_TIMESTEPS the number of
flow-matching steps: higher is slower and usually cleaner. RETRY_BADCASE
re-rolls a generation whose audio-to-text ratio looks wrong.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Kasem is
synthesised with this model on the next benchmark run.
"""


LANG_TAG = '<|lang:xsm|> '
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10
RETRY_BADCASE = True


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
