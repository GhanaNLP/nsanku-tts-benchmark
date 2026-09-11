"""Evaluation recipe for FarmerlineML/voxcpm2-akan-sft — Asante Twi (twi_asante).

Architecture: VoxCPM2 (2B)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

CFG_VALUE is the guidance scale and INFERENCE_TIMESTEPS the number of
flow-matching steps. MAX_LEN caps generated tokens; too low truncates
long sentences.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Asante Twi is
synthesised with this model on the next benchmark run.
"""


CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 15
RETRY_BADCASE = False
MAX_LEN = None


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
