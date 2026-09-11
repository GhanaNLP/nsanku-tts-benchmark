"""Evaluation recipe for ghananlpcommunity/VoxCPM2-Ghana — Nzema (nzi).

Architecture: VoxCPM2 (5B, IPA)
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

This model reads IPA, not orthography, so the sentence is phonemised
first. G2P_LANGUAGE is the ghana-g2p language used for that, and
G2P_SEPARATOR what goes between phonemes — this model was trained on
space-separated IPA. CFG_VALUE and INFERENCE_TIMESTEPS are the usual
VoxCPM guidance and step knobs.

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how Nzema is
synthesised with this model on the next benchmark run.
"""


G2P_LANGUAGE = 'Nzema'
G2P_SEPARATOR = ' '
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10
RETRY_BADCASE = True


# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
