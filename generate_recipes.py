"""Generate one recipe per (model, language) evaluation.

The benchmark's unit of work is a single model reading a single language, so
that is the unit a contributor should be able to change: each pair gets its own
file under recipes/, holding the knobs that decide how that model reads that
language. Editing one language cannot disturb another.

The knobs differ by architecture — a hosted API takes a language code, a
zero-shot model takes the transcript of its reference clip, a diffusion model
takes guidance and step counts — so each architecture has its own template.

Run:  python3 generate_recipes.py            # write missing recipes
      python3 generate_recipes.py --force    # rewrite all of them
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECIPES = ROOT / "recipes"

HEADER = '''"""Evaluation recipe for {model} — {language} ({iso}).

Architecture: {architecture}
Scope: THIS MODEL, THIS LANGUAGE. Every (model, language) pair has its own
recipe file, so changing this one does not affect any other.

{blurb}

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-tts-benchmark to change how {language} is
synthesised with this model on the next benchmark run.
"""
'''

# ghana-tts was trained on text prefixed with a language tag; the codes come
# from its training script (training/ghana-tts-training/precompute_voxcpm_latents.py
# in michsethowusu/VoxCPM), where the two Twi dialects are spelled out and every
# other language is its ISO code.
GHANA_TTS_TAG_CODES = {"twi_akuapem": "twi-akuapem", "twi_asante": "twi-asante"}


def lang_tag_for(meta, iso):
    """The language tag this model expects in front of the text, if any."""
    if meta.get("lang_tag") != "ghana-tts":
        return None
    return f"<|lang:{GHANA_TTS_TAG_CODES.get(iso, iso)}|> "


HOOK = '''

# def build_wrapper(model_id, device, meta):
#     """Take over loading entirely; must return a BaseTTSModel."""
#     from benchmark.models import load_tts_model
#     return load_tts_model(model_id, device=device, meta=meta)
'''

BLURBS = {
    "coqui": "SPEAKER picks a voice from the checkpoint's speaker file; left as None\n"
             "the first speaker is used, which is what a multi-speaker VITS needs to\n"
             "synthesise at all.",
    "piper": "VOICE picks one of the 12 exposed Piper voices. The model card's own\n"
             "measurements rank them differently for pure Twi (twi-6 best) and for\n"
             "code-switched text (twi-1 best, but 21st of 30 on pure Twi), so the\n"
             "right voice depends on what this language's sentences look like.\n"
             "LENGTH_SCALE above 1.0 slows the speech down.",
    "ipa_voxcpm": "This model reads IPA, not orthography, so the sentence is phonemised\n"
                  "first. G2P_LANGUAGE is the ghana-g2p language used for that, and\n"
                  "G2P_SEPARATOR what goes between phonemes — this model was trained on\n"
                  "space-separated IPA with punctuation kept as standalone tokens, which is\n"
                  "what G2P_PUNCTUATION preserves. CFG_VALUE and INFERENCE_TIMESTEPS are\n"
                  "the usual VoxCPM guidance and step knobs.",
    "ipa": "This model reads IPA, not orthography, so the sentence is phonemised\n"
           "first. G2P_LANGUAGE is the ghana-g2p language used for that, and\n"
           "G2P_SEPARATOR what goes between phonemes.",
    "khaya": "LANGUAGE_CODE is the Khaya API's language parameter. SPEAKER_ID\n"
             "picks the voice (male_low, male_high, female); None uses the default.",
    "voxcpm": "CFG_VALUE is the guidance scale and INFERENCE_TIMESTEPS the number of\n"
              "flow-matching steps: higher is slower and usually cleaner. RETRY_BADCASE\n"
              "re-rolls a generation whose audio-to-text ratio looks wrong.",
    "voxcpm2": "CFG_VALUE is the guidance scale and INFERENCE_TIMESTEPS the number of\n"
               "flow-matching steps. MAX_LEN caps generated tokens; too low truncates\n"
               "long sentences.",
    "f5": "F5-TTS is zero-shot: it copies the voice of a reference clip. REFERENCE_TEXT\n"
          "is that clip's transcript and MUST be in this language — the clip is\n"
          "synthesised once from this text and cached. SPEED scales the speaking rate.",
    "cosyvoice": "CosyVoice2 is zero-shot: it copies the voice of a reference clip.\n"
                 "REFERENCE_TEXT is that clip's transcript and MUST be in this language —\n"
                 "the clip is synthesised once from this text and cached.",
    "nanotwi": "NOISE_SCALE and LENGTH_SCALE are the Matcha-TTS sampling knobs:\n"
               "LENGTH_SCALE above 1.0 slows the speech down. SPEAKER_ID picks the voice.",
}

BODIES = {
    "coqui": "SPEAKER = None\n",
    "piper": "VOICE = 'twi-6'\nSYNTH_LANGUAGE = 'twi'\nLENGTH_SCALE = 1.0\n"
             "NOISE_SCALE = 0.667\nNOISE_W = 0.8\n",
    "ipa_voxcpm": "G2P_LANGUAGE = {g2p_language!r}\nG2P_SEPARATOR = ' '\n"
                  "G2P_PUNCTUATION = True\n"
                  "CFG_VALUE = 2.0\nINFERENCE_TIMESTEPS = 10\nRETRY_BADCASE = True\n",
    "ipa": "G2P_LANGUAGE = {g2p_language!r}\nG2P_SEPARATOR = ' '\nG2P_PUNCTUATION = True\n",
    "khaya": "LANGUAGE_CODE = {lang_code!r}\nSPEAKER_ID = None\n",
    "voxcpm": "CFG_VALUE = 2.0\nINFERENCE_TIMESTEPS = 10\nRETRY_BADCASE = True\n",
    "voxcpm2": "CFG_VALUE = 2.0\nINFERENCE_TIMESTEPS = 15\nRETRY_BADCASE = False\nMAX_LEN = None\n",
    "f5": "REFERENCE_TEXT = {reference_text!r}\nSPEED = 1.0\nNFE_STEP = 32\n",
    "cosyvoice": "REFERENCE_TEXT = {reference_text!r}\nSPEED = 1.0\n",
    "nanotwi": "NOISE_SCALE = 1.0\nLENGTH_SCALE = 1.0\nSPEAKER_ID = 0\n",
}


def kind_for(model_id, meta):
    lower = model_id.lower()
    if meta.get("runner") == "coqui-vits":
        return "coqui"
    if meta.get("runner") == "stable-twi-tts":
        return "piper"
    if meta.get("input_type") == "ipa":
        return "ipa_voxcpm" if "voxcpm" in lower else "ipa"
    if meta.get("runner") == "cosyvoice" or "cosyvoice" in lower:
        return "cosyvoice"
    if "khaya" in lower:
        return "khaya"
    if "nano-twi" in lower:
        return "nanotwi"
    if "f5-tts" in lower:
        return "f5"
    if "voxcpm2" in lower or meta.get("architecture", "").startswith("VoxCPM2"):
        return "voxcpm2"
    return "voxcpm"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rewrite existing recipes")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from benchmark.config import ISO_TO_NAME, TTS_LANG_MAP
    from benchmark.dataset import available_subsets, subset_to_iso
    from benchmark.evaluate import load_tts_models

    RECIPES.mkdir(exist_ok=True)
    written = skipped = 0
    for subset in available_subsets():
        iso = subset_to_iso(subset)
        language = ISO_TO_NAME.get(iso, iso)
        for meta in load_tts_models(subset):
            model_id = meta["name"]
            kind = kind_for(model_id, meta)
            safe = model_id.replace("/", "_").replace(":", "_")
            path = RECIPES / f"{safe}__{iso}.py"
            if path.exists() and not args.force:
                skipped += 1
                continue
            tag = lang_tag_for(meta, iso)
            body = BODIES[kind].format(
                lang_code=TTS_LANG_MAP.get(iso, iso),
                reference_text=meta.get("reference_text", ""),
                g2p_language=language,
            )
            if tag:
                body = f"LANG_TAG = {tag!r}\n" + body
            path.write_text(
                HEADER.format(
                    model=model_id,
                    language=language,
                    iso=iso,
                    architecture=meta.get("architecture", "unknown"),
                    blurb=BLURBS[kind],
                )
                + "\n\n"
                + body
                + HOOK,
                encoding="utf-8",
            )
            written += 1
    print(f"{written} recipe(s) written, {skipped} left alone")


if __name__ == "__main__":
    main()
