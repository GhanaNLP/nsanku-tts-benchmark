"""Configuration for the nsanku-TTS benchmark.

Every sample is drawn from ghana-speech-eval, which is *recorded human speech
with a transcript*. So each sample ships with a real recording of its own
sentence, and two metrics can grade the same synthesised clip:

  * CER, from an ASR judge transcribing the clip (intelligibility);
  * SpeechBERTScore (Saeki et al., 2024), generated speech against the real
    reference utterance (acoustic similarity, speaker, prosody).

The published score is the composite (SBS + (1 - CER)) / 2.
"""

import os
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Accept KHAYA_API / GEMINI_API as aliases for KHAYA_API_KEY / GEMINI_API_KEY
if "KHAYA_API" in os.environ and "KHAYA_API_KEY" not in os.environ:
    os.environ["KHAYA_API_KEY"] = os.environ["KHAYA_API"]
if "GEMINI_API" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_API"]

# ── Sample source ────────────────────────────────────────────────────────────
# Real recorded speech, with transcripts, in the languages being benchmarked.
SPEECH_EVAL = "ghananlpcommunity/ghana-speech-eval"

# Default samples per language. Bumping this re-uses already-scored samples
# and only scores the new ones (incremental, keyed by config row index).
NUM_SAMPLES = int(os.environ.get("NSANKU2_TTS_NUM_SAMPLES", "200"))

# HuggingFace authentication
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Sample filtering ─────────────────────────────────────────────────────────
# ghana-speech-eval holds read-aloud scripture, so the rows are already
# speakable sentences. These bounds only drop rows that make a reference
# recording unreliable to score against: too short to be a stable utterance,
# or too long to encode in one pass.
MIN_SECONDS = float(os.environ.get("NSANKU2_TTS_MIN_SECONDS", "3.0"))
MAX_SECONDS = float(os.environ.get("NSANKU2_TTS_MAX_SECONDS", "15.0"))
MIN_WORDS = 5
MAX_WORDS = 25

# ── Models whose recommended inference setting includes a reference clip ─────
# These are prompted with real recorded speech, as always. Crucially the
# prompt clip is a *different* row from the one being scored: a ref-mode model
# prompted with its own scoring reference would be graded on how well it copies
# the clip it was handed, not on synthesis quality. See PROMPT_ROW_OFFSET.
USE_REFERENCE_AUDIO = True
# The prompt clip for a sample is drawn from this far away in the config, so it
# is never the sample's own reference recording. The offset is fixed (not
# random) so a re-run prompts identically and stays comparable.
PROMPT_ROW_OFFSET = 500

# Only benchmark models published by organizations (drop personal accounts).
ORG_ONLY = True
ORG_OVERRIDES = {"FarmerlineML", "Qlerqly", "katrintomanek"}

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = Path(os.environ.get("NSANKU2_TTS_RESULTS_DIR", str(ROOT / "benchmarks")))
TRANSCRIPTIONS_DIR = Path(
    os.environ.get("NSANKU2_TTS_TRANSCRIPTIONS_DIR", str(ROOT / "transcriptions"))
)
# Every synthesised clip is kept: scoring reads them back, and they are the only
# way to actually listen to what a model produced.
AUDIO_DIR = Path(os.environ.get("NSANKU2_TTS_AUDIO_DIR", str(ROOT / "audio")))
# Reference recordings are extracted once from the parquet configs and cached
# here, so scoring does not re-download 15 GB per run.
REFERENCE_DIR = Path(
    os.environ.get("NSANKU2_TTS_REFERENCE_DIR", str(ROOT / "references"))
)
DATA_DIR = ROOT / "data"

# ── Audio settings ───────────────────────────────────────────────────────────
# SpeechBERTScore's features are computed at 16 kHz, the encoder's native rate
# and the rate the paper used. Synthesised clips are resampled from whatever
# their model emits, so a 48 kHz output is scored fairly.
SCORE_SAMPLE_RATE = 16000

# ── SpeechBERTScore configuration ────────────────────────────────────────────
# Encoder and layer were chosen by measured sweep, not assumption. See
# docs/encoder-selection.md.
#
# On 6 languages x 20 real samples, wavlm-large layer 6 gave the largest
# separation between a correct utterance and a mismatched one (0.317) while
# barely penalising speaking-rate changes (0.014). The multilingual
# omniASR-W2V-1B was close on content (0.292) but stronger on noise (0.330).
ENCODER_MODEL = os.environ.get("NSANKU2_TTS_ENCODER", "microsoft/wavlm-large")
ENCODER_LAYER = int(os.environ.get("NSANKU2_TTS_ENCODER_LAYER", "6"))
SBS_VARIANT = "precision"

# ── Languages ────────────────────────────────────────────────────────────────
# iso -> ghana-speech-eval config name. Each config holds 1000 rows of real
# recorded speech in that language. Verified present and non-empty upstream.
SPEECH_EVAL_CONFIGS = {
    "ada": "bible_Dangme_ada",
    "dag": "bible_Dagbani_dag",
    "dga": "bible_Dagaare_dga",
    "ewe": "bible_Ewe_ewe",
    "fat": "bible_Fante_fat",
    "gaa": "jw_ga_gaa",
    "gjn": "bible_Gonja_gjn",
    "gur": "bible_Ninkare_gur",
    "nzi": "bible_Nzema_nzi",
    "twi_akuapem": "bible_Akuapem_Twi",
    "twi_asante": "bible_Asante_Twi",
    "xsm": "bible_Kasem_xsm",
}

ISO_TO_NAME = {
    "ada": "Dangme",
    "dag": "Dagbani",
    "dga": "Dagaare",
    "ewe": "Ewe",
    "fat": "Fante",
    "gaa": "Ga",
    "gjn": "Gonja",
    "gur": "Gurene",
    "nzi": "Nzema",
    "twi_akuapem": "Akuapem Twi",
    "twi_asante": "Asante Twi",
    "xsm": "Kasem",
}

# Language codes accepted by TTS models / hosted APIs (ISO 639-3).
TTS_LANG_MAP = {
    "ada": "ada",
    "dag": "dag",
    "dga": "dga",
    "ewe": "ewe",
    "fat": "fat",
    "gaa": "gaa",
    "gjn": "gjn",
    "gur": "gur",
    "nzi": "nzi",
    "twi_akuapem": "atw",
    "twi_asante": "twi",
    "xsm": "xsm",
}

# The one domain: ghana-speech-eval is read-aloud scripture, so every sample is
# the same register. A single category also keeps scores comparable across
# languages, which a multi-register split would not.
DEFAULT_CATEGORY = "read_aloud"


def all_isos():
    return list(SPEECH_EVAL_CONFIGS)
