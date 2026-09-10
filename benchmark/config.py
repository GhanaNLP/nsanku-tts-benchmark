"""Configuration for nsanku-TTS benchmark."""

import os
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Dataset: sentence-level text corpus for Ghanaian languages
GHANA_SENTENCES = "ghanaopenai/ghana-sentences"
# Default samples per language. Bumping this re-uses already-scored samples and
# only scores the *new* ones (incremental, keyed by subset row index).
NUM_SAMPLES = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", "200"))

# Alignment model: MMS-300M CTC (1130 languages) via transformers
ALIGNMENT_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# HuggingFace authentication
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Paths
ROOT = Path(__file__).parent.parent
# Results dir overridable so Modal can persist it on a shared Volume.
BENCHMARK_DIR = Path(os.environ.get("NSANKU_TTS_RESULTS_DIR", str(ROOT / "benchmarks")))
AUDIO_DIR = ROOT / "audio"
DATA_DIR = ROOT / "data"
LANG_CONFIG = ROOT / "languages" / "ghana_languages.yaml"

# Audio settings for TTS output
SAMPLE_RATE = 24000

# Subset -> ISO 639-3 mapping for ghana-sentences
SUBSET_TO_ISO = {
    "ada": "ada",
    "dag": "dag",
    "dga": "dga",
    "ewe": "ewe",
    "fat": "fat",
    "gaa": "gaa",
    "gjn": "gjn",
    "gur": "gur",
    "nzi": "nzi",
    "twi-aku": "twi_akuapem",
    "twi-asa": "twi_asante",
    "xsm": "xsm",
}

# ISO -> human-readable language name
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

# Alignment ISO codes accepted by MMS-300M (ISO 639-3)
# Maps our internal ISO to the code the alignment model expects
ALIGNMENT_LANG_MAP = {
    "ada": "ada",
    "dag": "dag",
    "dga": "dga",
    "ewe": "ewe",
    "fat": "fat",
    "gaa": "gaa",
    "gjn": "gjn",
    "gur": "nhi",
    "nzi": "nzi",
    "twi_akuapem": "aka",
    "twi_asante": "aka",
    "xsm": "xsm",
}

# Language codes accepted by TTS models / hosted APIs (ISO 639-3).
# Khaya TTS v2 expects e.g. Asante Twi = "twi", Akuapem Twi = "atw".
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
