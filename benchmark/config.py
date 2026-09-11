"""Configuration for nsanku-TTS benchmark."""

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

# Dataset: sentence-level text corpus for Ghanaian languages
GHANA_SENTENCES = "ghanaopenai/ghana-sentences"
# Default samples per language. Bumping this re-uses already-scored samples and
# only scores the *new* ones (incremental, keyed by subset row index).
NUM_SAMPLES = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", "200"))

# HuggingFace authentication
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# No model is given a reference clip to imitate. A voice prompt changes what
# is being measured — partly the model, partly whoever recorded the prompt —
# and a model that cannot read a sentence without being shown one first is not
# being asked the same question as the rest.
USE_REFERENCE_AUDIO = False

# Only benchmark models published by organizations (drop personal accounts).
# A leaderboard is a claim about what is available to build on, and a personal
# checkpoint is not the same kind of artefact as an org release.
ORG_ONLY = True

# Namespaces to treat as organizations even though HuggingFace classifies them
# as personal accounts. Mirrors the ASR benchmark's list so the two agree on
# who counts as a publisher.
ORG_OVERRIDES = {"FarmerlineML", "Qlerqly", "katrintomanek"}

# Paths
ROOT = Path(__file__).parent.parent
# Results dir overridable so Modal can persist it on a shared Volume.
BENCHMARK_DIR = Path(os.environ.get("NSANKU_TTS_RESULTS_DIR", str(ROOT / "benchmarks")))
# Every synthesised clip is kept: stage 2 reads them back to score, and
# they are the only way to actually listen to what a model produced.
# On Modal this points at the shared results Volume.
AUDIO_DIR = Path(os.environ.get("NSANKU_TTS_AUDIO_DIR", str(ROOT / "audio")))
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

# ── Text domains ─────────────────────────────────────────────────────────────
# A domain ("category", to match the ASR benchmark's vocabulary) is the kind of
# text a model is asked to read. Scores are reported per domain and averaged, so
# a model that handles one register well is not credited for another.
EVAL_CONFIGS = DATA_DIR / "eval_configs.json"
DEFAULT_CATEGORY = "education"


def language_categories(iso):
    """Return the domain configs for *iso*, or the default if unregistered."""
    import json

    try:
        with open(EVAL_CONFIGS, encoding="utf-8") as f:
            entry = json.load(f)["languages"].get(iso)
    except (OSError, json.JSONDecodeError, KeyError):
        entry = None
    if not entry:
        return [{"category": DEFAULT_CATEGORY, "source": GHANA_SENTENCES, "subset": iso}]
    return entry["categories"]
