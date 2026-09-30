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
NUM_SAMPLES = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", "200"))

# HuggingFace authentication
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Sample filtering ─────────────────────────────────────────────────────────
# Deliberately wide: the text sources differ a lot (finance rows are a few words
# long, waxal and unicef rows run 15-45 s), and the bounds must not silently
# drop a whole source. They only exclude rows that cannot be scored at all:
# empty or one-word text, clips too short for the encoder to see a stable
# utterance, and clips beyond the ASR judge's input limit (MAX_JUDGE_SECONDS).
MIN_SECONDS = float(os.environ.get("NSANKU_TTS_MIN_SECONDS", "1.5"))
MAX_SECONDS = float(os.environ.get("NSANKU_TTS_MAX_SECONDS", "40.0"))
MIN_WORDS = 2
MAX_WORDS = 150

# Omnilingual ASR raises on audio past 40 s. A *generated* clip can be longer
# than its reference, so clips over this are recorded as a judge failure (no
# CER) rather than scored as a total miss.
MAX_JUDGE_SECONDS = 39.0

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
BENCHMARK_DIR = Path(os.environ.get("NSANKU_TTS_RESULTS_DIR", str(ROOT / "benchmarks")))
TRANSCRIPTIONS_DIR = Path(
    os.environ.get("NSANKU_TTS_TRANSCRIPTIONS_DIR", str(ROOT / "transcriptions"))
)
# Every synthesised clip is kept: scoring reads them back, and they are the only
# way to actually listen to what a model produced.
AUDIO_DIR = Path(os.environ.get("NSANKU_TTS_AUDIO_DIR", str(ROOT / "audio")))
# Reference recordings are extracted once from the parquet configs and cached
# here, so scoring does not re-download 15 GB per run.
REFERENCE_DIR = Path(
    os.environ.get("NSANKU_TTS_REFERENCE_DIR", str(ROOT / "references"))
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
ENCODER_MODEL = os.environ.get("NSANKU_TTS_ENCODER", "microsoft/wavlm-large")
ENCODER_LAYER = int(os.environ.get("NSANKU_TTS_ENCODER_LAYER", "6"))
SBS_VARIANT = "precision"

# ── Languages ────────────────────────────────────────────────────────────────
# iso -> {text source: ghana-speech-eval config}. The source is the config's
# prefix (bible, jw, finance, ...), i.e. the kind of text being read. A language
# with several sources is sampled across all of them (see dataset.allocate), so
# a model is not scored on one register only. Each config holds up to 1000 rows
# of real recorded speech. Only languages that also have an ASR judge
# (data/asr_judges.json) and at least one TTS model are listed.
SPEECH_EVAL_SOURCES = {
    # Original 12
    "ada": {"bible": "bible_Dangme_ada", "jw": "jw_dangme_ada"},
    "dag": {"bible": "bible_Dagbani_dag", "unicef": "unicef_dagbani",
            "waxal": "waxal_Dagbani_dag"},
    "dga": {"bible": "bible_Dagaare_dga", "jw": "jw_dagaare_dga",
            "waxal": "waxal_Dagaare_dga"},
    "ewe": {"bible": "bible_Ewe_ewe", "jw": "jw_ewe_ewe", "unicef": "unicef_ewe",
            "waxal": "waxal_Ewe_ewe"},
    "fat": {"bible": "bible_Fante_fat", "finance": "finance_fante",
            "jw": "jw_fante_fat", "lds": "lds_Fante_fat"},
    "gaa": {"jw": "jw_ga_gaa", "finance": "finance_ga"},
    "gjn": {"bible": "bible_Gonja_gjn"},
    "gur": {"bible": "bible_Ninkare_gur", "jw": "jw_frafra_gur"},
    "nzi": {"bible": "bible_Nzema_nzi", "jw": "jw_nzema_nzi"},
    "twi_akuapem": {"bible": "bible_Akuapem_Twi", "finance": "finance_Akuapem_Twi"},
    "twi_asante": {"bible": "bible_Asante_Twi", "finance": "finance_Asante_Twi",
                   "lds": "lds_Asante_Twi", "unicef": "unicef_Asante_Twi",
                   "waxal": "waxal_Asante_Twi"},
    "xsm": {"bible": "bible_Kasem_xsm"},
    # Added with the move to ghana-speech-eval
    "acd": {"bible": "bible_Gikyode_acd"},
    "aha": {"jw": "jw_ahanta_aha"},
    "akp": {"bible": "bible_Siwu_akp"},
    "any": {"bible": "bible_Anyin_any"},
    "avn": {"bible": "bible_Avatime_avn"},
    "bib": {"bible": "bible_Bissa_bib"},
    "bim": {"bible": "bible_Bimoba_bim"},
    "biv": {"bible": "bible_Birifor_Southern_biv"},
    "bov": {"bible": "bible_Tuwuli_bov"},
    "bud": {"bible": "bible_Bassar_Ntcham_bud"},
    "bwu": {"bible": "bible_Buli_bwu"},
    "ffm": {"bible": "bible_Fulfulde_Maasina_ffm"},
    "hau": {"bible": "bible_Hausa_hau"},
    "kbp": {"bible": "bible_Kabiye_kbp"},
    "kdh": {"bible": "bible_Tem_kdh"},
    "kma": {"bible": "bible_Konni_kma"},
    "kpo": {"waxal": "waxal_Ikposo_kpo"},
    "kus": {"bible": "bible_Kusaal_kus"},
    "lef": {"bible": "bible_Lelemi_lef"},
    "lip": {"bible": "bible_Sekpele_lip"},
    "maw": {"bible": "bible_Mampruli_maw"},
    "mzw": {"bible": "bible_Deg_mzw"},
    "naw": {"bible": "bible_Nawuri_naw"},
    "ncu": {"bible": "bible_Chumburung_ncu"},
    "nko": {"bible": "bible_Nkonya_nko"},
    "ntr": {"bible": "bible_Ntrubo_ntr"},
    "sfw": {"bible": "bible_Sehwi_sfw", "jw": "jw_sehwi_sfw"},
    "sig": {"bible": "bible_Paasaal_sig"},
    "sil": {"bible": "bible_Sisaala_Tumulung_sil"},
    "snw": {"bible": "bible_Selee_snw"},
    "tpm": {"bible": "bible_Tampulma_tpm"},
    "vag": {"bible": "bible_Vagla_vag"},
    "xon": {"bible": "bible_Konkomba_xon"},
}

# Before the move to multiple sources each language used one config, and
# samples were keyed by bare row index. scripts/migrate_keys.py renames those
# results to "<source>_<row>" using this table.
LEGACY_SOURCE = {
    "ada": "bible", "dag": "bible", "dga": "bible", "ewe": "bible", "fat": "bible",
    "gaa": "jw", "gjn": "bible", "gur": "bible", "nzi": "bible",
    "twi_akuapem": "bible", "twi_asante": "bible", "xsm": "bible",
}

ISO_TO_NAME = {
    "ada": "Dangme", "dag": "Dagbani", "dga": "Dagaare", "ewe": "Ewe",
    "fat": "Fante", "gaa": "Ga", "gjn": "Gonja", "gur": "Gurene",
    "nzi": "Nzema", "twi_akuapem": "Akuapem Twi", "twi_asante": "Asante Twi",
    "xsm": "Kasem",
    "acd": "Gikyode", "aha": "Ahanta", "akp": "Siwu", "any": "Anyin",
    "avn": "Avatime", "bib": "Bissa", "bim": "Bimoba", "biv": "Southern Birifor",
    "bov": "Tuwuli", "bud": "Bassar (Ntcham)", "bwu": "Buli", "ffm": "Fulfulde (Maasina)",
    "hau": "Hausa", "kbp": "Kabiye", "kdh": "Tem", "kma": "Konni",
    "kpo": "Ikposo", "kus": "Kusaal", "lef": "Lelemi", "lip": "Sekpele",
    "maw": "Mampruli", "mzw": "Deg", "naw": "Nawuri", "ncu": "Chumburung",
    "nko": "Nkonya", "ntr": "Ntrubo", "sfw": "Sehwi", "sig": "Paasaal",
    "sil": "Tumulung Sisaala", "snw": "Selee", "tpm": "Tampulma",
    "vag": "Vagla", "xon": "Konkomba",
}

# Language codes accepted by TTS models / hosted APIs (ISO 639-3).
TTS_LANG_MAP = {
    **{iso: iso for iso in SPEECH_EVAL_SOURCES},
    "twi_akuapem": "atw",
    "twi_asante": "twi",
}

# Directory namespace for clips and references. The text source of each sample
# (bible, jw, ...) is part of its key, not of the path.
DEFAULT_CATEGORY = "read_aloud"


def all_isos():
    return list(SPEECH_EVAL_SOURCES)
