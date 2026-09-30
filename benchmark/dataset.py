"""Load evaluation samples from ghana-speech-eval: text *and* a real recording.

This is what makes benchmark 2 possible. Each sample is a row of
ghana-speech-eval, which is real recorded human speech with a transcript, so
every sample carries its own reference utterance and SpeechBERTScore can be
computed against genuine speech rather than against text.

Rows are referenced by their index in the config, so scores are stable across
re-runs and bumping NUM_SAMPLES only scores rows not already seen.

Reference audio is materialised to ``references/{iso}/{index:05d}.wav`` once
and reused; scoring a 200-sample run should not re-read a 15 GB parquet.
"""

import io
import json
import os

from . import config
from .config import (
    MAX_SECONDS,
    MAX_WORDS,
    MIN_SECONDS,
    MIN_WORDS,
    NUM_SAMPLES,
    SPEECH_EVAL,
    SPEECH_EVAL_CONFIGS,
)

# Output locations are read through `config` at call time rather than imported
# at module scope, so a caller that overrides them (the tests do, to keep
# writes out of the repository) actually redirects the write. Importing them
# here silently pinned them to the repo's real directories, which put test
# output among real data.

# The parquet stores 16-bit mono WAV at 16 kHz, which is exactly the encoder's
# native rate, so reference audio is written through untouched.
_REF_SAMPLE_RATE = 16000

_TABLE_CACHE = {}


class Sample:
    """One evaluation sample: text to synthesise, and a real recording of it."""

    __slots__ = ("index", "text", "duration", "iso", "config")

    def __init__(self, index, text, duration, iso, config):
        self.index = index
        self.text = text
        self.duration = duration
        self.iso = iso
        self.config = config

    @property
    def key(self):
        return f"{self.index:05d}"

    @property
    def reference_path(self):
        return config.REFERENCE_DIR / self.iso / f"{self.key}.wav"

    def as_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "duration": self.duration,
            "iso": self.iso,
            "config": self.config,
        }


def is_usable(text, duration):
    """Is this row worth scoring? Drops rows that make a shaky reference."""
    text = (text or "").strip()
    if not text:
        return False
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if duration is None or not (MIN_SECONDS <= duration <= MAX_SECONDS):
        return False
    return True


def config_rows(iso):
    """Cached read of one language's parquet as a list of dicts.

    The configs run to ~1 GB each, so the table is read once per process and
    shared by the sample loader and the prompt-clip loader.
    """
    config = SPEECH_EVAL_CONFIGS[iso]
    if config not in _TABLE_CACHE:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            SPEECH_EVAL,
            f"{config}/eval-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.ParquetFile(path).read(columns=["audio", "text", "length"])
        _TABLE_CACHE[config] = table.to_pylist()
    return _TABLE_CACHE[config]


def _write_reference(sample, wav_bytes):
    """Persist one reference recording so scoring never re-reads the parquet."""
    import soundfile as sf

    out = sample.reference_path
    out.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sf.write(out, audio, sr if sr else _REF_SAMPLE_RATE, subtype="PCM_16")
    return out


def load_samples(iso, limit=NUM_SAMPLES, offset=0, extract_references=True):
    """Load up to *limit* scored samples for *iso*, with their reference audio.

    Args:
        iso: ISO 639-3 code, e.g. "twi_asante".
        limit: maximum samples to return.
        offset: skip the first *offset* accepted rows (incremental sampling).
        extract_references: write reference wavs to REFERENCE_DIR as we go.

    Returns:
        list of :class:`Sample`, ordered by row index.
    """
    limit = int(os.environ.get("NSANKU2_TTS_NUM_SAMPLES", limit))
    config = SPEECH_EVAL_CONFIGS[iso]
    samples = []
    for row_index, row in enumerate(config_rows(iso)):
        text = (row.get("text") or "").strip()
        if not is_usable(text, row.get("length")):
            continue
        sample = Sample(row_index, text, float(row["length"]), iso, config)
        if extract_references and not sample.reference_path.exists():
            _write_reference(sample, row["audio"]["bytes"])
        samples.append(sample)
        if len(samples) >= offset + limit:
            break
    return samples[offset:offset + limit]


def load_prompt_sample(iso, index):
    """A *different* real recording in the same language, for ref-mode prompting.

    A ref-mode model must never be prompted with the very clip it is scored
    against, or SpeechBERTScore would measure how well it copies the prompt
    rather than how well it synthesises. The prompt row is a fixed distance
    away, so re-runs prompt identically and stay comparable.
    """
    rows = config_rows(iso)
    if not rows:
        return None
    prompt_index = (index + config.PROMPT_ROW_OFFSET) % len(rows)
    row = rows[prompt_index]
    text = (row.get("text") or "").strip()
    if not is_usable(text, row.get("length")):
        return None

    sample = Sample(
        prompt_index, text, float(row["length"]), iso, SPEECH_EVAL_CONFIGS[iso]
    )
    if not sample.reference_path.exists():
        _write_reference(sample, row["audio"]["bytes"])
    return sample


def reference_path_for(iso, index):
    """Where the reference recording for a given row lives (or would live)."""
    return config.REFERENCE_DIR / iso / f"{index:05d}.wav"


def available_configs():
    return dict(SPEECH_EVAL_CONFIGS)


def write_manifest(iso, samples):
    """Record which rows were used, so a run can be reproduced exactly."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"manifest_{iso}.json"
    payload = {
        "iso": iso,
        "config": SPEECH_EVAL_CONFIGS[iso],
        "num_samples": len(samples),
        "source": SPEECH_EVAL,
        "prompt_row_offset": config.PROMPT_ROW_OFFSET,
        "samples": [s.as_dict() for s in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
