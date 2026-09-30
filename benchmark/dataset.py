"""Load evaluation samples from ghana-speech-eval: text *and* a real recording.

Each sample is a row of ghana-speech-eval, which is real recorded human speech
with a transcript, so every sample carries its own reference utterance and
SpeechBERTScore can be computed against genuine speech rather than against
text. The same transcript is what the ASR judge's output is compared with (CER).

A language can have several text sources (bible, jw, finance, ...). Its samples
are drawn across all of them, in equal shares, so a model is not scored on a
single register. A sample is identified by ``<source>_<row>`` (for example
``bible_00012``), which is stable across re-runs: raising NUM_SAMPLES only adds
rows, it never reshuffles the ones already scored.

Reference audio is materialised to ``references/{iso}/{key}.wav`` once and
reused; scoring a run should not re-read a 1 GB parquet per clip.
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
    SPEECH_EVAL_SOURCES,
)

# Output locations are read through `config` at call time rather than imported
# at module scope, so a caller that overrides them (the tests do, to keep
# writes out of the repository) actually redirects the write.

# The parquet stores 16-bit mono WAV at 16 kHz, which is exactly the encoder's
# native rate, so reference audio is written through untouched.
_REF_SAMPLE_RATE = 16000

# How far past the offset row to look for a usable prompt before giving up.
_PROMPT_SEARCH = 50

_META_CACHE = {}
_FILE_CACHE = {}
_GROUP_CACHE = {}


class Sample:
    """One evaluation sample: text to synthesise, and a real recording of it."""

    __slots__ = ("source", "row", "text", "duration", "iso", "config")

    def __init__(self, source, row, text, duration, iso, config):
        self.source = source
        self.row = row
        self.text = text
        self.duration = duration
        self.iso = iso
        self.config = config

    @property
    def key(self):
        return f"{self.source}_{self.row:05d}"

    # Results are keyed by this string; `index` is the name the scoring code uses.
    index = key

    @property
    def reference_path(self):
        return config.REFERENCE_DIR / self.iso / f"{self.key}.wav"

    def as_dict(self):
        return {
            "key": self.key,
            "source": self.source,
            "row": self.row,
            "text": self.text,
            "duration": self.duration,
            "iso": self.iso,
            "config": self.config,
        }


def split_key(key):
    """'bible_00012' -> ('bible', 12)."""
    source, row = key.rsplit("_", 1)
    return source, int(row)


def source_of(key):
    return key.rsplit("_", 1)[0]


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


_LISTING = []


def _shard_names(cfg):
    """Parquet files of one config. Most are a single file, a few are sharded."""
    if not _LISTING:
        from huggingface_hub import HfApi

        try:
            _LISTING.extend(HfApi().list_repo_files(SPEECH_EVAL, repo_type="dataset"))
        except Exception:
            _LISTING.append("")  # offline: fall back to the single-file name
    names = sorted(f for f in _LISTING if f.startswith(f"{cfg}/") and f.endswith(".parquet"))
    return names or [f"{cfg}/eval-00000-of-00001.parquet"]


def _open_shard(name):
    """Open one parquet shard.

    Read remotely by range request, so picking 200 rows out of a 1 GB config
    fetches the text/length columns and the few row groups that hold the chosen
    audio, not the whole file. Falls back to downloading the file if the
    remote read is unavailable.
    """
    import pyarrow.parquet as pq

    try:
        from huggingface_hub import HfFileSystem

        f = HfFileSystem().open(f"datasets/{SPEECH_EVAL}/{name}")
        return pq.ParquetFile(f)
    except Exception:
        from huggingface_hub import hf_hub_download

        return pq.ParquetFile(hf_hub_download(SPEECH_EVAL, name, repo_type="dataset"))


def _groups(cfg):
    """[(ParquetFile, row_group, first_row, num_rows)] across a config's shards.

    Row numbers are global to the config, so a sharded config indexes exactly
    like a single-file one.
    """
    if cfg not in _FILE_CACHE:
        groups, start = [], 0
        for name in _shard_names(cfg):
            pf = _open_shard(name)
            for g in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                groups.append((pf, g, start, n))
                start += n
        _FILE_CACHE[cfg] = groups
    return _FILE_CACHE[cfg]


def config_meta(cfg):
    """(text, length) for every row of a config, without reading the audio.

    The audio column is most of the file, so it is only read for the handful of
    rows that are actually used (see :func:`_audio_bytes`).
    """
    if cfg not in _META_CACHE:
        rows = []
        for pf, g, _, _ in _groups(cfg):
            rows.extend(pf.read_row_group(g, columns=["text", "length"]).to_pylist())
        _META_CACHE[cfg] = rows
    return _META_CACHE[cfg]


def _audio_bytes(cfg, row):
    """WAV bytes of one row, reading only the row group that holds it."""
    for i, (pf, g, first, n) in enumerate(_groups(cfg)):
        if row < first + n:
            if _GROUP_CACHE.get(cfg, (None,))[0] != i:
                table = pf.read_row_group(g, columns=["audio"])
                _GROUP_CACHE[cfg] = (i, table.column("audio").to_pylist(), first)
            _, audios, base = _GROUP_CACHE[cfg]
            return audios[row - base]["bytes"]
    raise IndexError(f"{cfg}: row {row} out of range")


def _write_reference(sample):
    """Persist one reference recording so scoring never re-reads the parquet."""
    import soundfile as sf

    out = sample.reference_path
    out.parent.mkdir(parents=True, exist_ok=True)
    wav_bytes = _audio_bytes(sample.config, sample.row)
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sf.write(out, audio, sr if sr else _REF_SAMPLE_RATE, subtype="PCM_16")
    return out


def allocate(limit, available):
    """Split *limit* samples across sources as evenly as their supply allows.

    Hands out one sample at a time round-robin, so shares differ by at most one
    (earlier sources take the remainder) and a source with too few usable rows
    gives its unmet share to the others. Raising *limit* only ever adds to each
    share, which is what keeps incremental runs stable.
    """
    take = [0] * len(available)
    remaining = limit
    while remaining > 0:
        progressed = False
        for i, supply in enumerate(available):
            if remaining == 0:
                break
            if take[i] < supply:
                take[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return take


def load_samples(iso, limit=NUM_SAMPLES, extract_references=True):
    """Load up to *limit* samples for *iso*, drawn across its text sources.

    Within a source the first usable rows in file order are taken, so the choice
    is deterministic. Reference audio is written to REFERENCE_DIR as we go.

    Returns:
        list of :class:`Sample`, ordered by source then row.
    """
    limit = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", limit))
    sources = SPEECH_EVAL_SOURCES[iso]

    usable = {}
    for source, cfg in sources.items():
        usable[source] = [
            i for i, r in enumerate(config_meta(cfg))
            if is_usable((r.get("text") or "").strip(), r.get("length"))
        ]

    shares = allocate(limit, [len(usable[s]) for s in sources])
    samples = []
    for (source, cfg), n in zip(sources.items(), shares):
        meta = config_meta(cfg)
        for row in usable[source][:n]:
            sample = Sample(source, row, meta[row]["text"].strip(),
                            float(meta[row]["length"]), iso, cfg)
            if extract_references and not sample.reference_path.exists():
                _write_reference(sample)
            samples.append(sample)
    return samples


def load_prompt_sample(iso, key):
    """A *different* real recording in the same language, for ref-mode prompting.

    A ref-mode model must never be prompted with the very clip it is scored
    against, or SpeechBERTScore would measure how well it copies the prompt
    rather than how well it synthesises. The prompt comes from the same source
    a fixed distance away (and the next usable row after that), so re-runs
    prompt identically and stay comparable.
    """
    source, row = split_key(key)
    cfg = SPEECH_EVAL_SOURCES[iso][source]
    meta = config_meta(cfg)
    if not meta:
        return None
    for step in range(_PROMPT_SEARCH):
        prompt_row = (row + config.PROMPT_ROW_OFFSET + step) % len(meta)
        r = meta[prompt_row]
        text = (r.get("text") or "").strip()
        if is_usable(text, r.get("length")):
            sample = Sample(source, prompt_row, text, float(r["length"]), iso, cfg)
            if not sample.reference_path.exists():
                _write_reference(sample)
            return sample
    return None


def reference_path_for(iso, key):
    """Where the reference recording for a given sample lives (or would live)."""
    return config.REFERENCE_DIR / iso / f"{key}.wav"


def write_manifest(iso, samples):
    """Record which rows were used, so a run can be reproduced exactly."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / f"manifest_{iso}.json"
    by_source = {}
    for s in samples:
        by_source[s.source] = by_source.get(s.source, 0) + 1
    payload = {
        "iso": iso,
        "sources": SPEECH_EVAL_SOURCES[iso],
        "num_samples": len(samples),
        "samples_per_source": by_source,
        "source": SPEECH_EVAL,
        "prompt_row_offset": config.PROMPT_ROW_OFFSET,
        "samples": [s.as_dict() for s in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
