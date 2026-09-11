"""Load evaluation text samples from ghana-sentences.

Supports incremental sampling: pass ``offset`` to start at a given index.
Scores are keyed by sample index, so bumping NUM_SAMPLES silently picks up
only the *new* samples on re-run.
"""

import os
import re

from datasets import load_dataset

from .config import GHANA_SENTENCES, NUM_SAMPLES, SUBSET_TO_ISO

# ghana-sentences is built from documents split into lines, so a lot of rows
# are not speakable sentences: headings, bylines, mid-sentence continuations,
# and phonetic notation from the linguistics texts.  Reading those measures
# the text more than the voice, so they are filtered out of the sample pool.
MIN_WORDS = 5
MAX_WORDS = 30

# Notation, list and markup characters that no TTS model is expected to read.
_REJECT_CHARS = re.compile(r"[\[\]{}<>|\\/=•·~^_*#@\d]")
_SENTENCE_END = (".", "!", "?")
_WORD_CHARS = re.compile(r"[^\W\d_]", re.UNICODE)


def is_clean_sentence(text):
    """Is *text* a self-contained sentence worth asking a TTS model to read?"""
    text = text.strip()
    if not text:
        return False

    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False

    # Mid-sentence fragments: the corpus wraps long sentences across rows.
    if not text.endswith(_SENTENCE_END):
        return False
    if not text[0].isupper():
        return False

    # Headings are often fully capitalised.
    letters = _WORD_CHARS.findall(text)
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False

    if _REJECT_CHARS.search(text):
        return False

    # Colons introduce examples/notation rather than prose.
    if ":" in text:
        return False

    # Guard against rows that are mostly punctuation.
    return len(letters) / len(text) > 0.7


def load_text_samples(subset, offset=0, limit=NUM_SAMPLES, num_rows=None, clean=True):
    """Load *limit* speakable sentences for a ghana-sentences subset.

    Args:
        subset: subset name (twi-aku, dag, ...)
        offset: skip the first *offset* accepted samples.
        limit: maximum number of samples to return.
        num_rows: stop after scanning this many rows of the subset.
        clean: keep only rows passing :func:`is_clean_sentence`.

    Returns:
        list of {"text": str, "index": int} — index is the row index in the
        subset, used to dedupe across incremental runs.  Row indices stay
        stable as long as the filter does, so bumping *limit* later picks up
        only sentences that have not been scored yet.
    """
    limit = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", NUM_SAMPLES)) if limit is None else limit
    ds = load_dataset(GHANA_SENTENCES, subset, split="train", streaming=True)
    samples = []
    accepted = 0
    for i, row in enumerate(ds):
        if num_rows is not None and i >= num_rows:
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if clean and not is_clean_sentence(text):
            continue
        accepted += 1
        if accepted <= offset:
            continue
        samples.append({"text": text, "index": i})
        if len(samples) >= limit:
            break
    return samples


def subset_to_iso(subset):
    """Map a ghana-sentences subset name to an ISO 639-3 code."""
    return SUBSET_TO_ISO.get(subset, subset)


def available_subsets():
    """Return the list of subset names in ghana-sentences."""
    return list(SUBSET_TO_ISO.keys())