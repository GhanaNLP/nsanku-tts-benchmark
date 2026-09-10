"""Load evaluation text samples from ghana-sentences.

Supports incremental sampling: pass ``offset`` to start at a given index.
Scores are keyed by sample index, so bumping NUM_SAMPLES silently picks up
only the *new* samples on re-run.
"""

import os

from datasets import load_dataset

from .config import GHANA_SENTENCES, NUM_SAMPLES, SUBSET_TO_ISO


def load_text_samples(subset, offset=0, limit=NUM_SAMPLES, num_rows=None):
    """Load *limit* text sentences for a ghana-sentences subset starting at offset.

    Args:
        subset: subset name (twi-aku, dag, ...)
        offset: skip the first *offset* rows of the subset.
        limit: maximum number of samples to return.
        num_rows: if given and the subset has fewer rows, only materialises
                  that many (letting callers know the total cheaply).

    Returns:
        list of {"text": str, "index": int} — index is the row index in the
        subset, used to dedupe across incremental runs.
    """
    limit = int(os.environ.get("NSANKU_TTS_NUM_SAMPLES", NUM_SAMPLES)) if limit is None else limit
    ds = load_dataset(GHANA_SENTENCES, subset, split="train", streaming=True)
    samples = []
    end = offset + limit
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if i >= end:
            break
        if num_rows is not None and i >= num_rows:
            break
        text = (row.get("text") or "").strip()
        if text:
            samples.append({"text": text, "index": i})
    return samples


def subset_to_iso(subset):
    """Map a ghana-sentences subset name to an ISO 639-3 code."""
    return SUBSET_TO_ISO.get(subset, subset)


def available_subsets():
    """Return the list of subset names in ghana-sentences."""
    return list(SUBSET_TO_ISO.keys())