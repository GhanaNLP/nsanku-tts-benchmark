"""Load evaluation text samples from ghana-sentences."""

from datasets import load_dataset

from .config import GHANA_SENTENCES, NUM_SAMPLES, SUBSET_TO_ISO


def load_text_samples(subset, num_samples=NUM_SAMPLES):
    """Load up to *num_samples* text sentences for a ghana-sentences subset.

    Returns a list of dicts: [{"text": str, "index": int}, ...].
    """
    ds = load_dataset(GHANA_SENTENCES, subset, split="train", streaming=True)
    samples = []
    for i, row in enumerate(ds):
        if i >= num_samples:
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
