"""WER and CER computation.

Kept byte-for-byte compatible with the ASR benchmark's metrics
(github.com/GhanaNLP/nsanku-asr-benchmark, benchmark/metrics.py) so a TTS
score and an ASR score for the same language are computed the same way and
can be read against each other.
"""

import re


def normalize_text(text):
    """Normalize transcript text for WER/CER computation."""
    text = text.upper()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference, hypothesis):
    """Word Error Rate."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return _edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)


def cer(reference, hypothesis):
    """Character Error Rate."""
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    return _edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)


def compute_metrics(reference, hypothesis):
    """Return dict with WER and CER."""
    return {
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
    }


def _edit_distance(ref_tokens, hyp_tokens):
    """Raw Levenshtein (edit) distance between two token lists."""
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
