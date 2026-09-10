"""CTC forced-alignment scoring for TTS audio quality.

Uses MMS-300M (1130-language CTC model) to align generated audio against
the source text.  The mean per-frame log-probability of the Viterbi path
serves as the alignment quality score: closer to 0 = better.

Two backends are available:

1. **python** (default) — pure-Python Viterbi, no C++ build required.
2. **ctc_forced_aligner** — the upstream C++ aligner (faster, needs build).

Select with the ``--alignment-backend`` flag or ``ALIGNMENT_BACKEND`` env var.
"""

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

SAMPLING_FREQ = 16000


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_from_bytes(audio_bytes, sample_rate=SAMPLING_FREQ):
    """Decode raw audio bytes (wav/mp3/ogg) to a mono float32 tensor."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        wav, sr = torchaudio.load(tmp)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav.squeeze(0)
    finally:
        os.unlink(tmp)


def load_audio_from_file(path, sample_rate=SAMPLING_FREQ):
    """Load an audio file path to a mono float32 tensor."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


# ---------------------------------------------------------------------------
# Emissions from MMS-300M via transformers
# ---------------------------------------------------------------------------

AlignmentModel = None
AlignmentTokenizer = None


def _load_alignment_model(device="cuda"):
    """Lazy-load the MMS-300M alignment model."""
    global AlignmentModel, AlignmentTokenizer
    if AlignmentModel is not None:
        return AlignmentModel, AlignmentTokenizer

    from transformers import AutoModelForCTC, AutoTokenizer
    from .config import ALIGNMENT_MODEL

    logger.info("Loading alignment model: %s", ALIGNMENT_MODEL)
    tok = AutoTokenizer.from_pretrained(ALIGNMENT_MODEL)
    mdl = AutoModelForCTC.from_pretrained(ALIGNMENT_MODEL)
    mdl = mdl.to(device).eval()
    AlignmentModel = mdl
    AlignmentTokenizer = tok
    return mdl, tok


def generate_emissions(waveform, device="cuda"):
    """Compute CTC log-probability emissions from a mono 16 kHz waveform.

    Returns (emissions: Tensor[T, C+1], stride_ms: int).
    The extra column is the <star> token (log-prob 0 = prob 1).
    """
    model, _ = _load_alignment_model(device)

    window_length = 30  # seconds
    context_length = 2
    window = int(window_length * SAMPLING_FREQ)
    context = int(context_length * SAMPLING_FREQ)

    if waveform.shape[0] < window:
        input_tensor = waveform.unsqueeze(0).to(device)
        extension = 0
        ctx = 0
    else:
        extension = math.ceil(waveform.shape[0] / window) * window - waveform.shape[0]
        padded = torch.nn.functional.pad(waveform, (context, context + extension))
        input_tensor = padded.unfold(0, window + 2 * context, window).to(device)
        ctx = context

    emissions_arr = []
    with torch.inference_mode():
        for i in range(0, input_tensor.size(0), 4):
            batch = input_tensor[i : i + 4]
            emissions_arr.append(model(batch).logits)

    emissions = torch.cat(emissions_arr, dim=0)

    if ctx > 0:
        stride_per_sec = 1000 / 20
        ctx_frames = int(context_length * stride_per_sec)
        emissions = emissions[:, ctx_frames : -ctx_frames + 1]

    emissions = emissions.flatten(0, 1)

    ext_frames = int((extension / SAMPLING_FREQ) * (1000 / 20))
    if ext_frames > 0:
        emissions = emissions[: -ext_frames]

    emissions = torch.log_softmax(emissions, dim=-1)
    emissions = torch.cat(
        [emissions, torch.zeros(emissions.size(0), 1, device=emissions.device)], dim=1
    )
    stride_ms = math.ceil(float(waveform.shape[0] * 1000 / emissions.size(0) / SAMPLING_FREQ))
    return emissions, stride_ms


# ---------------------------------------------------------------------------
# Pure-Python CTC Viterbi forced alignment
# ---------------------------------------------------------------------------

def _viterbi_align(emissions_np, target_ids, blank_id=0):
    """CTC forced alignment via Viterbi in log-space.

    Args:
        emissions_np: float32 array [T, V] of log-probabilities.
        target_ids: list[int] of target token indices (no blank).
        blank_id: CTC blank index.

    Returns:
        path: list[int] of emission indices for each timestep.
        scores: float32 array [T] of per-frame log-prob scores.
    """
    T, V = emissions_np.shape
    N = len(target_ids)

    # States: 0 = blank, 2*k+1 = on target[k] (even), 2*k+2 = repeat target[k] (odd)
    n_states = 2 * N + 1
    dp = np.full(n_states, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    pointers = np.zeros((T, n_states), dtype=np.int32)

    for t in range(T):
        new_dp = np.full(n_states, -np.inf, dtype=np.float64)
        for s in range(n_states):
            if dp[s] == -np.inf:
                continue
            if s == 0:
                # blank -> blank
                if new_dp[0] < dp[0] + emissions_np[t, blank_id]:
                    new_dp[0] = dp[0] + emissions_np[t, blank_id]
                    pointers[t, 0] = s
                # blank -> first target (state 1)
                if N > 0 and new_dp[1] < dp[0] + emissions_np[t, target_ids[0]]:
                    new_dp[1] = dp[0] + emissions_np[t, target_ids[0]]
                    pointers[t, 1] = s
            else:
                k = (s - 1) // 2
                is_odd = (s % 2 == 1)
                tid = target_ids[k]

                # stay on same state
                val = dp[s] + emissions_np[t, tid]
                if val > new_dp[s]:
                    new_dp[s] = val
                    pointers[t, s] = s

                # blank transition
                val = dp[s] + emissions_np[t, blank_id]
                if val > new_dp[0]:
                    new_dp[0] = val
                    pointers[t, 0] = s

                if is_odd:
                    # repeat -> advance to next target or blank
                    if k + 1 < N:
                        next_state = 2 * (k + 1) + 1
                        val = dp[s] + emissions_np[t, target_ids[k + 1]]
                        if val > new_dp[next_state]:
                            new_dp[next_state] = val
                            pointers[t, next_state] = s
                else:
                    # even -> can repeat (odd state)
                    val = dp[s] + emissions_np[t, tid]
                    if val > new_dp[s + 1]:
                        new_dp[s + 1] = val
                        pointers[t, s + 1] = s
                    # advance to next or blank
                    if k + 1 < N:
                        next_state = 2 * (k + 1) + 1
                        val = dp[s] + emissions_np[t, target_ids[k + 1]]
                        if val > new_dp[next_state]:
                            new_dp[next_state] = val
                            pointers[t, next_state] = s

        dp = new_dp

    # find best final state
    best_state = int(np.argmax(dp))
    best_score = dp[best_state]

    # backtrack
    path = np.zeros(T, dtype=np.int32)
    scores = np.zeros(T, dtype=np.float32)
    state = best_state
    for t in range(T - 1, -1, -1):
        path[t] = state
        scores[t] = emissions_np[t, blank_id] if state == 0 else emissions_np[t, target_ids[(state - 1) // 2]]
        state = pointers[t, state]

    return path.tolist(), scores


def _merge_repeats(path, idx_to_token):
    """Merge consecutive identical states into segments."""
    segments = []
    i = 0
    while i < len(path):
        j = i
        while j < len(path) and path[j] == path[i]:
            j += 1
        state_id = path[i]
        if state_id == 0:
            label = "<blank>"
        else:
            tid = (state_id - 1) // 2
            label = idx_to_token.get(tid, f"tok_{tid}")
        segments.append({"label": label, "start": i, "end": j - 1})
        i = j
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_alignment_score(waveform, text, lang="eng", device="cuda"):
    """Align *waveform* to *text* and return the mean log-probability score.

    Returns:
        dict with keys: score (float), segments (list), num_frames (int).
        score closer to 0 = better alignment.
    """
    model, tokenizer = _load_alignment_model(device)

    # Tokenise text
    tok_output = tokenizer(text.lower(), return_tensors="pt", padding=False)
    target_ids = tok_output.input_ids[0].tolist()

    # Remove pad/sos/eos tokens — keep only vocabulary tokens
    skip = {tokenizer.pad_token_id, tokenizer.sos_token_id, tokenizer.eos_token_id}
    target_ids = [t for t in target_ids if t not in skip]

    if not target_ids:
        return {"score": -10.0, "segments": [], "num_frames": 0}

    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Generate emissions
    emissions, _ = generate_emissions(waveform, device=device)

    # Run Viterbi alignment
    path, scores = _viterbi_align(
        emissions.cpu().numpy().astype(np.float32),
        target_ids,
        blank_id=blank_id,
    )

    # Build index map: state_id -> token string
    idx_to_token = {}
    for i, tid in enumerate(target_ids):
        idx_to_token[i] = tokenizer.decode([tid])
    segments = _merge_repeats(path, idx_to_token)

    mean_score = float(np.mean(scores)) if len(scores) > 0 else -10.0

    return {
        "score": mean_score,
        "segments": segments,
        "num_frames": len(path),
    }


def compute_alignment_score_from_bytes(audio_bytes, text, lang="eng", device="cuda"):
    """Convenience wrapper: decode audio bytes then align."""
    waveform = load_audio_from_bytes(audio_bytes)
    return compute_alignment_score(waveform, text, lang=lang, device=device)
