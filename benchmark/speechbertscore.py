"""SpeechBERTScore: reference-aware evaluation of synthesised speech.

Implements the metric from Saeki et al., "SpeechBERTScore: Reference-Aware
Automatic Evaluation of Speech Generation Leveraging NLP Evaluation Metrics"
(Interspeech 2024, arXiv:2401.16812).

    Z_hat = Encoder(X_hat)      Z = Encoder(X)
    SpeechBERTScore = (1/N_gen) * sum_i max_j cos(z_hat_i, z_j)      (Eq. 6)

The encoder is a self-supervised speech model and the two sequences are
matched greedily frame-by-frame, so generated and reference speech may differ
in length and need not be time-aligned. We use the precision variant because
the paper found it correlated best with human ratings (their Appendix A:
precision 0.545 vs F1 0.530 vs recall 0.495 utterance-level SRCC).

Encoder and layer were selected by measured sweep over 6 languages, not by
assumption -- see docs/encoder-selection.md. Default is wavlm-large layer 6.
"""

import logging

import numpy as np
import torch

from .config import ENCODER_LAYER, ENCODER_MODEL, SCORE_SAMPLE_RATE

logger = logging.getLogger(__name__)


def validate_layer(n_layers, layer, model_name=""):
    """Raise if *layer* is not a transformer block of this encoder.

    hidden_states[0] is the convolutional frontend output, so index N is the
    output of the Nth transformer block and the valid range is 0..N-1. An
    off-by-one here silently scores a different layer than intended.
    """
    if not 0 <= layer < n_layers:
        raise ValueError(
            f"layer {layer} out of range for {model_name or 'encoder'} "
            f"({n_layers} layers, valid 0..{n_layers - 1})"
        )
    return layer


class SpeechBERTScorer:
    """Scores synthesised speech against a real reference recording."""

    def __init__(self, model_name=ENCODER_MODEL, layer=ENCODER_LAYER,
                 device="cuda", dtype=torch.float32):
        from transformers import AutoModel

        self.model_name = model_name
        self.layer = layer
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        validate_layer(self.n_layers, layer, model_name)
        logger.info("Loaded %s layer %d/%d on %s",
                    model_name, layer, self.n_layers, self.device)

    @torch.inference_mode()
    def features(self, audio, sample_rate):
        """L2-normalised frame features from the configured layer.

        Returns a [T, D] float32 tensor. The waveform is resampled to the
        encoder's native rate here so a 48 kHz synthesised clip and a 16 kHz
        reference are compared in the same space.
        """
        audio = to_mono_16k(audio, sample_rate)
        if audio.size == 0:
            return None
        wav = torch.from_numpy(audio).to(self.device, dtype=self.dtype).unsqueeze(0)
        hidden = self.model(wav, output_hidden_states=True).hidden_states[self.layer][0]
        return torch.nn.functional.normalize(hidden.float(), dim=-1)

    def score(self, generated, generated_sr, reference, reference_sr):
        """SpeechBERTScore for one pair of clips. Higher is better (0-1)."""
        z_gen = self.features(generated, generated_sr)
        z_ref = self.features(reference, reference_sr)
        if z_gen is None or z_ref is None or z_gen.shape[0] == 0 or z_ref.shape[0] == 0:
            return None
        return speechbertscore(z_gen, z_ref)

    def cleanup(self):
        del self.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def speechbertscore(z_gen, z_ref, chunk=4096):
    """The metric itself, on precomputed normalised features (Eq. 6).

    Chunked over the generated frames so a long clip does not build a
    [T_gen, T_ref] matrix in memory at once.
    """
    if z_gen is None or z_ref is None or z_gen.numel() == 0 or z_ref.numel() == 0:
        return None
    total, n = 0.0, 0
    for start in range(0, z_gen.shape[0], chunk):
        block = z_gen[start:start + chunk]
        total += (block @ z_ref.T).max(dim=1).values.sum().item()
        n += block.shape[0]
    return total / n


def to_mono_16k(audio, sample_rate):
    """Decode any input to mono float32 at the encoder's native 16 kHz."""
    if isinstance(audio, (bytes, bytearray)):
        import io

        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sample_rate and sample_rate != SCORE_SAMPLE_RATE:
        audio = _resample(audio, sample_rate, SCORE_SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample(audio, src_rate, dst_rate):
    """Linear-interpolation resampling, avoiding a torchaudio dependency here."""
    if src_rate == dst_rate:
        return audio
    n_out = int(round(len(audio) * dst_rate / src_rate))
    if n_out <= 0 or len(audio) <= 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(audio), dtype=np.float64) / src_rate
    x_new = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(x_new, x_old, audio).astype(np.float32)


def self_check(scorer):
    """Assert the scorer is deterministic before trusting a leaderboard.

    Encoding identical audio must give bit-identical features and a
    self-similarity of exactly 1.0. This is not paranoia: an earlier version
    of this metric encoded through a pretraining path that applied
    SpecAugment-style masking, which made repeated encodes of the same audio
    differ by up to 7.9 and produced a self-similarity of 0.91. That bug
    looked like a working metric and silently corrupted every layer decision.
    """
    import numpy as np

    tone = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32)
    a = scorer.features(tone, 16000)
    b = scorer.features(tone, 16000)
    if a is None or b is None:
        return False
    if not torch.equal(a, b):
        raise AssertionError("encoder is not deterministic across calls")
    value = speechbertscore(b, a)
    if value is None or abs(value - 1.0) > 1e-4:
        raise AssertionError(f"self-similarity is {value!r}, expected 1.0")
    return True
