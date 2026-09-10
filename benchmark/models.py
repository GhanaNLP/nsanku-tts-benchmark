"""TTS model wrappers.

Each wrapper exposes a common interface:

    model = load_tts_model(model_id, device="cuda")
    audio_bytes = model.synthesize(text, lang="twi")

The ``model_id`` is a HuggingFace model identifier.  The wrapper is
auto-detected from the model's tags / config.
"""

import abc
import io
import logging
import struct
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def _wav_bytes(samples, sample_rate=24000):
    """Encode a float32 numpy array as WAV bytes."""
    import numpy as np

    if isinstance(samples, list):
        samples = np.array(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def load_tts_model(model_id, device="cuda", **kwargs):
    """Auto-detect and load a TTS model wrapper.

    Returns an instance of BaseTTSModel.
    """
    lower = model_id.lower()

    if "voxcpm2-ghana" in lower:
        raise UnsupportedModel(f"{model_id}: requires IPA input, skipping")
    if "stable-twi-tts" in lower:
        raise UnsupportedModel(f"{model_id}: requires IPA input, skipping")

    if "ghana-tts-72k" in lower:
        return VoxCPMWrapper(model_id, device=device, **kwargs)
    if "f5-tts" in lower and "openbible" in lower:
        return F5TTSWrapper(model_id, device=device, **kwargs)
    if "nano-twi" in lower:
        return NanoTwiWrapper(model_id, device=device, **kwargs)

    # Try generic kokoro / sherpa-onnx for any future models
    if "kokoro" in lower or "sherpa" in lower:
        raise UnsupportedModel(f"{model_id}: Kokoro/sherpa models not yet supported")

    raise UnsupportedModel(f"{model_id}: unknown TTS architecture")


class UnsupportedModel(Exception):
    pass


class BaseTTSModel(abc.ABC):
    """Abstract base for TTS model wrappers."""

    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device

    @abc.abstractmethod
    def synthesize(self, text, lang="eng"):
        """Synthesise *text* to audio.

        Returns:
            bytes: WAV-encoded audio.
        """
        ...

    def cleanup(self):
        """Release GPU memory."""
        import gc, torch
        del self.__dict__.get("model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VoxCPMWrapper(BaseTTSModel):
    """ghananlpcommunity/ghana-tts-72k — VoxCPM v1 (0.7B), orthographic input."""

    def __init__(self, model_id, device="cuda", **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download
        from pathlib import Path as P
        import os

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsanku-tts", "voxcpm")
        model_path = snapshot_download(self.model_id, cache_dir=cache_dir)
        logger.info("Loaded VoxCPM from %s", model_path)

        # VoxCPM uses a custom inference script — wrap it
        # For now, we use the model's generate API
        try:
            from voxcpm import VoxCPM
            self.model = VoxCPM.from_pretrained(model_path, device=self.device)
        except ImportError:
            raise ImportError(
                "VoxCPM library not installed. Install from the model's repo."
            )
        self._loaded = True

    def synthesize(self, text, lang="eng"):
        self._ensure_loaded()
        audio = self.model.generate(text)
        return _wav_bytes(audio)


class F5TTSWrapper(BaseTTSModel):
    """F5-TTS OpenBible fine-tunes — zero-shot TTS with reference audio."""

    def __init__(self, model_id, device="cuda", **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from f5_tts.api import F5TTS
        self.model = F5TTS(
            ckpt_file=self.model_id,
            device=self.device,
        )
        self._loaded = True
        logger.info("Loaded F5-TTS: %s", self.model_id)

    def synthesize(self, text, lang="eng", ref_audio=None, ref_text=None):
        """Synthesise text.  If ref_audio/ref_text are None, uses the
        model's built-in reference from the OpenBible fine-tune."""
        self._ensure_loaded()
        out = self.model.infer(
            text=text,
            ref_file=ref_audio,
            ref_text=ref_text,
        )
        audio = out.get("audio") if isinstance(out, dict) else out
        return _wav_bytes(audio)


class NanoTwiWrapper(BaseTTSModel):
    """ghananlpcommunity/nano-twi — Matcha-TTS + Vocos ONNX, lightweight CPU."""

    def __init__(self, model_id, device="cuda", **kwargs):
        super().__init__(model_id, "cpu")  # runs on CPU
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download
        import os

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsanku-tts", "nano-twi")
        model_path = snapshot_download(self.model_id, cache_dir=cache_dir)

        # Uses sherpa-onnx for inference
        try:
            import sherpa_onnx
            self.model = sherpa_onnx.OfflineTts(
                sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                            model=os.path.join(model_path, "model.onnx"),
                        ),
                    ),
                    tokens=os.path.join(model_path, "tokens.txt"),
                    num_threads=4,
                )
            )
        except ImportError:
            raise ImportError("sherpa-onnx not installed. pip install sherpa-onnx")
        self._loaded = True
        logger.info("Loaded nano-twi from %s", model_path)

    def synthesize(self, text, lang="eng"):
        self._ensure_loaded()
        audio = self.model.generate(text)
        return _wav_bytes(audio.sample, sample_rate=audio.sample_rate)


class KhayaTTSWrapper(BaseTTSModel):
    """Khaya AI TTS API — hosted REST endpoint."""

    def __init__(self, model_id="KhayaAI/khaya-tts", device="cpu", **kwargs):
        super().__init__(model_id, device)
        self.api_url = "https://translation.ghananlp.org/api/v2/tts"

    def synthesize(self, text, lang="twi"):
        import requests
        resp = requests.post(
            self.api_url,
            json={"text": text, "language": lang},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
