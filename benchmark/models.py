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
import os
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _wav_bytes(samples, sample_rate=24000):
    """Encode a float32 numpy array as WAV bytes."""
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


def load_tts_model(model_id, device="cuda", subset=None, **kwargs):
    """Auto-detect and load a TTS model wrapper.

    Args:
        model_id: huggingface model id
        device: torch device string
        subset: ghana-sentences subset name (for reference-audio lookup)

    Returns an instance of BaseTTSModel.
    """
    lower = model_id.lower()
    meta = _model_meta(model_id)

    if meta.get("input_type") == "ipa":
        raise UnsupportedModel(f"{model_id}: requires IPA input, skipping")

    vendor = model_id.split("/")[0].lower()

    if "voxcpm2" in lower or meta.get("architecture", "").startswith("VoxCPM2"):
        return VoxCPM2Wrapper(model_id, device=device, subset=subset, meta=meta, **kwargs)
    if "ghana-tts" in lower or vendor in ("techolise",):
        return VoxCPMWrapper(model_id, device=device, subset=subset, meta=meta, **kwargs)
    if "f5-tts" in lower and "openbible" in lower:
        return F5TTSWrapper(model_id, device=device, **kwargs)
    if "nano-twi" in lower:
        return NanoTwiWrapper(model_id, device=device, **kwargs)
    if lower.startswith("khaya") or "khaya" in lower:
        return KhayaTTSWrapper(model_id, device="cpu", **kwargs)

    if "kokoro" in lower or "sherpa" in lower:
        raise UnsupportedModel(f"{model_id}: Kokoro/sherpa models not yet supported")

    raise UnsupportedModel(f"{model_id}: unknown TTS architecture")


def _model_meta(model_id):
    """Grab metadata for *model_id* from data/tts_models.json."""
    import json

    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    try:
        with open(path) as f:
            for m in json.load(f):
                if m.get("name") == model_id:
                    return m
    except (OSError, json.JSONDecodeError):
        pass
    return {}


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
        import gc
        import torch

        model = self.__dict__.get("model")
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _reference_audio(meta, model_id):
    """Download & return (wav_path, ref_text) for a model's reference audio.

    Models that anchor a specific voice (LoRA TTS, VoxCPM2 SFT) ship a
    reference clip in their repo.  meta may carry:
        reference_audio: path relative to the repo
        reference_text:  gold transcript of the reference clip
    """
    ref_path = meta.get("reference_audio")
    if not ref_path:
        return None, None
    from huggingface_hub import hf_hub_download

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsanku-tts", "refs")
    wav_path = hf_hub_download(model_id, ref_path, cache_dir=cache_dir)
    return wav_path, meta.get("reference_text")


class VoxCPMWrapper(BaseTTSModel):
    """VoxCPM v1 (0.7B) — ghana-tts-72k / ghana-tts-36k and LoRA adapters.

    Orthographic input, 16 kHz output.  LoRA adapters (techolise speaker17)
    are voice-anchored: they require a reference clip + its transcript.
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_id, device="cuda", subset=None, meta=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.ref_wav = None
        self.ref_text = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(self.model_id, device=self.device)

        if self.meta.get("reference_audio"):
            ref_wav, ref_text = _reference_audio(self.meta, self.model_id)
            self.ref_wav, self.ref_text = ref_wav, ref_text
        self._loaded = True
        logger.info("Loaded VoxCPM v1: %s", self.model_id)

    def _generate(self, text, lang):
        """Call model.generate handling the optional lang kwarg."""
        kwargs = dict(
            cfg_value=2.4 if self.ref_wav else 2.0,
            inference_timesteps=26 if self.ref_wav else 10,
            retry_badcase=True,
        )
        if self.ref_wav:
            kwargs["prompt_wav_path"] = self.ref_wav
            if self.ref_text:
                kwargs["prompt_text"] = self.ref_text
        # Some forks accept language=, the upstream one does not.
        try:
            return self.model.generate(text, **{**kwargs, "language": lang})
        except TypeError:
            return self.model.generate(text, **kwargs)

    def synthesize(self, text, lang="eng"):
        self._ensure_loaded()
        audio = self._generate(text, lang)
        if isinstance(audio, tuple):
            audio = audio[0]
        return _wav_bytes(audio, sample_rate=self.SAMPLE_RATE)


class VoxCPM2Wrapper(BaseTTSModel):
    """VoxCPM2 (2B) SFT models — FarmerlineML voxcpm2-{akan,dagbani,ewe}-sft.

    Orthographic input, 48 kHz output.  Voice-anchored: best results use a
    reference clip; if none ships in the repo we run on the model's default.
    """

    SAMPLE_RATE = 48000

    def __init__(self, model_id, device="cuda", subset=None, meta=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.ref_wav = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False)
        if self.meta.get("reference_audio"):
            from huggingface_hub import hf_hub_download

            wav_path = hf_hub_download(self.model_id, self.meta["reference_audio"])
            self.ref_wav = wav_path
        self._loaded = True
        logger.info("Loaded VoxCPM2: %s", self.model_id)

    def _generate(self, text):
        kwargs = dict(
            text=text,
            cfg_value=2.0,
            inference_timesteps=15,
            retry_badcase=False,
            max_len=max(50, len(text) * 4),
        )
        if self.ref_wav:
            kwargs["reference_wav_path"] = self.ref_wav
        return self.model.generate(**kwargs)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        audio = self._generate(text)
        if isinstance(audio, tuple):
            audio = audio[0]
        return _wav_bytes(audio, sample_rate=self.SAMPLE_RATE)


class F5TTSWrapper(BaseTTSModel):
    """F5-TTS OpenBible fine-tunes — zero-shot TTS with reference audio."""

    SAMPLE_RATE = 24000

    def __init__(self, model_id, device="cuda", **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from f5_tts.api import F5TTS

        self.model = F5TTS(ckpt_file=self.model_id, device=self.device)
        self._loaded = True
        logger.info("Loaded F5-TTS: %s", self.model_id)

    def synthesize(self, text, lang="eng", ref_audio=None, ref_text=None):
        self._ensure_loaded()
        out = self.model.infer(text=text, ref_file=ref_audio, ref_text=ref_text)
        audio = out.get("audio") if isinstance(out, dict) else out
        return _wav_bytes(audio, sample_rate=self.SAMPLE_RATE)


class NanoTwiWrapper(BaseTTSModel):
    """ghananlpcommunity/nano-twi — Matcha-TTS + Vocos ONNX (CPU)."""

    SAMPLE_RATE = 24000

    def __init__(self, model_id, device="cuda", **kwargs):
        super().__init__(model_id, "cpu")  # runs on CPU
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download

        from .config import HF_TOKEN

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsanku-tts", "nano-twi")
        model_path = snapshot_download(self.model_id, cache_dir=cache_dir, token=HF_TOKEN or None)

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
    """Khaya AI TTS v2 — hosted REST endpoint.

    Endpoint: POST https://translation-api.ghananlp.org/tts/v2/synthesize
    Auth header: Ocp-Apim-Subscription-Key: <KHAYA_API_KEY>
    Body: {"text": ..., "language": <ISO 639-3>, "format": "wav"}
    """

    API_URL = "https://translation-api.ghananlp.org/tts/v2/synthesize"

    def __init__(self, model_id="KhayaAI/khaya-tts", device="cpu", **kwargs):
        super().__init__(model_id, "cpu")
        self.api_key = os.environ.get("KHAYA_API_KEY", "")
        if not self.api_key:
            raise ValueError("KHAYA_API_KEY not set — required for Khaya TTS")
        self._session = None

    def _ensure_session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/json",
            })
        return self._session

    def synthesize(self, text, lang="twi", speaker_id=None, output_format="wav"):
        session = self._ensure_session()
        body = {
            "text": text,
            "language": lang,
            "format": output_format,
        }
        if speaker_id:
            body["speaker_id"] = speaker_id
        resp = session.post(self.API_URL, json=body, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Khaya TTS {resp.status_code}: {resp.text[:200]}")
        return resp.content