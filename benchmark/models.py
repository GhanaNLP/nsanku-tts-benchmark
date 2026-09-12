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
import sys
import wave
from pathlib import Path

import numpy as np

from .config import USE_REFERENCE_AUDIO
from .recipes import recipe_get

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


def _cache_root():
    """Root for downloaded/derived assets — the HF cache volume on Modal."""
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "nsanku-tts"
    )


def _recipe_for(model_id, iso):
    """The (model, language) recipe module, or None."""
    if not iso:
        return None
    from .recipes import load_lang_recipe

    return load_lang_recipe(model_id, iso)


def _knob(meta, name, default=None):
    """A recipe's value for *name*, falling back to the built-in default."""
    return recipe_get(meta.get("recipe"), name, default)


def load_tts_model(model_id, device="cuda", subset=None, iso=None, **kwargs):
    """Auto-detect and load a TTS model wrapper.

    Args:
        model_id: huggingface model id
        device: torch device string
        subset: ghana-sentences subset name
        iso: language being synthesised, used to find the (model, language)
            recipe whose knobs override the built-in defaults

    Returns an instance of BaseTTSModel.
    """
    meta = kwargs.pop("meta", None) or _model_meta(model_id)
    # A variant ("…-ref"/"…-noref") reports separately but loads one repo.
    model_id = meta.get("model_id", model_id)
    lower = model_id.lower()
    meta = {**meta, "recipe": _recipe_for(model_id, iso)}

    # A recipe may take over loading entirely.
    builder = recipe_get(meta["recipe"], "build_wrapper")
    if builder is not None:
        return builder(model_id, device, meta)

    if "voxcpm2" in lower or meta.get("architecture", "").startswith("VoxCPM2"):
        return VoxCPM2Wrapper(model_id, device=device, subset=subset, meta=meta,
                              iso=iso, **kwargs)
    if "ghana-tts" in lower:
        return VoxCPMWrapper(model_id, device=device, subset=subset, meta=meta,
                             iso=iso, **kwargs)
    if "f5-tts" in lower:
        if not USE_REFERENCE_AUDIO:
            raise UnsupportedModel(
                f"{model_id}: its recommended inference setting uses a reference "
                "clip, which this run has disabled"
            )
        return F5TTSWrapper(model_id, device=device, meta=meta, iso=iso, **kwargs)
    if meta.get("runner") == "omnivoice":
        return OmniVoiceWrapper(model_id, device=device, meta=meta, iso=iso, **kwargs)
    if meta.get("runner") == "coqui-vits":
        return CoquiVITSWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "stable-twi-tts" or "stable-twi-tts" in lower:
        return StableTwiWrapper(model_id, device=device, meta=meta, **kwargs)
    if "nano-twi" in lower:
        return NanoTwiWrapper(model_id, device=device, meta=meta, **kwargs)
    if lower.startswith("khaya") or "khaya" in lower:
        return KhayaTTSWrapper(model_id, device="cpu", meta=meta, **kwargs)

    if "kokoro" in lower or "sherpa" in lower:
        raise UnsupportedModel(f"{model_id}: Kokoro/sherpa models not yet supported")

    raise UnsupportedModel(f"{model_id}: unknown TTS architecture")


def _model_meta(model_id):
    """Grab metadata for *model_id* from data/tts_models.json."""
    import json

    path = Path(__file__).parent.parent / "data" / "tts_models.json"
    try:
        with open(path, encoding="utf-8") as f:
            for m in json.load(f):
                if m.get("name") == model_id:
                    return m
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def phonemize(text, iso, meta):
    """Convert orthographic text to the IPA a phoneme model expects.

    Models trained on IPA are not excluded from the benchmark: they are run
    with the G2P they were trained with, which is how anyone would use them.
    The G2P language can be overridden per (model, language) in the recipe.
    """
    from .config import ISO_TO_NAME
    from ghana_g2p import GhanaG2P

    language = _knob(meta, "G2P_LANGUAGE") or ISO_TO_NAME.get(iso, iso)
    separator = _knob(meta, "G2P_SEPARATOR", " ")
    # ghana-g2p strips punctuation by default, but a model trained with
    # punctuation kept as standalone tokens needs it: drop it and the model
    # sees a convention it never learned, with nothing to pause on.
    punctuation = _knob(meta, "G2P_PUNCTUATION", True)
    cache = meta.setdefault("_g2p_cache", {})
    if language not in cache:
        cache[language] = GhanaG2P(language)
    return cache[language].ipa(text, sep=separator, punctuation=punctuation)


class UnsupportedModel(Exception):
    pass


def _local_snapshot(model_id, token=None):
    """Download a model, skipping training-only files.

    VoxCPM repos ship the optimizer state beside the weights — 18 GB of it for
    VoxCPM2-Ghana — which inference never touches. from_pretrained would fetch
    the whole repo, so the snapshot is taken here and the local path handed on.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        ignore_patterns=["optimizer.pth", "optimizer*.pt", "*.ckpt", "samples/*"],
        token=token or None,
    )


REFERENCE_REPO = os.environ.get(
    "NSANKU_TTS_AUDIO_REPO", "ghananlpcommunity/nsanku-tts-benchmark-audio"
)


def reference_clip(iso, meta, token=None):
    """The reference clip and transcript for a language.

    Real recorded speech from ghana-speech-eval, in the language being read
    and from a different corpus than the benchmark sentences. A recipe can
    override either half with REFERENCE_CLIP / REFERENCE_TEXT.
    """
    import json

    from huggingface_hub import hf_hub_download

    override_clip = _knob(meta, "REFERENCE_CLIP")
    override_text = _knob(meta, "REFERENCE_TEXT")
    if override_clip and override_text:
        return override_clip, override_text, "recipe override"

    manifest_path = hf_hub_download(
        REFERENCE_REPO, "references/manifest.json", repo_type="dataset", token=token or None
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entry = manifest.get(iso)
    if not entry:
        raise UnsupportedModel(f"no reference clip published for {iso}")

    wav = hf_hub_download(
        REFERENCE_REPO, f"references/{entry['audio']}", repo_type="dataset", token=token or None
    )
    return override_clip or wav, override_text or entry["text"], entry["source"]


class BaseTTSModel(abc.ABC):
    """Abstract base for TTS model wrappers."""

    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device

    def prepare_text(self, text, iso):
        """Text as this model expects it.

        Two transforms, both taken from how the model was trained: phoneme
        models are given IPA, and a model trained on language-tagged text is
        given its tag. Feeding a tagged model untagged text asks it to guess
        the language from the orthography alone.
        """
        meta = getattr(self, "meta", {}) or {}
        if meta.get("input_type") == "ipa":
            text = phonemize(text, iso, meta)
        tag = _knob(meta, "LANG_TAG")
        if tag:
            text = f"{tag}{text}"
        return text

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


class VoxCPMWrapper(BaseTTSModel):
    """VoxCPM v1 (0.7B) — ghana-tts-72k / ghana-tts-36k.

    Orthographic input, 16 kHz output.  Voice-prompted through
    ``prompt_wav_path`` + ``prompt_text``: the library concatenates the two
    texts as ``prompt_text + target_text`` with no separator, so the prompt
    transcript is given a trailing space to keep the two from running
    together.
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_id, device="cuda", subset=None, meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso
        self.ref_wav = None
        self.ref_text = None
        self.ref_source = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from voxcpm import VoxCPM

        from .config import HF_TOKEN

        self.model = VoxCPM.from_pretrained(
            _local_snapshot(self.model_id, HF_TOKEN),
            device=self.device,
            # torch.compile needs a C toolchain and buys little on a
            # benchmark that loads a model once; the model cards' own
            # inference scripts disable it too.
            optimize=_knob(self.meta, "OPTIMIZE", False),
        )

        if USE_REFERENCE_AUDIO and self.meta.get("uses_reference"):
            from .config import HF_TOKEN

            self.ref_wav, self.ref_text, self.ref_source = reference_clip(
                self.iso, self.meta, HF_TOKEN
            )
        self._loaded = True
        logger.info("Loaded VoxCPM v1: %s (ref=%s)", self.model_id, bool(self.ref_wav))

    def _generate(self, text, lang):
        """Call model.generate handling the optional lang kwarg."""
        kwargs = dict(
            cfg_value=_knob(self.meta, "CFG_VALUE", 2.4 if self.ref_wav else 2.0),
            inference_timesteps=_knob(
                self.meta, "INFERENCE_TIMESTEPS", 26 if self.ref_wav else 10
            ),
            retry_badcase=_knob(self.meta, "RETRY_BADCASE", True),
        )
        if self.ref_wav:
            kwargs["prompt_wav_path"] = self.ref_wav
            if self.ref_text:
                # The library joins prompt and target with no separator.
                kwargs["prompt_text"] = self.ref_text.rstrip() + " "
        # voxcpm's generate() has no language kwarg — the model is
        # language-conditioned through the text/prompt only.
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

    def __init__(self, model_id, device="cuda", subset=None, meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso
        self.ref_wav = None
        self.ref_source = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from voxcpm import VoxCPM

        from .config import HF_TOKEN

        self.model = VoxCPM.from_pretrained(
            _local_snapshot(self.model_id, HF_TOKEN),
            load_denoiser=False,
            device=self.device,
            optimize=_knob(self.meta, "OPTIMIZE", False),
        )
        if USE_REFERENCE_AUDIO and self.meta.get("uses_reference"):
            self.ref_wav, _text, self.ref_source = reference_clip(
                self.iso, self.meta, HF_TOKEN
            )
        self._loaded = True
        logger.info("Loaded VoxCPM2: %s (ref=%s)", self.model_id, bool(self.ref_wav))

    def _generate(self, text):
        kwargs = dict(
            text=text,
            cfg_value=_knob(self.meta, "CFG_VALUE", 2.0),
            inference_timesteps=_knob(self.meta, "INFERENCE_TIMESTEPS", 15),
            retry_badcase=_knob(self.meta, "RETRY_BADCASE", False),
            max_len=_knob(self.meta, "MAX_LEN") or max(50, len(text) * 4),
            # The text normaliser is built for orthography and corrupts IPA.
            normalize=self.meta.get("input_type") != "ipa",
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
    """F5-TTS OpenBible fine-tunes — reads in the voice of a reference clip.

    The repos ship only ``model_last.pt`` + ``vocab.txt`` (no reference
    audio), so we mint one per model by synthesising ``reference_text``
    (from data/tts_models.json) with Khaya and caching the wav under
    ``$HF_HOME/nsanku-refs``.
    """

    SAMPLE_RATE = 24000
    BASE_CONFIG = "F5TTS_v1_Base"

    def __init__(self, model_id, device="cuda", meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso
        self.ref_wav = None
        self.ref_text = None
        self.ref_source = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download
        from f5_tts.api import F5TTS

        from .config import HF_TOKEN

        repo_dir = snapshot_download(
            self.model_id,
            allow_patterns=["*.pt", "vocab.txt", "*.yaml"],
            token=HF_TOKEN or None,
        )
        ckpt = os.path.join(repo_dir, "model_last.pt")
        if not os.path.exists(ckpt):
            pts = sorted(Path(repo_dir).glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"{self.model_id}: no .pt checkpoint in {repo_dir}")
            ckpt = str(pts[0])
        vocab = os.path.join(repo_dir, "vocab.txt")

        self.model = F5TTS(
            model=self.BASE_CONFIG,
            ckpt_file=ckpt,
            vocab_file=vocab if os.path.exists(vocab) else "",
            device=self.device,
        )
        self._loaded = True
        logger.info("Loaded F5-TTS: %s (ckpt=%s)", self.model_id, os.path.basename(ckpt))

    def _ensure_reference(self):
        """Return (wav_path, transcript) for this language."""
        if self.ref_wav is None:
            from .config import HF_TOKEN

            self.ref_wav, self.ref_text, self.ref_source = reference_clip(
                self.iso, self.meta, HF_TOKEN
            )
        return self.ref_wav, self.ref_text

    def synthesize(self, text, lang="eng", ref_audio=None, ref_text=None):
        self._ensure_loaded()
        if ref_audio is None:
            ref_audio, ref_text = self._ensure_reference()
        wav, sr, _spec = self.model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            speed=_knob(self.meta, "SPEED", 1.0),
            nfe_step=_knob(self.meta, "NFE_STEP", 32),
            show_info=lambda *a, **k: None,
        )
        return _wav_bytes(wav, sample_rate=sr or self.SAMPLE_RATE)


def _merged_espeak_data(model_data_dir):
    """Overlay a model's slimmed espeak-ng-data on the system one.

    nano-twi ships only the handful of lfn voice files it needs, but
    espeak-ng always loads the English dictionary at startup and aborts the
    process when ``en_dict`` is missing.  Merge the two so both are present.
    """
    import glob
    import shutil

    # Debian puts it under /usr/lib/<triplet>/, other distros under /usr/share.
    candidates = ["/usr/share/espeak-ng-data", *glob.glob("/usr/lib/*/espeak-ng-data")]
    system_dir = next((d for d in candidates if os.path.isdir(d)), None)
    if system_dir is None:
        logger.warning("No system espeak-ng-data found; using the model's slimmed copy")
        return model_data_dir

    merged = Path(_cache_root()) / "espeak-ng-data"
    if not (merged / "en_dict").exists():
        merged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(system_dir, merged, dirs_exist_ok=True)
    shutil.copytree(model_data_dir, merged, dirs_exist_ok=True)
    return str(merged)


class OmniVoiceWrapper(BaseTTSModel):
    """k2-fsa/OmniVoice — voice cloning from a reference clip, 24 kHz.

    Needs transformers 5.x, so it runs in its own image (see the "stack"
    field in the registry).
    """

    SAMPLE_RATE = 24000

    def __init__(self, model_id, device="cuda", meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso
        self.ref_wav = None
        self.ref_text = None
        self.ref_source = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from omnivoice import OmniVoice

        self.model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        )
        self._loaded = True
        logger.info("Loaded OmniVoice: %s", self.model_id)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        if self.ref_wav is None:
            from .config import HF_TOKEN

            self.ref_wav, self.ref_text, self.ref_source = reference_clip(
                self.iso, self.meta, HF_TOKEN
            )
        audio = self.model.generate(
            text=text, ref_audio=self.ref_wav, ref_text=self.ref_text
        )
        if isinstance(audio, (list, tuple)):
            if not audio:
                raise RuntimeError("OmniVoice returned no audio")
            audio = audio[0]
        return _wav_bytes(np.asarray(audio, dtype=np.float32), sample_rate=self.SAMPLE_RATE)


class CoquiVITSWrapper(BaseTTSModel):
    """Coqui-TTS VITS checkpoints (multilingual-tts OpenBible voices).

    The repo ships a raw training checkpoint — config.json, model_last.pth and
    a speakers file — rather than anything loadable by name, so it is driven
    through Coqui's Synthesizer directly.
    """

    SAMPLE_RATE = 22050

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download
        from TTS.utils.synthesizer import Synthesizer

        from .config import HF_TOKEN

        repo = Path(snapshot_download(
            self.model_id,
            allow_patterns=["*.json", "*.pth"],
            token=HF_TOKEN or None,
        ))
        checkpoint = next(iter(sorted(repo.glob("model*.pth"))), None)
        if checkpoint is None:
            raise FileNotFoundError(f"{self.model_id}: no model*.pth checkpoint")
        speakers = repo / "speakers.pth"

        self.model = Synthesizer(
            tts_checkpoint=str(checkpoint),
            tts_config_path=str(repo / "config.json"),
            tts_speakers_file=str(speakers) if speakers.exists() else None,
            use_cuda=self.device.startswith("cuda"),
        )
        self.SAMPLE_RATE = getattr(self.model, "output_sample_rate", self.SAMPLE_RATE)
        self._loaded = True
        logger.info("Loaded Coqui VITS: %s", self.model_id)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        kwargs = {}
        speaker = _knob(self.meta, "SPEAKER")
        if speaker:
            kwargs["speaker_name"] = speaker
        elif getattr(self.model.tts_model, "num_speakers", 0) > 1:
            # A multi-speaker checkpoint refuses to synthesise without one.
            names = list(getattr(self.model.tts_model.speaker_manager, "name_to_id", {}))
            if names:
                kwargs["speaker_name"] = names[0]
        wav = self.model.tts(text, **kwargs)
        return _wav_bytes(np.asarray(wav, dtype=np.float32), sample_rate=self.SAMPLE_RATE)


class StableTwiWrapper(BaseTTSModel):
    """ghananlpcommunity/stable-twi-tts — Piper VITS, ONNX, CPU.

    IPA-driven, but it phonemises internally, so it takes orthographic text.
    Twelve voices are exposed and they disagree sharply about what is best:
    the strongest code-switching voice is 21st of 30 on pure Twi, so the voice
    is a per-language recipe knob rather than a fixed default.
    """

    SAMPLE_RATE = 22050

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, "cpu")  # ONNX, CPU
        self.model = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        from huggingface_hub import snapshot_download
        from stable_twi_tts import StableTwiTTS

        from .config import HF_TOKEN

        model_dir = snapshot_download(
            self.model_id,
            allow_patterns=["*.onnx", "*.json", "tokens.txt"],
            token=HF_TOKEN or None,
        )
        self.model = StableTwiTTS(model_dir)
        self._loaded = True
        logger.info("Loaded stable-twi-tts: %s", self.model_id)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        out = self.model.synthesize(
            text,
            voice=_knob(self.meta, "VOICE", "twi-6"),
            language=_knob(self.meta, "SYNTH_LANGUAGE", "twi"),
            length_scale=_knob(self.meta, "LENGTH_SCALE", 1.0),
            noise_scale=_knob(self.meta, "NOISE_SCALE", 0.667),
            noise_w=_knob(self.meta, "NOISE_W", 0.8),
        )
        return _wav_bytes(out.audio, sample_rate=out.sample_rate or self.SAMPLE_RATE)


class NanoTwiWrapper(BaseTTSModel):
    """ghananlpcommunity/nano-twi — Matcha-TTS + Vocos ONNX (CPU).

    Repo layout (``sherpa-onnx/`` subdir): ``twi_ep045_steps4.onnx``
    (acoustic), ``vocos-22khz-univ.onnx`` (vocoder), ``tokens.txt``,
    ``espeak-ng-data/``.
    """

    SAMPLE_RATE = 22050
    ACOUSTIC = "twi_ep045_steps4.onnx"
    VOCODER = "vocos-22khz-univ.onnx"

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, "cpu")  # runs on CPU
        self.model = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        import sherpa_onnx
        from huggingface_hub import snapshot_download

        from .config import HF_TOKEN

        repo_dir = snapshot_download(
            self.model_id,
            allow_patterns=["sherpa-onnx/*"],
            token=HF_TOKEN or None,
        )
        base = os.path.join(repo_dir, "sherpa-onnx")
        data_dir = _merged_espeak_data(os.path.join(base, "espeak-ng-data"))

        # sherpa-onnx >=1.13: field is `acoustic_model`, and the vocoder,
        # tokens and espeak data live inside the Matcha config itself.
        matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=os.path.join(base, self.ACOUSTIC),
            vocoder=os.path.join(base, self.VOCODER),
            tokens=os.path.join(base, "tokens.txt"),
            data_dir=data_dir,
            noise_scale=_knob(self.meta, "NOISE_SCALE", 1.0),
            length_scale=_knob(self.meta, "LENGTH_SCALE", 1.0),
        )
        self.model = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(matcha=matcha, num_threads=4),
            )
        )
        self._loaded = True
        logger.info("Loaded nano-twi from %s", base)

    def synthesize(self, text, lang="eng"):
        self._ensure_loaded()
        audio = self.model.generate(
            text, sid=_knob(self.meta, "SPEAKER_ID", 0) or 0, speed=1.0
        )
        return _wav_bytes(audio.samples, sample_rate=audio.sample_rate or self.SAMPLE_RATE)


class KhayaTTSWrapper(BaseTTSModel):
    """Khaya AI TTS v2 — hosted REST endpoint.

    Endpoint: POST https://translation-api.ghananlp.org/tts/v2/synthesize
    Auth header: Ocp-Apim-Subscription-Key: <KHAYA_API_KEY>
    Body: {"text": ..., "language": <ISO 639-3>, "format": "wav"}
    """

    API_URL = "https://translation-api.ghananlp.org/tts/v2/synthesize"

    def __init__(self, model_id="KhayaAI/khaya-tts", device="cpu", meta=None, **kwargs):
        super().__init__(model_id, "cpu")
        self.meta = meta or {}
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
            "language": _knob(self.meta, "LANGUAGE_CODE", lang) or lang,
            "format": output_format,
        }
        speaker_id = speaker_id or _knob(self.meta, "SPEAKER_ID")
        if speaker_id:
            body["speaker_id"] = speaker_id
        resp = session.post(self.API_URL, json=body, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Khaya TTS {resp.status_code}: {resp.text[:200]}")
        return resp.content

def _check_wrappers():
    """Every wrapper the dispatcher can return must exist.

    A missing one used to surface as NameError once per sample, inside a job,
    after the model had downloaded — 200 identical failures for a typo.
    """
    import re

    source = Path(__file__).read_text(encoding="utf-8")
    named = set(re.findall(r"return (\w+Wrapper)\(", source))
    missing = sorted(n for n in named if n not in globals())
    if missing:
        raise RuntimeError(f"models.py dispatches to undefined wrapper(s): {missing}")


_check_wrappers()
