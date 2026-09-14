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
import threading
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
    if meta.get("runner") == "orpheus":
        return OrpheusWrapper(model_id, device=device, meta=meta, iso=iso, **kwargs)
    if meta.get("runner") == "coqui-vits":
        return CoquiVITSWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "mms-tts" or lower.startswith("facebook/mms-tts"):
        return MmsTTSWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "stable-twi-tts" or "stable-twi-tts" in lower:
        return StableTwiWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "kasanoma" or "kasanoma" in lower:
        return KasanomaWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "transformers-vits" or "tekyerema" in lower:
        return TransformersVitsWrapper(model_id, device=device, meta=meta, **kwargs)
    if meta.get("runner") == "spark-tts" or "spark-tts" in lower:
        return SparkTTSWrapper(model_id, device=device, meta=meta, iso=iso, **kwargs)
    if "nano-twi" in lower:
        return NanoTwiWrapper(model_id, device=device, meta=meta, **kwargs)
    if lower.startswith("khaya") or "khaya" in lower:
        return KhayaTTSWrapper(model_id, device="cpu", meta=meta, **kwargs)
    if "gemini" in lower and "tts" in lower:
        return GeminiTTSWrapper(model_id, device="cpu", meta=meta, **kwargs)

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
    """k2-fsa/OmniVoice — voice cloning and voice design, 24 kHz.

    In the default (ref) mode it clones the voice of a reference clip. In
    design mode it reads the voice from the ``VOICE_DESIGN`` instruct text
    instead and lets OmniVoice auto-detect everything else (language, any
    attribute the instruct doesn't mention). Needs transformers 5.x, so it
    runs in its own image (see the "stack" field in the registry).
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
        mode = self.meta.get("mode")
        if mode == "design":
            # Voice design: only text (required) and the design instruct are
            # set; language and everything unmentioned is auto-detected by the
            # model (instruct=None means "auto" for the whole voice too).
            instruct = _knob(self.meta, "VOICE_DESIGN")
            audio = self.model.generate(text=text, instruct=instruct)
        elif mode == "noref":
            # Pure auto: the model picks the voice (and detects the language)
            # itself — neither a reference clip nor a voice design.
            audio = self.model.generate(text=text)
        else:
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


class OrpheusWrapper(BaseTTSModel):
    """Sunbird/orpheus-3b-tts-multilingual — autoregressive LLM over SNAC codes.

    A Llama-3.2-3B fine-tune that emits SNAC audio-codebook tokens, decoded to
    24 kHz speech.  There is no language knob: the voice — and therefore the
    language — travels through the ``speaker_id`` prompt tag, e.g.
    ``slr129_ewe_0001: <text>`` for Ewe.  Which speaker a language uses is a
    per-language recipe knob (SPEAKER_ID), since Orpheus has no other
    language-conditioning input.
    """

    SAMPLE_RATE = 24000

    # Special tokens from the model card; the tokenizer knows their ids.
    END_OF_TEXT = 128009
    START_OF_SPEECH = 128257
    END_OF_SPEECH = 128258
    START_OF_HUMAN = 128259
    END_OF_HUMAN = 128260
    AUDIO_TOKEN_LO = 128266
    AUDIO_TOKEN_HI = 128266 + 7 * 4096

    def __init__(self, model_id, device="cuda", meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self.tokenizer = None
        self.snac = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from huggingface_hub import snapshot_download
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .config import HF_TOKEN

        # SNAC decoding is tiny and runs fine on CPU, freeing the GPU for the LM.
        snac_dir = snapshot_download("hubertsiuzdak/snac_24khz", token=HF_TOKEN or None)
        self.snac = SNAC.from_pretrained(snac_dir).to("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=HF_TOKEN or None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device.startswith("cuda") else "cpu",
            token=HF_TOKEN or None,
        ).eval()
        self._loaded = True
        logger.info("Loaded Orpheus: %s", self.model_id)

    def _synthesize(self, text, speaker_id):
        import numpy as np
        import torch

        tagged = f"{speaker_id}: {text}"
        text_ids = self.tokenizer(tagged, return_tensors="pt").input_ids
        soh = torch.tensor([[self.START_OF_HUMAN]], dtype=torch.int64)
        end = torch.tensor(
            [[self.END_OF_TEXT, self.END_OF_HUMAN]], dtype=torch.int64
        )
        input_ids = torch.cat([soh, text_ids, end], dim=1)
        if self.device.startswith("cuda"):
            input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        seed = _knob(self.meta, "SEED", 42)
        if seed is not None:
            torch.manual_seed(seed)

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=_knob(self.meta, "MAX_NEW_TOKENS", 1200),
            do_sample=True,
            temperature=_knob(self.meta, "TEMPERATURE", 0.6),
            top_p=_knob(self.meta, "TOP_P", 0.95),
            repetition_penalty=_knob(self.meta, "REPETITION_PENALTY", 1.1),
            eos_token_id=self.END_OF_SPEECH,
            use_cache=True,
        )

        # Crop on the last START_OF_SPEECH, keep only audio codebook tokens.
        generated = generated.to("cpu")
        sos = (generated == self.START_OF_SPEECH).nonzero(as_tuple=True)
        if len(sos[1]) > 0:
            generated = generated[:, sos[1][-1].item() + 1:]
        row = generated[0]
        audio = row[(row >= self.AUDIO_TOKEN_LO) & (row < self.AUDIO_TOKEN_HI)] \
            - self.AUDIO_TOKEN_LO
        n = (audio.size(0) // 7) * 7
        if n == 0:
            return _wav_bytes(np.zeros(12000, dtype=np.float32), sample_rate=self.SAMPLE_RATE)

        # Redistribute the flattened 7-code frames into SNAC's 3-layer layout.
        l1, l2, l3 = [], [], []
        for i in range(n // 7):
            l1.append(audio[7 * i].item())
            l2.append(audio[7 * i + 1].item() - 4096)
            l3.extend([
                audio[7 * i + 2].item() - 2 * 4096,
                audio[7 * i + 3].item() - 3 * 4096,
            ])
            l2.append(audio[7 * i + 4].item() - 4 * 4096)
            l3.extend([
                audio[7 * i + 5].item() - 5 * 4096,
                audio[7 * i + 6].item() - 6 * 4096,
            ])
        clamp = lambda vals: [max(0, min(4095, int(v))) for v in vals]
        codes = [
            torch.tensor(clamp(l1), dtype=torch.int32).unsqueeze(0),
            torch.tensor(clamp(l2), dtype=torch.int32).unsqueeze(0),
            torch.tensor(clamp(l3), dtype=torch.int32).unsqueeze(0),
        ]
        waveform = self.snac.decode(codes)
        return _wav_bytes(
            waveform.detach().squeeze().to("cpu").numpy().astype(np.float32),
            sample_rate=self.SAMPLE_RATE,
        )

    def synthesize(self, text, lang="ewe"):
        self._ensure_loaded()
        speaker_id = _knob(self.meta, "SPEAKER_ID")
        if not speaker_id:
            raise RuntimeError(
                "orpheus-3b needs a SPEAKER_ID recipe knob per language "
                "(it has no language-conditioning input of its own)"
            )
        return self._synthesize(text, speaker_id)


class MmsTTSWrapper(BaseTTSModel):
    """facebook/mms-tts-* — Meta MMS per-language VITS checkpoints, 16 kHz.

    Native to transformers (VitsModel): a char-level tokenizer and a single
    speaker, so no recipe knobs are needed and text is read as spelled.
    """

    SAMPLE_RATE = 16000

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = VitsModel.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Loaded MMS-TTS: %s", self.model_id)

    def synthesize(self, text, lang="eng"):
        self._ensure_loaded()
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        if self.device.startswith("cuda"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            wav = self.model(**inputs).waveform
        wav = np.asarray(wav.detach().cpu().numpy().squeeze(), dtype=np.float32)
        sr = getattr(self.model.config, "sampling_rate", None) or self.SAMPLE_RATE
        return _wav_bytes(wav, sample_rate=sr)


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


class KasanomaWrapper(BaseTTSModel):
    """AfriSpeech Kasanoma Twi — Piper VITS ONNX (CPU)."""

    SAMPLE_RATE = 16000

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, "cpu")  # ONNX, CPU
        self.model = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        import json
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from .config import HF_TOKEN

        model_dir = Path(snapshot_download(
            self.model_id,
            allow_patterns=["model.onnx", "model.onnx.json"],
            token=HF_TOKEN or None,
        ))
        self.dir = model_dir
        cfg_path = model_dir / "model.onnx.json"
        self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.id_map = self.config["phoneme_id_map"]
        self.espeak_voice = self.config["espeak"]["voice"]
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self._loaded = True
        logger.info("Loaded Kasanoma model: %s", self.model_id)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        import subprocess
        import numpy as np

        cmd = ["espeak-ng", "-v", self.espeak_voice, "-q", "--ipa", text]
        res = subprocess.run(cmd, capture_output=True, text=True)
        phonemes = res.stdout.strip()

        ids = []
        for char in phonemes:
            if char in self.id_map:
                ids.extend(self.id_map[char])
        if "$" in self.id_map:
            ids = self.id_map["$"] + ids + self.id_map["$"]
        if not ids:
            raise ValueError(f"no pronounceable phonemes in {text!r}")

        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([
                _knob(self.meta, "NOISE_SCALE", 0.667),
                _knob(self.meta, "LENGTH_SCALE", 1.0),
                _knob(self.meta, "NOISE_W", 0.8),
            ], dtype=np.float32),
        }
        out = self.session.run(None, feed)[0]
        audio = np.asarray(out, dtype=np.float32).squeeze()
        return _wav_bytes(audio, sample_rate=self.sample_rate)


class TransformersVitsWrapper(BaseTTSModel):
    """Hugging Face Transformers VitsModel (CPU/CUDA)."""

    def __init__(self, model_id, device="cuda", meta=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.meta = meta or {}

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel
        from .config import HF_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=HF_TOKEN or None
        )
        self.model = VitsModel.from_pretrained(
            self.model_id, token=HF_TOKEN or None
        ).to(self.device)
        self.model.eval()
        self.sample_rate = getattr(self.model.config, "sampling_rate", 16000)
        self._loaded = True
        logger.info("Loaded Transformers VitsModel: %s on %s", self.model_id, self.device)

    def synthesize(self, text, lang="twi"):
        self._ensure_loaded()
        import torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        audio = output[0].cpu().numpy()
        return _wav_bytes(audio, sample_rate=self.sample_rate)


class SparkTTSWrapper(BaseTTSModel):
    """walusungungulube/Spark-TTS-0.5B-twi-ewe-dagbani — Spark-TTS (CUDA/CPU)."""

    def __init__(self, model_id, device="cuda", meta=None, iso=None, **kwargs):
        super().__init__(model_id, device)
        self.model = None
        self._loaded = False
        self.meta = meta or {}
        self.iso = iso

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        from huggingface_hub import snapshot_download
        from .config import HF_TOKEN
        from .sparktts.cli.SparkTTS import SparkTTS

        base_model_id = self.model_id.replace("-ref", "").replace("-noref", "")
        model_dir = snapshot_download(base_model_id, token=HF_TOKEN or None)
        self.model_dir = model_dir
        dev = torch.device(self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu")
        self.spark = SparkTTS(Path(model_dir), device=dev)
        self.sample_rate = self.spark.sample_rate
        self._loaded = True
        logger.info("Loaded SparkTTS model: %s on %s", base_model_id, dev)

    def synthesize(self, text, lang="twi", ref_audio=None, ref_text=None):
        self._ensure_loaded()
        import torch

        mode = self.meta.get("mode") or ("noref" if "-noref" in self.model_id else "ref")

        if mode == "ref":
            if ref_audio is None and self.iso:
                ref_audio, ref_text, _ = reference_clip(self.iso, self.meta)
            if ref_audio is not None:
                wav = self.spark.inference(
                    text=text,
                    prompt_speech_path=Path(ref_audio),
                    prompt_text=ref_text,
                )
            else:
                wav = self.spark.inference(
                    text=text,
                    gender=_knob(self.meta, "GENDER", "female"),
                    pitch=_knob(self.meta, "PITCH", "moderate"),
                    speed=_knob(self.meta, "SPEED", "moderate"),
                )
        else:
            wav = self.spark.inference(
                text=text,
                gender=_knob(self.meta, "GENDER", "female"),
                pitch=_knob(self.meta, "PITCH", "moderate"),
                speed=_knob(self.meta, "SPEED", "moderate"),
            )

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        audio = wav.squeeze()
        return _wav_bytes(audio, sample_rate=self.sample_rate)


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


class GeminiTTSWrapper(BaseTTSModel):
    """Google Gemini 3.1 Flash TTS — hosted `generateContent` REST API.

    Endpoint: POST https://generativelanguage.googleapis.com/v1beta/
             models/gemini-3.1-flash-tts-preview:generateContent
    Auth header: x-goog-api-key: <GEMINI_API_KEY>
    Body: {"contents":[{"parts":[{"text": ...}],"role":"user"}],
           "generationConfig":{
             "responseModalities":["AUDIO"],
             "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":
                          {"voice_name":"Zephyr","language_code": None}}}}
    Response: candidates[0].content.parts[0].inlineData.data is base64 PCM
              audio (L16, 24kHz, 1ch, mono, 16-bit) — no WAV header, so we
              build one at 24000 Hz.

    Language is auto-detected by the model from the input text (BCP-47 not
    required), which is exactly why a single wrapper covers every benchmark
    language with no per-language recipe.
    """

    API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-flash-tts-preview:generateContent"
    )

    # Token bucket: 200 requests / minute, burst 60. Whatever the job's
    # concurrency, we never exceed the paid-plan rpm.
    _bucket = {"tokens": 60.0, "last": 0.0, "lock": threading.Lock()}
    RATE = 200.0 / 60.0  # tokens per second

    def __init__(self, model_id="Google/gemini-3.1-flash-tts-preview",
                 device="cpu", meta=None, **kwargs):
        super().__init__(model_id, "cpu")
        self.meta = meta or {}
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set — required for Gemini TTS (export it "
                "at job runtime; never commit the key)"
            )
        self._session = None
        self._session_name = _knob(self.meta, "SESSION_NAME", "Ask a friend")
        self._voice_name = _knob(self.meta, "VOICE_NAME", None)

    def _ensure_session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            })
        return self._session

    def _acquire(self):
        """Block until a request token is available (200 rpm, burst 60)."""
        import time

        now = time.monotonic()
        b = self._bucket
        with b["lock"]:
            if b["last"]:
                b["tokens"] = min(60.0, b["tokens"] + (now - b["last"]) * self.RATE)
            b["last"] = now
            if b["tokens"] >= 1.0:
                b["tokens"] -= 1.0
                return
        time.sleep(0.05)
        self._acquire()

    def synthesize(self, text, lang=None, speaker_id=None, output_format="wav"):
        import base64

        self._acquire()
        session = self._ensure_session()
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voice_name": self._voice_name or "Zephyr",
                        }
                    }
                }
            },
        }
        resp = session.post(self.API_URL, json=body, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504):
            # Shared-key 429s, or a flaky preview — back off and retry. Gemini
            # throttles globally per key at 200 rpm across every parallel
            # language job, so one job's 429 is NOT a per-model error.
            time.sleep(5.0)
            return self.synthesize(text, lang=lang, speaker_id=speaker_id,
                                   output_format=output_format)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini TTS {resp.status_code}: {resp.text[:200]}")
        data = resp.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        pcm = np.frombuffer(base64.b64decode(data), dtype=np.int16).astype(np.float32) / 32768.0
        return _wav_bytes(pcm, sample_rate=24000)


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
