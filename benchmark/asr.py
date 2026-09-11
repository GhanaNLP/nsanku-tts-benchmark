"""ASR judges used to score TTS intelligibility.

Each language is judged by the lowest-CER model for that language on the
GhanaNLP ASR leaderboard (see data/asr_judges.json).  A judge exposes:

    judge = load_judge(iso)
    text = judge.transcribe(wav_bytes)

The runners mirror the reference implementations in the ASR benchmark
(github.com/GhanaNLP/nsanku-asr-benchmark, recipes/ + benchmark/) so a TTS run and an
ASR run transcribe audio the same way.
"""

import gc
import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

JUDGE_SAMPLE_RATE = 16000


def _judges():
    path = Path(__file__).parent.parent / "data" / "asr_judges.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["judges"]


def judge_for(iso):
    """Return the judge spec for *iso*, or None if the language has none."""
    return _judges().get(iso)


def load_judge(iso, device="cuda"):
    """Instantiate the ASR judge for *iso*."""
    spec = judge_for(iso)
    if spec is None:
        raise UnsupportedJudge(f"{iso}: no ASR judge configured")

    kind = spec["kind"]
    if kind == "khaya":
        return KhayaASR(spec)
    if kind == "griot":
        return GriotASR(spec, device=device)
    if kind == "omniasr":
        return OmniASR(spec, device=device)
    raise UnsupportedJudge(f"{iso}: unknown judge kind {kind!r}")


class UnsupportedJudge(Exception):
    pass


def _to_mono_16k(wav_bytes):
    """Decode WAV bytes to a mono float32 array at the judge's sample rate."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != JUDGE_SAMPLE_RATE:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, JUDGE_SAMPLE_RATE
        ).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def _encode_wav(audio, sample_rate=JUDGE_SAMPLE_RATE):
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class BaseJudge:
    """Common surface: bytes in, transcript out."""

    model_id = None

    def transcribe(self, wav_bytes):
        raise NotImplementedError

    def transcribe_many(self, wav_bytes_list):
        """Transcribe a list of clips. "" marks one the judge could not read."""
        out = []
        for wav_bytes in wav_bytes_list:
            try:
                out.append(self.transcribe(wav_bytes))
            except Exception:
                out.append("")
        return out

    def cleanup(self):
        pass


class KhayaASR(BaseJudge):
    """Hosted Khaya ASR v3.

    POST /asr/v3/transcribe?language=<code>, body audio/wav -> {"text": ...}.
    An empty transcript is a failure, not a zero-effort answer, so requests
    are retried before giving up.
    """

    API_URL = "https://translation-api.ghananlp.org/asr/v3/transcribe"
    MAX_RETRIES = 6

    def __init__(self, spec):
        self.model_id = spec["model"]
        self.code = spec["code"]
        self.api_key = os.environ.get("KHAYA_API_KEY", "")
        if not self.api_key:
            raise ValueError("KHAYA_API_KEY not set — required for the Khaya ASR judge")
        self._session = None

    def _get_session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "audio/wav",
            })
        return self._session

    def transcribe(self, wav_bytes):
        import time

        # Khaya expects 16 kHz PCM; TTS output ranges from 16 to 48 kHz.
        wav_bytes = _encode_wav(_to_mono_16k(wav_bytes))
        session = self._get_session()
        url = f"{self.API_URL}?language={self.code}"
        for attempt in range(self.MAX_RETRIES):
            try:
                r = session.post(url, data=wav_bytes, timeout=120)
                if r.status_code == 200:
                    return (r.json().get("text") or "").strip()
                if r.status_code in (429, 500, 503) and attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"Khaya ASR {r.status_code}: {r.text[:200]}")
            except Exception:
                if attempt >= self.MAX_RETRIES - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return ""


class GriotASR(BaseJudge):
    """Qlerqly/griot-nano-1 — 153M Conformer CTC, greedy decoding.

    The repo ships its own ``src/conformer_ctc`` package; the model does not
    load through transformers.
    """

    def __init__(self, spec, device="cuda"):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        from .config import HF_TOKEN

        self.model_id = spec["model"]
        self.device = torch.device(device)

        model_dir = Path(snapshot_download(
            self.model_id,
            allow_patterns=[
                "config.json", "vocab.json", "tokenizer.json",
                "model.safetensors", "src/**", "*.json",
            ],
            token=HF_TOKEN or None,
        ))
        # The author's model code lives inside the repo, not in a package.
        sys.path.insert(0, str(model_dir / "src"))
        from conformer_ctc.data import FeatureConfig
        from conformer_ctc.model import ConformerCTC, ConformerCTCConfig

        config = ConformerCTCConfig(**json.loads((model_dir / "config.json").read_text()))
        vocab = {
            str(tok): int(i)
            for tok, i in json.loads((model_dir / "vocab.json").read_text()).items()
        }
        self.id_to_token = {int(i): str(tok) for tok, i in vocab.items()}
        self.feat = FeatureConfig(sample_rate=JUDGE_SAMPLE_RATE, n_mels=config.n_mels)

        self.model = ConformerCTC(config)
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
        self.model.to(self.device).eval()
        logger.info("Loaded ASR judge %s", self.model_id)

    def transcribe(self, wav_bytes):
        import torch
        from conformer_ctc.data import audio_to_log_mel
        from conformer_ctc.model import greedy_decode

        audio = _to_mono_16k(wav_bytes)
        with torch.inference_mode():
            features = audio_to_log_mel(audio, JUDGE_SAMPLE_RATE, self.feat)
            lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)
            output = self.model(features.unsqueeze(0).to(device=self.device), lengths)
            text = greedy_decode(
                output.logits.cpu(),
                output.output_lengths.cpu(),
                self.id_to_token,
                blank_id=self.model.config.blank_id,
                pad_id=self.model.config.pad_id,
            )[0]
        return (text or "").strip()

    def cleanup(self):
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OmniASR(BaseJudge):
    """Meta Omnilingual ASR, LLM decoder (facebook/omniASR-LLM-7B-v2).

    Language-conditioned through ids like ``ewe_Latn``.
    """

    BATCH_SIZE = 8

    def __init__(self, spec, device="cuda"):
        import torch
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.model_id = spec["model"]
        self.lang_id = spec["code"]
        card = self.model_id.split("/")[-1].replace("-", "_")
        self.pipeline = ASRInferencePipeline(card, device=torch.device(device), dtype=torch.bfloat16)
        logger.info("Loaded ASR judge %s (%s)", self.model_id, self.lang_id)

    def _item(self, wav_bytes):
        import torch

        # 1-D (time,) — the pipeline's collater reads axis 0 as time, so a
        # (1, N) channel-major clip would look like a single frame.
        waveform = torch.as_tensor(_to_mono_16k(wav_bytes), dtype=torch.float32).flatten()
        return {"waveform": waveform, "sample_rate": JUDGE_SAMPLE_RATE}

    def transcribe(self, wav_bytes):
        out = self.pipeline.transcribe(
            [self._item(wav_bytes)], lang=[self.lang_id], batch_size=1
        )
        return (out[0] or "").strip() if out else ""

    def transcribe_many(self, wav_bytes_list):
        results = []
        for start in range(0, len(wav_bytes_list), self.BATCH_SIZE):
            chunk = [self._item(b) for b in wav_bytes_list[start:start + self.BATCH_SIZE]]
            langs = [self.lang_id] * len(chunk)
            try:
                out = self.pipeline.transcribe(chunk, lang=langs, batch_size=len(chunk))
                results.extend((t or "").strip() for t in out)
            except Exception:
                # One bad clip would otherwise lose its whole batch — fall
                # back to clip-by-clip so only genuinely bad clips come back
                # empty.
                for item in chunk:
                    try:
                        out = self.pipeline.transcribe(
                            [item], lang=[self.lang_id], batch_size=1
                        )
                        results.append((out[0] or "").strip() if out else "")
                    except Exception:
                        results.append("")
        return results

    def cleanup(self):
        import torch

        del self.pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
