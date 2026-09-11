"""Modal runner for the nsanku-TTS benchmark.

Two stages, because the ASR judges cannot share an environment with the TTS
models — omniASR pins torch 2.8 through fairseq2 while voxcpm/f5-tts run on
torch 2.5 — and because re-scoring should never mean re-synthesising:

    stage 1  synthesize   GPU, tts_image   ->  /results/audio/{iso}/{model}/
    stage 2  score        GPU/CPU, asr_image -> /results/{iso}.yaml

Usage:
    modal run modal_app.py                          # all languages, both stages
    modal run modal_app.py --langs dag ewe          # specific languages
    modal run modal_app.py --model f5               # filter TTS models
    modal run modal_app.py --samples 500            # bump the sample count
    modal run modal_app.py --stage score            # re-score existing clips

Secrets (Modal secret named "nsanku-khaya"): KHAYA_API_KEY, HF_TOKEN.
"""

import os
import sys

import modal

app = modal.App("nsanku-tts-benchmark")

results_volume = modal.Volume.from_name("nsanku-tts-results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("nsanku-tts-hf-cache", create_if_missing=True)

RESULTS_DIR = "/results"
HF_HOME = "/hf-cache"
# Synthesised clips live on the results Volume, keyed by language and model.
AUDIO_SUBDIR = "/results/audio"
# Per-sample ASR output, kept next to the results it explains.
TRANSCRIPTIONS_SUBDIR = "/results/transcriptions"

_IGNORED = {".env", ".git", "benchmarks", "__pycache__", ".venv", "space", "audio"}

# FunAudioLLM/CosyVoice @ 2026-05-25 — see cosy_image below.
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"


def _with_repo(image):
    """Bake the repo in so `benchmark` and `data/` are importable."""
    return image.add_local_dir(
        ".",
        "/workspace",
        ignore=lambda p: any(part in _IGNORED for part in p.parts),
    )


# ── Images ───────────────────────────────────────────────────────────────────
# Versions are pinned and installed in separate steps so pip does not try to
# backtrack across incompatible ranges (transformers 5.x <-> tokenizers,
# f5-tts gradio <-> pydantic, voxcpm datasets).
tts_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg: pydub, used by F5-TTS to load the reference clip.
    # espeak-ng-data: nano-twi ships a slimmed data dir missing en_dict.
    .apt_install("ffmpeg", "espeak-ng-data")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1")
    .pip_install(
        "transformers==4.57.2",
        "datasets==3.6.0",
        "huggingface_hub==0.36.2",
        "numpy",
        "pyyaml",
        "requests",
        "soundfile",
    )
    .pip_install("voxcpm==2.0.3")
    .pip_install("f5-tts==1.0.3")
    .pip_install("sherpa-onnx")
    # IPA models are phonemised with the G2P they were trained on.
    .pip_install("ghana-g2p")
    # Base image locale is POSIX → Python defaults to ascii for text IO,
    # which breaks reading the Twi/Ewe YAML results.
    .env({"PYTHONUTF8": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
)

# The judge environment, following the ASR benchmark's own env recipe
# (github.com/GhanaNLP/nsanku-asr-benchmark, run_omniasr.py): fairseq2 pins torch 2.8.
asr_image = _with_repo(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install("fairseq2==0.6", "omnilingual-asr==0.2.0")
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub",
        "numpy",
        "pyyaml",
        "requests",
        "safetensors",
        "soundfile",
        # griot-nano-1's in-repo conformer code imports it.
        "tokenizers",
    )
    .env({"PYTHONUTF8": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
)


# CosyVoice is not on PyPI (the PyPI "cosyvoice" is an unrelated 2024 fork
# predating CosyVoice2), so the official repo is cloned with its vendored
# Matcha-TTS.  Only the inference subset of its requirements is installed:
# TensorRT is imported lazily, and deepspeed/gradio/tensorboard are training
# and demo-only — together they would add gigabytes for nothing.
cosy_image = _with_repo(
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "sox", "libsox-dev", "ffmpeg", "build-essential")
    # Pinned: upstream renamed inference_zero_shot's prompt_speech_16k to
    # prompt_wav under main, silently breaking every clip. A benchmark has to
    # keep producing the same numbers months from now.
    .run_commands(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        f"cd /opt/CosyVoice && git checkout {COSYVOICE_COMMIT} && "
        "git submodule update --init --recursive",
    )
    .pip_install(
        "torch==2.3.1", "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # openai-whisper ships only an sdist whose setup.py imports
    # pkg_resources, which the isolated build env's setuptools no longer
    # provides — build it against the environment instead.
    .pip_install("setuptools<81", "wheel")
    .run_commands("pip install --no-build-isolation openai-whisper==20231117")
    .pip_install(
        "conformer==0.3.2",
        "diffusers==0.29.0",
        "gdown==5.1.0",
        "HyperPyYAML==1.2.3",
        "hydra-core==1.3.2",
        "inflect==7.3.1",
        "librosa==0.10.2",
        "lightning==2.2.4",
        "matplotlib==3.7.5",
        "modelscope==1.20.0",
        "networkx==3.1",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "onnx==1.16.0",
        "onnxruntime-gpu==1.18.0",
        "protobuf==4.25",
        "pydantic==2.7.0",
        "pyworld==0.3.4",
        "rich==13.7.1",
        "soundfile==0.12.1",
        "transformers==4.51.3",
        "wetext==0.0.4",
        "wget==3.2",
        "x-transformers==2.11.24",
        extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    .pip_install("huggingface_hub", "pyyaml", "requests", "datasets==3.6.0")
    .env({"PYTHONUTF8": "1", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
)

VOLUMES = {RESULTS_DIR: results_volume, HF_HOME: hf_cache_volume}
SECRETS = [modal.Secret.from_name("nsanku-khaya")]


def _prepare_env(samples):
    sys.path.insert(0, "/workspace")
    os.environ["HF_HOME"] = HF_HOME
    os.environ["NSANKU_TTS_RESULTS_DIR"] = RESULTS_DIR
    os.environ["NSANKU_TTS_AUDIO_DIR"] = AUDIO_SUBDIR
    os.environ["NSANKU_TTS_TRANSCRIPTIONS_DIR"] = TRANSCRIPTIONS_SUBDIR
    if samples:
        os.environ["NSANKU_TTS_NUM_SAMPLES"] = str(samples)


@app.function(
    image=tts_image,
    gpu="A10G",
    # 200 samples on the slowest models needs far more than 4h.
    timeout=60 * 60 * 12,
    # A native crash in one model (espeak aborting the process) must not
    # make Modal replay the work.
    retries=0,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def synthesize(subset, model_name, samples=None, force=False):
    """Stage 1 — synthesise every sample for one model on one language."""
    _prepare_env(samples)
    from benchmark.evaluate import synthesize_language

    out = synthesize_language(subset, model_filter=model_name, device="cuda", force=force)
    results_volume.commit()
    return out


@app.function(
    image=cosy_image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    retries=0,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def synthesize_cosy(subset, model_name, samples=None, force=False):
    """Stage 1 for CosyVoice models, which need their own environment."""
    _prepare_env(samples)
    from benchmark.evaluate import synthesize_language

    out = synthesize_language(subset, model_filter=model_name, device="cuda", force=force)
    results_volume.commit()
    return out


@app.function(
    image=asr_image,
    # omniASR-LLM-7B is ~14 GB in bf16, which fits an A10G's 24 GB at
    # batch 1. Bigger cards need a payment method on this account.
    gpu="A10G",
    timeout=60 * 60 * 6,
    retries=0,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def score_gpu(subset, samples=None, force=False):
    """Stage 2 — transcribe and score one language with a local ASR judge."""
    _prepare_env(samples)
    from benchmark.evaluate import score_language

    out = score_language(subset, device="cuda", force=force)
    results_volume.commit()
    return [(r["model"], r["cer"]) for r in out]


@app.function(
    image=asr_image,
    timeout=60 * 60 * 6,
    retries=0,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def score_api(subset, samples=None, force=False):
    """Stage 2 for API judges — no GPU needed, just the network."""
    _prepare_env(samples)
    from benchmark.evaluate import score_language

    out = score_language(subset, device="cpu", force=force)
    results_volume.commit()
    return [(r["model"], r["cer"]) for r in out]



@app.function(
    image=asr_image,
    # The driver waits on every task, so it needs Modal's maximum: a 20h
    # cap cut the first full run off partway through stage 2.
    timeout=60 * 60 * 24 - 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def orchestrate(subsets, model=None, samples=None, force=False, stage="all"):
    """Fan the run out from inside Modal, one pipeline per language.

    Driving this from the local machine means a laptop OOM halfway through
    silently drops every task not yet submitted.  Languages are pipelined
    rather than staged globally, so a language is scored as soon as its own
    clips exist instead of waiting on the slowest model of some other
    language — the first full run timed out with 8 languages unscored.
    """
    sys.path.insert(0, "/workspace")
    from concurrent.futures import ThreadPoolExecutor

    from benchmark.asr import judge_for
    from benchmark.dataset import subset_to_iso
    from benchmark.evaluate import load_tts_models

    def run_language(subset):
        report = []
        if stage in ("all", "synth"):
            tasks = []
            for info in load_tts_models(subset):
                if model and model.lower() not in info["name"].lower():
                    continue
                fn = synthesize_cosy if info.get("runner") == "cosyvoice" else synthesize
                tasks.append((f"synth {subset} / {info['name']}", fn, (subset, info["name"])))
            report += _fan_out(tasks, samples=samples, force=force)

        if stage in ("all", "score"):
            spec = judge_for(subset_to_iso(subset))
            if spec is None:
                print(f"  no judge for {subset} — skipping", flush=True)
            else:
                fn = score_api if spec["kind"] == "khaya" else score_gpu
                report += _fan_out(
                    [(f"score {subset} [{spec['model']}]", fn, (subset,))],
                    samples=samples, force=force,
                )
        return report

    print(f"Running {len(subsets)} language pipeline(s), stage={stage}", flush=True)
    report = []
    with ThreadPoolExecutor(max_workers=len(subsets)) as pool:
        for language_report in pool.map(run_language, subsets):
            report += language_report
    return report


def _fan_out(tasks, samples=None, force=False, max_workers=16):
    """Run the tasks in parallel, reporting each as it finishes."""
    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        print("  nothing to run")
        return []
    report = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), max_workers)) as pool:
        futs = {
            pool.submit(fn.remote, *args, samples=samples, force=force): label
            for label, fn, args in tasks
        }
        for fut, label in futs.items():
            try:
                fut.result()
                print(f"  done {label}", flush=True)
                report.append((label, "ok"))
            except Exception as e:
                print(f"  FAILED {label}: {str(e)[:200]}", flush=True)
                report.append((label, str(e)[:200]))
    return report



@app.function(image=asr_image, timeout=60 * 30, volumes=VOLUMES)
def migrate_audio_layout(category="education", dry_run=True):
    """Move /audio/{iso}/{model}/ under /audio/{iso}/{category}/{model}/.

    Clips predate the per-domain layout; re-synthesising 8k of them to gain a
    path segment would be hours of GPU for nothing.
    """
    import shutil
    from pathlib import Path

    root = Path(AUDIO_SUBDIR)
    moved = []
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cat_dir = lang_dir / category
        for model_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            if model_dir.name == category:
                continue
            dest = cat_dir / model_dir.name
            moved.append(f"{model_dir} -> {dest}")
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(model_dir), str(dest))
        index = lang_dir / "TEXTS.txt"
        if index.exists() and not dry_run:
            cat_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(index), str(cat_dir / "TEXTS.txt"))
    if not dry_run:
        results_volume.commit()
    return moved



@app.function(
    image=asr_image,
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def publish_audio(repo_id="ghananlpcommunity/nsanku-tts-benchmark-audio", private=False):
    """Push the clips and transcriptions to a HF dataset repo.

    Uploading from the container rather than a laptop: the volume is already
    here, and 1.7 GB over a home connection is the slow way round.

    upload_large_folder, not upload_folder: 9000 small files committed one at
    a time is throttled into hours, and a dropped connection loses the lot.
    This uploads in parallel and resumes where it left off.
    """
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=RESULTS_DIR,
        repo_type="dataset",
        # The YAMLs live in git, and shards/ is dead weight from an earlier
        # layout; only the clips and transcriptions belong here.
        allow_patterns=["audio/**", "transcriptions/**"],
        num_workers=16,
        print_report=False,
    )
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    wavs = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".wav")]
    return {"repo": repo_id, "files": len(info.siblings or []), "clips": len(wavs)}


@app.local_entrypoint()
def main(
    langs: str = "",
    model: str = "",
    samples: int = 0,
    force: bool = False,
    stage: str = "all",
):
    """Kick off the run inside Modal, then report what landed on the Volume."""
    from benchmark.dataset import available_subsets

    subsets = [s.strip() for s in langs.replace(",", " ").split() if s.strip()] or available_subsets()
    if stage not in ("all", "synth", "score"):
        raise SystemExit(f"unknown stage {stage!r} — use all, synth or score")

    print(f"nsanku-TTS benchmark — {len(subsets)} language(s), stage={stage}, "
          f"model_filter={model or 'all'}, samples={samples or 'default'}")
    print("Driving the run from inside Modal; safe to lose this terminal.")

    report = orchestrate.remote(
        subsets, model=model or None, samples=samples or None, force=force, stage=stage,
    )
    ok = sum(1 for _, status in report if status == "ok")
    print(f"\n{ok} task(s) ok, {len(report) - ok} failed")
    for label, status in report:
        if status != "ok":
            print(f"  FAILED {label}: {status}")

    print("\nResults on volume:")
    try:
        for entry in results_volume.iterdir("/"):
            if entry.type == modal.volume.FileEntryType.DIRECTORY:
                continue
            print(f"  {entry.path} ({entry.size} bytes)")
    except Exception as e:
        print(f"  error listing volume: {e}")
    print("\nFetch them with:")
    print("  modal volume get nsanku-tts-results / --local-dir benchmarks/")
    print("  modal volume get nsanku-tts-results /audio audio/")
