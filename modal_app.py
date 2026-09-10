"""Modal entrypoint for nsanku-TTS benchmark.

Runs the full benchmark (all ghana-sentences subsets x all fitted TTS models)
on Modal GPUs, persisting incremental per-language YAML results and the HF
model cache on shared Volumes.

Usage:
    modal run modal_app.py                        # all languages (incremental)
    modal run modal_app.py --langs dag ewe        # specific languages
    modal run modal_app.py --model f5             # filter models by substring
    modal run modal_app.py --samples 500          # bump sample count (incremental)

The workspace must be `ghana-nlp3`:
    modal profile use ghana-nlp3   (sets MODAL_ENVIRONMENT / workspace)

Secrets (Modal secret named "nsanku-khaya"):
    HF_TOKEN       — for gated models (ghana-tts-72k)
    KHAYA_API_KEY  — for the Khaya TTS API
"""

import os

import modal

app = modal.App("nsanku-tts-benchmark")

# ── Volumes ──────────────────────────────────────────────────────────────────
# Results persist between runs so bumping sample counts is incremental.
results_volume = modal.Volume.from_name("nsanku-tts-results", create_if_missing=True)
# HF cache survives across runs (aligner + TTS weights reuse).
hf_cache_volume = modal.Volume.from_name("nsanku-tts-hf-cache", create_if_missing=True)

RESULTS_DIR = "/results"
HF_HOME = "/hf-cache"

# ── Image ────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5,<2.6",
        "torchaudio>=2.5,<2.6",
    )
    .pip_install(
        "voxcpm",
        "transformers>=4.40.0",
        "datasets>=3.0.0",
        "pyyaml>=6.0",
        "numpy",
        "requests",
        "huggingface_hub>=0.26.0",
        "soundfile>=0.12.0",
        "f5-tts",
        "sherpa-onnx",
    )
)


@app.function(
    image=image,
    gpu=modal.gpu.H100(memory=80),
    timeout=60 * 60 * 4,
    volumes={RESULTS_DIR: results_volume, HF_HOME: hf_cache_volume},
    secrets=[modal.Secret.from_name("nsanku-khaya", required=False)],
    mounts=[modal.Mount.from_local_dir(".", remote_path="/repo")],
    workdir="/repo",
)
def evaluate_subset(subset, model_filter=None, samples=None, force=False):
    """Evaluate one language subset (all fitted models) on a GPU."""
    os.environ["HF_HOME"] = HF_HOME
    os.environ["NSANKU_TTS_RESULTS_DIR"] = RESULTS_DIR
    if samples:
        os.environ["NSANKU_TTS_NUM_SAMPLES"] = str(samples)

    from benchmark.evaluate import evaluate_language

    return evaluate_language(
        subset,
        model_filter=model_filter,
        device="cuda",
        force=force,
    )


@app.local_entrypoint()
def main(
    langs: list[str] = None,
    model: str = None,
    samples: int = None,
    force: bool = False,
):
    """Run the benchmark, then report results left on the shared Volume."""
    from benchmark.dataset import available_subsets

    subsets = langs or available_subsets()
    print(f"nsanku-TTS benchmark on Modal — {len(subsets)} language(s), "
          f"model_filter={model or 'all'}, samples={samples or 'default'}")

    for subset in subsets:
        evaluate_subset.remote(subset, model_filter=model, samples=samples, force=force)

    results_volume.reload()
    total = 0
    print("\nResults on volume:")
    for entry in results_volume.iterdir("/"):
        if entry.is_dir():
            continue
        total += entry.size
        print(f"  {entry.path} ({entry.size} bytes)")
    if total == 0:
        print("  (empty — no benchmark YAMLs produced yet)")
    print("\nCommit them locally with:")
    print("  modal volume get nsanku-tts-results / --local-dir benchmarks/")


def sync_results_local():
    """Copy volume YAMLs into ./benchmarks for git commit."""
    modal.Volume.from_name("nsanku-tts-results").get(
        remote_path="/", local_dir="benchmarks"
    )