#!/usr/bin/env bash
# Run a benchmark stage inside the prebuilt synthesis/scoring images.
#
# The images are the same ones benchmark 1 uses for HuggingFace Jobs, so local
# H200 results and published job results come from an identical environment.
# That matters here: SpeechBERTScore is only comparable across runs if the
# encoder stack is identical, and rebuilding a venv by hand silently drifts
# (the `tts` image ships torch 2.5.1, which transformers 4.57 refuses to load
# WavLM's .bin checkpoint from).
#
# Two images, not one, because the TTS stack and the encoder stack genuinely
# cannot coexist: voxcpm/f5-tts pin transformers to 4.57, and WavLM under
# transformers 4.57 on torch < 2.6 hits the CVE-2025-32434 torch.load guard.
#
# Usage:
#   scripts/h200_run.sh synth  python -m benchmark.evaluate synth --iso twi_asante --model X
#   scripts/h200_run.sh score  python -m benchmark.evaluate score-sbs --iso twi_asante
#   scripts/h200_run.sh asr    python -m benchmark.evaluate score-cer --iso twi_asante
#   scripts/h200_run.sh score  python -m benchmark.evaluate assemble --iso twi_asante
#   scripts/h200_run.sh verify python scripts/verify_scorer.py
set -euo pipefail

STAGE="${1:?usage: h200_run.sh <synth|omni|asr|score|verify|shell> [cmd...]}"
shift || true

REPO="${NSANKU_REPO:-/mnt/volume_d2wey28/projects/nsanku-tts-benchmark}"
IMAGES="${NSANKU_IMAGES:-ghcr.io/ghanaopenai/nsanku-tts-benchmark}"
HF_CACHE="${HF_HOME:-/mnt/volume_d2wey28/hf-cache}"
TMP="${NSANKU_TMP:-/mnt/volume_d2wey28/tmp}"

# `score` defaults to the locally built nsanku-score image. The published
# `asr` image is deliberately not used here: it is the fairseq2/omnilingual ASR
# judge and ships no transformers, so it cannot load WavLM at all.
SCORE_IMAGE="${NSANKU_SCORE_IMAGE:-nsanku-score:latest}"

case "$STAGE" in
  synth|shell) IMAGE="$IMAGES:tts" ;;
  omni) IMAGE="$IMAGES:omni" ;;
  asr) IMAGE="$IMAGES:asr" ;;
  score|verify) IMAGE="$SCORE_IMAGE" ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

mkdir -p "$HF_CACHE" "$TMP"

ENV_FILE_FLAG=()
if [ -f "$REPO/.env" ]; then
  ENV_FILE_FLAG=(--env-file "$REPO/.env")
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

# HF_TOKEN is read from the host environment and passed through rather than
# written into this repo. Gated checkpoints (F5-TTS, VoxCPM) need it.
exec docker run --rm --gpus all --ipc=host --shm-size=8g \
  --network=host \
  "${ENV_FILE_FLAG[@]}" \
  -v "$REPO:/app" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$TMP:/tmp/work" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e TORCH_HOME=/root/.cache/torch \
  -e TMPDIR=/tmp/work \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTHONUTF8=1 \
  -e PYTHONPATH=/app \
  -w /app \
  "$IMAGE" \
  "$@"
