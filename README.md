# nsanku-TTS Benchmark

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Leaderboard-HF%20Space-blue)](https://huggingface.co/spaces/ghananlpcommunity/nsanku-tts-benchmark)

TTS benchmark for Ghanaian languages. Every synthesised clip is graded two ways and
ranked by the **composite** `(SpeechBERTScore + (1 - CER)) / 2` (higher is better):

- **CER** (intelligibility): an ASR judge transcribes the clip; the transcript is
  compared with the sentence that was synthesised.
- **SpeechBERTScore** (acoustic similarity): the clip is compared with a real human
  recording of the same sentence in a frozen SSL encoder's feature space
  (`microsoft/wavlm-large`, layer 6, precision variant; see `docs/encoder-selection.md`).

This repository is the single source of truth: running the benchmark writes both metrics
into `benchmarks/{iso}.yaml`, and the [leaderboard Space](https://huggingface.co/spaces/ghananlpcommunity/nsanku-tts-benchmark)
reads those files. (It merges the former *nsanku-tts-benchmark* CER benchmark and
*nsanku-tts-benchmark-2*; the original CER-only results are archived in
`data/legacy_benchmark1/`.)

## How it works

1. **Samples**: 200 rows per language from
   [ghana-speech-eval](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
   (recorded speech + transcript). Each row is a sentence *and* its reference recording.
2. **Synthesis** (stage 1): each model synthesises every sentence; the clips are kept.
   Models whose recommended setting includes a reference clip are prompted with a
   *different* row of the same language (`PROMPT_ROW_OFFSET`), never the recording they
   are scored against. Those rows are marked `-ref` on the leaderboard.
3. **CER** (stage 2a): each clip is transcribed by the lowest-CER ASR model for that
   language (`data/asr_judges.json`, from the
   [nsanku ASR benchmark](https://github.com/GhanaNLP/nsanku-asr-benchmark)). What the
   judge heard is kept in `benchmarks/{iso}.cer.json`.
4. **SpeechBERTScore** (stage 2b): the clip is scored against the row's real recording.
5. **Assemble** (stage 3): both metrics are joined into `benchmarks/{iso}.yaml`. A model
   with only one metric is listed with `partial: true`.

CER uses the same normalisation as the ASR benchmark, and each judge's own CER on real
speech is recorded alongside the results: a TTS model cannot meaningfully score below its
judge's error rate.

Samples are keyed by row index, so runs are **incremental**: raising the sample count
only scores new rows. Stages are split because the TTS models, the ASR judges and the
encoder cannot share an environment (see `scripts/h200_run.sh`).

Every (model, language) evaluation has its own **recipe** under `recipes/`, holding
the settings used to synthesise that language with that model — guidance scale, step
counts, the reference clip's transcript, the API language code. Editing one language
cannot disturb another, and the leaderboard links each row to its recipe. Regenerate
missing ones with `python3 generate_recipes.py`.

## Languages (12, from ghana-speech-eval)

| Code | Language | ghana-speech-eval config |
|------|----------|--------|
| ada | Dangme | bible_Dangme_ada |
| dag | Dagbani | bible_Dagbani_dag |
| dga | Dagaare | bible_Dagaare_dga |
| ewe | Ewe | bible_Ewe_ewe |
| fat | Fante | bible_Fante_fat |
| gaa | Ga | jw_ga_gaa |
| gjn | Gonja | bible_Gonja_gjn |
| gur | Gurene | bible_Ninkare_gur |
| nzi | Nzema | bible_Nzema_nzi |
| twi_akuapem | Akuapem Twi | bible_Akuapem_Twi |
| twi_asante | Asante Twi | bible_Asante_Twi |
| xsm | Kasem | bible_Kasem_xsm |

## TTS models (orthographic input only)

| Model | Architecture | Languages |
|-------|-------------|-----------|
| `ghananlpcommunity/ghana-tts-72k` | VoxCPM v1 (0.7B) | 44 langs |
| `ghananlpcommunity/ghana-tts-36k` | VoxCPM v1 (0.7B) | 41 langs |
| `techolise/akan-twi-speaker17-tts-v2` | VoxCPM v1 + LoRA | Asante Twi |
| `FarmerlineML/voxcpm2-akan-sft` | VoxCPM2 (2B) | Akan |
| `FarmerlineML/voxcpm2-dagbani-sft` | VoxCPM2 (2B) | Dagbani |
| `FarmerlineML/voxcpm2-ewe-sft` | VoxCPM2 (2B) | Ewe |
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Asante` | F5-TTS | Asante Twi |
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Akuapem` | F5-TTS | Akuapem Twi |
| `ghananlpcommunity/F5-TTS-OpenBible-Ewe` | F5-TTS | Ewe |
| `ghananlpcommunity/nano-twi` | Matcha-TTS + Vocos | Asante Twi |
| `KhayaAI/khaya-tts-v2` | Khaya AI TTS v2 API (hosted) | 32 langs/dialects |

Models requiring IPA input (VoxCPM2-Ghana, stable-twi-tts) are excluded.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
cp .env.example .env   # set HF_TOKEN (gated models) and KHAYA_API_KEY
```

## Usage

Each stage runs in its own container image (`scripts/h200_run.sh`):

```bash
# One language, every stage (synth -> omni synth -> CER -> SBS -> assemble)
python3 scripts/run_benchmark.py --iso twi_asante
python3 scripts/run_benchmark.py --all-languages

# Individual stages
scripts/h200_run.sh synth python -m benchmark.evaluate synthesize --iso ewe --model voxcpm
scripts/h200_run.sh asr   python -m benchmark.evaluate score-cer  --iso ewe
scripts/h200_run.sh score python -m benchmark.evaluate score-sbs  --iso ewe
scripts/h200_run.sh score python -m benchmark.evaluate assemble   --iso ewe

# Cross-language leaderboard in the terminal
PYTHONPATH=. python3 scripts/leaderboard.py

# Tests (no GPU, no network)
PYTHONPATH=. python3 scripts/test_pipeline.py && PYTHONPATH=. python3 scripts/test_metric.py
```

`python -m benchmark.evaluate score` runs score-cer, score-sbs and assemble in one
process for environments that have both stacks installed.

## Leaderboard

The HF Space [ghananlpcommunity/nsanku-tts-benchmark](https://huggingface.co/spaces/ghananlpcommunity/nsanku-tts-benchmark)
(source in `space/`) reads `benchmarks/*.yaml` from this repo. Ranking is by composite score.

## Adding a new language

1. Add the language's config to `SPEECH_EVAL_CONFIGS`, `ISO_TO_NAME` and `TTS_LANG_MAP` in `benchmark/config.py`
2. Add an ASR judge in `data/asr_judges.json`
3. Run `python3 scripts/run_benchmark.py --iso <code>`

## Adding a new TTS model

Add an entry to `data/tts_models.json`:

```json
{
  "name": "org/model-id",
  "architecture": "Architecture Name",
  "languages": ["twi_asante", "ewe"],
  "input_type": "orthographic",
  "license": "MIT",
  "url": "https://huggingface.co/org/model-id",
  "notes": "Description"
}
```

If the model needs a custom wrapper, add a class in `benchmark/models.py`.

## Project structure

```
nsanku-tts-benchmark/
├── benchmark/      Core library (config, dataset, models, asr, metrics, speechbertscore, recipes, evaluate)
├── recipes/        One synthesis recipe per (model, language)
├── benchmarks/     Per-language results: {iso}.yaml (published), {iso}.scores.json / {iso}.cer.json (caches)
├── data/           Model registry, ASR judges; legacy_benchmark1/ = archived CER-only results
├── docs/           Methodology and encoder selection
├── space/          HF Space leaderboard (static HTML)
├── scripts/        Runner, leaderboard, tests, model search utilities
├── docker/         Images for the score / asr / tts stages
└── legacy/         Pre-merge runners (Modal, HF Jobs); target the old API
```
