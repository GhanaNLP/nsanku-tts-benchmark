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

1. **Samples**: 200 rows per language, spread across the language's text sources, from
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

## Languages (45, from ghana-speech-eval)

Every language that has recordings in ghana-speech-eval, an ASR judge in the
[nsanku ASR benchmark](https://github.com/GhanaNLP/nsanku-asr-benchmark) and at least one
TTS model. The sources are the config prefixes (bible, jw, finance, lds, unicef, waxal).
**When a language has several sources its 200 samples are split evenly across them**
(`allocate` in `benchmark/dataset.py`; a source that runs short hands its share to the
others), and results are also reported per source (`per_source` in each YAML row).
Sample keys are `<source>_<row>` (for example `jw_00012`).

| Code | Language | Text sources (ghana-speech-eval) |
|------|----------|-----------|
| ada | Dangme | bible, jw |
| dag | Dagbani | bible, unicef, waxal |
| dga | Dagaare | bible, jw, waxal |
| ewe | Ewe | bible, jw, unicef, waxal |
| fat | Fante | bible, finance, jw, lds |
| gaa | Ga | jw, finance |
| gjn | Gonja | bible |
| gur | Gurene | bible, jw |
| nzi | Nzema | bible, jw |
| twi_akuapem | Akuapem Twi | bible, finance |
| twi_asante | Asante Twi | bible, finance, lds, unicef, waxal |
| xsm | Kasem | bible |
| acd | Gikyode | bible |
| aha | Ahanta | jw |
| akp | Siwu | bible |
| any | Anyin | bible |
| avn | Avatime | bible |
| bib | Bissa | bible |
| bim | Bimoba | bible |
| biv | Southern Birifor | bible |
| bov | Tuwuli | bible |
| bud | Bassar (Ntcham) | bible |
| bwu | Buli | bible |
| ffm | Fulfulde (Maasina) | bible |
| hau | Hausa | bible |
| kbp | Kabiye | bible |
| kdh | Tem | bible |
| kma | Konni | bible |
| kpo | Ikposo | waxal |
| kus | Kusaal | bible |
| lef | Lelemi | bible |
| lip | Sekpele | bible |
| maw | Mampruli | bible |
| mzw | Deg | bible |
| naw | Nawuri | bible |
| ncu | Chumburung | bible |
| nko | Nkonya | bible |
| ntr | Ntrubo | bible |
| sfw | Sehwi | bible, jw |
| sig | Paasaal | bible |
| sil | Tumulung Sisaala | bible |
| snw | Selee | bible |
| tpm | Tampulma | bible |
| vag | Vagla | bible |
| xon | Konkomba | bible |

## TTS models (orthographic input unless noted)

The registry is `data/tts_models.json`; each model's per-language settings are in `recipes/`.

| Model | Architecture | Languages |
|-------|-------------|-----------|
| `ghananlpcommunity/ghana-tts-72k` | VoxCPM v1 (0.7B) | 43 languages |
| `ghananlpcommunity/ghana-tts-36k` | VoxCPM v1 (0.7B) | 43 languages |
| `FarmerlineML/voxcpm2-akan-sft` | VoxCPM2 (2B) | 2 languages |
| `FarmerlineML/voxcpm2-dagbani-sft` | VoxCPM2 (2B) | 1 language |
| `FarmerlineML/voxcpm2-ewe-sft` | VoxCPM2 (2B) | 1 language |
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Asante` | F5-TTS (DiT + Vocos) | 1 language |
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Akuapem` | F5-TTS (DiT + Vocos) | 1 language |
| `ghananlpcommunity/F5-TTS-OpenBible-Ewe` | F5-TTS (DiT + Vocos) | 1 language |
| `ghananlpcommunity/nano-twi` | Matcha-TTS + Vocos ONNX | 1 language |
| `ghananlpcommunity/stable-twi-tts` | Piper VITS (ONNX) | 2 languages |
| `multilingual-tts/VITS-OpenBible-Twi-Asante` | VITS (Coqui) | 1 language |
| `multilingual-tts/VITS-OpenBible-Twi-Akuapem` | VITS (Coqui) | 1 language |
| `multilingual-tts/VITS-OpenBible-Ewe` | VITS (Coqui) | 1 language |
| `k2-fsa/OmniVoice` | OmniVoice (voice cloning + voice design) | 45 languages |
| `KhayaAI/khaya-tts-v2` | Khaya AI TTS v2 API | 16 languages |
| `Sunbird/orpheus-3b-tts-multilingual` | Orpheus-3B (autoregressive LLM over SNAC 24 kHz) | 1 language |
| `Google/gemini-3.1-flash-tts-preview` | Google Gemini 3.1 Flash TTS (API) | 45 languages |
| `Google/gemini-3.1-flash-tts-preview-universal` | Google Gemini 3.1 Flash TTS (Universal Graphemes via africa-g2p) | 44 languages |
| `ghananlpcommunity/tekyerema-tts-ewe` | Tekyerema Ewe VITS (0.03B) | 1 language |
| `ghananlpcommunity/tekyerema-tts-twi` | Tekyerema Twi VITS (0.03B) | 2 languages |
| `walusungungulube/Spark-TTS-0.5B-twi-ewe-dagbani` | Spark-TTS (0.5B) | 4 languages |
| `facebook/mms-tts-<iso>` | Meta MMS VITS (per-language) | 36 checkpoints, one language each |

Candidates found on Hugging Face are listed by `scripts/scan_tts_models.py`; a model is added once it has a wrapper. Models requiring IPA input that we could not drive correctly are in `data/excluded_models.json`.

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

# Results from before multi-source sampling (bare row keys) -> <source>_<row>
python3 scripts/migrate_keys.py

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

1. Add its `{source: config}` entry to `SPEECH_EVAL_SOURCES`, and `ISO_TO_NAME`, in `benchmark/config.py`
2. Add an ASR judge in `data/asr_judges.json`
3. Add the language to the models that speak it in `data/tts_models.json`, then `python3 generate_recipes.py`
4. Run `python3 scripts/run_benchmark.py --iso <code>`

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
