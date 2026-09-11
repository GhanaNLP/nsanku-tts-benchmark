# nsanku-TTS Benchmark

TTS intelligibility benchmark for Ghanaian languages, scored by **ASR character error rate**.

## How it works

1. **Text source**: 200 sentences per language from [ghanaopenai/ghana-sentences](https://huggingface.co/datasets/ghanaopenai/ghana-sentences) (default; bump via `NSANKU_TTS_NUM_SAMPLES`)
2. **Synthesis** (stage 1): each model synthesises every sentence; the clips are kept
3. **ASR scoring** (stage 2): each clip is transcribed by the lowest-CER ASR model for
   that language, taken from the [nsanku ASR benchmark](https://github.com/GhanaNLP/nsanku-asr-benchmark)
   (see `data/asr_judges.json`), and compared to the sentence it was asked to read
4. **Ranking**: mean character error rate — **lower is better**

Scores are reported **per text domain** and averaged with equal weight, so a model
that handles one register well is not credited for another. The first domain is
`education` (textbook prose); more will be added. Domains are registered in
`data/eval_configs.json`.

Every (model, language) evaluation has its own **recipe** under `recipes/`, holding
the settings used to synthesise that language with that model — guidance scale, step
counts, the reference clip's transcript, the API language code. Editing one language
cannot disturb another, and the leaderboard links each row to its recipe. Regenerate
missing ones with `python3 generate_recipes.py`.

Per-sample references and what the judge heard are written to
`transcriptions/{iso}_{category}_{model}.csv`, so every score can be checked line by
line rather than taken on trust.

CER is computed with the same normalisation as the ASR benchmark, so a TTS score and
an ASR score for a language are directly comparable. Each judge's own CER on real
speech is recorded alongside the results: a TTS model cannot meaningfully score below
its judge's error rate.

Synthesis and scoring are split because the judges cannot share an environment with
the TTS models (omniASR pins torch 2.8 via fairseq2; voxcpm/f5-tts need torch 2.5),
and because re-scoring should never mean re-synthesising.

Samples are keyed by their row index in the ghana-sentences subset, so runs are
**incremental**: raising the sample count only scores the *new* sentences and
reuses previously scored ones.

## Languages (12, from ghana-sentences)

| Code | Language | Subset |
|------|----------|--------|
| ada | Dangme | ada |
| dag | Dagbani | dag |
| dga | Dagaare | dga |
| ewe | Ewe | ewe |
| fat | Fante | fat |
| gaa | Ga | gaa |
| gjn | Gonja | gjn |
| gur | Gurene | gur |
| nzi | Nzema | nzi |
| twi_akuapem | Akuapem Twi | twi-aku |
| twi_asante | Asante Twi | twi-asa |
| xsm | Kasem | xsm |

All 43 nsanku-asr-benchmark language codes are pre-registered for easy expansion.

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

```bash
# Full benchmark (all languages, all models)
python pipeline.py

# Specific languages
python pipeline.py --langs twi-asa ewe dag

# Specific model
python run_benchmark.py --langs twi-asa --model "KhayaAI/khaya-tts-v2"

# Bump samples (only new ones are scored — incremental)
NSANKU_TTS_NUM_SAMPLES=500 python pipeline.py

# Dry run (list work without GPU)
python pipeline.py --dry-run
```

## Running on Modal

The benchmark runs on Modal GPUs (workspace `ghana-nlp3`). Results persist on a
shared Volume so re-runs are incremental:

```bash
modal profile use ghana-nlp3                  # or MODAL_ENVIRONMENT=ghana-nlp3
modal secret create nsanku-khaya HF_TOKEN=... KHAYA_API_KEY=...   # once
modal run modal_app.py                        # all languages
modal run modal_app.py --langs dag ewe        # specific languages
modal run modal_app.py --samples 500          # incremental sample bump
modal volume get nsanku-tts-results / --local-dir benchmarks/   # pull YAMLs
```

## Leaderboard

The HF Space at [ghananlpcommunity/nsanku-tts-benchmark](https://huggingface.co/spaces/ghananlpcommunity/nsanku-tts-benchmark) reads `benchmarks/*.yaml` from this repo and renders a per-language leaderboard.

## Adding a new language

1. Add sentences to [ghanaopenai/ghana-sentences](https://huggingface.co/datasets/ghanaopenai/ghana-sentences)
2. Add a subset mapping in `benchmark/config.py` (`SUBSET_TO_ISO`)
3. The language is already pre-registered in `languages/ghana_languages.yaml`
4. Run `python pipeline.py --langs <subset>`

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
├── benchmark/          Core library (config, dataset, models, asr, metrics, recipes, evaluate)
├── recipes/            One synthesis recipe per (model, language)
├── transcriptions/     Per-sample judge output behind every score
├── benchmarks/         Per-language YAML results
├── languages/          Language metadata (ghana_languages.yaml)
├── data/               Model registry (tts_models.json)
├── space/              HF Space leaderboard (static HTML)
├── scripts/            Utilities (search_tts_models.py, requirements.txt)
├── pipeline.py         Full benchmark CLI
└── run_benchmark.py    Targeted benchmark CLI
```
