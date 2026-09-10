# nsanku-TTS Benchmark

TTS quality benchmark for Ghanaian languages, scored by **CTC forced-alignment**.

## How it works

1. **Text source**: 200 sentences per language from [ghanaopenai/ghana-sentences](https://huggingface.co/datasets/ghanaopenai/ghana-sentences) (default; bump via `NSANKU_TTS_NUM_SAMPLES`)
2. **TTS generation**: Each model synthesises every sentence to audio
3. **Alignment scoring**: [MMS-300M](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) CTC forced-alignment computes per-frame log-probability of the Viterbi path
4. **Ranking**: Mean alignment score (closer to 0 = better TTS quality)

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

All 43 nsanku-ASR language codes are pre-registered for easy expansion.

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
├── benchmark/          Core library (config, dataset, models, alignment, evaluate)
├── benchmarks/         Per-language YAML results
├── languages/          Language metadata (ghana_languages.yaml)
├── data/               Model registry (tts_models.json)
├── space/              HF Space leaderboard (static HTML)
├── scripts/            Utilities (search_tts_models.py, requirements.txt)
├── pipeline.py         Full benchmark CLI
└── run_benchmark.py    Targeted benchmark CLI
```
