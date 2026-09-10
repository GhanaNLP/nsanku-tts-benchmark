# nsanku-TTS Benchmark

TTS quality benchmark for Ghanaian languages, scored by **CTC forced-alignment**.

## How it works

1. **Text source**: 1000 sentences per language from [ghanaopenai/ghana-sentences](https://huggingface.co/datasets/ghanaopenai/ghana-sentences)
2. **TTS generation**: Each model synthesises every sentence to audio
3. **Alignment scoring**: [MMS-300M](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) CTC forced-alignment computes per-frame log-probability of the Viterbi path
4. **Ranking**: Mean alignment score (closer to 0 = better TTS quality)

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
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Asante` | F5-TTS | Asante Twi |
| `ghananlpcommunity/F5-TTS-OpenBible-Twi-Akuapem` | F5-TTS | Akuapem Twi |
| `ghananlpcommunity/F5-TTS-OpenBible-Ewe` | F5-TTS | Ewe |
| `ghananlpcommunity/nano-twi` | Matcha-TTS + Vocos | Asante Twi |

Models requiring IPA input (VoxCPM2-Ghana, stable-twi-tts) are excluded.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
cp .env.example .env   # set HF_TOKEN for gated models
```

## Usage

```bash
# Full benchmark (all languages, all models)
python pipeline.py

# Specific languages
python pipeline.py --langs twi_asante ewe dag

# Specific model
python run_benchmark.py --langs twi_asante --model "ghananlpcommunity/nano-twi"

# Dry run (list work without GPU)
python pipeline.py --dry-run
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
