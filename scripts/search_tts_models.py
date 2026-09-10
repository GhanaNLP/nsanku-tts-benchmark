#!/usr/bin/env python3
"""Search HuggingFace for TTS models supporting Ghanaian languages.

Writes discovered models to data/tts_models.json.
"""

import json
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = DATA_DIR / "tts_models.json"

# Ghanaian language tags on HuggingFace
GHANA_LANG_TAGS = [
    "twi", "akan", "ewe", "ga", "dagbani", "fante",
    "dangme", "dagaare", "gonja", "kasem", "nzema", "gurene",
    "ghanaian", "ghana",
]

GHANA_ORGS = ["ghananlpcommunity", "ghanaopenai", "Ghana-NLP", "KhayaAI"]


def search_tts_models():
    """Search HF for TTS models tagged with Ghanaian languages."""
    models = {}

    # Search by language tag
    for tag in GHANA_LANG_TAGS:
        url = f"https://huggingface.co/api/models?pipeline_tag=text-to-speech&search={tag}&limit=50"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for m in resp.json():
                mid = m.get("id", "")
                if mid not in models:
                    models[mid] = {
                        "name": mid,
                        "tags": m.get("tags", []),
                        "pipeline_tag": m.get("pipeline_tag"),
                    }
        except Exception as e:
            print(f"  Warning: search for '{tag}' failed: {e}", file=sys.stderr)

    # Search org repos
    for org in GHANA_ORGS:
        url = f"https://huggingface.co/api/models?author={org}&pipeline_tag=text-to-speech&limit=50"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for m in resp.json():
                mid = m.get("id", "")
                if mid not in models:
                    models[mid] = {
                        "name": mid,
                        "tags": m.get("tags", []),
                        "pipeline_tag": m.get("pipeline_tag"),
                    }
        except Exception as e:
            print(f"  Warning: search for org '{org}' failed: {e}", file=sys.stderr)

    return list(models.values())


def main():
    print("Searching HuggingFace for TTS models...")
    raw = search_tts_models()
    print(f"Found {len(raw)} candidate models")

    # Filter: must be text-to-speech, not require IPA, not a base model
    skip_keywords = ["ipa", "phoneme", "phonemizer", "g2p", "base", "pretrained"]
    accepted = []
    for m in raw:
        name = m["name"].lower()
        tags = [t.lower() for t in m.get("tags", [])]

        # skip non-TTS
        if m.get("pipeline_tag") != "text-to-speech":
            continue

        # skip IPA-only models
        if any(kw in name for kw in skip_keywords):
            continue

        accepted.append(m)

    # Merge with existing hand-curated list
    existing = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for m in json.load(f):
                existing[m["name"]] = m

    for m in accepted:
        if m["name"] not in existing:
            existing[m["name"]] = {
                "name": m["name"],
                "architecture": "unknown",
                "languages": [],
                "input_type": "orthographic",
                "license": "",
                "url": f"https://huggingface.co/{m['name']}",
                "notes": "auto-discovered",
            }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(list(existing.values()), f, indent=2)
    print(f"Wrote {len(existing)} models to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
