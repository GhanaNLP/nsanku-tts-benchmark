#!/usr/bin/env python3
"""Scan HuggingFace for organization-published TTS models in our languages.

Discovery is by language tag rather than free-text search: a model that
declares it speaks Ewe is a candidate, whatever it is called. Personal
accounts are dropped (see ORG_ONLY) — a leaderboard is a claim about what is
available to build on, and a personal checkpoint is not that kind of artefact.

Candidates are printed for review, not auto-registered: adding a model means
knowing how to drive it, which is a wrapper and a recipe, not a JSON line.

Run:  python3 scripts/scan_tts_models.py
      python3 scripts/scan_tts_models.py --json candidates.json
"""

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.config import ISO_TO_NAME, ORG_OVERRIDES, SUBSET_TO_ISO  # noqa: E402

API = "https://huggingface.co/api"

# "ga" is ISO 639-1 for Irish, but publishers routinely use it for Ga. A match
# on it alone pulls in every multilingual giant, so it counts only when the
# model also looks Ghanaian — another of our language tags, or a Ghana tag.
AMBIGUOUS_TAGS = {"ga"}
CORROBORATING_TAGS = {
    "ghana", "ghanaian", "akan", "twi", "asante", "akuapem", "ewe", "dagbani",
    "dangme", "adangme", "dagaare", "fante", "gonja", "kasem", "nzema",
    "gurene", "frafra", "gaa", "aka", "tw", "ee", "dag", "dga", "fat", "gjn",
    "xsm", "nzi", "gur", "ada",
}


def corroborated(model_tags):
    """True if a model tagged `ga` also looks Ghanaian."""
    tags = {t.lower() for t in model_tags}
    return bool((tags & CORROBORATING_TAGS) - AMBIGUOUS_TAGS)

# HF language tags to look under, per language we evaluate. ISO 639-3 is the
# tag most models use; the aliases are what the rest actually use in practice.
LANG_TAGS = {
    "twi_akuapem": ["tw", "ak", "aka", "twi", "akuapem"],
    "twi_asante": ["tw", "ak", "aka", "twi", "asante"],
    "ada": ["ada", "adangme", "dangme"],
    "dag": ["dag", "dagbani"],
    "dga": ["dga", "dagaare"],
    "ewe": ["ee", "ewe"],
    "fat": ["fat", "fante"],
    "gaa": ["gaa", "ga"],   # "ga" is ambiguous — see AMBIGUOUS_TAGS
    "gjn": ["gjn", "gonja"],
    "gur": ["gur", "gurune", "frafra"],
    "nzi": ["nzi", "nzema"],
    "xsm": ["xsm", "kasem"],
}


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException as e:
        print(f"  warning: {url} failed ({e})", file=sys.stderr)
    return None


def is_org(owner, cache):
    """True if *owner* is an HF organization (or one we treat as one)."""
    if owner in ORG_OVERRIDES:
        return True
    if owner not in cache:
        cache[owner] = _get(f"{API}/organizations/{owner}/overview") is not None
    return cache[owner]


def _excluded():
    """Models deliberately not benchmarked, and why."""
    path = ROOT / "data" / "excluded_models.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["excluded"]


def scan():
    registered = {
        m["name"] for m in json.loads((ROOT / "data" / "tts_models.json").read_text())
    }
    excluded = _excluded()
    org_cache = {}
    found = {}

    for iso in sorted(set(SUBSET_TO_ISO.values())):
        for tag in LANG_TAGS.get(iso, [iso]):
            models = _get(
                f"{API}/models",
                {"filter": tag, "pipeline_tag": "text-to-speech", "limit": 100},
            ) or []
            for m in models:
                mid = m.get("id", "")
                owner = mid.split("/")[0] if "/" in mid else ""
                if not owner or not is_org(owner, org_cache):
                    continue
                if tag in AMBIGUOUS_TAGS and not corroborated(m.get("tags", [])):
                    continue
                entry = found.setdefault(mid, {
                    "name": mid,
                    "owner": owner,
                    "languages": set(),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": m.get("tags", []),
                    "registered": mid in registered,
                    "excluded": excluded.get(mid),
                })
                entry["languages"].add(iso)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="also write the candidates to this file")
    args = ap.parse_args()

    found = scan()
    new = {k: v for k, v in found.items()
           if not v["registered"] and not v.get("excluded")}
    known = {k: v for k, v in found.items() if v["registered"]}
    skipped = {k: v for k, v in found.items() if v.get("excluded")}

    print(f"\n{len(found)} org-published TTS model(s) tagged with our languages "
          f"({len(known)} already benchmarked, {len(new)} new)\n")
    for mid, v in sorted(new.items(), key=lambda kv: -kv[1]["downloads"]):
        langs = ", ".join(sorted(ISO_TO_NAME.get(i, i) for i in v["languages"]))
        print(f"  NEW  {mid}")
        print(f"       {langs}  ·  {v['downloads']} downloads, {v['likes']} likes")
    if skipped:
        print("\n  deliberately excluded:")
        for mid, v in sorted(skipped.items()):
            print(f"       {mid} — {v['excluded']}")
    if known:
        print("\n  already benchmarked:")
        for mid in sorted(known):
            print(f"       {mid}")

    if args.json:
        out = {k: {**v, "languages": sorted(v["languages"])} for k, v in found.items()}
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
