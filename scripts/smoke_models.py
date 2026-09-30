#!/usr/bin/env python3
"""Synthesize one sample per model to find what actually works.

The registry lists 23 entries, but an entry being in a JSON file says nothing
about whether its wrapper still runs against the current library versions. A
full 12-language sweep costs hours, so it is worth spending a few minutes
finding out which entries are broken before committing to one.

Reports each model as OK or the exception it raised. A model that fails on the
first sample usually fails on all 200, so a one-sample smoke test is a sound
filter -- it is not evidence about quality, only about whether the code path
runs at all.

Usage:
    python scripts/smoke_models.py --iso twi_asante
    python scripts/smoke_models.py --iso twi_asante --only voxcpm omni
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def registry_models(iso):
    """Registry entries that claim to support iso."""
    data = json.loads((REPO / "data" / "tts_models.json").read_text())
    entries = data["models"] if isinstance(data, dict) and "models" in data else data
    out = []
    for m in entries:
        langs = m.get("languages") or []
        if isinstance(langs, str):
            langs = [langs]
        if iso in langs:
            out.append(m)
    return out


def model_id(m):
    return m.get("name") or m.get("model_id") or m.get("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", default="twi_asante")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on model ids")
    ap.add_argument("--results", default="smoke_results.json")
    args = ap.parse_args()

    models = registry_models(args.iso)
    if args.only:
        models = [m for m in models
                  if any(s.lower() in json.dumps(m).lower() for s in args.only)]
    print(f"smoke: {len(models)} models for {args.iso}\n")

    results = []
    for i, m in enumerate(models, 1):
        name = model_id(m)
        print(f"[{i}/{len(models)}] {name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run(
                [sys.executable, "-m", "benchmark.evaluate", "synthesize",
                 "--iso", args.iso, "--model", name, "--limit", "1"],
                capture_output=True, text=True, timeout=1800, cwd=REPO)
            ok = p.returncode == 0
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
            results.append({"model": name, "ok": ok, "secs": round(time.time() - t0, 1),
                            "detail": tail[0][:200]})
            print(("OK   " if ok else "FAIL ") + f"{time.time()-t0:.0f}s  {'' if ok else tail[0][:120]}")
        except subprocess.TimeoutExpired:
            results.append({"model": name, "ok": False, "secs": 1800,
                            "detail": "timeout >30min"})
            print("TIMEOUT")
        except Exception as e:
            results.append({"model": name, "ok": False, "secs": 0,
                            "detail": f"{type(e).__name__}: {e}"[:200]})
            print(f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()

    Path(args.results).write_text(json.dumps(results, indent=2))
    good = [r for r in results if r["ok"]]
    print(f"\n{len(good)}/{len(results)} models synthesised successfully")
    if len(good) < len(results):
        print("\nfailures:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['model']}\n      {r['detail']}")


if __name__ == "__main__":
    main()
